import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from critic_adam_component_ablation import _clone_adam_hparams
from critic_loss_component_ablation import (
    _build_payloads_from_capture_and_rollout,
    _flatten_tensors,
    _parameter_list_delta_norm,
    _semantic_metrics_from_payload,
)
from critic_optimizer_rule_ablation import (
    _assign_copied_critic_grads,
    _collect_single_rollout_epoch,
)
from dense_actor_input_experiment import build_state_layout
from dense_policy_ratio_mode_experiment import build_policy_ratio_mode_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from joint_training_critic_drift_trace import (
    MODE,
    _collect_fixed_probe_payload,
    _find_current_mainline_checkpoint,
)
from rl.ppo_agent import PPOAgent, CriticOptimizerAblationCaptured
from simulator.simulator import Simulator


DEFAULT_EPOCH = 0
DEFAULT_UPDATE_EPOCH = 2
DEFAULT_MINIBATCH_ID = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose the captured bad Adam step into raw-grad, momentum, preconditioned, "
            "and full-Adam geometry on the same checkpoint / rollout / held-out / probe."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument("--update-epoch", type=int, default=DEFAULT_UPDATE_EPOCH)
    parser.add_argument("--minibatch-id", type=int, default=DEFAULT_MINIBATCH_ID)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    return parser.parse_args()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(torch.dot(left, right).item() / (left_norm * right_norm))


def _evaluate_payload(
    agent: PPOAgent,
    payload: dict[str, torch.Tensor],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, float]:
    value_targets = agent._value_targets_from_returns_with_stats(
        payload["returns"],
        target_mean,
        target_std,
    )
    critic_inputs, _, _ = agent._prepare_critic_inputs(
        payload["states"],
        update_stats=False,
    )
    with torch.no_grad():
        value_predictions = agent._trace_raw_value_predictions_from_critic_inputs(
            critic_inputs,
            target_mean,
            target_std,
        )
        semantics = agent._critic_trace_semantic_summary(
            value_predictions,
            value_targets,
            payload["returns"],
            payload["advantages"],
            payload["rewards"],
        )
        critic_features = agent.network.critic_features_from_critic_input(critic_inputs)
        normalized_values = agent.network.critic_head(critic_features).squeeze(-1)
        if agent.config.value_target_mode == "popart_return_norm":
            critic_predictions = normalized_values
        else:
            raw_values = agent.network.value_from_critic_input(critic_inputs).squeeze(-1)
            if agent.config.value_target_mode in {"normalized_return", "running_return_norm"}:
                critic_predictions = (raw_values - target_mean) / (target_std + 1e-8)
            else:
                critic_predictions = raw_values
        critic_loss = agent._compute_value_loss(critic_predictions, value_targets)
    return {
        "critic_loss": float(critic_loss.item()),
        **semantics,
    }


def _evaluate_sets(
    agent: PPOAgent,
    payloads: dict[str, dict[str, torch.Tensor]],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, dict[str, float]]:
    return {
        set_type: _evaluate_payload(agent, payload, target_mean, target_std)
        for set_type, payload in payloads.items()
    }


def _parameter_delta_norms(
    base_agent: PPOAgent,
    updated_agent: PPOAgent,
) -> dict[str, float]:
    return {
        "critic_param_delta_norm": _parameter_list_delta_norm(
            [param.detach().clone() for param in base_agent.critic_params],
            [param.detach().clone() for param in updated_agent.critic_params],
        ),
        "critic_backbone_delta_norm": _parameter_list_delta_norm(
            [param.detach().clone() for param in base_agent.network.critic_backbone.parameters()],
            [param.detach().clone() for param in updated_agent.network.critic_backbone.parameters()],
        ),
        "critic_head_delta_norm": _parameter_list_delta_norm(
            [param.detach().clone() for param in base_agent.network.critic_head.parameters()],
            [param.detach().clone() for param in updated_agent.network.critic_head.parameters()],
        ),
    }


def _build_adam_geometry_directions(
    agent_with_state: PPOAgent,
    base_grads: list[torch.Tensor | None],
) -> dict[str, list[torch.Tensor | None]]:
    group = agent_with_state.critic_optimizer.param_groups[0]
    beta1, beta2 = group["betas"]
    eps = float(group["eps"])
    weight_decay = float(group.get("weight_decay", 0.0))

    raw_direction: list[torch.Tensor | None] = []
    momentum_direction: list[torch.Tensor | None] = []
    preconditioned_direction: list[torch.Tensor | None] = []
    full_adam_direction: list[torch.Tensor | None] = []

    for parameter, grad in zip(agent_with_state.critic_params, base_grads):
        if grad is None:
            raw_direction.append(None)
            momentum_direction.append(None)
            preconditioned_direction.append(None)
            full_adam_direction.append(None)
            continue

        grad_for_adam = grad.detach().clone()
        if weight_decay != 0.0:
            grad_for_adam = grad_for_adam + weight_decay * parameter.detach()

        state = agent_with_state.critic_optimizer.state[parameter]
        exp_avg = state["exp_avg"].detach().clone()
        exp_avg_sq = state["exp_avg_sq"].detach().clone()
        step_value = state["step"]
        if torch.is_tensor(step_value):
            step_value = int(step_value.item())
        else:
            step_value = int(step_value)
        next_step = step_value + 1

        exp_avg_t = exp_avg * beta1 + grad_for_adam * (1.0 - beta1)
        exp_avg_sq_t = exp_avg_sq * beta2 + grad_for_adam.square() * (1.0 - beta2)
        bias_correction1 = 1.0 - beta1 ** next_step
        bias_correction2 = 1.0 - beta2 ** next_step
        m_hat = exp_avg_t / bias_correction1
        v_hat = exp_avg_sq_t / bias_correction2
        denom = v_hat.sqrt() + eps

        raw_direction.append(grad.detach().clone())
        momentum_direction.append(m_hat.detach().clone())
        preconditioned_direction.append((grad_for_adam / denom).detach().clone())
        full_adam_direction.append((m_hat / denom).detach().clone())

    return {
        "raw_grad_direction": raw_direction,
        "momentum_direction": momentum_direction,
        "preconditioned_grad_direction": preconditioned_direction,
        "full_adam_direction": full_adam_direction,
    }


def _flat_direction(
    tensors: list[torch.Tensor | None],
) -> torch.Tensor:
    return _flatten_tensors(tensors)


def _flat_backbone_direction(
    agent: PPOAgent,
    direction_tensors: list[torch.Tensor | None],
) -> torch.Tensor:
    count = len(list(agent.network.critic_backbone.parameters()))
    return _flatten_tensors(direction_tensors[:count])


def _flat_head_direction(
    agent: PPOAgent,
    direction_tensors: list[torch.Tensor | None],
) -> torch.Tensor:
    backbone_count = len(list(agent.network.critic_backbone.parameters()))
    head_count = len(list(agent.network.critic_head.parameters()))
    return _flatten_tensors(direction_tensors[backbone_count : backbone_count + head_count])


def _apply_direction_with_target_norm(
    base_agent: PPOAgent,
    direction_tensors: list[torch.Tensor | None],
    target_update_norm: float,
) -> tuple[PPOAgent, dict[str, float]]:
    updated_agent = copy.deepcopy(base_agent)
    direction_vector = _flat_direction(direction_tensors)
    direction_norm = float(direction_vector.norm().item())
    scale = 0.0 if direction_norm <= 1e-12 else float(target_update_norm) / direction_norm
    with torch.no_grad():
        for parameter, direction in zip(updated_agent.critic_params, direction_tensors):
            if direction is None:
                continue
            parameter.add_(direction, alpha=-scale)
    return updated_agent, {
        **_parameter_delta_norms(base_agent, updated_agent),
        "grad_norm": float(direction_norm),
        "update_over_grad_ratio": float(target_update_norm / max(direction_norm, 1e-12)),
        "scale_factor": float(scale),
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Critic Optimizer Geometry Audit",
        "",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        (
            f"- captured minibatch: epoch `{summary['target_epoch']}` / update_epoch "
            f"`{summary['target_update_epoch']}` / minibatch `{summary['target_minibatch_id']}`"
        ),
        f"- target update norm: `{summary['target_update_norm']}`",
        f"- actual Adam update norm: `{summary['actual_adam_update_norm']}`",
        f"- interpretation: {summary['interpretation']}",
        "",
        "## Direction Cosines",
        "",
        "```json",
        json.dumps(summary["direction_cosines"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Direction Effects",
        "",
        "```json",
        json.dumps(summary["direction_effects"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    root_dir = output_root / datetime.now().strftime(
        "critic_optimizer_geometry_audit_%Y%m%d_%H%M%S"
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    set_global_seeds(args.seed)
    config = build_policy_ratio_mode_config(policy_ratio_mode=MODE, seed=args.seed)
    checkpoint_path = _find_current_mainline_checkpoint(output_root)
    state_layout = build_state_layout(config)
    agent = PPOAgent(
        config=config.ppo,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        critic_state_layout=state_layout,
    )
    agent.load(str(checkpoint_path), load_optimizer=True)
    agent.freeze_actor_training_updates = True

    probe_payload = _collect_fixed_probe_payload(agent, config, args.seed)
    simulator = Simulator(config)
    _collect_single_rollout_epoch(agent, simulator, config, epoch=args.epoch)
    full_rollout = agent.buffer.as_tensors(agent.device)
    full_rollout["rewards"] = torch.tensor(
        agent.buffer.rewards,
        dtype=torch.float32,
        device=agent.device,
    )

    agent.critic_optimizer_ablation_request = {
        "epoch": int(args.epoch),
        "update_epoch": int(args.update_epoch),
        "minibatch_id": int(args.minibatch_id),
    }
    try:
        agent.train()
        raise RuntimeError("Expected critic optimizer geometry capture did not trigger.")
    except CriticOptimizerAblationCaptured:
        capture = agent.critic_optimizer_ablation_capture

    payloads = _build_payloads_from_capture_and_rollout(
        agent=agent,
        capture=capture,
        full_data=full_rollout,
        probe_payload=probe_payload,
        seed=args.seed,
    )
    eval_payloads = {
        "current_batch": payloads["current_batch"],
        "held_out_batch": payloads["held_out_batch"],
        "probe": payloads["probe"],
    }

    before_sets = _evaluate_sets(
        agent,
        eval_payloads,
        capture["target_mean"],
        capture["target_std"],
    )
    base_agent = copy.deepcopy(agent)
    base_grads = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in agent.critic_params
    ]

    adam_agent = copy.deepcopy(base_agent)
    adam_agent.critic_optimizer.load_state_dict(
        copy.deepcopy(capture["critic_optimizer_state_dict"])
    )
    _assign_copied_critic_grads(adam_agent, base_grads)
    actual_adam_before = copy.deepcopy(adam_agent)
    adam_agent.critic_optimizer.step()
    actual_adam_update_norm = _parameter_delta_norms(
        actual_adam_before,
        adam_agent,
    )["critic_param_delta_norm"]

    state_agent = copy.deepcopy(base_agent)
    state_agent.critic_optimizer.load_state_dict(
        copy.deepcopy(capture["critic_optimizer_state_dict"])
    )
    direction_tensors = _build_adam_geometry_directions(state_agent, base_grads)

    direction_vectors = {
        name: _flat_direction(tensors) for name, tensors in direction_tensors.items()
    }
    backbone_vectors = {
        name: _flat_backbone_direction(base_agent, tensors)
        for name, tensors in direction_tensors.items()
    }
    head_vectors = {
        name: _flat_head_direction(base_agent, tensors)
        for name, tensors in direction_tensors.items()
    }

    direction_cosines = {
        "cosine_raw_vs_momentum": _cosine(
            direction_vectors["raw_grad_direction"],
            direction_vectors["momentum_direction"],
        ),
        "cosine_raw_vs_preconditioned": _cosine(
            direction_vectors["raw_grad_direction"],
            direction_vectors["preconditioned_grad_direction"],
        ),
        "cosine_raw_vs_full_adam": _cosine(
            direction_vectors["raw_grad_direction"],
            direction_vectors["full_adam_direction"],
        ),
        "cosine_momentum_vs_full_adam": _cosine(
            direction_vectors["momentum_direction"],
            direction_vectors["full_adam_direction"],
        ),
        "cosine_preconditioned_vs_full_adam": _cosine(
            direction_vectors["preconditioned_grad_direction"],
            direction_vectors["full_adam_direction"],
        ),
        "backbone_cosine_raw_vs_momentum": _cosine(
            backbone_vectors["raw_grad_direction"],
            backbone_vectors["momentum_direction"],
        ),
        "backbone_cosine_raw_vs_preconditioned": _cosine(
            backbone_vectors["raw_grad_direction"],
            backbone_vectors["preconditioned_grad_direction"],
        ),
        "backbone_cosine_raw_vs_full_adam": _cosine(
            backbone_vectors["raw_grad_direction"],
            backbone_vectors["full_adam_direction"],
        ),
        "head_cosine_raw_vs_momentum": _cosine(
            head_vectors["raw_grad_direction"],
            head_vectors["momentum_direction"],
        ),
        "head_cosine_raw_vs_preconditioned": _cosine(
            head_vectors["raw_grad_direction"],
            head_vectors["preconditioned_grad_direction"],
        ),
        "head_cosine_raw_vs_full_adam": _cosine(
            head_vectors["raw_grad_direction"],
            head_vectors["full_adam_direction"],
        ),
        "raw_grad_norm": float(direction_vectors["raw_grad_direction"].norm().item()),
        "momentum_norm": float(direction_vectors["momentum_direction"].norm().item()),
        "preconditioned_norm": float(
            direction_vectors["preconditioned_grad_direction"].norm().item()
        ),
        "full_adam_norm": float(direction_vectors["full_adam_direction"].norm().item()),
    }

    metrics_rows: list[dict[str, Any]] = []

    def _append_rows(
        group_name: str,
        direction_name: str,
        target_update_norm: float,
        step_type: str,
        set_metrics: dict[str, dict[str, float]],
        delta_metrics: dict[str, float],
    ) -> None:
        for set_type in ("current_batch", "held_out_batch", "probe"):
            metrics_rows.append(
                {
                    "group_name": group_name,
                    "update_direction": direction_name,
                    "target_update_norm": float(target_update_norm),
                    "step_type": step_type,
                    "set_type": set_type,
                    **set_metrics[set_type],
                    **delta_metrics,
                    "cosine_raw_vs_momentum": direction_cosines["cosine_raw_vs_momentum"],
                    "cosine_raw_vs_preconditioned": direction_cosines[
                        "cosine_raw_vs_preconditioned"
                    ],
                    "cosine_raw_vs_full_adam": direction_cosines["cosine_raw_vs_full_adam"],
                    "cosine_momentum_vs_full_adam": direction_cosines[
                        "cosine_momentum_vs_full_adam"
                    ],
                    "cosine_preconditioned_vs_full_adam": direction_cosines[
                        "cosine_preconditioned_vs_full_adam"
                    ],
                    "backbone_cosine_raw_vs_momentum": direction_cosines[
                        "backbone_cosine_raw_vs_momentum"
                    ],
                    "backbone_cosine_raw_vs_preconditioned": direction_cosines[
                        "backbone_cosine_raw_vs_preconditioned"
                    ],
                    "backbone_cosine_raw_vs_full_adam": direction_cosines[
                        "backbone_cosine_raw_vs_full_adam"
                    ],
                    "head_cosine_raw_vs_momentum": direction_cosines[
                        "head_cosine_raw_vs_momentum"
                    ],
                    "head_cosine_raw_vs_preconditioned": direction_cosines[
                        "head_cosine_raw_vs_preconditioned"
                    ],
                    "head_cosine_raw_vs_full_adam": direction_cosines[
                        "head_cosine_raw_vs_full_adam"
                    ],
                }
            )

    _append_rows(
        "no_op",
        "no_op",
        actual_adam_update_norm,
        "before_update",
        before_sets,
        {
            "critic_param_delta_norm": 0.0,
            "critic_backbone_delta_norm": 0.0,
            "critic_head_delta_norm": 0.0,
            "grad_norm": 0.0,
            "update_over_grad_ratio": 0.0,
            "cosine_update_vs_current_grad": 0.0,
            "cosine_update_vs_heldout_grad": 0.0,
            "cosine_update_vs_global_grad": 0.0,
            "cosine_update_vs_blended_grad": 0.0,
        },
    )
    _append_rows(
        "no_op",
        "no_op",
        actual_adam_update_norm,
        "after_update",
        before_sets,
        {
            "critic_param_delta_norm": 0.0,
            "critic_backbone_delta_norm": 0.0,
            "critic_head_delta_norm": 0.0,
            "grad_norm": 0.0,
            "update_over_grad_ratio": 0.0,
            "cosine_update_vs_current_grad": 0.0,
            "cosine_update_vs_heldout_grad": 0.0,
            "cosine_update_vs_global_grad": 0.0,
            "cosine_update_vs_blended_grad": 0.0,
        },
    )

    for direction_name in (
        "raw_grad_direction",
        "momentum_direction",
        "preconditioned_grad_direction",
        "full_adam_direction",
    ):
        updated_agent, delta_metrics = _apply_direction_with_target_norm(
            base_agent,
            direction_tensors[direction_name],
            actual_adam_update_norm,
        )
        after_sets = _evaluate_sets(
            updated_agent,
            eval_payloads,
            capture["target_mean"],
            capture["target_std"],
        )
        direction_vector = direction_vectors[direction_name]
        _append_rows(
            direction_name,
            direction_name,
            actual_adam_update_norm,
            "before_update",
            before_sets,
            {
                "critic_param_delta_norm": 0.0,
                "critic_backbone_delta_norm": 0.0,
                "critic_head_delta_norm": 0.0,
                "grad_norm": float(direction_vector.norm().item()),
                "update_over_grad_ratio": 0.0,
                "cosine_update_vs_current_grad": 0.0,
                "cosine_update_vs_heldout_grad": 0.0,
                "cosine_update_vs_global_grad": 0.0,
                "cosine_update_vs_blended_grad": 0.0,
            },
        )
        _append_rows(
            direction_name,
            direction_name,
            actual_adam_update_norm,
            "after_update",
            after_sets,
            {
                **delta_metrics,
                "cosine_update_vs_current_grad": _cosine(
                    direction_vector,
                    direction_vectors["raw_grad_direction"],
                ),
                "cosine_update_vs_heldout_grad": 0.0,
                "cosine_update_vs_global_grad": 0.0,
                "cosine_update_vs_blended_grad": 0.0,
            },
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        root_dir / "critic_optimizer_geometry_audit_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for set_type, filename in (
        ("current_batch", "critic_optimizer_geometry_audit_current_batch.csv"),
        ("held_out_batch", "critic_optimizer_geometry_audit_heldout_batch.csv"),
        ("probe", "critic_optimizer_geometry_audit_probe.csv"),
    ):
        metrics_df[metrics_df["set_type"] == set_type].to_csv(
            root_dir / filename,
            index=False,
            encoding="utf-8-sig",
        )

    direction_effects: dict[str, Any] = {}
    for group_name in (
        "raw_grad_direction",
        "momentum_direction",
        "preconditioned_grad_direction",
        "full_adam_direction",
    ):
        group_rows = metrics_df[metrics_df["group_name"] == group_name]
        before_rows = group_rows[group_rows["step_type"] == "before_update"].set_index("set_type")
        after_rows = group_rows[group_rows["step_type"] == "after_update"].set_index("set_type")
        direction_effects[group_name] = {}
        for set_type in ("current_batch", "held_out_batch", "probe"):
            direction_effects[group_name][set_type] = {
                "critic_loss_delta": float(
                    after_rows.loc[set_type, "critic_loss"] - before_rows.loc[set_type, "critic_loss"]
                ),
                "target_pearson_delta": float(
                    after_rows.loc[set_type, "pearson_value_vs_value_target"]
                    - before_rows.loc[set_type, "pearson_value_vs_value_target"]
                ),
                "return_pearson_delta": float(
                    after_rows.loc[set_type, "pearson_value_vs_return"]
                    - before_rows.loc[set_type, "pearson_value_vs_return"]
                ),
                "advantage_pearson_delta": float(
                    after_rows.loc[set_type, "pearson_value_vs_advantage"]
                    - before_rows.loc[set_type, "pearson_value_vs_advantage"]
                ),
            }
        direction_effects[group_name]["delta_metrics"] = {
            "critic_param_delta_norm": float(after_rows["critic_param_delta_norm"].iloc[0]),
            "critic_backbone_delta_norm": float(
                after_rows["critic_backbone_delta_norm"].iloc[0]
            ),
            "critic_head_delta_norm": float(after_rows["critic_head_delta_norm"].iloc[0]),
            "grad_norm": float(after_rows["grad_norm"].iloc[0]),
            "update_over_grad_ratio": float(after_rows["update_over_grad_ratio"].iloc[0]),
        }

    interpretation = (
        "If full_adam_direction is clearly worse than raw_grad at the same update norm, "
        "the bad step is primarily coming from optimizer geometry rather than from the "
        "raw value-gradient direction itself."
    )
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "target_epoch": int(args.epoch),
        "target_update_epoch": int(args.update_epoch),
        "target_minibatch_id": int(args.minibatch_id),
        "actual_adam_update_norm": float(actual_adam_update_norm),
        "target_update_norm": float(actual_adam_update_norm),
        "direction_cosines": direction_cosines,
        "direction_effects": direction_effects,
        "interpretation": interpretation,
    }
    (root_dir / "critic_optimizer_geometry_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, root_dir / "critic_optimizer_geometry_audit_summary.md")


if __name__ == "__main__":
    main()
