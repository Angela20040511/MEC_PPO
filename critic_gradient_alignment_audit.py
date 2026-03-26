import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from critic_loss_component_ablation import (
    _append_rows,
    _build_payloads_from_capture_and_rollout,
    _cosine_similarity,
    _evaluate_all_sets,
    _flatten_tensors,
    _main_value_loss_only,
    _parameter_list_delta_norm,
    _semantic_metrics_from_payload,
)
from critic_optimizer_rule_ablation import _collect_single_rollout_epoch
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
TARGET_UPDATE_NORM = 5e-5
SET_TYPES = ("current_batch", "held_out_batch", "probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit main value-loss gradient alignment across current minibatch, fixed held-out "
            "minibatches, and a large same-rollout global batch on the same captured bad step."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument("--update-epoch", type=int, default=DEFAULT_UPDATE_EPOCH)
    parser.add_argument("--minibatch-id", type=int, default=DEFAULT_MINIBATCH_ID)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    return parser.parse_args()


def _build_global_payload(
    full_data: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "states": full_data["states"].clone(),
        "returns": full_data["returns"].clone(),
        "advantages": full_data["advantages"].clone(),
        "rewards": full_data["rewards"].clone(),
        "source_step_count": int(full_data["states"].size(0)),
        "global_sample_count": int(full_data["states"].size(0)),
    }


def _main_value_objective_from_payload(
    agent: PPOAgent,
    payload: dict[str, torch.Tensor],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    value_targets = agent._value_targets_from_returns_with_stats(
        payload["returns"],
        target_mean,
        target_std,
    )
    critic_inputs, _, _ = agent._prepare_critic_inputs(
        payload["states"],
        update_stats=False,
    )
    objective, details = _main_value_loss_only(
        agent,
        critic_inputs,
        value_targets,
        target_mean,
        target_std,
    )
    return objective, details


def _collect_gradients(
    base_agent: PPOAgent,
    payload: dict[str, torch.Tensor],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, Any]:
    working_agent = copy.deepcopy(base_agent)
    working_agent.critic_optimizer.zero_grad()
    objective, details = _main_value_objective_from_payload(
        working_agent,
        payload,
        target_mean,
        target_std,
    )
    objective.backward()
    raw_grad_norm = float(working_agent._grad_norm(working_agent.critic_params))
    grads = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in working_agent.critic_params
    ]
    grad_vector = _flatten_tensors(grads)
    backbone_grads = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in working_agent.network.critic_backbone.parameters()
    ]
    head_grads = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in working_agent.network.critic_head.parameters()
    ]
    return {
        "objective": objective.detach().item(),
        "details": details,
        "raw_grad_norm": raw_grad_norm,
        "grads": grads,
        "grad_vector": grad_vector,
        "backbone_grad_vector": _flatten_tensors(backbone_grads),
        "head_grad_vector": _flatten_tensors(head_grads),
    }


def _apply_small_matched_update(
    base_agent: PPOAgent,
    grads: list[torch.Tensor | None],
    target_update_norm: float,
) -> tuple[PPOAgent, dict[str, float]]:
    updated_agent = copy.deepcopy(base_agent)
    with torch.no_grad():
        grad_vector = _flatten_tensors(grads)
        grad_norm = float(grad_vector.norm().item())
        if grad_norm <= 1e-12:
            scale = 0.0
        else:
            scale = float(target_update_norm) / grad_norm
        for parameter, grad in zip(updated_agent.critic_params, grads):
            if grad is None:
                continue
            parameter.add_(grad, alpha=-scale)

    before_params = [parameter.detach().clone() for parameter in base_agent.critic_params]
    after_params = [parameter.detach().clone() for parameter in updated_agent.critic_params]
    before_backbone = [
        parameter.detach().clone() for parameter in base_agent.network.critic_backbone.parameters()
    ]
    after_backbone = [
        parameter.detach().clone()
        for parameter in updated_agent.network.critic_backbone.parameters()
    ]
    before_head = [
        parameter.detach().clone() for parameter in base_agent.network.critic_head.parameters()
    ]
    after_head = [
        parameter.detach().clone() for parameter in updated_agent.network.critic_head.parameters()
    ]
    update_vector = []
    for before_param, after_param, grad in zip(before_params, after_params, grads):
        if grad is None:
            continue
        update_vector.append((after_param - before_param).reshape(-1).cpu())
    update_vector = torch.cat(update_vector) if update_vector else torch.zeros(1, dtype=torch.float32)
    grad_vector = _flatten_tensors(grads)
    delta_metrics = {
        "critic_param_delta_norm": _parameter_list_delta_norm(before_params, after_params),
        "critic_backbone_delta_norm": _parameter_list_delta_norm(before_backbone, after_backbone),
        "critic_head_delta_norm": _parameter_list_delta_norm(before_head, after_head),
        "update_over_grad_ratio": float(
            _parameter_list_delta_norm(before_params, after_params) / max(grad_norm, 1e-12)
        ),
        "scale_factor": float(scale),
        "cosine_update_vs_raw_grad": _cosine_similarity(update_vector, grad_vector),
    }
    return updated_agent, delta_metrics


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Critic Gradient Alignment Audit",
        "",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        (
            f"- captured minibatch: epoch `{summary['target_epoch']}` / update_epoch "
            f"`{summary['target_update_epoch']}` / minibatch `{summary['target_minibatch_id']}`"
        ),
        f"- target update norm: `{summary['target_update_norm']}`",
        f"- interpretation: {summary['interpretation']}",
        "",
        "## Gradient Cosines",
        "",
        "```json",
        json.dumps(summary["gradient_alignment"], ensure_ascii=False, indent=2),
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
        "critic_gradient_alignment_audit_%Y%m%d_%H%M%S"
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
        raise RuntimeError("Expected critic gradient alignment capture did not trigger.")
    except CriticOptimizerAblationCaptured:
        capture = agent.critic_optimizer_ablation_capture

    payloads = _build_payloads_from_capture_and_rollout(
        agent=agent,
        capture=capture,
        full_data=full_rollout,
        probe_payload=probe_payload,
        seed=args.seed,
    )
    payloads["global_batch"] = _build_global_payload(full_rollout)

    before_sets = _evaluate_all_sets(
        agent,
        {
            "current_batch": payloads["current_batch"],
            "held_out_batch": payloads["held_out_batch"],
            "probe": payloads["probe"],
        },
        capture["target_mean"],
        capture["target_std"],
    )
    global_before = _semantic_metrics_from_payload(
        agent,
        payloads["global_batch"],
        capture["target_mean"],
        capture["target_std"],
    )

    base_agent = copy.deepcopy(agent)
    gradient_groups = {
        "current_batch_value_grad": payloads["current_batch"],
        "heldout_value_grad": payloads["held_out_batch"],
        "global_value_grad": payloads["global_batch"],
    }
    gradient_results = {
        name: _collect_gradients(
            base_agent,
            payload,
            capture["target_mean"],
            capture["target_std"],
        )
        for name, payload in gradient_groups.items()
    }

    blended_agent = copy.deepcopy(base_agent)
    blended_agent.critic_optimizer.zero_grad()
    current_objective, current_details = _main_value_objective_from_payload(
        blended_agent,
        payloads["current_batch"],
        capture["target_mean"],
        capture["target_std"],
    )
    heldout_objective, heldout_details = _main_value_objective_from_payload(
        blended_agent,
        payloads["held_out_batch"],
        capture["target_mean"],
        capture["target_std"],
    )
    blended_objective = 0.5 * current_objective + 0.5 * heldout_objective
    blended_objective.backward()
    blended_grads = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in blended_agent.critic_params
    ]
    gradient_results["blended_current_heldout_grad"] = {
        "objective": float(blended_objective.detach().item()),
        "details": {
            "main_value_loss": 0.5 * current_details["main_value_loss"]
            + 0.5 * heldout_details["main_value_loss"],
            "heldout_main_value_loss": heldout_details["main_value_loss"],
            "blend_current_weight": 0.5,
            "blend_heldout_weight": 0.5,
        },
        "raw_grad_norm": float(blended_agent._grad_norm(blended_agent.critic_params)),
        "grads": blended_grads,
        "grad_vector": _flatten_tensors(blended_grads),
        "backbone_grad_vector": _flatten_tensors(
            [
                parameter.grad.detach().clone() if parameter.grad is not None else None
                for parameter in blended_agent.network.critic_backbone.parameters()
            ]
        ),
        "head_grad_vector": _flatten_tensors(
            [
                parameter.grad.detach().clone() if parameter.grad is not None else None
                for parameter in blended_agent.network.critic_head.parameters()
            ]
        ),
    }

    gradient_alignment = {
        "current_vs_heldout_cosine": _cosine_similarity(
            gradient_results["current_batch_value_grad"]["grad_vector"],
            gradient_results["heldout_value_grad"]["grad_vector"],
        ),
        "current_vs_global_cosine": _cosine_similarity(
            gradient_results["current_batch_value_grad"]["grad_vector"],
            gradient_results["global_value_grad"]["grad_vector"],
        ),
        "current_vs_blended_cosine": _cosine_similarity(
            gradient_results["current_batch_value_grad"]["grad_vector"],
            gradient_results["blended_current_heldout_grad"]["grad_vector"],
        ),
        "heldout_vs_global_cosine": _cosine_similarity(
            gradient_results["heldout_value_grad"]["grad_vector"],
            gradient_results["global_value_grad"]["grad_vector"],
        ),
        "current_vs_heldout_backbone_cosine": _cosine_similarity(
            gradient_results["current_batch_value_grad"]["backbone_grad_vector"],
            gradient_results["heldout_value_grad"]["backbone_grad_vector"],
        ),
        "current_vs_heldout_head_cosine": _cosine_similarity(
            gradient_results["current_batch_value_grad"]["head_grad_vector"],
            gradient_results["heldout_value_grad"]["head_grad_vector"],
        ),
        "grad_norm_current": gradient_results["current_batch_value_grad"]["raw_grad_norm"],
        "grad_norm_heldout": gradient_results["heldout_value_grad"]["raw_grad_norm"],
        "grad_norm_global": gradient_results["global_value_grad"]["raw_grad_norm"],
        "grad_norm_blended": gradient_results["blended_current_heldout_grad"]["raw_grad_norm"],
    }

    metrics_rows: list[dict[str, Any]] = []
    before_global_row = {"global_batch": global_before}
    extended_before_sets = {
        "current_batch": before_sets["current_batch"],
        "held_out_batch": before_sets["held_out_batch"],
        "probe": before_sets["probe"],
        "global_batch": global_before,
    }

    def _append_extended_rows(
        group_name: str,
        step_type: str,
        set_metrics: dict[str, dict[str, float]],
        grad_name: str,
        grad_norm: float,
        delta_metrics: dict[str, float],
    ) -> None:
        for set_type in ("current_batch", "held_out_batch", "probe", "global_batch"):
            metrics = set_metrics[set_type]
            row = {
                "group_name": group_name,
                "update_direction": grad_name,
                "step_type": step_type,
                "set_type": set_type,
                **metrics,
                "grad_norm": float(grad_norm),
                **delta_metrics,
                "cosine_update_vs_current_grad": (
                    delta_metrics["cosine_update_vs_current_grad"]
                    if step_type == "after_update"
                    else 0.0
                ),
                "cosine_update_vs_heldout_grad": (
                    delta_metrics["cosine_update_vs_heldout_grad"]
                    if step_type == "after_update"
                    else 0.0
                ),
                "cosine_update_vs_global_grad": (
                    delta_metrics["cosine_update_vs_global_grad"]
                    if step_type == "after_update"
                    else 0.0
                ),
                "cosine_update_vs_blended_grad": (
                    delta_metrics["cosine_update_vs_blended_grad"]
                    if step_type == "after_update"
                    else 0.0
                ),
            }
            metrics_rows.append(row)

    def _run_direction_group(group_name: str, grad_name: str) -> None:
        grad_bundle = gradient_results[grad_name]
        updated_agent, delta_metrics = _apply_small_matched_update(
            base_agent,
            grad_bundle["grads"],
            TARGET_UPDATE_NORM,
        )
        after_sets = _evaluate_all_sets(
            updated_agent,
            {
                "current_batch": payloads["current_batch"],
                "held_out_batch": payloads["held_out_batch"],
                "probe": payloads["probe"],
            },
            capture["target_mean"],
            capture["target_std"],
        )
        after_global = _semantic_metrics_from_payload(
            updated_agent,
            payloads["global_batch"],
            capture["target_mean"],
            capture["target_std"],
        )
        extended_after_sets = {
            "current_batch": after_sets["current_batch"],
            "held_out_batch": after_sets["held_out_batch"],
            "probe": after_sets["probe"],
            "global_batch": after_global,
        }
        extra_cosines = {
            "cosine_update_vs_current_grad": _cosine_similarity(
                gradient_results["current_batch_value_grad"]["grad_vector"],
                gradient_results[grad_name]["grad_vector"],
            ),
            "cosine_update_vs_heldout_grad": _cosine_similarity(
                gradient_results["heldout_value_grad"]["grad_vector"],
                gradient_results[grad_name]["grad_vector"],
            ),
            "cosine_update_vs_global_grad": _cosine_similarity(
                gradient_results["global_value_grad"]["grad_vector"],
                gradient_results[grad_name]["grad_vector"],
            ),
            "cosine_update_vs_blended_grad": _cosine_similarity(
                gradient_results["blended_current_heldout_grad"]["grad_vector"],
                gradient_results[grad_name]["grad_vector"],
            ),
        }
        _append_extended_rows(
            group_name,
            "before_update",
            extended_before_sets,
            grad_name,
            grad_bundle["raw_grad_norm"],
            {
                "critic_param_delta_norm": 0.0,
                "critic_backbone_delta_norm": 0.0,
                "critic_head_delta_norm": 0.0,
                "update_over_grad_ratio": 0.0,
                "scale_factor": 0.0,
                **{key: 0.0 for key in extra_cosines},
            },
        )
        _append_extended_rows(
            group_name,
            "after_update",
            extended_after_sets,
            grad_name,
            grad_bundle["raw_grad_norm"],
            {
                **delta_metrics,
                **extra_cosines,
            },
        )

    _append_extended_rows(
        "no_op",
        "before_update",
        extended_before_sets,
        "no_op",
        0.0,
        {
            "critic_param_delta_norm": 0.0,
            "critic_backbone_delta_norm": 0.0,
            "critic_head_delta_norm": 0.0,
            "update_over_grad_ratio": 0.0,
            "scale_factor": 0.0,
            "cosine_update_vs_current_grad": 0.0,
            "cosine_update_vs_heldout_grad": 0.0,
            "cosine_update_vs_global_grad": 0.0,
            "cosine_update_vs_blended_grad": 0.0,
        },
    )
    _append_extended_rows(
        "no_op",
        "after_update",
        extended_before_sets,
        "no_op",
        0.0,
        {
            "critic_param_delta_norm": 0.0,
            "critic_backbone_delta_norm": 0.0,
            "critic_head_delta_norm": 0.0,
            "update_over_grad_ratio": 0.0,
            "scale_factor": 0.0,
            "cosine_update_vs_current_grad": 0.0,
            "cosine_update_vs_heldout_grad": 0.0,
            "cosine_update_vs_global_grad": 0.0,
            "cosine_update_vs_blended_grad": 0.0,
        },
    )

    _run_direction_group("current_grad_small_step", "current_batch_value_grad")
    _run_direction_group("heldout_grad_small_step", "heldout_value_grad")
    _run_direction_group("global_grad_small_step", "global_value_grad")
    _run_direction_group("blended_grad_small_step", "blended_current_heldout_grad")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = root_dir / "critic_gradient_alignment_audit_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    for set_type, filename in (
        ("current_batch", "critic_gradient_alignment_current_batch.csv"),
        ("held_out_batch", "critic_gradient_alignment_heldout_batch.csv"),
        ("probe", "critic_gradient_alignment_probe.csv"),
    ):
        metrics_df[metrics_df["set_type"] == set_type].to_csv(
            root_dir / filename,
            index=False,
            encoding="utf-8-sig",
        )

    direction_effects: dict[str, Any] = {}
    for group_name in (
        "current_grad_small_step",
        "heldout_grad_small_step",
        "global_grad_small_step",
        "blended_grad_small_step",
    ):
        group_rows = metrics_df[metrics_df["group_name"] == group_name]
        before_rows = group_rows[group_rows["step_type"] == "before_update"].set_index("set_type")
        after_rows = group_rows[group_rows["step_type"] == "after_update"].set_index("set_type")
        direction_effects[group_name] = {}
        for set_type in ("current_batch", "held_out_batch", "probe", "global_batch"):
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
            "critic_param_delta_norm": float(
                after_rows["critic_param_delta_norm"].iloc[0]
            ),
            "critic_backbone_delta_norm": float(
                after_rows["critic_backbone_delta_norm"].iloc[0]
            ),
            "critic_head_delta_norm": float(after_rows["critic_head_delta_norm"].iloc[0]),
            "update_over_grad_ratio": float(after_rows["update_over_grad_ratio"].iloc[0]),
            "scale_factor": float(after_rows["scale_factor"].iloc[0]),
        }

    current_delta = direction_effects["current_grad_small_step"]
    heldout_delta = direction_effects["heldout_grad_small_step"]
    blended_delta = direction_effects["blended_grad_small_step"]
    global_delta = direction_effects["global_grad_small_step"]

    interpretation = (
        "Current-batch gradient direction appears misaligned with held-out/global semantics "
        "if its cosine to held-out/global is low or negative and its matched small-step update "
        "improves current_batch while worsening held_out_batch/probe more than heldout/blended/global directions."
    )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "target_epoch": int(args.epoch),
        "target_update_epoch": int(args.update_epoch),
        "target_minibatch_id": int(args.minibatch_id),
        "target_update_norm": float(TARGET_UPDATE_NORM),
        "gradient_alignment": gradient_alignment,
        "direction_effects": direction_effects,
        "interpretation": interpretation,
    }

    summary_path = root_dir / "critic_gradient_alignment_audit_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, root_dir / "critic_gradient_alignment_audit_summary.md")


if __name__ == "__main__":
    main()
