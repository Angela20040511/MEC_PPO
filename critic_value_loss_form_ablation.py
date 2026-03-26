import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

from critic_loss_component_ablation import (
    _append_rows,
    _build_payloads_from_capture_and_rollout,
    _cosine_similarity,
    _delta_vector,
    _evaluate_all_sets,
    _flatten_tensors,
    _parameter_list_delta_norm,
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
HUBER_DELTA = 1.0
SET_TYPES = ("current_batch", "held_out_batch", "probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare different main critic value-loss forms on the same captured bad-step "
            "minibatch, fixed held-out minibatches, and fixed probe set."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument("--update-epoch", type=int, default=DEFAULT_UPDATE_EPOCH)
    parser.add_argument("--minibatch-id", type=int, default=DEFAULT_MINIBATCH_ID)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    return parser.parse_args()


def _critic_predictions(
    agent: PPOAgent,
    critic_inputs: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    critic_features = agent.network.critic_features_from_critic_input(critic_inputs)
    normalized_values = agent.network.critic_head(critic_features).squeeze(-1)
    if agent.config.value_target_mode == "popart_return_norm":
        return normalized_values
    raw_values = agent.network.value_from_critic_input(critic_inputs).squeeze(-1)
    if agent.config.value_target_mode in {"normalized_return", "running_return_norm"}:
        return (raw_values - target_mean) / (target_std + 1e-8)
    return raw_values


def _current_value_loss_form(
    agent: PPOAgent,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return agent._compute_value_loss(predictions, targets)


def _plain_mse_value_loss(
    _agent: PPOAgent,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(predictions, targets)


def _plain_huber_value_loss(
    _agent: PPOAgent,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return F.huber_loss(predictions, targets, delta=HUBER_DELTA)


def _build_objective(
    agent: PPOAgent,
    capture: dict[str, Any],
    loss_form: str,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    predictions = _critic_predictions(
        agent,
        capture["batch_critic_inputs"],
        capture["target_mean"],
        capture["target_std"],
    )
    targets = capture["batch_value_targets"]
    if loss_form == "current_value_loss_form":
        main_value_loss = _current_value_loss_form(agent, predictions, targets)
        note = f"current_config_{agent.config.value_loss_mode}"
    elif loss_form == "plain_mse_value_loss":
        main_value_loss = _plain_mse_value_loss(agent, predictions, targets)
        note = "plain_mse"
    elif loss_form == "plain_huber_value_loss":
        main_value_loss = _plain_huber_value_loss(agent, predictions, targets)
        note = f"plain_huber_delta_{HUBER_DELTA}"
    elif loss_form == "unclipped_value_loss_only":
        main_value_loss = _current_value_loss_form(agent, predictions, targets)
        note = "equivalent_to_current_no_value_clipping_present"
    else:
        raise ValueError(f"Unsupported loss form: {loss_form}")
    objective = agent.config.value_coeff * main_value_loss
    return objective, {
        "main_value_loss": float(main_value_loss.detach().item()),
        "loss_form_note": note,
        "group_applicable": 1.0,
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Critic Value Loss Form Ablation",
        "",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        (
            f"- captured minibatch: epoch `{summary['target_epoch']}` / update_epoch "
            f"`{summary['target_update_epoch']}` / minibatch `{summary['target_minibatch_id']}`"
        ),
        f"- current value loss mode: `{summary['current_value_loss_mode']}`",
        f"- interpretation: {summary['interpretation']}",
        "",
        "## Group Comparison",
        "",
        "```json",
        json.dumps(summary["group_comparison"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    root_dir = output_root / datetime.now().strftime(
        "critic_value_loss_form_ablation_%Y%m%d_%H%M%S"
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
        raise RuntimeError("Expected critic value-loss-form ablation capture did not trigger.")
    except CriticOptimizerAblationCaptured:
        capture = agent.critic_optimizer_ablation_capture

    payloads = _build_payloads_from_capture_and_rollout(
        agent=agent,
        capture=capture,
        full_data=full_rollout,
        probe_payload=probe_payload,
        seed=args.seed,
    )
    before_sets = _evaluate_all_sets(
        agent,
        payloads,
        capture["target_mean"],
        capture["target_std"],
    )
    base_agent = copy.deepcopy(agent)
    before_params = [parameter.detach().clone() for parameter in base_agent.critic_params]

    metrics_rows: list[dict[str, Any]] = []

    def _run_group(group_name: str) -> None:
        updated_agent = copy.deepcopy(base_agent)
        updated_agent.critic_optimizer.load_state_dict(
            copy.deepcopy(capture["critic_optimizer_state_dict"])
        )
        updated_agent.critic_optimizer.zero_grad()
        objective, before_info = _build_objective(updated_agent, capture, group_name)
        objective.backward()
        grad_norm = float(updated_agent._grad_norm(updated_agent.critic_params))
        torch.nn.utils.clip_grad_norm_(
            updated_agent.critic_params,
            updated_agent.config.max_grad_norm,
        )
        updated_agent.critic_optimizer.step()
        after_sets = _evaluate_all_sets(
            updated_agent,
            payloads,
            capture["target_mean"],
            capture["target_std"],
        )
        after_objective, after_info = _build_objective(updated_agent, capture, group_name)
        after_params = [parameter.detach().clone() for parameter in updated_agent.critic_params]
        backbone_before = [
            parameter.detach().clone()
            for parameter in base_agent.network.critic_backbone.parameters()
        ]
        backbone_after = [
            parameter.detach().clone()
            for parameter in updated_agent.network.critic_backbone.parameters()
        ]
        head_before = [
            parameter.detach().clone() for parameter in base_agent.network.critic_head.parameters()
        ]
        head_after = [
            parameter.detach().clone()
            for parameter in updated_agent.network.critic_head.parameters()
        ]
        grads = [
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for parameter in updated_agent.critic_params
        ]
        grad_vector = _flatten_tensors(grads)
        update_vector = _delta_vector(before_params, after_params, grads)
        delta_metrics = {
            "critic_param_delta_norm": _parameter_list_delta_norm(before_params, after_params),
            "critic_backbone_delta_norm": _parameter_list_delta_norm(
                backbone_before,
                backbone_after,
            ),
            "critic_head_delta_norm": _parameter_list_delta_norm(head_before, head_after),
            "cosine_update_vs_grad": _cosine_similarity(update_vector, grad_vector),
        }
        _append_rows(
            rows=metrics_rows,
            group_name=group_name,
            step_type="before_update",
            set_metrics=before_sets,
            grad_norm=grad_norm,
            delta_metrics={
                "critic_param_delta_norm": 0.0,
                "critic_backbone_delta_norm": 0.0,
                "critic_head_delta_norm": 0.0,
                "cosine_update_vs_grad": 0.0,
            },
            objective_info={
                "update_objective": float(objective.detach().item()),
                **before_info,
            },
        )
        _append_rows(
            rows=metrics_rows,
            group_name=group_name,
            step_type="after_update",
            set_metrics=after_sets,
            grad_norm=grad_norm,
            delta_metrics=delta_metrics,
            objective_info={
                "update_objective": float(after_objective.detach().item()),
                **after_info,
            },
        )

    def _append_na_group(group_name: str, reason: str) -> None:
        objective_info = {
            "update_objective": 0.0,
            "main_value_loss": 0.0,
            "loss_form_note": reason,
            "group_applicable": 0.0,
        }
        zero_delta = {
            "critic_param_delta_norm": 0.0,
            "critic_backbone_delta_norm": 0.0,
            "critic_head_delta_norm": 0.0,
            "cosine_update_vs_grad": 0.0,
        }
        _append_rows(
            metrics_rows,
            group_name,
            "before_update",
            before_sets,
            0.0,
            zero_delta,
            objective_info,
        )
        _append_rows(
            metrics_rows,
            group_name,
            "after_update",
            before_sets,
            0.0,
            zero_delta,
            objective_info,
        )

    _append_na_group("no_op", "no_update")
    _run_group("current_value_loss_form")
    _run_group("plain_mse_value_loss")
    _run_group("plain_huber_value_loss")
    _run_group("unclipped_value_loss_only")
    _append_na_group(
        "clipped_value_loss_only",
        "not_applicable_current_main_value_loss_has_no_value_clipping_branch",
    )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(root_dir / "critic_value_loss_form_ablation_metrics.csv", index=False)
    for set_type, filename in {
        "probe": "critic_value_loss_form_ablation_probe.csv",
        "current_batch": "critic_value_loss_form_ablation_current_batch.csv",
        "held_out_batch": "critic_value_loss_form_ablation_heldout_batch.csv",
    }.items():
        metrics_df[metrics_df["set_type"] == set_type].to_csv(root_dir / filename, index=False)

    def _extract(group_name: str, set_type: str, metric: str) -> dict[str, float]:
        before_row = metrics_df[
            (metrics_df["group_name"] == group_name)
            & (metrics_df["set_type"] == set_type)
            & (metrics_df["step_type"] == "before_update")
        ].iloc[0]
        after_row = metrics_df[
            (metrics_df["group_name"] == group_name)
            & (metrics_df["set_type"] == set_type)
            & (metrics_df["step_type"] == "after_update")
        ].iloc[0]
        return {
            "before": float(before_row[metric]),
            "after": float(after_row[metric]),
            "delta": float(after_row[metric] - before_row[metric]),
        }

    def _group_summary(group_name: str) -> dict[str, Any]:
        current_before = metrics_df[
            (metrics_df["group_name"] == group_name)
            & (metrics_df["set_type"] == "current_batch")
            & (metrics_df["step_type"] == "before_update")
        ].iloc[0]
        current_after = metrics_df[
            (metrics_df["group_name"] == group_name)
            & (metrics_df["set_type"] == "current_batch")
            & (metrics_df["step_type"] == "after_update")
        ].iloc[0]
        return {
            "current_batch": {
                "critic_loss": _extract(group_name, "current_batch", "critic_loss"),
                "target_pearson": _extract(
                    group_name,
                    "current_batch",
                    "pearson_value_vs_value_target",
                ),
            },
            "held_out_batch": {
                "critic_loss": _extract(group_name, "held_out_batch", "critic_loss"),
                "target_pearson": _extract(
                    group_name,
                    "held_out_batch",
                    "pearson_value_vs_value_target",
                ),
            },
            "probe": {
                "critic_loss": _extract(group_name, "probe", "critic_loss"),
                "target_pearson": _extract(
                    group_name,
                    "probe",
                    "pearson_value_vs_value_target",
                ),
            },
            "update_details": {
                "objective_before": float(current_before["update_objective"]),
                "objective_after": float(current_after["update_objective"]),
                "main_value_loss_before": float(current_before["main_value_loss"]),
                "main_value_loss_after": float(current_after["main_value_loss"]),
                "group_applicable": float(current_after["group_applicable"]),
                "loss_form_note": str(current_after["loss_form_note"]),
                "grad_norm": float(current_after["grad_norm"]),
                "critic_param_delta_norm": float(current_after["critic_param_delta_norm"]),
                "update_over_grad_ratio": float(current_after["update_over_grad_ratio"]),
                "cosine_update_vs_grad": float(current_after["cosine_update_vs_grad"]),
            },
        }

    group_comparison = {
        "no_op": _group_summary("no_op"),
        "current_value_loss_form": _group_summary("current_value_loss_form"),
        "plain_mse_value_loss": _group_summary("plain_mse_value_loss"),
        "plain_huber_value_loss": _group_summary("plain_huber_value_loss"),
        "unclipped_value_loss_only": _group_summary("unclipped_value_loss_only"),
        "clipped_value_loss_only": _group_summary("clipped_value_loss_only"),
    }

    current_probe_drop = group_comparison["current_value_loss_form"]["probe"]["target_pearson"][
        "delta"
    ]
    mse_probe_drop = group_comparison["plain_mse_value_loss"]["probe"]["target_pearson"][
        "delta"
    ]
    huber_probe_drop = group_comparison["plain_huber_value_loss"]["probe"]["target_pearson"][
        "delta"
    ]
    mse_heldout_drop = group_comparison["plain_mse_value_loss"]["held_out_batch"][
        "target_pearson"
    ]["delta"]
    huber_heldout_drop = group_comparison["plain_huber_value_loss"]["held_out_batch"][
        "target_pearson"
    ]["delta"]

    if abs(
        group_comparison["current_value_loss_form"]["probe"]["target_pearson"]["delta"]
        - group_comparison["plain_huber_value_loss"]["probe"]["target_pearson"]["delta"]
    ) < 1e-6 and abs(current_probe_drop) > 0.005:
        if huber_probe_drop > mse_probe_drop + 0.01 and huber_heldout_drop > mse_heldout_drop + 0.01:
            interpretation = (
                "The current main value-loss form is plain Huber(delta=1.0), and it is materially "
                "more stable than plain MSE on held-out/probe; value-loss mathematics matters."
            )
        elif mse_probe_drop > huber_probe_drop + 0.01:
            interpretation = (
                "Plain MSE is worse than the current Huber form on held-out/probe, so the current "
                "value-loss mathematics is not the main source of the batch-better/global-worse conflict."
            )
        else:
            interpretation = (
                "Current, plain Huber, and unclipped are effectively the same here; loss-form differences "
                "alone do not explain the bad step strongly."
            )
    else:
        interpretation = (
            "Changing the main value-loss form changes the bad-step behavior, which means value-loss "
            "mathematics contributes materially to the batch-fit/global-semantics conflict."
        )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "target_epoch": int(args.epoch),
        "target_update_epoch": int(args.update_epoch),
        "target_minibatch_id": int(args.minibatch_id),
        "current_value_loss_mode": str(agent.config.value_loss_mode),
        "current_value_loss_has_clipping": False,
        "plain_huber_delta": HUBER_DELTA,
        "group_comparison": group_comparison,
        "interpretation": interpretation,
    }
    (root_dir / "critic_value_loss_form_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, root_dir / "critic_value_loss_form_ablation_summary.md")


if __name__ == "__main__":
    main()
