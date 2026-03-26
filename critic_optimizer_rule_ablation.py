import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from dense_actor_input_experiment import build_state_layout
from dense_critic_diagnosis import compute_joint_counterfactual_scores
from dense_policy_ratio_mode_experiment import build_policy_ratio_mode_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from joint_training_critic_drift_trace import (
    MODE,
    PROBE_STEP_COUNT,
    _collect_fixed_probe_payload,
    _find_current_mainline_checkpoint,
)
from rl.ppo_agent import PPOAgent, CriticOptimizerAblationCaptured
from simulator.simulator import Simulator


DEFAULT_EPOCH = 0
DEFAULT_UPDATE_EPOCH = 2
DEFAULT_MINIBATCH_ID = 9
SMALL_SGD_LR = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare no-op, small-SGD, and current-Adam critic updates on the "
            "same captured minibatch/gradient near the first critic drift point."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument("--update-epoch", type=int, default=DEFAULT_UPDATE_EPOCH)
    parser.add_argument("--minibatch-id", type=int, default=DEFAULT_MINIBATCH_ID)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    return parser.parse_args()


def _collect_single_rollout_epoch(
    agent: PPOAgent,
    simulator: Simulator,
    config: Any,
    epoch: int,
) -> None:
    state = simulator.reset(seed=config.training.seed + epoch)
    done = False
    for _ in range(config.training.time_steps):
        action, log_prob, value = agent.select_action(state)
        joint_reward_aligned_scores, _joint_td_scores = compute_joint_counterfactual_scores(
            agent,
            simulator,
            action,
        )
        next_state, reward, done, _info = simulator.step(action)
        agent.store_transition(
            state,
            action,
            log_prob,
            reward,
            done,
            value,
            next_state,
            joint_reward_aligned_scores=joint_reward_aligned_scores,
        )
        state = next_state
        if done:
            break

    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)


def _semantic_metrics(
    agent: PPOAgent,
    critic_inputs: torch.Tensor,
    value_targets: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        predictions = agent._trace_raw_value_predictions_from_critic_inputs(
            critic_inputs,
            target_mean,
            target_std,
        )
        return agent._critic_trace_semantic_summary(
            predictions,
            value_targets,
            returns,
            advantages,
            rewards,
        )


def _critic_loss_value(
    agent: PPOAgent,
    critic_inputs: torch.Tensor,
    value_targets: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> float:
    with torch.no_grad():
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
        return float(agent._compute_value_loss(critic_predictions, value_targets).item())


def _evaluate_group(
    agent: PPOAgent,
    capture: dict[str, Any],
    probe_payload: dict[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, float]]:
    batch_metrics = _semantic_metrics(
        agent,
        capture["batch_critic_inputs"],
        capture["batch_value_targets"],
        capture["batch_returns"],
        capture["batch_raw_advantages"],
        capture["batch_rewards"],
        capture["target_mean"],
        capture["target_std"],
    )
    probe_value_targets = agent._value_targets_from_returns_with_stats(
        probe_payload["returns"],
        capture["target_mean"],
        capture["target_std"],
    )
    probe_critic_inputs, _, _ = agent._prepare_critic_inputs(
        probe_payload["states"],
        update_stats=False,
    )
    probe_metrics = _semantic_metrics(
        agent,
        probe_critic_inputs,
        probe_value_targets,
        probe_payload["returns"],
        probe_payload["advantages"],
        probe_payload["rewards"],
        capture["target_mean"],
        capture["target_std"],
    )
    return batch_metrics, probe_metrics


def _critic_param_delta_norms(base_agent: PPOAgent, updated_agent: PPOAgent) -> dict[str, float]:
    critic_delta = base_agent._parameter_list_delta_norm(
        base_agent.critic_params,
        updated_agent.critic_params,
    )
    backbone_delta = base_agent._parameter_list_delta_norm(
        list(base_agent.network.critic_backbone.parameters()),
        list(updated_agent.network.critic_backbone.parameters()),
    )
    head_delta = base_agent._parameter_list_delta_norm(
        list(base_agent.network.critic_head.parameters()),
        list(updated_agent.network.critic_head.parameters()),
    )
    return {
        "critic_param_delta_norm": float(critic_delta),
        "critic_backbone_delta_norm": float(backbone_delta),
        "critic_head_delta_norm": float(head_delta),
    }


def _assign_copied_critic_grads(agent: PPOAgent, base_grads: list[torch.Tensor | None]) -> None:
    for parameter, grad in zip(agent.critic_params, base_grads):
        if grad is None:
            parameter.grad = None
        else:
            parameter.grad = grad.detach().clone().to(parameter.device)


def _append_group_rows(
    rows: list[dict[str, Any]],
    group_name: str,
    before_batch: dict[str, float],
    before_probe: dict[str, float],
    after_batch: dict[str, float],
    after_probe: dict[str, float],
    before_critic_loss: float,
    after_critic_loss: float,
    delta_metrics: dict[str, float],
    grad_norm: float,
) -> None:
    common = {
        "group_name": group_name,
        "grad_norm": float(grad_norm),
    }
    rows.append(
        {
            **common,
            "step_type": "before_update",
            "critic_loss": float(before_critic_loss),
            **before_batch,
            **{f"probe_{key}": value for key, value in before_probe.items()},
            "critic_param_delta_norm": 0.0,
            "critic_backbone_delta_norm": 0.0,
            "critic_head_delta_norm": 0.0,
            "update_over_grad_ratio": 0.0,
        }
    )
    update_over_grad_ratio = (
        delta_metrics["critic_param_delta_norm"] / max(float(grad_norm), 1e-12)
    )
    rows.append(
        {
            **common,
            "step_type": "after_update",
            "critic_loss": float(after_critic_loss),
            **after_batch,
            **{f"probe_{key}": value for key, value in after_probe.items()},
            **delta_metrics,
            "update_over_grad_ratio": float(update_over_grad_ratio),
        }
    )


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Critic Optimizer Rule Ablation",
        "",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        f"- target minibatch: epoch `{summary['target_epoch']}` / update_epoch `{summary['target_update_epoch']}` / minibatch `{summary['target_minibatch_id']}`",
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
        "critic_optimizer_rule_ablation_%Y%m%d_%H%M%S"
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
    agent.critic_optimizer_ablation_request = {
        "epoch": int(args.epoch),
        "update_epoch": int(args.update_epoch),
        "minibatch_id": int(args.minibatch_id),
    }

    try:
        agent.train()
        raise RuntimeError("Expected critic optimizer ablation capture did not trigger.")
    except CriticOptimizerAblationCaptured:
        capture = agent.critic_optimizer_ablation_capture

    base_grads = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in agent.critic_params
    ]
    grad_norm = float(agent._grad_norm(agent.critic_params))
    before_batch_metrics, before_probe_metrics = _evaluate_group(agent, capture, probe_payload)
    before_critic_loss = _critic_loss_value(
        agent,
        capture["batch_critic_inputs"],
        capture["batch_value_targets"],
        capture["target_mean"],
        capture["target_std"],
    )
    base_agent = copy.deepcopy(agent)

    metrics_rows: list[dict[str, Any]] = []

    noop_agent = copy.deepcopy(base_agent)
    noop_batch_after, noop_probe_after = _evaluate_group(noop_agent, capture, probe_payload)
    noop_critic_loss_after = _critic_loss_value(
        noop_agent,
        capture["batch_critic_inputs"],
        capture["batch_value_targets"],
        capture["target_mean"],
        capture["target_std"],
    )
    _append_group_rows(
        metrics_rows,
        "no_op",
        before_batch_metrics,
        before_probe_metrics,
        noop_batch_after,
        noop_probe_after,
        before_critic_loss,
        noop_critic_loss_after,
        _critic_param_delta_norms(base_agent, noop_agent),
        grad_norm,
    )

    sgd_agent = copy.deepcopy(base_agent)
    _assign_copied_critic_grads(sgd_agent, base_grads)
    with torch.no_grad():
        for parameter in sgd_agent.critic_params:
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-SMALL_SGD_LR)
    sgd_batch_after, sgd_probe_after = _evaluate_group(sgd_agent, capture, probe_payload)
    sgd_critic_loss_after = _critic_loss_value(
        sgd_agent,
        capture["batch_critic_inputs"],
        capture["batch_value_targets"],
        capture["target_mean"],
        capture["target_std"],
    )
    _append_group_rows(
        metrics_rows,
        "small_sgd",
        before_batch_metrics,
        before_probe_metrics,
        sgd_batch_after,
        sgd_probe_after,
        before_critic_loss,
        sgd_critic_loss_after,
        _critic_param_delta_norms(base_agent, sgd_agent),
        grad_norm,
    )

    adam_agent = copy.deepcopy(base_agent)
    adam_agent.critic_optimizer.load_state_dict(
        copy.deepcopy(capture["critic_optimizer_state_dict"])
    )
    _assign_copied_critic_grads(adam_agent, base_grads)
    adam_agent.critic_optimizer.step()
    adam_batch_after, adam_probe_after = _evaluate_group(adam_agent, capture, probe_payload)
    adam_critic_loss_after = _critic_loss_value(
        adam_agent,
        capture["batch_critic_inputs"],
        capture["batch_value_targets"],
        capture["target_mean"],
        capture["target_std"],
    )
    _append_group_rows(
        metrics_rows,
        "current_adam",
        before_batch_metrics,
        before_probe_metrics,
        adam_batch_after,
        adam_probe_after,
        before_critic_loss,
        adam_critic_loss_after,
        _critic_param_delta_norms(base_agent, adam_agent),
        grad_norm,
    )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(root_dir / "critic_optimizer_rule_ablation_metrics.csv", index=False)
    metrics_df[
        [
            "group_name",
            "step_type",
            "probe_value_pred_mean",
            "probe_value_pred_std",
            "probe_pearson_value_vs_value_target",
            "probe_spearman_value_vs_value_target",
            "probe_pearson_value_vs_return",
            "probe_spearman_value_vs_return",
            "probe_pearson_value_vs_advantage",
            "probe_spearman_value_vs_advantage",
            "critic_param_delta_norm",
            "critic_backbone_delta_norm",
            "critic_head_delta_norm",
            "grad_norm",
            "update_over_grad_ratio",
        ]
    ].to_csv(root_dir / "critic_optimizer_rule_ablation_probe.csv", index=False)
    metrics_df[
        [
            "group_name",
            "step_type",
            "critic_loss",
            "value_pred_mean",
            "value_pred_std",
            "value_target_mean",
            "value_target_std",
            "pearson_value_vs_value_target",
            "spearman_value_vs_value_target",
            "pearson_value_vs_return",
            "spearman_value_vs_return",
            "pearson_value_vs_advantage",
            "spearman_value_vs_advantage",
            "critic_param_delta_norm",
            "critic_backbone_delta_norm",
            "critic_head_delta_norm",
            "grad_norm",
            "update_over_grad_ratio",
        ]
    ].to_csv(root_dir / "critic_optimizer_rule_ablation_minibatch.csv", index=False)

    def _after(group_name: str) -> dict[str, Any]:
        row = metrics_df[
            (metrics_df["group_name"] == group_name)
            & (metrics_df["step_type"] == "after_update")
        ].iloc[0]
        before_row = metrics_df[
            (metrics_df["group_name"] == group_name)
            & (metrics_df["step_type"] == "before_update")
        ].iloc[0]
        return {
            "batch_target_pearson_before": float(before_row["pearson_value_vs_value_target"]),
            "batch_target_pearson_after": float(row["pearson_value_vs_value_target"]),
            "probe_target_pearson_before": float(
                before_row["probe_pearson_value_vs_value_target"]
            ),
            "probe_target_pearson_after": float(row["probe_pearson_value_vs_value_target"]),
            "probe_return_pearson_before": float(before_row["probe_pearson_value_vs_return"]),
            "probe_return_pearson_after": float(row["probe_pearson_value_vs_return"]),
            "critic_param_delta_norm": float(row["critic_param_delta_norm"]),
            "critic_backbone_delta_norm": float(row["critic_backbone_delta_norm"]),
            "critic_head_delta_norm": float(row["critic_head_delta_norm"]),
            "grad_norm": float(row["grad_norm"]),
            "update_over_grad_ratio": float(row["update_over_grad_ratio"]),
        }

    group_comparison = {
        "no_op": _after("no_op"),
        "small_sgd": _after("small_sgd"),
        "current_adam": _after("current_adam"),
    }

    sgd_probe_drop = (
        group_comparison["small_sgd"]["probe_target_pearson_after"]
        - group_comparison["small_sgd"]["probe_target_pearson_before"]
    )
    adam_probe_drop = (
        group_comparison["current_adam"]["probe_target_pearson_after"]
        - group_comparison["current_adam"]["probe_target_pearson_before"]
    )
    if abs(sgd_probe_drop) < 1e-5 and abs(adam_probe_drop) < 1e-5:
        interpretation = "No material probe drift appears under either SGD or Adam on this captured step."
    elif sgd_probe_drop < -0.01 and adam_probe_drop < -0.01:
        interpretation = (
            "Both small SGD and current Adam degrade the fixed probe semantics on the same gradient; "
            "the gradient direction itself is already bad."
        )
    elif sgd_probe_drop > -0.01 and adam_probe_drop < sgd_probe_drop - 0.01:
        interpretation = (
            "Small SGD barely harms probe semantics, but current Adam is clearly worse; "
            "the optimizer rule is amplifying or distorting the bad direction."
        )
    else:
        interpretation = (
            "Both gradient direction and optimizer rule contribute, but Adam is not the only source."
        )

    summary = {
        "mode": MODE,
        "seed": args.seed,
        "checkpoint_path": str(checkpoint_path),
        "probe_step_count": PROBE_STEP_COUNT,
        "target_epoch": int(args.epoch),
        "target_update_epoch": int(args.update_epoch),
        "target_minibatch_id": int(args.minibatch_id),
        "captured_location": {
            "epoch": int(capture["epoch"]),
            "update_epoch": int(capture["update_epoch"]),
            "minibatch_id": int(capture["minibatch_id"]),
        },
        "small_sgd_lr": float(SMALL_SGD_LR),
        "group_comparison": group_comparison,
        "interpretation": interpretation,
        "output_files": {
            "metrics_csv": str(root_dir / "critic_optimizer_rule_ablation_metrics.csv"),
            "probe_csv": str(root_dir / "critic_optimizer_rule_ablation_probe.csv"),
            "minibatch_csv": str(root_dir / "critic_optimizer_rule_ablation_minibatch.csv"),
        },
    }
    (root_dir / "critic_optimizer_rule_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, root_dir / "critic_optimizer_rule_ablation_summary.md")
    print(f"[critic_optimizer_rule_ablation] wrote results to {root_dir}")


if __name__ == "__main__":
    main()
