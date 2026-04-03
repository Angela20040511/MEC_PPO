import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dense_actor_input_experiment import build_state_layout
from dense_critic_diagnosis import compute_joint_counterfactual_scores
from dense_policy_ratio_mode_experiment import build_policy_ratio_mode_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator


MODE = "hierarchical_actor_joint_reward_aligned_credit"
TRACE_EPOCHS = 2
PROBE_STEP_COUNT = 48
PROBE_SEED_OFFSET = 17000
TRACE_STAGE_ORDER = {
    "before_actor": 0,
    "after_actor_before_critic": 1,
    "after_critic": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace the first two continued training epochs of the current joint "
            "reward-aligned actor mainline to find where critic semantics drift."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--trace-epochs", type=int, default=TRACE_EPOCHS)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    parser.add_argument(
        "--load-current-checkpoint",
        action="store_true",
        help="Continue trace from the latest saved joint_reward_aligned checkpoint instead of tracing from fresh initialization.",
    )
    return parser.parse_args()


def _find_current_mainline_checkpoint(root: Path) -> Path:
    candidates = sorted(
        root.glob(
            "dense_policy_joint_reward_aligned_credit_experiment_*/"
            "policy_ratio_hierarchical_actor_joint_reward_aligned_credit_*/best_model.pt"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "Unable to find a hierarchical_actor_joint_reward_aligned_credit checkpoint."
        )
    return candidates[0]


def _compute_gae_targets(
    rewards: list[float],
    dones: list[bool],
    values: list[float],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    returns = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    next_value = float(last_value)
    for index in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[index])
        delta = rewards[index] + gamma * next_value * mask - values[index]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[index] = gae
        returns[index] = advantages[index] + values[index]
        next_value = values[index]
    return advantages, returns


def _collect_fixed_probe_payload(
    agent: PPOAgent,
    config: Any,
    seed: int,
) -> dict[str, torch.Tensor]:
    simulator = Simulator(config)
    state = simulator.reset(seed=seed + PROBE_SEED_OFFSET)
    states: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    values: list[float] = []

    for _ in range(PROBE_STEP_COUNT):
        states.append(np.array(state, dtype=np.float32, copy=True))
        action, _log_prob, value, _policy_cache = agent.select_action_with_info(state)
        next_state, reward, done, _info = simulator.step(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        values.append(float(value))
        state = next_state
        if done:
            state = simulator.reset(seed=seed + PROBE_SEED_OFFSET + len(states))

    last_value = float(agent.evaluate_value(state))
    advantages, returns = _compute_gae_targets(
        rewards=rewards,
        dones=dones,
        values=values,
        last_value=last_value,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
    )
    return {
        "states": torch.tensor(np.stack(states), dtype=torch.float32, device=agent.device),
        "returns": torch.tensor(returns, dtype=torch.float32, device=agent.device),
        "advantages": torch.tensor(advantages, dtype=torch.float32, device=agent.device),
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=agent.device),
    }


def _collect_one_epoch(
    agent: PPOAgent,
    simulator: Simulator,
    config: Any,
    epoch: int,
) -> dict[str, Any]:
    state = simulator.reset(seed=config.training.seed + epoch)
    done = False
    for _ in range(config.training.time_steps):
        action, log_prob, value, policy_cache = agent.select_action_with_info(state)
        joint_reward_aligned_scores, _joint_td_scores = compute_joint_counterfactual_scores(
            agent,
            simulator,
            np.asarray(action, dtype=np.float32),
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
            policy_cache=policy_cache,
        )
        state = next_state
        if done:
            break

    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)
    return agent.train()


def _first_stage_drop(
    pivot_df: pd.DataFrame,
    delta_prefix: str,
    threshold: float = -0.01,
) -> dict[str, Any] | None:
    for _, row in pivot_df.iterrows():
        deltas = [
            float(row[f"{delta_prefix}_probe_pearson_value_vs_value_target"]),
            float(row[f"{delta_prefix}_probe_pearson_value_vs_return"]),
            float(row[f"{delta_prefix}_probe_spearman_value_vs_value_target"]),
            float(row[f"{delta_prefix}_probe_spearman_value_vs_return"]),
        ]
        if min(deltas) <= threshold:
            return row.to_dict()
    return None


def _analyse_trace(trace_df: pd.DataFrame, epoch_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if trace_df.empty:
        return {
            "classification": "no_trace_rows",
            "detail": "No critic drift trace rows were captured.",
            "first_divergence": None,
            "initial_probe_metrics": None,
            "epoch_summaries": epoch_summaries,
        }

    ordered_df = trace_df.copy()
    ordered_df["trace_stage_order"] = ordered_df["trace_stage"].map(TRACE_STAGE_ORDER)
    ordered_df = ordered_df.sort_values(
        ["epoch", "update_epoch", "minibatch_id", "trace_stage_order"]
    ).reset_index(drop=True)

    initial_row = ordered_df.iloc[0].to_dict()
    minibatch_pivot = ordered_df.pivot_table(
        index=["epoch", "update_epoch", "minibatch_id"],
        columns="trace_stage",
        values=[
            "probe_pearson_value_vs_value_target",
            "probe_spearman_value_vs_value_target",
            "probe_pearson_value_vs_return",
            "probe_spearman_value_vs_return",
            "probe_pearson_value_vs_advantage",
            "probe_spearman_value_vs_advantage",
            "probe_value_pred_mean",
            "probe_value_pred_std",
            "critic_param_delta_norm",
            "actor_param_delta_norm",
        ],
        aggfunc="first",
    )
    minibatch_pivot.columns = [
        f"{metric}_{stage}" for metric, stage in minibatch_pivot.columns.to_flat_index()
    ]
    minibatch_pivot = minibatch_pivot.reset_index().sort_values(
        ["epoch", "update_epoch", "minibatch_id"]
    )

    for metric in [
        "probe_pearson_value_vs_value_target",
        "probe_spearman_value_vs_value_target",
        "probe_pearson_value_vs_return",
        "probe_spearman_value_vs_return",
        "probe_pearson_value_vs_advantage",
        "probe_spearman_value_vs_advantage",
    ]:
        minibatch_pivot[f"actor_delta_{metric}"] = (
            minibatch_pivot[f"{metric}_after_actor_before_critic"]
            - minibatch_pivot[f"{metric}_before_actor"]
        )
        minibatch_pivot[f"critic_delta_{metric}"] = (
            minibatch_pivot[f"{metric}_after_critic"]
            - minibatch_pivot[f"{metric}_after_actor_before_critic"]
        )

    first_actor_drop = _first_stage_drop(minibatch_pivot, "actor_delta")
    first_critic_drop = _first_stage_drop(minibatch_pivot, "critic_delta")

    initial_bad = (
        float(initial_row["probe_pearson_value_vs_value_target"]) < -0.05
        and float(initial_row["probe_pearson_value_vs_return"]) < -0.05
    )

    if initial_bad:
        if first_critic_drop is not None and first_actor_drop is None:
            classification = "A"
            detail = (
                "The probe critic semantics are already reversed at the first traced stage, "
                "and the first additional measurable worsening appears after a critic step."
            )
            first_divergence = {
                "where": "after_critic",
                "epoch": int(first_critic_drop["epoch"]),
                "update_epoch": int(first_critic_drop["update_epoch"]),
                "minibatch_id": int(first_critic_drop["minibatch_id"]),
            }
        elif first_actor_drop is not None and first_critic_drop is None:
            classification = "B"
            detail = (
                "The probe critic semantics are already reversed at the first traced stage, "
                "and the first additional measurable worsening appears after an actor step."
            )
            first_divergence = {
                "where": "after_actor_before_critic",
                "epoch": int(first_actor_drop["epoch"]),
                "update_epoch": int(first_actor_drop["update_epoch"]),
                "minibatch_id": int(first_actor_drop["minibatch_id"]),
            }
        elif first_actor_drop is not None and first_critic_drop is not None:
            classification = "D"
            detail = (
                "The probe critic semantics are already reversed at the first traced stage, "
                "and both actor and critic steps contribute additional degradation."
            )
            earlier = first_actor_drop
            stage = "after_actor_before_critic"
            if (
                int(first_critic_drop["epoch"]),
                int(first_critic_drop["update_epoch"]),
                int(first_critic_drop["minibatch_id"]),
            ) < (
                int(first_actor_drop["epoch"]),
                int(first_actor_drop["update_epoch"]),
                int(first_actor_drop["minibatch_id"]),
            ):
                earlier = first_critic_drop
                stage = "after_critic"
            first_divergence = {
                "where": stage,
                "epoch": int(earlier["epoch"]),
                "update_epoch": int(earlier["update_epoch"]),
                "minibatch_id": int(earlier["minibatch_id"]),
            }
        else:
            classification = "C"
            detail = (
                "The probe critic semantics are already reversed at the first traced stage, "
                "and no single actor/critic step causes a sharp break within the traced window; "
                "the bad semantics persist with only mild additional drift."
            )
            first_divergence = {
                "where": "before_actor",
                "epoch": int(initial_row["epoch"]),
                "update_epoch": int(initial_row["update_epoch"]),
                "minibatch_id": int(initial_row["minibatch_id"]),
            }
    else:
        if first_critic_drop is not None and first_actor_drop is None:
            classification = "A"
            detail = "The first clear critic semantic drop appears immediately after a critic step."
            first_divergence = {
                "where": "after_critic",
                "epoch": int(first_critic_drop["epoch"]),
                "update_epoch": int(first_critic_drop["update_epoch"]),
                "minibatch_id": int(first_critic_drop["minibatch_id"]),
            }
        elif first_actor_drop is not None and first_critic_drop is None:
            classification = "B"
            detail = "The first clear critic semantic drop appears after an actor step."
            first_divergence = {
                "where": "after_actor_before_critic",
                "epoch": int(first_actor_drop["epoch"]),
                "update_epoch": int(first_actor_drop["update_epoch"]),
                "minibatch_id": int(first_actor_drop["minibatch_id"]),
            }
        elif first_actor_drop is not None and first_critic_drop is not None:
            classification = "D"
            detail = "Both actor and critic steps contribute to the first visible semantic drift."
            first_divergence = {
                "where": "after_actor_before_critic",
                "epoch": int(first_actor_drop["epoch"]),
                "update_epoch": int(first_actor_drop["update_epoch"]),
                "minibatch_id": int(first_actor_drop["minibatch_id"]),
            }
        else:
            classification = "C"
            detail = "No sharp single-step break is visible; the drift is gradual across minibatches."
            first_divergence = None

    return {
        "classification": classification,
        "detail": detail,
        "first_divergence": first_divergence,
        "initial_probe_metrics": {
            key: initial_row[key]
            for key in [
                "epoch",
                "update_epoch",
                "minibatch_id",
                "trace_stage",
                "probe_pearson_value_vs_value_target",
                "probe_spearman_value_vs_value_target",
                "probe_pearson_value_vs_return",
                "probe_spearman_value_vs_return",
                "probe_pearson_value_vs_advantage",
                "probe_spearman_value_vs_advantage",
            ]
        },
        "first_actor_drop": first_actor_drop,
        "first_critic_drop": first_critic_drop,
        "epoch_summaries": epoch_summaries,
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Joint Training Critic Drift Trace",
        "",
        f"- classification: `{summary['classification']}`",
        f"- detail: {summary['detail']}",
        "",
        "## Initial Probe Metrics",
        "",
        "```json",
        json.dumps(summary["initial_probe_metrics"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## First Divergence",
        "",
    ]
    if summary["first_divergence"] is None:
        lines.append("- none within traced epochs")
    else:
        lines.extend(
            [
                "```json",
                json.dumps(summary["first_divergence"], ensure_ascii=False, indent=2),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## Epoch Summary",
            "",
            "| epoch | route_update_count | actor_loss | critic_loss | probe value-target pearson | probe return pearson |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["epoch_summaries"]:
        lines.append(
            f"| {int(row['epoch'])} | {int(row['route_update_count'])} | "
            f"{float(row['actor_loss']):.6f} | {float(row['critic_loss']):.6f} | "
            f"{float(row['probe_after_critic_pearson_value_vs_value_target']):.6f} | "
            f"{float(row['probe_after_critic_pearson_value_vs_return']):.6f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    root_dir = output_root / datetime.now().strftime(
        "joint_training_critic_drift_trace_%Y%m%d_%H%M%S"
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    set_global_seeds(args.seed)
    config = build_policy_ratio_mode_config(policy_ratio_mode=MODE, seed=args.seed)
    checkpoint_path = (
        _find_current_mainline_checkpoint(output_root)
        if args.load_current_checkpoint
        else None
    )
    state_layout = build_state_layout(config)
    agent = PPOAgent(
        config=config.ppo,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        critic_state_layout=state_layout,
    )
    if checkpoint_path is not None:
        agent.load(str(checkpoint_path), load_optimizer=True)
    simulator = Simulator(config)
    probe_payload = _collect_fixed_probe_payload(agent, config, args.seed)
    agent.critic_training_drift_trace_rows = []
    agent.critic_training_drift_probe_payload = probe_payload

    epoch_summaries: list[dict[str, Any]] = []
    for epoch in range(args.trace_epochs):
        rows_before = len(agent.critic_training_drift_trace_rows)
        losses = _collect_one_epoch(agent, simulator, config, epoch)
        rows_after = len(agent.critic_training_drift_trace_rows)
        epoch_rows = agent.critic_training_drift_trace_rows[rows_before:rows_after]
        epoch_df = pd.DataFrame(epoch_rows)
        probe_after_critic = (
            epoch_df[epoch_df["trace_stage"] == "after_critic"].reset_index(drop=True)
            if not epoch_df.empty
            else pd.DataFrame()
        )
        epoch_summaries.append(
            {
                "mode": MODE,
                "epoch": int(epoch),
                "actor_loss": float(losses.get("actor_loss", 0.0)),
                "critic_loss": float(losses.get("critic_loss", 0.0)),
                "route_update_count": int(losses.get("route_update_count", 0)),
                "theta_update_count": int(losses.get("theta_update_count", 0)),
                "probe_after_critic_pearson_value_vs_value_target": float(
                    probe_after_critic["probe_pearson_value_vs_value_target"].mean()
                    if not probe_after_critic.empty
                    else 0.0
                ),
                "probe_after_critic_pearson_value_vs_return": float(
                    probe_after_critic["probe_pearson_value_vs_return"].mean()
                    if not probe_after_critic.empty
                    else 0.0
                ),
            }
        )

    trace_df = pd.DataFrame(agent.critic_training_drift_trace_rows)
    minibatch_cols = [
        "mode",
        "epoch",
        "update_epoch",
        "minibatch_id",
        "trace_stage",
        "critic_loss",
        "actor_loss",
        "critic_grad_norm",
        "actor_grad_norm",
        "critic_param_delta_norm",
        "actor_param_delta_norm",
        "critic_backbone_delta_norm",
        "critic_head_delta_norm",
        "actor_backbone_delta_norm",
        "actor_head_delta_norm",
        "value_pred_mean",
        "value_pred_std",
        "value_target_mean",
        "value_target_std",
        "return_mean",
        "return_std",
        "advantage_mean",
        "advantage_std",
        "pearson_value_vs_value_target",
        "spearman_value_vs_value_target",
        "pearson_value_vs_return",
        "spearman_value_vs_return",
        "pearson_value_vs_advantage",
        "spearman_value_vs_advantage",
        "pearson_value_vs_one_step_reward",
        "spearman_value_vs_one_step_reward",
        "popart_mean",
        "popart_std",
    ]
    probe_cols = [
        "mode",
        "epoch",
        "update_epoch",
        "minibatch_id",
        "trace_stage",
        "probe_value_pred_mean",
        "probe_value_pred_std",
        "probe_value_target_mean",
        "probe_value_target_std",
        "probe_return_mean",
        "probe_return_std",
        "probe_advantage_mean",
        "probe_advantage_std",
        "probe_pearson_value_vs_value_target",
        "probe_spearman_value_vs_value_target",
        "probe_pearson_value_vs_return",
        "probe_spearman_value_vs_return",
        "probe_pearson_value_vs_advantage",
        "probe_spearman_value_vs_advantage",
        "probe_pearson_value_vs_one_step_reward",
        "probe_spearman_value_vs_one_step_reward",
    ]
    trace_df.to_csv(root_dir / "joint_training_critic_drift_trace.csv", index=False)
    trace_df[minibatch_cols].to_csv(
        root_dir / "joint_training_critic_drift_trace_minibatch.csv",
        index=False,
    )
    trace_df[probe_cols].to_csv(
        root_dir / "joint_training_critic_drift_trace_probe.csv",
        index=False,
    )

    analysis = _analyse_trace(trace_df, epoch_summaries)
    summary = {
        "mode": MODE,
        "seed": args.seed,
        "trace_epochs": int(args.trace_epochs),
        "load_current_checkpoint": bool(args.load_current_checkpoint),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
        "probe_step_count": PROBE_STEP_COUNT,
        **analysis,
        "output_files": {
            "trace_csv": str(root_dir / "joint_training_critic_drift_trace.csv"),
            "probe_csv": str(root_dir / "joint_training_critic_drift_trace_probe.csv"),
            "minibatch_csv": str(root_dir / "joint_training_critic_drift_trace_minibatch.csv"),
        },
    }
    (root_dir / "joint_training_critic_drift_trace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, root_dir / "joint_training_critic_drift_trace_summary.md")
    print(f"[joint_training_critic_drift_trace] wrote results to {root_dir}")


if __name__ == "__main__":
    main()
