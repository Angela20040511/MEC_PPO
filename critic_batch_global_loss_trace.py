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
from joint_training_critic_drift_trace import (
    MODE,
    PROBE_SEED_OFFSET,
    PROBE_STEP_COUNT,
    _compute_gae_targets,
    _find_current_mainline_checkpoint,
)
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator


TRACE_EPOCHS = 2
HELDOUT_BATCH_COUNT = 2
HELDOUT_SEED_OFFSET = 23000
TRACE_STAGES = [
    "before_actor",
    "after_actor_before_critic",
    "before_critic",
    "after_backward_before_step",
    "after_optimizer_step",
    "after_critic",
]
SET_TYPES = ("current_batch", "held_out_batch", "probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace whether critic optimizer.step improves the current minibatch while "
            "hurting fixed held-out minibatches and a fixed probe set."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--trace-epochs", type=int, default=TRACE_EPOCHS)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    return parser.parse_args()


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

    for step_index in range(PROBE_STEP_COUNT):
        states.append(np.array(state, dtype=np.float32, copy=True))
        action, _log_prob, value = agent.select_action(state)
        next_state, reward, done, _info = simulator.step(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        values.append(float(value))
        state = next_state
        if done:
            state = simulator.reset(seed=seed + PROBE_SEED_OFFSET + step_index + 1)

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


def _collect_fixed_heldout_payload(
    agent: PPOAgent,
    config: Any,
    seed: int,
) -> dict[str, torch.Tensor]:
    simulator = Simulator(config)
    state = simulator.reset(seed=seed + HELDOUT_SEED_OFFSET)
    states: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    values: list[float] = []

    for step_index in range(config.training.time_steps):
        states.append(np.array(state, dtype=np.float32, copy=True))
        action, _log_prob, value = agent.select_action(state)
        next_state, reward, done, _info = simulator.step(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        values.append(float(value))
        state = next_state
        if done:
            state = simulator.reset(seed=seed + HELDOUT_SEED_OFFSET + step_index + 1)

    last_value = float(agent.evaluate_value(state))
    advantages, returns = _compute_gae_targets(
        rewards=rewards,
        dones=dones,
        values=values,
        last_value=last_value,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
    )
    states_tensor = torch.tensor(np.stack(states), dtype=torch.float32, device=agent.device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=agent.device)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=agent.device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=agent.device)

    generator = torch.Generator(device=agent.device)
    generator.manual_seed(seed + HELDOUT_SEED_OFFSET)
    permutation = torch.randperm(states_tensor.size(0), generator=generator, device=agent.device)
    heldout_count = min(
        states_tensor.size(0),
        int(config.ppo.mini_batch_size) * HELDOUT_BATCH_COUNT,
    )
    indices = permutation[:heldout_count]
    return {
        "states": states_tensor[indices].clone(),
        "returns": returns_tensor[indices].clone(),
        "advantages": advantages_tensor[indices].clone(),
        "rewards": rewards_tensor[indices].clone(),
        "source_step_count": int(states_tensor.size(0)),
        "heldout_sample_count": int(heldout_count),
        "heldout_batch_count": int(max(1, heldout_count // int(config.ppo.mini_batch_size))),
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
        action, log_prob, value = agent.select_action(state)
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
        )
        state = next_state
        if done:
            break

    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)
    return agent.train()


def _wide_row_to_long_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    set_prefixes = {
        "current_batch": "",
        "held_out_batch": "heldout_",
        "probe": "probe_",
    }
    long_rows: list[dict[str, Any]] = []
    for set_type, prefix in set_prefixes.items():
        prefix_name = prefix
        long_rows.append(
            {
                "mode": row["mode"],
                "epoch": int(row["epoch"]),
                "update_epoch": int(row["update_epoch"]),
                "minibatch_id": int(row["minibatch_id"]),
                "trace_stage": row["trace_stage"],
                "trace_scope": row["trace_scope"],
                "set_type": set_type,
                "critic_loss": float(row[f"{prefix_name}critic_loss"] if prefix_name else row["critic_loss"]),
                "value_pred_mean": float(row[f"{prefix_name}value_pred_mean"]),
                "value_pred_std": float(row[f"{prefix_name}value_pred_std"]),
                "value_target_mean": float(row[f"{prefix_name}value_target_mean"]),
                "value_target_std": float(row[f"{prefix_name}value_target_std"]),
                "return_mean": float(row[f"{prefix_name}return_mean"]),
                "return_std": float(row[f"{prefix_name}return_std"]),
                "advantage_mean": float(row[f"{prefix_name}advantage_mean"]),
                "advantage_std": float(row[f"{prefix_name}advantage_std"]),
                "pearson_value_vs_value_target": float(
                    row[f"{prefix_name}pearson_value_vs_value_target"]
                ),
                "spearman_value_vs_value_target": float(
                    row[f"{prefix_name}spearman_value_vs_value_target"]
                ),
                "pearson_value_vs_return": float(row[f"{prefix_name}pearson_value_vs_return"]),
                "spearman_value_vs_return": float(row[f"{prefix_name}spearman_value_vs_return"]),
                "pearson_value_vs_advantage": float(
                    row[f"{prefix_name}pearson_value_vs_advantage"]
                ),
                "spearman_value_vs_advantage": float(
                    row[f"{prefix_name}spearman_value_vs_advantage"]
                ),
                "pearson_value_vs_one_step_reward": float(
                    row[f"{prefix_name}pearson_value_vs_one_step_reward"]
                ),
                "spearman_value_vs_one_step_reward": float(
                    row[f"{prefix_name}spearman_value_vs_one_step_reward"]
                ),
                "critic_grad_norm": float(row["critic_grad_norm"]),
                "actor_grad_norm": float(row["actor_grad_norm"]),
                "critic_param_delta_norm": float(row["critic_param_delta_norm"]),
                "actor_param_delta_norm": float(row["actor_param_delta_norm"]),
                "critic_backbone_delta_norm": float(row["critic_backbone_delta_norm"]),
                "critic_head_delta_norm": float(row["critic_head_delta_norm"]),
                "popart_mean": float(row["popart_mean"]),
                "popart_std": float(row["popart_std"]),
            }
        )
    return long_rows


def _compute_stage_delta(
    pivot_row: pd.Series,
    set_type: str,
    metric: str,
    before_stage: str = "after_backward_before_step",
    after_stage: str = "after_optimizer_step",
) -> float:
    return float(pivot_row[f"{metric}_{before_stage}"] - pivot_row[f"{metric}_{after_stage}"]) * -1.0


def _analyse_long_trace(long_df: pd.DataFrame, epoch_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = long_df[
        long_df["trace_stage"].isin(TRACE_STAGES) & (long_df["trace_scope"] == "minibatch")
    ].copy()
    stage_order = {stage: index for index, stage in enumerate(TRACE_STAGES)}
    ordered["trace_stage_order"] = ordered["trace_stage"].map(stage_order)
    ordered = ordered.sort_values(
        ["set_type", "epoch", "update_epoch", "minibatch_id", "trace_stage_order"]
    ).reset_index(drop=True)

    pivot = ordered.pivot_table(
        index=["epoch", "update_epoch", "minibatch_id"],
        columns=["set_type", "trace_stage"],
        values=[
            "critic_loss",
            "pearson_value_vs_value_target",
            "spearman_value_vs_value_target",
            "pearson_value_vs_return",
            "spearman_value_vs_return",
            "pearson_value_vs_advantage",
            "spearman_value_vs_advantage",
        ],
        aggfunc="first",
    )
    pivot.columns = [
        f"{metric}_{set_type}_{stage}"
        for metric, set_type, stage in pivot.columns.to_flat_index()
    ]
    pivot = pivot.reset_index().sort_values(["epoch", "update_epoch", "minibatch_id"])

    for set_type in SET_TYPES:
        pivot[f"{set_type}_delta_critic_loss"] = (
            pivot[f"critic_loss_{set_type}_after_optimizer_step"]
            - pivot[f"critic_loss_{set_type}_after_backward_before_step"]
        )
        pivot[f"{set_type}_delta_target_pearson"] = (
            pivot[f"pearson_value_vs_value_target_{set_type}_after_optimizer_step"]
            - pivot[f"pearson_value_vs_value_target_{set_type}_after_backward_before_step"]
        )
        pivot[f"{set_type}_delta_return_pearson"] = (
            pivot[f"pearson_value_vs_return_{set_type}_after_optimizer_step"]
            - pivot[f"pearson_value_vs_return_{set_type}_after_backward_before_step"]
        )
        pivot[f"{set_type}_delta_advantage_pearson"] = (
            pivot[f"pearson_value_vs_advantage_{set_type}_after_optimizer_step"]
            - pivot[f"pearson_value_vs_advantage_{set_type}_after_backward_before_step"]
        )

    first_batch_better_global_worse = None
    for _, row in pivot.iterrows():
        current_batch_improves = (
            float(row["current_batch_delta_critic_loss"]) < 0.0
            and float(row["current_batch_delta_target_pearson"]) > 0.0
        )
        heldout_worsens = (
            float(row["held_out_batch_delta_critic_loss"]) > 0.0
            or float(row["held_out_batch_delta_target_pearson"]) < 0.0
        )
        probe_worsens = (
            float(row["probe_delta_critic_loss"]) > 0.0
            or float(row["probe_delta_target_pearson"]) < 0.0
        )
        if current_batch_improves and heldout_worsens and probe_worsens:
            first_batch_better_global_worse = {
                "epoch": int(row["epoch"]),
                "update_epoch": int(row["update_epoch"]),
                "minibatch_id": int(row["minibatch_id"]),
                "trace_stage": "after_optimizer_step",
                "current_batch": {
                    "delta_critic_loss": float(row["current_batch_delta_critic_loss"]),
                    "delta_target_pearson": float(row["current_batch_delta_target_pearson"]),
                    "delta_return_pearson": float(row["current_batch_delta_return_pearson"]),
                    "delta_advantage_pearson": float(row["current_batch_delta_advantage_pearson"]),
                },
                "held_out_batch": {
                    "delta_critic_loss": float(row["held_out_batch_delta_critic_loss"]),
                    "delta_target_pearson": float(row["held_out_batch_delta_target_pearson"]),
                    "delta_return_pearson": float(row["held_out_batch_delta_return_pearson"]),
                    "delta_advantage_pearson": float(row["held_out_batch_delta_advantage_pearson"]),
                },
                "probe": {
                    "delta_critic_loss": float(row["probe_delta_critic_loss"]),
                    "delta_target_pearson": float(row["probe_delta_target_pearson"]),
                    "delta_return_pearson": float(row["probe_delta_return_pearson"]),
                    "delta_advantage_pearson": float(row["probe_delta_advantage_pearson"]),
                },
            }
            break

    delta_summary = {}
    for set_type in SET_TYPES:
        delta_summary[set_type] = {
            "mean_delta_critic_loss_after_optimizer_step": float(
                pivot[f"{set_type}_delta_critic_loss"].mean()
            ),
            "mean_delta_target_pearson_after_optimizer_step": float(
                pivot[f"{set_type}_delta_target_pearson"].mean()
            ),
            "mean_delta_return_pearson_after_optimizer_step": float(
                pivot[f"{set_type}_delta_return_pearson"].mean()
            ),
            "mean_delta_advantage_pearson_after_optimizer_step": float(
                pivot[f"{set_type}_delta_advantage_pearson"].mean()
            ),
            "min_delta_target_pearson_after_optimizer_step": float(
                pivot[f"{set_type}_delta_target_pearson"].min()
            ),
            "max_delta_target_pearson_after_optimizer_step": float(
                pivot[f"{set_type}_delta_target_pearson"].max()
            ),
        }

    return {
        "first_batch_better_global_worse": first_batch_better_global_worse,
        "after_optimizer_step_delta_summary": delta_summary,
        "epoch_summaries": epoch_summaries,
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Critic Batch Global Loss Trace",
        "",
        f"- mode: `{summary['mode']}`",
        f"- actor_frozen: `{summary['actor_frozen']}`",
        f"- heldout_sample_count: `{summary['heldout_payload']['heldout_sample_count']}`",
        f"- interpretation: {summary['interpretation']}",
        "",
        "## First Batch-Better / Global-Worse Point",
        "",
    ]
    if summary["first_batch_better_global_worse"] is None:
        lines.append("- none within traced epochs")
    else:
        lines.extend(
            [
                "```json",
                json.dumps(summary["first_batch_better_global_worse"], ensure_ascii=False, indent=2),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## After Optimizer Delta Summary",
            "",
            "```json",
            json.dumps(summary["after_optimizer_step_delta_summary"], ensure_ascii=False, indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    root_dir = output_root / datetime.now().strftime(
        "critic_batch_global_loss_trace_%Y%m%d_%H%M%S"
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
    agent.critic_training_internal_stage_trace = True

    probe_payload = _collect_fixed_probe_payload(agent, config, args.seed)
    heldout_payload = _collect_fixed_heldout_payload(agent, config, args.seed)
    agent.critic_training_drift_trace_rows = []
    agent.critic_training_drift_probe_payload = probe_payload
    agent.critic_training_drift_heldout_payload = heldout_payload

    simulator = Simulator(config)
    epoch_summaries: list[dict[str, Any]] = []
    for epoch in range(args.trace_epochs):
        rows_before = len(agent.critic_training_drift_trace_rows)
        losses = _collect_one_epoch(agent, simulator, config, epoch)
        rows_after = len(agent.critic_training_drift_trace_rows)
        epoch_rows = agent.critic_training_drift_trace_rows[rows_before:rows_after]
        epoch_df = pd.DataFrame(epoch_rows)
        after_critic_rows = (
            epoch_df[epoch_df["trace_stage"] == "after_critic"].reset_index(drop=True)
            if not epoch_df.empty
            else pd.DataFrame()
        )
        epoch_summaries.append(
            {
                "epoch": int(epoch),
                "actor_loss": float(losses.get("actor_loss", 0.0)),
                "critic_loss": float(losses.get("critic_loss", 0.0)),
                "route_update_count": int(losses.get("route_update_count", 0)),
                "theta_update_count": int(losses.get("theta_update_count", 0)),
                "after_critic_current_batch_target_pearson_mean": float(
                    after_critic_rows["pearson_value_vs_value_target"].mean()
                    if not after_critic_rows.empty
                    else 0.0
                ),
                "after_critic_heldout_target_pearson_mean": float(
                    after_critic_rows["heldout_pearson_value_vs_value_target"].mean()
                    if not after_critic_rows.empty
                    else 0.0
                ),
                "after_critic_probe_target_pearson_mean": float(
                    after_critic_rows["probe_pearson_value_vs_value_target"].mean()
                    if not after_critic_rows.empty
                    else 0.0
                ),
            }
        )

    wide_df = pd.DataFrame(agent.critic_training_drift_trace_rows)
    wide_df.to_csv(root_dir / "critic_batch_global_loss_trace_wide.csv", index=False)

    long_rows: list[dict[str, Any]] = []
    for row in wide_df.to_dict(orient="records"):
        long_rows.extend(_wide_row_to_long_rows(row))
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(root_dir / "critic_batch_global_loss_trace.csv", index=False)
    long_df[long_df["set_type"] == "probe"].to_csv(
        root_dir / "critic_batch_global_loss_trace_probe.csv",
        index=False,
    )
    long_df[long_df["set_type"] == "current_batch"].to_csv(
        root_dir / "critic_batch_global_loss_trace_current_batch.csv",
        index=False,
    )
    long_df[long_df["set_type"] == "held_out_batch"].to_csv(
        root_dir / "critic_batch_global_loss_trace_heldout_batch.csv",
        index=False,
    )

    analysis = _analyse_long_trace(long_df, epoch_summaries)
    interpretation = (
        "Optimizer steps repeatedly improve current-batch fit while hurting held-out and probe "
        "semantics."
        if analysis["first_batch_better_global_worse"] is not None
        else "No clear current-batch-better / global-worse pattern appears within the traced window."
    )
    summary = {
        "mode": MODE,
        "seed": args.seed,
        "trace_epochs": int(args.trace_epochs),
        "checkpoint_path": str(checkpoint_path),
        "actor_frozen": True,
        "probe_step_count": PROBE_STEP_COUNT,
        "heldout_payload": {
            "heldout_sample_count": int(heldout_payload["heldout_sample_count"]),
            "heldout_batch_count": int(heldout_payload["heldout_batch_count"]),
            "source_step_count": int(heldout_payload["source_step_count"]),
            "selection_reused_globally": True,
        },
        "probe_payload": {
            "probe_step_count": PROBE_STEP_COUNT,
            "selection_reused_globally": True,
        },
        "actor_param_delta_norm_max": float(
            long_df["actor_param_delta_norm"].max() if not long_df.empty else 0.0
        ),
        **analysis,
        "interpretation": interpretation,
        "output_files": {
            "trace_csv": str(root_dir / "critic_batch_global_loss_trace.csv"),
            "probe_csv": str(root_dir / "critic_batch_global_loss_trace_probe.csv"),
            "current_batch_csv": str(
                root_dir / "critic_batch_global_loss_trace_current_batch.csv"
            ),
            "heldout_batch_csv": str(
                root_dir / "critic_batch_global_loss_trace_heldout_batch.csv"
            ),
        },
    }
    summary = _to_builtin(summary)
    (root_dir / "critic_batch_global_loss_trace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, root_dir / "critic_batch_global_loss_trace_summary.md")
    print(f"[critic_batch_global_loss_trace] wrote results to {root_dir}")


if __name__ == "__main__":
    main()
