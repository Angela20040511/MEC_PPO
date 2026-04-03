import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dense_actor_input_experiment import build_state_layout
from dense_policy_ratio_mode_experiment import build_policy_ratio_mode_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator


BASELINE_MODE = "hierarchical_actor_true_conditional_route_policy"
CANDIDATE_MODE = "hierarchical_actor_true_conditional_route_candidate_score_credit"
TRACE_EPOCHS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace the first two training epochs to find where candidate-score route "
            "updates drift from single-step viability to training-time non-updates."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    return parser.parse_args()


def _collect_one_epoch(agent: PPOAgent, simulator: Simulator, config: Any, epoch: int) -> dict[str, Any]:
    state = simulator.reset(seed=config.training.seed + epoch)
    done = False
    for _ in range(config.training.time_steps):
        action, log_prob, value, policy_cache = agent.select_action_with_info(state)
        next_state, reward, done, _ = simulator.step(action)
        agent.store_transition(
            state,
            action,
            log_prob,
            reward,
            done,
            value,
            next_state,
            policy_cache=policy_cache,
        )
        state = next_state
        if done:
            break
    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)
    return agent.train()


def run_mode_trace(mode: str, seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    set_global_seeds(seed)
    config = build_policy_ratio_mode_config(policy_ratio_mode=mode, seed=seed)
    state_layout = build_state_layout(config)
    agent = PPOAgent(
        config=config.ppo,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        critic_state_layout=state_layout,
    )
    simulator = Simulator(config)
    agent.route_training_drift_trace_rows = []

    epoch_summaries: list[dict[str, Any]] = []
    for epoch in range(TRACE_EPOCHS):
        rows_before = len(agent.route_training_drift_trace_rows)
        losses = _collect_one_epoch(agent, simulator, config, epoch)
        rows_after = len(agent.route_training_drift_trace_rows)
        epoch_rows = agent.route_training_drift_trace_rows[rows_before:rows_after]
        trace_df = pd.DataFrame(epoch_rows)
        epoch_summaries.append(
            {
                "mode": mode,
                "epoch": epoch,
                "train_route_update_count": int(losses.get("route_update_count", 0)),
                "train_route_head_grad_norm": float(losses.get("route_head_grad_norm", 0.0)),
                "train_route_backbone_grad_norm": float(
                    losses.get("route_backbone_grad_norm", 0.0)
                ),
                "trace_rows": int(len(epoch_rows)),
                "trace_optimizer_step_executed_sum": float(
                    trace_df["optimizer_step_executed"].sum() if not trace_df.empty else 0.0
                ),
                "trace_counted_into_route_update_count_sum": float(
                    trace_df["counted_into_route_update_count"].sum()
                    if not trace_df.empty
                    else 0.0
                ),
                "trace_route_param_delta_norm_sum": float(
                    trace_df["route_param_delta_norm"].sum() if not trace_df.empty else 0.0
                ),
                "trace_bare_route_head_grad_norm_mean": float(
                    trace_df["bare_route_head_grad_norm"].mean()
                    if not trace_df.empty
                    else 0.0
                ),
                "trace_bare_route_backbone_grad_norm_mean": float(
                    trace_df["bare_route_backbone_grad_norm"].mean()
                    if not trace_df.empty
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(agent.route_training_drift_trace_rows), epoch_summaries


def _first_row_with_condition(df: pd.DataFrame, condition) -> dict[str, Any] | None:
    if df.empty:
        return None
    matches = df[condition(df)]
    if matches.empty:
        return None
    row = matches.sort_values(["epoch", "update_epoch", "minibatch_id"]).iloc[0]
    return row.to_dict()


def analyse_trace(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    baseline_epoch_summaries: list[dict[str, Any]],
    candidate_epoch_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_epoch_df = pd.DataFrame(baseline_epoch_summaries)
    candidate_epoch_df = pd.DataFrame(candidate_epoch_summaries)

    candidate_summary_mismatch = _first_row_with_condition(
        candidate_epoch_df,
        lambda df: df["trace_counted_into_route_update_count_sum"]
        != df["train_route_update_count"],
    )
    baseline_summary_mismatch = _first_row_with_condition(
        baseline_epoch_df,
        lambda df: df["trace_counted_into_route_update_count_sum"]
        != df["train_route_update_count"],
    )

    candidate_failed_step = _first_row_with_condition(
        candidate_df,
        lambda df: (
            (df["bare_route_head_grad_norm"] > 0.0)
            | (df["bare_route_backbone_grad_norm"] > 0.0)
        )
        & (
            (df["optimizer_step_executed"] <= 0.0)
            | (df["route_param_delta_norm"] <= 0.0)
            | (df["counted_into_route_update_count"] <= 0.0)
        ),
    )

    candidate_zero_grad_step = _first_row_with_condition(
        candidate_df,
        lambda df: (df["bare_route_head_grad_norm"] <= 0.0)
        & (df["bare_route_backbone_grad_norm"] <= 0.0),
    )

    candidate_late_collapse = _first_row_with_condition(
        candidate_df,
        lambda df: (df["epoch"] >= 1)
        & (
            (df["route_param_delta_norm"] <= 0.0)
            | (df["counted_into_route_update_count"] <= 0.0)
        ),
    )

    if candidate_df.empty and not baseline_df.empty:
        classification = "完整训练控制路径未接入"
        detail = (
            "candidate 组在前两轮里没有生成任何 route-only trace row；这不是统计口径丢记，"
            "而是完整训练路径在进入 route branch trace 之前就已经被 mode 分发挡住了。"
        )
        first_divergence = {
            "where": "_uses_blockwise_policy_surrogate gate before the first route minibatch",
            "epoch": 0,
            "update_epoch": 0,
            "minibatch_id": 0,
        }
    elif candidate_summary_mismatch is not None or baseline_summary_mismatch is not None:
        classification = "统计口径问题"
        detail = (
            "trace 里确实执行并计数了 route step，但 train summary 的 route_update_count 不一致。"
        )
        first_divergence = candidate_summary_mismatch or baseline_summary_mismatch
    elif candidate_failed_step is not None:
        classification = "theta route 多步交互问题"
        detail = (
            "candidate 组在完整训练中出现了 bare backward 仍有梯度、但真实 step/计数掉线的首个 route minibatch。"
        )
        first_divergence = candidate_failed_step
    elif candidate_zero_grad_step is not None:
        classification = "完整训练动态塌缩"
        detail = "candidate 组在完整训练前两轮内出现了 route bare gradient 直接塌到 0 的首个 minibatch。"
        first_divergence = candidate_zero_grad_step
    else:
        classification = "完整训练动态塌缩"
        detail = (
            "前两轮 trace 内没有出现统计口径失配，也没有出现单步 route step 被 guard 压掉；"
            "如果长程训练最后变成 route_update_count=0，更像是更晚发生的训练动态塌缩。"
        )
        first_divergence = candidate_late_collapse

    return {
        "classification": classification,
        "detail": detail,
        "first_divergence": first_divergence,
        "baseline_epoch_summaries": baseline_epoch_summaries,
        "candidate_epoch_summaries": candidate_epoch_summaries,
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    baseline_epoch_df = pd.DataFrame(summary["baseline_epoch_summaries"])
    candidate_epoch_df = pd.DataFrame(summary["candidate_epoch_summaries"])
    first_divergence = summary.get("first_divergence")

    lines = [
        "# Route Training Drift Trace",
        "",
        f"- classification: `{summary['classification']}`",
        f"- detail: {summary['detail']}",
        "",
        "## Epoch Summary",
        "",
        "| mode | epoch | train route_update_count | trace counted sum | trace optimizer sum | trace route_param_delta sum |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for df in (baseline_epoch_df, candidate_epoch_df):
        for _, row in df.iterrows():
            lines.append(
                f"| {row['mode']} | {int(row['epoch'])} | {int(row['train_route_update_count'])} | "
                f"{row['trace_counted_into_route_update_count_sum']:.0f} | "
                f"{row['trace_optimizer_step_executed_sum']:.0f} | "
                f"{row['trace_route_param_delta_norm_sum']:.6f} |"
            )

    lines.extend(["", "## First Divergence", ""])
    if first_divergence is None:
        lines.append("- none within traced epochs 0-1")
    else:
        lines.append("```json")
        lines.append(json.dumps(first_divergence, ensure_ascii=False, indent=2))
        lines.append("```")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    root_dir = output_root / datetime.now().strftime("route_training_drift_trace_%Y%m%d_%H%M%S")
    root_dir.mkdir(parents=True, exist_ok=True)

    baseline_df, baseline_epoch_summaries = run_mode_trace(BASELINE_MODE, args.seed)
    candidate_df, candidate_epoch_summaries = run_mode_trace(CANDIDATE_MODE, args.seed)

    combined_df = pd.concat([baseline_df, candidate_df], ignore_index=True)
    combined_df.to_csv(root_dir / "route_training_drift_trace.csv", index=False)
    baseline_df.to_csv(root_dir / "route_training_drift_trace_baseline.csv", index=False)
    candidate_df.to_csv(root_dir / "route_training_drift_trace_candidate.csv", index=False)

    analysis = analyse_trace(
        baseline_df,
        candidate_df,
        baseline_epoch_summaries,
        candidate_epoch_summaries,
    )
    summary = {
        "seed": args.seed,
        "trace_epochs": TRACE_EPOCHS,
        **analysis,
        "output_files": {
            "combined_csv": str(root_dir / "route_training_drift_trace.csv"),
            "baseline_csv": str(root_dir / "route_training_drift_trace_baseline.csv"),
            "candidate_csv": str(root_dir / "route_training_drift_trace_candidate.csv"),
        },
    }

    (root_dir / "route_training_drift_trace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, root_dir / "route_training_drift_trace_summary.md")

    print(f"[route_training_drift_trace] wrote results to {root_dir}")


if __name__ == "__main__":
    main()
