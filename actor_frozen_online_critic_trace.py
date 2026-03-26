import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dense_actor_input_experiment import build_state_layout
from dense_policy_ratio_mode_experiment import build_policy_ratio_mode_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from joint_training_critic_drift_trace import (
    MODE,
    PROBE_STEP_COUNT,
    _analyse_trace,
    _collect_fixed_probe_payload,
    _collect_one_epoch,
    _find_current_mainline_checkpoint,
)
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator


TRACE_EPOCHS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace the current joint actor mainline with actor updates frozen, while "
            "keeping online rollouts and critic updates active."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--trace-epochs", type=int, default=TRACE_EPOCHS)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    return parser.parse_args()


def _find_previous_normal_trace_summary(root: Path) -> Path | None:
    candidates = sorted(
        root.glob("joint_training_critic_drift_trace_*/joint_training_critic_drift_trace_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        summary = json.loads(candidate.read_text(encoding="utf-8"))
        if bool(summary.get("load_current_checkpoint", False)) or bool(
            summary.get("checkpoint_path", "")
        ):
            return candidate
    return None


def _probe_stage_summary(trace_df: pd.DataFrame) -> dict[str, float]:
    if trace_df.empty:
        return {
            "initial_before_actor_probe_target_pearson": 0.0,
            "final_after_critic_probe_target_pearson": 0.0,
            "initial_before_actor_probe_return_pearson": 0.0,
            "final_after_critic_probe_return_pearson": 0.0,
        }
    ordered_df = trace_df.copy()
    ordered_df["trace_stage_order"] = ordered_df["trace_stage"].map(
        {"before_actor": 0, "after_actor_before_critic": 1, "after_critic": 2}
    )
    ordered_df = ordered_df.sort_values(
        ["epoch", "update_epoch", "minibatch_id", "trace_stage_order"]
    )
    initial_before_actor = ordered_df[ordered_df["trace_stage"] == "before_actor"].iloc[0]
    final_after_critic = ordered_df[ordered_df["trace_stage"] == "after_critic"].iloc[-1]
    return {
        "initial_before_actor_probe_target_pearson": float(
            initial_before_actor["probe_pearson_value_vs_value_target"]
        ),
        "final_after_critic_probe_target_pearson": float(
            final_after_critic["probe_pearson_value_vs_value_target"]
        ),
        "initial_before_actor_probe_return_pearson": float(
            initial_before_actor["probe_pearson_value_vs_return"]
        ),
        "final_after_critic_probe_return_pearson": float(
            final_after_critic["probe_pearson_value_vs_return"]
        ),
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Actor Frozen Online Critic Trace",
        "",
        f"- classification: `{summary['classification']}`",
        f"- detail: {summary['detail']}",
        f"- actor_param_delta_norm_max: `{summary['actor_param_delta_norm_max']:.6f}`",
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
    if summary.get("comparison_to_normal_trace") is not None:
        lines.extend(
            [
                "",
                "## Comparison To Previous Normal Trace",
                "",
                "```json",
                json.dumps(summary["comparison_to_normal_trace"], ensure_ascii=False, indent=2),
                "```",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    root_dir = output_root / datetime.now().strftime(
        "actor_frozen_online_critic_trace_%Y%m%d_%H%M%S"
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
    trace_df.to_csv(root_dir / "actor_frozen_online_critic_trace.csv", index=False)
    trace_df.to_csv(root_dir / "actor_frozen_online_critic_trace_minibatch.csv", index=False)
    trace_df[
        [
            "mode",
            "epoch",
            "update_epoch",
            "minibatch_id",
            "trace_stage",
            "probe_value_pred_mean",
            "probe_value_pred_std",
            "probe_pearson_value_vs_value_target",
            "probe_spearman_value_vs_value_target",
            "probe_pearson_value_vs_return",
            "probe_spearman_value_vs_return",
            "probe_pearson_value_vs_advantage",
            "probe_spearman_value_vs_advantage",
            "probe_pearson_value_vs_one_step_reward",
            "probe_spearman_value_vs_one_step_reward",
        ]
    ].to_csv(root_dir / "actor_frozen_online_critic_trace_probe.csv", index=False)

    analysis = _analyse_trace(trace_df, epoch_summaries)
    actor_param_delta_norm_max = float(trace_df["actor_param_delta_norm"].max()) if not trace_df.empty else 0.0
    previous_normal_summary_path = _find_previous_normal_trace_summary(output_root)
    comparison_to_normal_trace = None
    if previous_normal_summary_path is not None:
        previous_summary = json.loads(previous_normal_summary_path.read_text(encoding="utf-8"))
        previous_trace_csv = Path(previous_summary["output_files"]["trace_csv"])
        if not previous_trace_csv.is_absolute():
            previous_trace_csv = output_root / previous_trace_csv
        previous_trace_df = pd.read_csv(previous_trace_csv)
        comparison_to_normal_trace = {
            "normal_trace_summary_path": str(previous_normal_summary_path),
            "normal_trace_classification": previous_summary.get("classification", ""),
            "normal_trace_first_divergence": previous_summary.get("first_divergence"),
            "normal_trace_probe_summary": _probe_stage_summary(previous_trace_df),
            "frozen_trace_probe_summary": _probe_stage_summary(trace_df),
        }

    summary = {
        "mode": MODE,
        "seed": args.seed,
        "trace_epochs": int(args.trace_epochs),
        "checkpoint_path": str(checkpoint_path),
        "probe_step_count": PROBE_STEP_COUNT,
        "actor_frozen": True,
        "actor_param_delta_norm_max": actor_param_delta_norm_max,
        **analysis,
        "comparison_to_normal_trace": comparison_to_normal_trace,
        "output_files": {
            "trace_csv": str(root_dir / "actor_frozen_online_critic_trace.csv"),
            "probe_csv": str(root_dir / "actor_frozen_online_critic_trace_probe.csv"),
            "minibatch_csv": str(root_dir / "actor_frozen_online_critic_trace_minibatch.csv"),
        },
    }
    (root_dir / "actor_frozen_online_critic_trace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, root_dir / "actor_frozen_online_critic_trace_summary.md")
    print(f"[actor_frozen_online_critic_trace] wrote results to {root_dir}")


if __name__ == "__main__":
    main()
