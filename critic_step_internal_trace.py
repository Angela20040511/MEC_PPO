import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from actor_frozen_online_critic_trace import _find_previous_normal_trace_summary
from dense_actor_input_experiment import build_state_layout
from dense_policy_ratio_mode_experiment import build_policy_ratio_mode_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from joint_training_critic_drift_trace import (
    MODE,
    PROBE_STEP_COUNT,
    _collect_fixed_probe_payload,
    _collect_one_epoch,
    _find_current_mainline_checkpoint,
)
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator


TRACE_EPOCHS = 2
TRACE_STAGE_ORDER = {
    "before_popart_update": 0,
    "after_popart_update": 1,
    "after_target_build": 2,
    "before_actor": 3,
    "after_actor_before_critic": 4,
    "before_critic": 5,
    "after_backward_before_step": 6,
    "after_optimizer_step": 7,
    "after_critic": 8,
}
DELTA_METRICS = [
    "probe_pearson_value_vs_value_target",
    "probe_spearman_value_vs_value_target",
    "probe_pearson_value_vs_return",
    "probe_spearman_value_vs_return",
    "probe_pearson_value_vs_advantage",
    "probe_spearman_value_vs_advantage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace internal critic-step sub-stages on the frozen-actor joint "
            "reward-aligned mainline."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--trace-epochs", type=int, default=TRACE_EPOCHS)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    return parser.parse_args()


def _find_previous_actor_frozen_summary(root: Path) -> Path | None:
    candidates = sorted(
        root.glob(
            "actor_frozen_online_critic_trace_*/actor_frozen_online_critic_trace_summary.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _ordered_trace_df(trace_df: pd.DataFrame) -> pd.DataFrame:
    ordered_df = trace_df.copy()
    ordered_df["trace_stage_order"] = ordered_df["trace_stage"].map(TRACE_STAGE_ORDER)
    ordered_df = ordered_df.sort_values(
        ["epoch", "update_epoch", "minibatch_id", "trace_stage_order"]
    ).reset_index(drop=True)
    return ordered_df


def _stage_label(stage: str) -> str:
    if stage == "after_popart_update":
        return "A"
    if stage == "after_optimizer_step" or stage == "after_critic":
        return "B"
    if stage == "after_target_build":
        return "C"
    if stage == "after_backward_before_step":
        return "D"
    return "D"


def _analyse_internal_trace(trace_df: pd.DataFrame, epoch_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if trace_df.empty:
        return {
            "classification": "no_trace_rows",
            "detail": "No critic internal trace rows were captured.",
            "first_divergence": None,
            "initial_probe_metrics": None,
            "epoch_summaries": epoch_summaries,
        }

    ordered_df = _ordered_trace_df(trace_df)
    initial_row = ordered_df.iloc[0].to_dict()
    first_divergence = None
    threshold = -0.01
    for row_index in range(1, len(ordered_df)):
        previous_row = ordered_df.iloc[row_index - 1]
        current_row = ordered_df.iloc[row_index]
        deltas = {
            metric: float(current_row[metric]) - float(previous_row[metric])
            for metric in DELTA_METRICS
        }
        if min(deltas.values()) <= threshold:
            first_divergence = {
                "where": str(current_row["trace_stage"]),
                "from_stage": str(previous_row["trace_stage"]),
                "epoch": int(current_row["epoch"]),
                "update_epoch": int(current_row["update_epoch"]),
                "minibatch_id": int(current_row["minibatch_id"]),
                "trace_scope": str(current_row.get("trace_scope", "")),
                "delta_metrics": deltas,
            }
            break

    if first_divergence is None:
        classification = "D"
        detail = "No sharp single-stage drop appears within the traced window."
    else:
        classification = _stage_label(first_divergence["where"])
        if classification == "A":
            detail = "The first clear semantic drop appears immediately after PopArt rescale."
        elif classification == "B":
            detail = "The first clear semantic drop appears only after critic optimizer.step()."
        elif classification == "C":
            detail = "The first clear semantic drop appears right after target construction."
        else:
            detail = (
                "No parameter update is needed for the first visible drop; the bad turn "
                "appears before optimizer.step(), so multiple critic-step sub-stages likely combine."
            )

    final_probe_after_critic = ordered_df[ordered_df["trace_stage"] == "after_critic"].iloc[-1]
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
        "final_after_critic_probe_metrics": {
            key: final_probe_after_critic[key]
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
        "epoch_summaries": epoch_summaries,
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Critic Step Internal Trace",
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
    if summary.get("comparison_to_actor_frozen_trace") is not None:
        lines.extend(
            [
                "",
                "## Comparison To Previous Frozen-Actor Trace",
                "",
                "```json",
                json.dumps(
                    summary["comparison_to_actor_frozen_trace"],
                    ensure_ascii=False,
                    indent=2,
                ),
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
        "critic_step_internal_trace_%Y%m%d_%H%M%S"
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
    trace_df.to_csv(root_dir / "critic_step_internal_trace.csv", index=False)

    minibatch_cols = [
        "mode",
        "epoch",
        "update_epoch",
        "minibatch_id",
        "trace_stage",
        "trace_scope",
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
        "critic_param_delta_norm_after_popart",
        "critic_backbone_delta_norm_after_popart",
        "critic_head_delta_norm_after_popart",
        "critic_param_delta_norm_after_optimizer_step",
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
        "trace_scope",
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
    trace_df[minibatch_cols].to_csv(
        root_dir / "critic_step_internal_trace_minibatch.csv",
        index=False,
    )
    trace_df[probe_cols].to_csv(
        root_dir / "critic_step_internal_trace_probe.csv",
        index=False,
    )

    analysis = _analyse_internal_trace(trace_df, epoch_summaries)
    actor_param_delta_norm_max = (
        float(trace_df["actor_param_delta_norm"].max()) if not trace_df.empty else 0.0
    )
    previous_actor_frozen_summary_path = _find_previous_actor_frozen_summary(output_root)
    comparison_to_actor_frozen_trace = None
    if previous_actor_frozen_summary_path is not None:
        previous_summary = json.loads(
            previous_actor_frozen_summary_path.read_text(encoding="utf-8")
        )
        comparison_to_actor_frozen_trace = {
            "actor_frozen_trace_summary_path": str(previous_actor_frozen_summary_path),
            "actor_frozen_classification": previous_summary.get("classification", ""),
            "actor_frozen_first_divergence": previous_summary.get("first_divergence"),
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
        "comparison_to_actor_frozen_trace": comparison_to_actor_frozen_trace,
        "output_files": {
            "trace_csv": str(root_dir / "critic_step_internal_trace.csv"),
            "probe_csv": str(root_dir / "critic_step_internal_trace_probe.csv"),
            "minibatch_csv": str(root_dir / "critic_step_internal_trace_minibatch.csv"),
        },
    }
    summary = _to_builtin(summary)
    (root_dir / "critic_step_internal_trace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, root_dir / "critic_step_internal_trace_summary.md")
    print(f"[critic_step_internal_trace] wrote results to {root_dir}")


if __name__ == "__main__":
    main()
