import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
STAGE_ORDER = {
    "before_actor": 0,
    "after_actor_before_critic": 1,
    "before_critic": 2,
    "after_backward_before_step": 3,
    "after_optimizer_step": 4,
    "after_critic": 5,
}
STEP_SIZE_GROUPS = [
    ("current_step_size", 1.0),
    ("quarter_step_size", 0.25),
    ("tenth_step_size", 0.10),
]
PROBE_DROP_THRESHOLD = -0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace online critic drift under frozen actor while scaling only the "
            "critic optimizer step size."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--trace-epochs", type=int, default=TRACE_EPOCHS)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    return parser.parse_args()


def _clone_probe_payload(probe_payload: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in probe_payload.items():
        if hasattr(value, "detach"):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _scale_critic_optimizer_lr(agent: PPOAgent, scale: float) -> dict[str, float]:
    original: dict[str, float] = {}
    for index, group in enumerate(agent.critic_optimizer.param_groups):
        key = f"group_{index}"
        original[key] = float(group["lr"])
        group["lr"] = float(group["lr"]) * float(scale)
    return original


def _ordered_minibatch_trace(trace_df: pd.DataFrame) -> pd.DataFrame:
    filtered = trace_df[
        (trace_df["trace_scope"] == "minibatch")
        & (trace_df["trace_stage"].isin(STAGE_ORDER))
    ].copy()
    filtered["trace_stage_order"] = filtered["trace_stage"].map(STAGE_ORDER)
    filtered = filtered.sort_values(
        ["epoch", "update_epoch", "minibatch_id", "trace_stage_order"]
    ).reset_index(drop=True)
    return filtered


def _build_stage_pivot(trace_df: pd.DataFrame) -> pd.DataFrame:
    ordered = _ordered_minibatch_trace(trace_df)
    if ordered.empty:
        return pd.DataFrame()
    pivot = ordered.pivot_table(
        index=["step_size_group", "epoch", "update_epoch", "minibatch_id"],
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
            "critic_backbone_delta_norm",
            "critic_head_delta_norm",
            "critic_grad_norm",
            "actor_param_delta_norm",
            "value_pred_mean",
            "value_pred_std",
            "value_target_mean",
            "value_target_std",
            "return_mean",
            "return_std",
            "advantage_mean",
            "advantage_std",
            "critic_loss",
            "actor_loss",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{stage}" for metric, stage in pivot.columns.to_flat_index()]
    pivot = pivot.reset_index().sort_values(
        ["step_size_group", "epoch", "update_epoch", "minibatch_id"]
    )
    return pivot


def _attach_deltas(pivot_df: pd.DataFrame) -> pd.DataFrame:
    if pivot_df.empty:
        return pivot_df
    metrics = [
        "probe_pearson_value_vs_value_target",
        "probe_spearman_value_vs_value_target",
        "probe_pearson_value_vs_return",
        "probe_spearman_value_vs_return",
        "probe_pearson_value_vs_advantage",
        "probe_spearman_value_vs_advantage",
    ]
    for metric in metrics:
        pivot_df[f"optimizer_delta_{metric}"] = (
            pivot_df[f"{metric}_after_optimizer_step"]
            - pivot_df[f"{metric}_after_backward_before_step"]
        )
        pivot_df[f"critic_delta_{metric}"] = (
            pivot_df[f"{metric}_after_critic"]
            - pivot_df[f"{metric}_before_critic"]
        )
    pivot_df["update_over_grad_ratio"] = (
        pivot_df["critic_param_delta_norm_after_optimizer_step"]
        / pivot_df["critic_grad_norm_after_backward_before_step"].clip(lower=1e-12)
    )
    return pivot_df


def _first_optimizer_divergence(group_pivot: pd.DataFrame) -> dict[str, Any] | None:
    for _, row in group_pivot.iterrows():
        deltas = {
            metric: float(row[f"optimizer_delta_{metric}"])
            for metric in [
                "probe_pearson_value_vs_value_target",
                "probe_spearman_value_vs_value_target",
                "probe_pearson_value_vs_return",
                "probe_spearman_value_vs_return",
            ]
        }
        if min(deltas.values()) <= PROBE_DROP_THRESHOLD:
            return {
                "epoch": int(row["epoch"]),
                "update_epoch": int(row["update_epoch"]),
                "minibatch_id": int(row["minibatch_id"]),
                "trace_stage": "after_optimizer_step",
                "delta_metrics": deltas,
                "critic_param_delta_norm": float(
                    row["critic_param_delta_norm_after_optimizer_step"]
                ),
                "update_over_grad_ratio": float(row["update_over_grad_ratio"]),
            }
    return None


def _group_summary(
    group_name: str,
    scale: float,
    trace_df: pd.DataFrame,
    pivot_df: pd.DataFrame,
    epoch_summaries: list[dict[str, Any]],
    original_lr_by_group: dict[str, float],
) -> dict[str, Any]:
    group_trace = trace_df[trace_df["step_size_group"] == group_name].copy()
    group_pivot = pivot_df[pivot_df["step_size_group"] == group_name].copy()
    actor_param_delta_norm_max = (
        float(group_trace["actor_param_delta_norm"].max()) if not group_trace.empty else 0.0
    )
    first_divergence = _first_optimizer_divergence(group_pivot)
    optimizer_probe_drop_min = (
        float(group_pivot["optimizer_delta_probe_pearson_value_vs_value_target"].min())
        if not group_pivot.empty
        else 0.0
    )
    optimizer_probe_drop_mean = (
        float(group_pivot["optimizer_delta_probe_pearson_value_vs_value_target"].mean())
        if not group_pivot.empty
        else 0.0
    )
    initial_probe = None
    final_probe = None
    ordered_group_trace = _ordered_minibatch_trace(group_trace)
    if not ordered_group_trace.empty:
        initial_row = ordered_group_trace.iloc[0]
        final_row = ordered_group_trace[ordered_group_trace["trace_stage"] == "after_critic"].iloc[-1]
        initial_probe = {
            "epoch": int(initial_row["epoch"]),
            "update_epoch": int(initial_row["update_epoch"]),
            "minibatch_id": int(initial_row["minibatch_id"]),
            "trace_stage": str(initial_row["trace_stage"]),
            "probe_pearson_value_vs_value_target": float(
                initial_row["probe_pearson_value_vs_value_target"]
            ),
            "probe_pearson_value_vs_return": float(initial_row["probe_pearson_value_vs_return"]),
            "probe_pearson_value_vs_advantage": float(
                initial_row["probe_pearson_value_vs_advantage"]
            ),
        }
        final_probe = {
            "epoch": int(final_row["epoch"]),
            "update_epoch": int(final_row["update_epoch"]),
            "minibatch_id": int(final_row["minibatch_id"]),
            "trace_stage": str(final_row["trace_stage"]),
            "probe_pearson_value_vs_value_target": float(
                final_row["probe_pearson_value_vs_value_target"]
            ),
            "probe_pearson_value_vs_return": float(final_row["probe_pearson_value_vs_return"]),
            "probe_pearson_value_vs_advantage": float(
                final_row["probe_pearson_value_vs_advantage"]
            ),
        }
    return {
        "step_size_group": group_name,
        "critic_step_size_scale": float(scale),
        "critic_optimizer_lr_by_group": original_lr_by_group,
        "actor_param_delta_norm_max": actor_param_delta_norm_max,
        "first_divergence": first_divergence,
        "optimizer_probe_target_drop_min": optimizer_probe_drop_min,
        "optimizer_probe_target_drop_mean": optimizer_probe_drop_mean,
        "initial_probe_metrics": initial_probe,
        "final_after_critic_probe_metrics": final_probe,
        "epoch_summaries": epoch_summaries,
    }


def _compare_groups(group_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    current = group_summaries["current_step_size"]
    quarter = group_summaries["quarter_step_size"]
    tenth = group_summaries["tenth_step_size"]
    return {
        "current_step_size": {
            "first_divergence": current["first_divergence"],
            "optimizer_probe_target_drop_min": current["optimizer_probe_target_drop_min"],
            "optimizer_probe_target_drop_mean": current["optimizer_probe_target_drop_mean"],
        },
        "quarter_step_size": {
            "first_divergence": quarter["first_divergence"],
            "optimizer_probe_target_drop_min": quarter["optimizer_probe_target_drop_min"],
            "optimizer_probe_target_drop_mean": quarter["optimizer_probe_target_drop_mean"],
        },
        "tenth_step_size": {
            "first_divergence": tenth["first_divergence"],
            "optimizer_probe_target_drop_min": tenth["optimizer_probe_target_drop_min"],
            "optimizer_probe_target_drop_mean": tenth["optimizer_probe_target_drop_mean"],
        },
        "relative_drop_reduction_vs_current": {
            "quarter_step_size_min_drop_ratio": (
                float(quarter["optimizer_probe_target_drop_min"] / current["optimizer_probe_target_drop_min"])
                if current["optimizer_probe_target_drop_min"] != 0.0
                else 0.0
            ),
            "tenth_step_size_min_drop_ratio": (
                float(tenth["optimizer_probe_target_drop_min"] / current["optimizer_probe_target_drop_min"])
                if current["optimizer_probe_target_drop_min"] != 0.0
                else 0.0
            ),
        },
    }


def _interpret(group_summaries: dict[str, dict[str, Any]]) -> str:
    current_drop = abs(group_summaries["current_step_size"]["optimizer_probe_target_drop_min"])
    quarter_drop = abs(group_summaries["quarter_step_size"]["optimizer_probe_target_drop_min"])
    tenth_drop = abs(group_summaries["tenth_step_size"]["optimizer_probe_target_drop_min"])
    quarter_has_divergence = group_summaries["quarter_step_size"]["first_divergence"] is not None
    tenth_has_divergence = group_summaries["tenth_step_size"]["first_divergence"] is not None

    if current_drop > 0 and quarter_drop < current_drop * 0.6 and tenth_drop < current_drop * 0.35:
        if not tenth_has_divergence:
            return (
                "Shrinking only the critic step size clearly suppresses the first after-optimizer "
                "semantic drop; critic step radius looks like the primary driver."
            )
        return (
            "Shrinking only the critic step size materially weakens the after-optimizer semantic "
            "drop; critic step radius looks like the dominant driver, with residual gradient/loss "
            "issues still present."
        )
    if current_drop > 0 and quarter_drop < current_drop and tenth_drop < quarter_drop:
        return (
            "Smaller critic steps help, but do not remove drift; both step radius and critic "
            "gradient direction likely contribute, with step radius still the stronger lever."
        )
    return (
        "Even large step-size reductions do not materially reduce the after-optimizer drift; "
        "the critic loss/gradient direction looks more primary than update radius."
    )


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Critic Step Size Online Trace",
        "",
        f"- mode: `{summary['mode']}`",
        f"- actor_frozen: `{summary['actor_frozen']}`",
        f"- trace_epochs: `{summary['trace_epochs']}`",
        f"- interpretation: {summary['interpretation']}",
        "",
        "## Group Comparison",
        "",
        "```json",
        json.dumps(summary["group_comparison"], ensure_ascii=False, indent=2),
        "```",
    ]
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
        "critic_step_size_online_trace_%Y%m%d_%H%M%S"
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    set_global_seeds(args.seed)
    config = build_policy_ratio_mode_config(policy_ratio_mode=MODE, seed=args.seed)
    checkpoint_path = _find_current_mainline_checkpoint(output_root)

    base_agent = PPOAgent(
        config=config.ppo,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        critic_state_layout=build_state_layout(config),
    )
    base_agent.load(str(checkpoint_path), load_optimizer=True)
    shared_probe_payload = _collect_fixed_probe_payload(base_agent, config, args.seed)

    all_trace_rows: list[dict[str, Any]] = []
    group_summaries: dict[str, dict[str, Any]] = {}

    for group_name, scale in STEP_SIZE_GROUPS:
        set_global_seeds(args.seed)
        group_agent = PPOAgent(
            config=config.ppo,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            critic_state_layout=build_state_layout(config),
        )
        group_agent.load(str(checkpoint_path), load_optimizer=True)
        group_agent.freeze_actor_training_updates = True
        group_agent.critic_training_internal_stage_trace = True
        lr_by_group = _scale_critic_optimizer_lr(group_agent, scale)
        group_agent.critic_training_drift_trace_rows = []
        group_agent.critic_training_drift_probe_payload = _clone_probe_payload(shared_probe_payload)
        simulator = Simulator(config)

        epoch_summaries: list[dict[str, Any]] = []
        for epoch in range(args.trace_epochs):
            rows_before = len(group_agent.critic_training_drift_trace_rows)
            losses = _collect_one_epoch(group_agent, simulator, config, epoch)
            rows_after = len(group_agent.critic_training_drift_trace_rows)
            epoch_rows = group_agent.critic_training_drift_trace_rows[rows_before:rows_after]
            epoch_df = pd.DataFrame(epoch_rows)
            if not epoch_df.empty:
                epoch_df["step_size_group"] = group_name
                epoch_df["critic_step_size_scale"] = float(scale)
            for row in epoch_df.to_dict(orient="records"):
                all_trace_rows.append(row)
            probe_after_critic = (
                epoch_df[epoch_df["trace_stage"] == "after_critic"].reset_index(drop=True)
                if not epoch_df.empty
                else pd.DataFrame()
            )
            epoch_summaries.append(
                {
                    "step_size_group": group_name,
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

        group_trace_df = pd.DataFrame(group_agent.critic_training_drift_trace_rows)
        if not group_trace_df.empty:
            group_trace_df["step_size_group"] = group_name
            group_trace_df["critic_step_size_scale"] = float(scale)
        group_pivot = _attach_deltas(_build_stage_pivot(group_trace_df))
        group_summaries[group_name] = _group_summary(
            group_name,
            scale,
            group_trace_df,
            group_pivot,
            epoch_summaries,
            lr_by_group,
        )

    trace_df = pd.DataFrame(all_trace_rows)
    pivot_df = _attach_deltas(_build_stage_pivot(trace_df))

    trace_df.to_csv(root_dir / "critic_step_size_online_trace.csv", index=False)
    minibatch_cols = [
        "mode",
        "step_size_group",
        "critic_step_size_scale",
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
        "critic_backbone_delta_norm",
        "critic_head_delta_norm",
        "actor_param_delta_norm",
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
        "step_size_group",
        "critic_step_size_scale",
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
        root_dir / "critic_step_size_online_trace_minibatch.csv",
        index=False,
    )
    trace_df[probe_cols].to_csv(
        root_dir / "critic_step_size_online_trace_probe.csv",
        index=False,
    )

    comparison = _compare_groups(group_summaries)
    summary = {
        "mode": MODE,
        "seed": args.seed,
        "trace_epochs": int(args.trace_epochs),
        "checkpoint_path": str(checkpoint_path),
        "probe_step_count": PROBE_STEP_COUNT,
        "actor_frozen": True,
        "group_summaries": group_summaries,
        "group_comparison": comparison,
        "interpretation": _interpret(group_summaries),
        "output_files": {
            "trace_csv": str(root_dir / "critic_step_size_online_trace.csv"),
            "probe_csv": str(root_dir / "critic_step_size_online_trace_probe.csv"),
            "minibatch_csv": str(root_dir / "critic_step_size_online_trace_minibatch.csv"),
        },
    }
    summary = _to_builtin(summary)
    (root_dir / "critic_step_size_online_trace_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, root_dir / "critic_step_size_online_trace_summary.md")
    print(f"[critic_step_size_online_trace] wrote results to {root_dir}")


if __name__ == "__main__":
    main()
