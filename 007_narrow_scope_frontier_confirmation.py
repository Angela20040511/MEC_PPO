from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from critic_backbone_preconditioner_clip_telemetry_validation import (
    SplitCriticOptimizer,
    TelemetryActivePreconditionerAdam,
)
from critic_batch_global_loss_trace import (
    _collect_fixed_heldout_payload,
    _collect_fixed_probe_payload,
)
from dense_actor_input_experiment import build_state_layout
from dense_critic_diagnosis import (
    PROBE_STATE_COUNT,
    collect_probe_states,
    dataframe_to_markdown,
    extract_metrics,
    plot_single_run_outputs,
    train_with_diagnosis,
)
from dense_policy_joint_reward_aligned_credit_experiment import (
    JOINT_REWARD_ALIGNED_MODE_DEFINITIONS,
)
from dense_policy_ratio_mode_experiment import (
    FIXED_CRITIC_LEARNING_RATE,
    build_action_block_layout,
    build_policy_ratio_mode_config,
    compute_bucket_slope,
)
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds
from rl.ppo_agent import PPOAgent


EXPERIMENT_ID = "007"
EXPERIMENT_STEM = "007_narrow_scope_frontier_confirmation"
POLICY_RATIO_MODE = "hierarchical_actor_joint_reward_aligned_credit"
TRAINING_SEMANTICS = "compatible_default"
SEEDS = (2024, 2025)
NUM_EPOCHS = 12
TIME_STEPS = 200
TRACKED_HOTSPOT_LAYERS = (
    "critic_backbone.0.weight",
    "critic_backbone.6.weight",
    "critic_backbone.7.weight",
    "critic_backbone.4.bias",
    "critic_backbone.7.bias",
)
NARROW_SCOPE_LAYERS = (
    "critic_backbone.0.weight",
    "critic_backbone.6.weight",
)
NARROW_SCOPE_REFERENCE_PRECONDITIONERS = {
    "critic_backbone.0.weight": 1780020.0437499732,
    "critic_backbone.6.weight": 4129.422778320308,
}
NARROW_SCOPE_ALPHA = 0.15
NARROW_SCOPE_BETA = 8.0
REFERENCE_BASELINE_GROUP_NAME = "007_pure_blended_baseline_reference_adam"
PRIORITY_NARROW_GROUP_NAME = "007_pure_blended_narrow_scope_soft_geometry_priority_adam"

GROUP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "group_name": "007_pure_blended_narrow_scope_soft_geometry_priority_adam",
        "description": (
            "Compatible default training semantics with the retained 006 narrow-scope "
            "soft geometry frontier branch on top of the pure blended critic baseline."
        ),
        "soft_geometry_enabled": True,
        "soft_geometry_alpha": NARROW_SCOPE_ALPHA,
        "soft_geometry_beta": NARROW_SCOPE_BETA,
        "soft_geometry_focus_parameter_names": NARROW_SCOPE_LAYERS,
        "layerwise_soft_reference_preconditioner_by_name": (
            NARROW_SCOPE_REFERENCE_PRECONDITIONERS
        ),
    },
    {
        "group_name": "007_pure_blended_baseline_reference_adam",
        "description": (
            "Compatible default training semantics with the pure blended critic baseline "
            "reference anchor: main_value_loss = 0.8 * current_batch_loss + "
            "0.2 * fixed_heldout_batch_loss."
        ),
        "soft_geometry_enabled": False,
    },
)
SUMMARY_METRICS = (
    "best_epoch",
    "best_reward",
    "final_reward",
    "reward_gap",
    "overall_advantage_action_alignment",
    "theta_advantage_alignment",
    "route_advantage_alignment",
    "value_explained_variance",
    "prediction_target_corr",
    "prediction_std_over_target_std",
    "target_bucket_prediction_slope",
    "joint_action_decision_agreement_ratio_under_reward_aligned",
    "critic_blended_value_loss_enabled",
    "critic_loss_current_batch",
    "critic_loss_heldout_batch",
    "would_reject_rate",
    "backbone_active_preconditioner_p95",
    "backbone_active_preconditioner_p99",
    "backbone_clipped_active_fraction",
    "soft_scale_mean",
    "soft_scale_min",
    "soft_scale_p50",
    "soft_scale_p95",
    "soft_scale_trigger_fraction",
    "overflow_mean",
    "overflow_p95",
    "critic_backbone_0_weight_delta_norm",
    "critic_backbone_0_weight_active_preconditioner_p95",
    "critic_backbone_0_weight_active_preconditioner_p99",
    "critic_backbone_6_weight_delta_norm",
    "critic_backbone_6_weight_active_preconditioner_p95",
    "critic_backbone_6_weight_active_preconditioner_p99",
    "critic_backbone_7_weight_delta_norm",
    "critic_backbone_7_weight_active_preconditioner_p95",
    "critic_backbone_7_weight_active_preconditioner_p99",
    "critic_backbone_4_bias_delta_norm",
    "critic_backbone_4_bias_active_preconditioner_p95",
    "critic_backbone_4_bias_active_preconditioner_p99",
    "critic_backbone_7_bias_delta_norm",
    "critic_backbone_7_bias_active_preconditioner_p95",
    "critic_backbone_7_bias_active_preconditioner_p99",
)
BOX_METRICS = (
    "best_reward",
    "final_reward",
    "value_explained_variance",
    "prediction_target_corr",
    "critic_loss_current_batch",
    "critic_loss_heldout_batch",
    "would_reject_rate",
    "backbone_active_preconditioner_p99",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="007 compatible-default narrow-scope frontier confirmation."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="D:\\MEC_PPO\\checkpoints",
        help="Directory where the numbered 007 output root will be created.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device passed into PPOAgent.",
    )
    parser.add_argument(
        "--resume-root",
        type=str,
        default="",
        help="Existing 007 output root to resume from; completed runs are reused and missing runs continue in-place.",
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _build_experiment_config(seed: int) -> Any:
    config = build_policy_ratio_mode_config(policy_ratio_mode=POLICY_RATIO_MODE, seed=seed)
    training = replace(config.training, num_epochs=NUM_EPOCHS, time_steps=TIME_STEPS, seed=seed)
    return replace(config, training=training)


def _disabled_early_stopping(config: Any) -> EarlyStoppingConfig:
    return EarlyStoppingConfig(
        patience=max(999, config.training.num_epochs + 10),
        min_delta=1e9,
        monitor_start_epoch=config.training.num_epochs + 1,
    )


def _build_agent_preview(config: Any, device: torch.device) -> PPOAgent:
    return PPOAgent(
        config=config.ppo,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        critic_state_layout=build_state_layout(config),
        device=device,
    )


def _build_backbone_optimizer(
    named_backbone_params: list[tuple[str, torch.nn.Parameter]],
    group_spec: dict[str, Any],
) -> TelemetryActivePreconditionerAdam:
    return TelemetryActivePreconditionerAdam(
        named_params=named_backbone_params,
        lr=FIXED_CRITIC_LEARNING_RATE,
        active_grad_threshold=1e-12,
        tracked_hotspot_layers=TRACKED_HOTSPOT_LAYERS,
        layerwise_soft_reference_preconditioner_by_name=group_spec.get(
            "layerwise_soft_reference_preconditioner_by_name"
        ),
        soft_geometry_focus_parameter_names=group_spec.get(
            "soft_geometry_focus_parameter_names"
        ),
        soft_geometry_alpha=float(group_spec.get("soft_geometry_alpha", 0.0)),
        soft_geometry_beta=float(group_spec.get("soft_geometry_beta", 8.0)),
    )


def _configure_agent(
    agent: PPOAgent,
    config: Any,
    group_spec: dict[str, Any],
    seed: int,
) -> None:
    agent.critic_blended_value_loss_enabled = True
    agent.critic_blended_current_weight = 0.8
    agent.critic_blended_heldout_weight = 0.2

    heldout_payload = _collect_fixed_heldout_payload(agent, config, seed)
    probe_payload = _collect_fixed_probe_payload(agent, config, seed)
    agent.critic_blended_fixed_heldout_payload = heldout_payload
    agent.critic_blended_heldout_batch_count = int(heldout_payload["heldout_batch_count"])
    agent.critic_training_drift_heldout_payload = heldout_payload
    agent.critic_training_drift_probe_payload = probe_payload
    agent.critic_step_acceptance_config = {
        "enabled": True,
        "enforce": False,
        "current_loss_improve_epsilon": 1e-6,
        "heldout_loss_tolerance": 1e-4,
        "probe_pearson_tolerance": 1e-4,
    }
    agent.critic_training_drift_trace_rows = []

    named_backbone_params = [
        (f"critic_backbone.{name}", param)
        for name, param in agent.network.critic_backbone.named_parameters()
    ]
    head_params = (
        list(agent.network.critic_head.parameters())
        + list(agent.network.critic_block_value_head.parameters())
        + list(agent.network.critic_block_path_value_head.parameters())
    )
    backbone_optimizer = _build_backbone_optimizer(named_backbone_params, group_spec)
    head_optimizer = torch.optim.Adam(head_params, lr=FIXED_CRITIC_LEARNING_RATE)
    agent.critic_optimizer = SplitCriticOptimizer(
        backbone_optimizer=backbone_optimizer,
        head_optimizer=head_optimizer,
    )


def _finalize_telemetry(agent: PPOAgent, run_dir: Path) -> dict[str, Any]:
    critic_optimizer = getattr(agent, "critic_optimizer", None)
    backbone_optimizer = getattr(critic_optimizer, "backbone_optimizer", None)
    if backbone_optimizer is None or not hasattr(backbone_optimizer, "finalize_telemetry"):
        return {}
    telemetry_summary = backbone_optimizer.finalize_telemetry(run_dir)
    analysis_path = run_dir / "analysis_summary.json"
    if analysis_path.exists():
        analysis_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis_payload["preconditioner_telemetry_summary"] = telemetry_summary
        analysis_path.write_text(
            json.dumps(analysis_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return telemetry_summary


def _write_run_experiment_config(
    run_dir: Path,
    config: Any,
    group_spec: dict[str, Any],
    device: torch.device,
) -> None:
    preview_agent = _build_agent_preview(config, device)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "training_semantics": TRAINING_SEMANTICS,
        "group_name": group_spec["group_name"],
        "group_description": group_spec["description"],
        "training_length_reason": (
            "Use 12 epochs x 200 time steps for direct comparability against 005 while staying "
            "within the same compatible frontier continuation scale."
        ),
        "training": asdict(config.training),
        "ppo": asdict(config.ppo),
        "dt": asdict(config.dt),
        "system": asdict(config.system),
        "critic_state_layout": build_state_layout(config),
        "action_block_layout": build_action_block_layout(config),
        "actor_structure": preview_agent.describe_actor_structure(),
        "actor_raw_layout": preview_agent.describe_actor_raw_layout(),
        "policy_mode_definition": JOINT_REWARD_ALIGNED_MODE_DEFINITIONS.get(
            POLICY_RATIO_MODE,
            "",
        ),
        "critic_baseline_definition": {
            "critic_blended_value_loss_enabled": True,
            "main_value_loss": "0.8 * current_batch_loss + 0.2 * fixed_heldout_batch_loss",
            "compatible_default_runner": True,
            "legacy_runner": False,
            "vectorized_runner": False,
        },
        "soft_geometry": {
            "enabled": bool(group_spec.get("soft_geometry_enabled", False)),
            "alpha": float(group_spec.get("soft_geometry_alpha", 0.0)),
            "beta": float(group_spec.get("soft_geometry_beta", 8.0)),
            "focus_parameter_names": list(group_spec.get("soft_geometry_focus_parameter_names", ())),
            "reference_preconditioners": group_spec.get(
                "layerwise_soft_reference_preconditioner_by_name", {}
            ),
        },
        "device": str(device),
    }
    (run_dir / "experiment_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_group_run_dir(seed_root: Path, group_name: str) -> Path:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = seed_root / f"{group_name}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _augment_metrics(
    run_dir: Path,
    base_metrics: dict[str, Any],
    group_spec: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    logs = pd.read_csv(run_dir / "train_logs.csv")
    final_row = logs.iloc[-1]
    metrics = dict(base_metrics)
    telemetry_summary: dict[str, Any] = {}
    telemetry_path = run_dir / "preconditioner_telemetry_summary.json"
    if telemetry_path.exists():
        telemetry_summary = json.loads(telemetry_path.read_text(encoding="utf-8"))
    final_backbone = telemetry_summary.get("final_backbone_telemetry", {})
    final_layers = telemetry_summary.get("final_hotspot_layer_telemetry", {})

    metrics.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "training_semantics": TRAINING_SEMANTICS,
            "seed": int(seed),
            "group_name": group_spec["group_name"],
            "overall_advantage_action_alignment": _safe_float(
                final_row.get("advantage_action_alignment")
            ),
            "theta_advantage_alignment": _safe_float(final_row.get("theta_advantage_alignment")),
            "route_advantage_alignment": _safe_float(final_row.get("route_advantage_alignment")),
            "value_explained_variance": _safe_float(final_row.get("value_explained_variance")),
            "prediction_target_corr": _safe_float(final_row.get("prediction_target_corr")),
            "prediction_std_over_target_std": _safe_float(
                final_row.get("prediction_std_over_target_std")
            ),
            "target_bucket_prediction_slope": compute_bucket_slope(run_dir),
            "joint_action_decision_agreement_ratio_under_reward_aligned": _safe_float(
                final_row.get("joint_action_decision_agreement_ratio_under_reward_aligned")
            ),
            "critic_blended_value_loss_enabled": _safe_float(
                final_row.get("critic_blended_value_loss_enabled")
            ),
            "critic_loss_current_batch": _safe_float(final_row.get("critic_loss_current_batch")),
            "critic_loss_heldout_batch": _safe_float(final_row.get("critic_loss_heldout_batch")),
            "would_reject_rate": _safe_float(final_row.get("critic_step_would_reject_rate")),
            "backbone_active_preconditioner_p95": _safe_float(
                final_backbone.get("backbone_active_preconditioner_p95")
            ),
            "backbone_active_preconditioner_p99": _safe_float(
                final_backbone.get("backbone_active_preconditioner_p99")
            ),
            "backbone_clipped_active_fraction": _safe_float(
                final_backbone.get("backbone_clipped_active_fraction")
            ),
            "soft_scale_mean": _safe_float(
                final_backbone.get("backbone_soft_geometry_scale_factor_mean"),
                default=1.0,
            ),
            "soft_scale_min": _safe_float(
                final_backbone.get("backbone_soft_geometry_scale_factor_min"),
                default=1.0,
            ),
            "soft_scale_p50": _safe_float(
                final_backbone.get("backbone_soft_geometry_scale_factor_p50"),
                default=1.0,
            ),
            "soft_scale_p95": _safe_float(
                final_backbone.get("backbone_soft_geometry_scale_factor_p95"),
                default=1.0,
            ),
            "soft_scale_trigger_fraction": _safe_float(
                final_backbone.get("backbone_soft_geometry_trigger_active_fraction")
            ),
            "overflow_mean": _safe_float(
                final_backbone.get("backbone_soft_geometry_overflow_stat_mean")
            ),
            "overflow_p95": _safe_float(
                final_backbone.get("backbone_soft_geometry_overflow_stat_p95")
            ),
            "soft_geometry_enabled": bool(group_spec.get("soft_geometry_enabled", False)),
            "soft_geometry_alpha": float(group_spec.get("soft_geometry_alpha", 0.0)),
            "soft_geometry_beta": float(group_spec.get("soft_geometry_beta", 8.0)),
            "soft_geometry_focus_parameter_names": ",".join(
                group_spec.get("soft_geometry_focus_parameter_names", ())
            ),
            "soft_geometry_reference_preconditioners_json": json.dumps(
                group_spec.get("layerwise_soft_reference_preconditioner_by_name", {}),
                ensure_ascii=False,
                sort_keys=True,
            ),
            "compatible_default_runner": _safe_float(final_row.get("rollout_cache_enabled")),
            "rollout_cache_enabled": _safe_float(final_row.get("rollout_cache_enabled")),
            "counterfactual_batch_value_inference": _safe_float(
                final_row.get("counterfactual_batch_value_inference")
            ),
            "digital_twin_fit_enabled": _safe_float(final_row.get("digital_twin_fit_enabled")),
        }
    )

    layer_aliases = {
        "critic_backbone.0.weight": "critic_backbone_0_weight",
        "critic_backbone.6.weight": "critic_backbone_6_weight",
        "critic_backbone.7.weight": "critic_backbone_7_weight",
        "critic_backbone.4.bias": "critic_backbone_4_bias",
        "critic_backbone.7.bias": "critic_backbone_7_bias",
    }
    for layer_name, alias in layer_aliases.items():
        payload = final_layers.get(layer_name, {})
        metrics[f"{alias}_delta_norm"] = _safe_float(payload.get("layer_actual_param_delta_norm"))
        metrics[f"{alias}_active_preconditioner_p95"] = _safe_float(
            payload.get("layer_active_preconditioner_p95")
        )
        metrics[f"{alias}_active_preconditioner_p99"] = _safe_float(
            payload.get("layer_active_preconditioner_p99")
        )
    return metrics


def _summarize_seed(seed: int, rows: list[dict[str, Any]], root_dir: Path) -> pd.DataFrame:
    records = []
    for row in rows:
        record = {
            "seed": seed,
            "group_name": row["group_name"],
            "training_semantics": row["training_semantics"],
            "run_dir": row["run_dir"],
        }
        for metric in SUMMARY_METRICS:
            record[metric] = row.get(metric, 0.0)
        records.append(record)
    df = pd.DataFrame(records)
    (root_dir / f"007_seed_{seed}_summary.json").write_text(
        df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    df.to_csv(root_dir / f"007_seed_{seed}_summary.csv", index=False, encoding="utf-8-sig")
    (root_dir / f"007_seed_{seed}_summary.md").write_text(
        dataframe_to_markdown(df),
        encoding="utf-8",
    )
    return df


def _build_cross_seed_summary(all_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(all_rows)
    group_order = [spec["group_name"] for spec in GROUP_SPECS]
    preferred_max = {
        "best_reward",
        "final_reward",
        "overall_advantage_action_alignment",
        "theta_advantage_alignment",
        "route_advantage_alignment",
        "value_explained_variance",
        "prediction_target_corr",
        "prediction_std_over_target_std",
        "target_bucket_prediction_slope",
        "joint_action_decision_agreement_ratio_under_reward_aligned",
    }
    baseline_name = REFERENCE_BASELINE_GROUP_NAME
    summary_rows = []
    long_rows = []
    baseline_by_seed = df[df["group_name"] == baseline_name].set_index("seed")

    for group_name in group_order:
        group_df = df[df["group_name"] == group_name]
        summary_row = {"group_name": group_name, "seed_count": int(len(group_df))}
        for metric in SUMMARY_METRICS:
            values = group_df[metric].astype(float)
            summary_row[f"{metric}_mean"] = float(values.mean())
            summary_row[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            summary_row[f"{metric}_min"] = float(values.min())
            summary_row[f"{metric}_max"] = float(values.max())
            wins = 0
            for seed in sorted(df["seed"].unique()):
                seed_df = df[df["seed"] == seed]
                seed_value = float(seed_df[seed_df["group_name"] == group_name][metric].iloc[0])
                if metric in preferred_max:
                    if abs(seed_value - float(seed_df[metric].max())) < 1e-9:
                        wins += 1
                else:
                    if abs(seed_value - float(seed_df[metric].min())) < 1e-9:
                        wins += 1
            summary_row[f"{metric}_win_seed_count"] = int(wins)
            if group_name == baseline_name:
                summary_row[f"{metric}_mean_delta_vs_baseline"] = 0.0
            else:
                deltas = []
                for seed, group_seed_row in group_df.set_index("seed").iterrows():
                    deltas.append(
                        float(group_seed_row[metric]) - float(baseline_by_seed.loc[seed, metric])
                    )
                summary_row[f"{metric}_mean_delta_vs_baseline"] = float(np.mean(deltas))
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    for metric in SUMMARY_METRICS:
        for group_name in group_order:
            row = summary_df.loc[summary_df["group_name"] == group_name].iloc[0]
            long_rows.append(
                {
                    "metric_name": metric,
                    "group_name": group_name,
                    "mean": float(row[f"{metric}_mean"]),
                    "std": float(row[f"{metric}_std"]),
                    "min": float(row[f"{metric}_min"]),
                    "max": float(row[f"{metric}_max"]),
                    "win_seed_count": int(row[f"{metric}_win_seed_count"]),
                    "mean_delta_vs_baseline": float(row[f"{metric}_mean_delta_vs_baseline"]),
                }
            )
    return summary_df, pd.DataFrame(long_rows)


def _plot_metric_curves(
    run_rows: list[dict[str, Any]],
    metric_names: tuple[str, ...],
    output_path: Path,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(9, 4.2 * len(metric_names)), sharex=True)
    if len(metric_names) == 1:
        axes = [axes]
    for axis, metric_name in zip(axes, metric_names, strict=True):
        for row in run_rows:
            logs = pd.read_csv(Path(row["run_dir"]) / "train_logs.csv")
            if metric_name in logs.columns:
                axis.plot(logs["epoch"], logs[metric_name], marker="o", label=row["group_name"])
        axis.set_title(f"{title_prefix}: {metric_name}")
        axis.set_ylabel(metric_name)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_cross_seed_mean_curve(
    all_rows: list[dict[str, Any]],
    metric_names: tuple[str, ...],
    output_path: Path,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(9, 4.2 * len(metric_names)), sharex=True)
    if len(metric_names) == 1:
        axes = [axes]
    for axis, metric_name in zip(axes, metric_names, strict=True):
        for group_name in [spec["group_name"] for spec in GROUP_SPECS]:
            epoch_frames = []
            for row in all_rows:
                if row["group_name"] != group_name:
                    continue
                logs = pd.read_csv(Path(row["run_dir"]) / "train_logs.csv")
                if metric_name in logs.columns:
                    epoch_frames.append(logs[["epoch", metric_name]].copy())
            if not epoch_frames:
                continue
            merged = pd.concat(epoch_frames, ignore_index=True)
            mean_df = merged.groupby("epoch", as_index=False)[metric_name].mean()
            axis.plot(mean_df["epoch"], mean_df[metric_name], marker="o", label=group_name)
        axis.set_title(f"{title_prefix}: {metric_name}")
        axis.set_ylabel(metric_name)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_box_compare(summary_df: pd.DataFrame, output_path: Path) -> None:
    group_names = list(summary_df["group_name"])
    fig, axes = plt.subplots(len(BOX_METRICS), 1, figsize=(9, 4.2 * len(BOX_METRICS)))
    if len(BOX_METRICS) == 1:
        axes = [axes]
    for axis, metric in zip(axes, BOX_METRICS, strict=True):
        values = [
            float(summary_df.loc[summary_df["group_name"] == group_name, f"{metric}_mean"].iloc[0])
            for group_name in group_names
        ]
        axis.bar(group_names, values)
        axis.set_ylabel(metric)
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_narrow_vs_baseline(summary_df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        "value_explained_variance",
        "prediction_target_corr",
        "critic_loss_current_batch",
        "critic_loss_heldout_batch",
        "would_reject_rate",
        "backbone_active_preconditioner_p99",
    ]
    baseline_name = REFERENCE_BASELINE_GROUP_NAME
    narrow_name = PRIORITY_NARROW_GROUP_NAME
    deltas = []
    for metric in metrics:
        baseline_value = float(
            summary_df.loc[summary_df["group_name"] == baseline_name, f"{metric}_mean"].iloc[0]
        )
        narrow_value = float(
            summary_df.loc[summary_df["group_name"] == narrow_name, f"{metric}_mean"].iloc[0]
        )
        deltas.append(narrow_value - baseline_value)
    plt.figure(figsize=(9, 4.8))
    plt.bar(metrics, deltas)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("narrow - baseline")
    plt.title("007 Narrow Scope vs Baseline Mean Delta")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _run_one_group(seed: int, group_spec: dict[str, Any], seed_root: Path, device: torch.device) -> dict[str, Any]:
    config = _build_experiment_config(seed)
    run_dir = _build_group_run_dir(seed_root, group_spec["group_name"])
    _write_run_experiment_config(run_dir, config, group_spec, device)
    probe_states = collect_probe_states(config=config, seed=seed, probe_count=PROBE_STATE_COUNT)
    np.save(run_dir / "007_probe_states.npy", probe_states)

    early_stopping = _disabled_early_stopping(config)
    agent, _logs, _summary, _payload_path = train_with_diagnosis(
        config=config,
        checkpoint_dir=str(run_dir),
        early_stopping=early_stopping,
        probe_states=probe_states,
        agent_kwargs={
            "critic_state_layout": build_state_layout(config),
            "device": device,
        },
        agent_setup_hook=lambda a: _configure_agent(a, config, group_spec, seed),
        rollout_options={
            "rollout_cache_enabled": True,
            "enable_joint_counterfactual_scores": True,
            "counterfactual_batch_value_inference": True,
        },
    )
    _finalize_telemetry(agent, run_dir)
    plot_single_run_outputs(run_dir)
    metrics = extract_metrics(run_dir)
    return _augment_metrics(run_dir, metrics, group_spec, seed)


def _load_existing_completed_group(
    seed: int,
    group_spec: dict[str, Any],
    seed_root: Path,
) -> dict[str, Any] | None:
    if not seed_root.exists():
        return None
    candidate_dirs = sorted(
        seed_root.glob(f"{group_spec['group_name']}_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidate_dirs:
        train_summary = run_dir / "train_summary.json"
        analysis_summary = run_dir / "analysis_summary.json"
        if train_summary.exists() and analysis_summary.exists():
            metrics = extract_metrics(run_dir)
            return _augment_metrics(run_dir, metrics, group_spec, seed)
    return None


def _build_manifest(root_dir: Path, all_rows: list[dict[str, Any]]) -> None:
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "output_root": str(root_dir),
        "training_semantics": TRAINING_SEMANTICS,
        "policy_ratio_mode": POLICY_RATIO_MODE,
        "seeds": list(SEEDS),
        "num_epochs": NUM_EPOCHS,
        "time_steps": TIME_STEPS,
        "groups": [spec["group_name"] for spec in GROUP_SPECS],
        "runs": [
            {"group_name": row["group_name"], "seed": row["seed"], "run_dir": row["run_dir"]}
            for row in all_rows
        ],
    }
    (root_dir / "007_artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if args.resume_root:
        root_dir = Path(args.resume_root)
    else:
        root_dir = output_root / datetime.now().strftime(
            "007_narrow_scope_frontier_confirmation_output_%Y%m%d_%H%M%S"
        )
    root_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    all_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        set_global_seeds(seed)
        seed_runs_root = root_dir / f"007_seed_{seed}_runs"
        seed_runs_root.mkdir(parents=True, exist_ok=True)
        seed_rows: list[dict[str, Any]] = []
        for group_spec in GROUP_SPECS:
            run_metrics = _load_existing_completed_group(seed, group_spec, seed_runs_root)
            if run_metrics is None:
                run_metrics = _run_one_group(seed, group_spec, seed_runs_root, device)
            seed_rows.append(run_metrics)
            all_rows.append(run_metrics)
        _summarize_seed(seed, seed_rows, root_dir)
        _plot_metric_curves(
            seed_rows,
            ("episode_reward",),
            root_dir / f"007_seed_{seed}_reward_curve_compare.png",
            f"seed {seed}",
        )
        _plot_metric_curves(
            seed_rows,
            ("advantage_action_alignment", "theta_advantage_alignment", "route_advantage_alignment"),
            root_dir / f"007_seed_{seed}_advantage_alignment_curve_compare.png",
            f"seed {seed}",
        )
        _plot_metric_curves(
            seed_rows,
            ("value_explained_variance", "prediction_target_corr", "prediction_std_over_target_std"),
            root_dir / f"007_seed_{seed}_critic_health_curve_compare.png",
            f"seed {seed}",
        )
        _plot_metric_curves(
            seed_rows,
            ("critic_loss_current_batch", "critic_loss_heldout_batch", "critic_step_would_reject_rate"),
            root_dir / f"007_seed_{seed}_critic_drift_compare.png",
            f"seed {seed}",
        )

    summary_df, long_df = _build_cross_seed_summary(all_rows)
    (root_dir / "007_narrow_scope_frontier_confirmation_summary.json").write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_df.to_csv(
        root_dir / "007_narrow_scope_frontier_confirmation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    md_lines = [
        f"# {EXPERIMENT_STEM}",
        "",
        "- training_semantics: compatible_default",
        "- seeds: 2024, 2025",
        "- groups: pure blended baseline vs retained 005 narrow scope branch",
        "- training_length_reason: direct comparability to 005 using 12 epochs x 200 time steps.",
        "",
        "## Cross-seed Summary",
        "",
        dataframe_to_markdown(summary_df),
        "",
        "## Long Metric Table",
        "",
        dataframe_to_markdown(long_df),
        "",
    ]
    (root_dir / "007_narrow_scope_frontier_confirmation_summary.md").write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )

    _plot_cross_seed_mean_curve(
        all_rows,
        ("episode_reward",),
        root_dir / "007_reward_curve_compare.png",
        "cross-seed mean",
    )
    _plot_cross_seed_mean_curve(
        all_rows,
        ("advantage_action_alignment", "theta_advantage_alignment", "route_advantage_alignment"),
        root_dir / "007_advantage_alignment_curve_compare.png",
        "cross-seed mean",
    )
    _plot_cross_seed_mean_curve(
        all_rows,
        ("value_explained_variance", "prediction_target_corr", "prediction_std_over_target_std"),
        root_dir / "007_critic_health_curve_compare.png",
        "cross-seed mean",
    )
    _plot_cross_seed_mean_curve(
        all_rows,
        ("critic_loss_current_batch", "critic_loss_heldout_batch", "critic_step_would_reject_rate"),
        root_dir / "007_critic_drift_compare.png",
        "cross-seed mean",
    )
    _plot_narrow_vs_baseline(summary_df, root_dir / "007_narrow_scope_vs_baseline_compare.png")
    _plot_box_compare(summary_df, root_dir / "007_multiseed_metric_box_compare.png")
    _build_manifest(root_dir, all_rows)


if __name__ == "__main__":
    main()



