import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MECConfig, build_config
from dense_actor_input_experiment import build_state_layout
from dense_critic_diagnosis import (
    PROBE_STATE_COUNT,
    collect_probe_states,
    dataframe_to_markdown,
    extract_metrics,
    plot_single_run_outputs,
    train_with_diagnosis,
)
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds
from rl.ppo_agent import PPOAgent

FIXED_SEED = 2025
FIXED_TOPOLOGY = "dense"
FIXED_REWARD_MODE = "per_sensor"
FIXED_VALUE_TARGET_MODE = "popart_return_norm"
FIXED_VALUE_LOSS_MODE = "huber"
FIXED_CRITIC_ARCH_MODE = "stronger_critic_backbone"
FIXED_CRITIC_INPUT_MODE = "normalized_augmented_critic_input"
FIXED_ACTOR_INPUT_MODE = "normalized_augmented_actor_input"
FIXED_ACTOR_RAW_INPUT_MODE = "baseline_actor_raw"

FIXED_ACTOR_LEARNING_RATE = 5e-4
FIXED_CRITIC_LEARNING_RATE = 5e-4
FIXED_ENTROPY_COEFF = 1e-3
FIXED_VALUE_COEFF = 0.5
FIXED_UPDATE_EPOCHS = 10
FIXED_TIME_STEPS = 100

FIXED_DT_HISTORY_WINDOW = 8
FIXED_DT_PREDICTION_HORIZON = 2
FIXED_DT_HIDDEN_SIZE = 128
FIXED_DT_RETRAIN_INTERVAL = 5
FIXED_DT_TRAIN_EPOCHS = 5
FIXED_DT_NUM_LAYERS = 1
FIXED_DT_LEARNING_RATE = 1e-3
FIXED_DT_BATCH_SIZE = 32
FIXED_DT_MIN_HISTORY_TO_TRAIN = 20

FIXED_NOMA_QUANTILE = 0.65
FIXED_MAX_CLUSTER_SIZE = 2
FIXED_ALPHA_P = 0.4
FIXED_RHO0 = 0.75
FIXED_GAMMA_Q = 0.10
FIXED_GAMMA_E = 0.15

FIXED_START_CHECK_EPOCH = 1
FIXED_MAX_EPOCHS = 6
FIXED_PATIENCE = 3
FIXED_MIN_DELTA = 1.0

DEFAULT_POLICY_RATIO_MODES = (
    "current_joint_sum_ratio",
    "blockwise_surrogate_mean",
    "blockwise_weighted_surrogate_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled dense comparison for PPO policy-ratio aggregation mode."
    )
    parser.add_argument(
        "--policy-ratio-modes",
        nargs="+",
        default=list(DEFAULT_POLICY_RATIO_MODES),
        help="Policy ratio aggregation modes to compare.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FIXED_SEED,
        help="Fixed seed for this controlled experiment.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="checkpoints",
        help="Root directory for experiment outputs.",
    )
    return parser.parse_args()


def build_policy_ratio_mode_config(policy_ratio_mode: str, seed: int) -> MECConfig:
    config = build_config(topology_mode=FIXED_TOPOLOGY)
    actor_structure_mode = (
        "hierarchical_actor_theta_route"
        if policy_ratio_mode in {
            "hierarchical_actor_theta_route",
            "hierarchical_actor_offload_gated_route",
            "hierarchical_actor_hard_offload_route",
        }
        else (
            "hierarchical_actor_separate_theta_route_backbones"
            if policy_ratio_mode in {
                "hierarchical_actor_separate_theta_route_backbones",
                "hierarchical_actor_branchwise_balanced_pg",
                "hierarchical_actor_alternating_branch_pg",
                "hierarchical_actor_dual_optimizer_alternating_branch_pg",
                "hierarchical_actor_factorized_ratio_pg",
                "hierarchical_actor_factorized_trust_region_pg",
                "hierarchical_actor_factorized_trust_region_pg_route_first",
                "hierarchical_actor_factorized_trust_region_pg_coupled_stop",
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop",
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor",
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask",
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap",
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise",
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate",
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1",
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv",
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
                "hierarchical_actor_route_step_alignment_gate",
                "hierarchical_actor_factorized_conditional_route_pg",
                "hierarchical_actor_true_conditional_route_policy",
                "hierarchical_actor_route_credit_theta_gate",
                "hierarchical_actor_route_credit_theta_soft_weight",
                "hierarchical_actor_route_residual_credit_vectorized",
                "hierarchical_actor_true_conditional_route_candidate_score_credit",
                "hierarchical_actor_theta_candidate_score_credit",
                "hierarchical_actor_theta_route_candidate_score_credit",
                "hierarchical_actor_joint_reward_aligned_credit",
                "hierarchical_actor_joint_td_aligned_credit",
            }
            else "flat_joint_actor"
        )
    )
    training = replace(
        config.training,
        num_epochs=FIXED_MAX_EPOCHS,
        time_steps=FIXED_TIME_STEPS,
        seed=seed,
    )
    ppo = replace(
        config.ppo,
        critic_arch_mode=FIXED_CRITIC_ARCH_MODE,
        critic_input_mode=FIXED_CRITIC_INPUT_MODE,
        actor_input_mode=FIXED_ACTOR_INPUT_MODE,
        actor_structure_mode=actor_structure_mode,
        actor_raw_input_mode=FIXED_ACTOR_RAW_INPUT_MODE,
        policy_ratio_mode=policy_ratio_mode,
        actor_learning_rate=FIXED_ACTOR_LEARNING_RATE,
        critic_learning_rate=FIXED_CRITIC_LEARNING_RATE,
        entropy_coeff=FIXED_ENTROPY_COEFF,
        value_coeff=FIXED_VALUE_COEFF,
        value_target_mode=FIXED_VALUE_TARGET_MODE,
        value_loss_mode=FIXED_VALUE_LOSS_MODE,
        update_epochs=FIXED_UPDATE_EPOCHS,
        policy_update_diagnosis_enabled=True,
    )
    dt = replace(
        config.dt,
        history_window=FIXED_DT_HISTORY_WINDOW,
        prediction_horizon=FIXED_DT_PREDICTION_HORIZON,
        hidden_size=FIXED_DT_HIDDEN_SIZE,
        retrain_interval=FIXED_DT_RETRAIN_INTERVAL,
        train_epochs=FIXED_DT_TRAIN_EPOCHS,
        num_layers=FIXED_DT_NUM_LAYERS,
        learning_rate=FIXED_DT_LEARNING_RATE,
        batch_size=FIXED_DT_BATCH_SIZE,
        min_history_to_train=FIXED_DT_MIN_HISTORY_TO_TRAIN,
    )
    system = replace(
        config.system,
        reward_mode=FIXED_REWARD_MODE,
        noma_quantile=FIXED_NOMA_QUANTILE,
        max_cluster_size=FIXED_MAX_CLUSTER_SIZE,
        alpha_p=FIXED_ALPHA_P,
        rho0=FIXED_RHO0,
        gamma_q=FIXED_GAMMA_Q,
        gamma_e=FIXED_GAMMA_E,
    )
    return replace(config, training=training, ppo=ppo, dt=dt, system=system)


def compute_bucket_slope(run_dir: Path) -> float:
    bucket_df = pd.read_csv(run_dir / "target_bucket_prediction_mean.csv")
    if len(bucket_df) < 2:
        return 0.0
    slope, _ = np.polyfit(
        bucket_df["bucket_target_mean"].to_numpy(),
        bucket_df["bucket_prediction_mean"].to_numpy(),
        deg=1,
    )
    return float(slope)


def build_action_block_layout(config: MECConfig) -> list[dict[str, object]]:
    """Describe the true action layout using real sensor/task/base-station semantics."""
    base_station_ids = list(config.base_station_ids)
    action_blocks: list[dict[str, object]] = []
    cursor = 0
    block_index = 0
    for sensor_id in config.sensor_ids:
        for task_name in config.task_names:
            theta_start = cursor
            theta_stop = theta_start + 1
            logits_start = theta_stop
            logits_stop = logits_start + len(base_station_ids)
            action_blocks.append(
                {
                    "block_index": block_index,
                    "block_label": f"{sensor_id}_{task_name}_dispatch",
                    "sensor_id": sensor_id,
                    "task_name": task_name,
                    "semantic": "dispatch_decision_block",
                    "start": theta_start,
                    "stop_exclusive": logits_stop,
                    "theta_slice": [theta_start, theta_stop],
                    "routing_logits_slice": [logits_start, logits_stop],
                    "routing_logit_labels": base_station_ids,
                }
            )
            cursor = logits_stop
            block_index += 1
    return action_blocks


def load_mode_logs(summary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mode_logs: dict[str, pd.DataFrame] = {}
    for row in summary_df.itertuples(index=False):
        mode_logs[str(row.policy_ratio_mode)] = pd.read_csv(Path(row.run_dir) / "train_logs.csv")
    return mode_logs


def load_mode_bucket_stats(summary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mode_bucket_stats: dict[str, pd.DataFrame] = {}
    for row in summary_df.itertuples(index=False):
        bucket_stats_path = Path(row.run_dir) / "advantage_bucket_update_stats.csv"
        if bucket_stats_path.exists():
            mode_bucket_stats[str(row.policy_ratio_mode)] = pd.read_csv(bucket_stats_path)
    return mode_bucket_stats


def plot_metric_compare(
    mode_logs: dict[str, pd.DataFrame],
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for policy_ratio_mode, logs in mode_logs.items():
        values = logs[column] if column in logs.columns else np.zeros(len(logs))
        plt.plot(logs["epoch"], values, marker="o", label=policy_ratio_mode)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_two_panel_compare(
    mode_logs: dict[str, pd.DataFrame],
    panel_specs: list[tuple[str, str]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(panel_specs), 1, figsize=(8, 4.5 * len(panel_specs)), sharex=True)
    if len(panel_specs) == 1:
        axes = [axes]
    for axis, (column, title) in zip(axes, panel_specs, strict=True):
        for policy_ratio_mode, logs in mode_logs.items():
            values = logs[column] if column in logs.columns else np.zeros(len(logs))
            axis.plot(logs["epoch"], values, marker="o", label=policy_ratio_mode)
        axis.set_ylabel(column)
        axis.set_title(title)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_logprob_scale_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    panel_specs = [
        (
            ("logprob_delta_sum_mean", "sum mean", "o"),
            ("logprob_delta_mean_mean", "mean mean", "s"),
            "value",
            "Log-Prob Delta Mean",
        ),
        (
            ("logprob_delta_sum_std", "sum std", "o"),
            ("logprob_delta_mean_std", "mean std", "s"),
            "value",
            "Log-Prob Delta Std",
        ),
        (
            ("sum_to_mean_scale_ratio", "sum/mean scale", "o"),
            None,
            "ratio",
            "Sum to Mean Scale Ratio",
        ),
    ]
    for axis, (primary_spec, secondary_spec, ylabel, title) in zip(axes, panel_specs, strict=True):
        for policy_ratio_mode, logs in mode_logs.items():
            primary_column, primary_label, primary_marker = primary_spec
            primary_values = (
                logs[primary_column] if primary_column in logs.columns else np.zeros(len(logs))
            )
            axis.plot(
                logs["epoch"],
                primary_values,
                marker=primary_marker,
                label=f"{policy_ratio_mode} {primary_label}",
            )
            if secondary_spec is not None:
                secondary_column, secondary_label, secondary_marker = secondary_spec
                secondary_values = (
                    logs[secondary_column]
                    if secondary_column in logs.columns
                    else np.zeros(len(logs))
                )
                axis.plot(
                    logs["epoch"],
                    secondary_values,
                    marker=secondary_marker,
                    linestyle="--",
                    label=f"{policy_ratio_mode} {secondary_label}",
                )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_block_logprob_scale_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    panel_specs = [
        (
            ("block_logprob_delta_mean", "block delta mean", "o"),
            ("block_logprob_delta_std", "block delta std", "s"),
            "value",
            "Block-Mean Log-Prob Delta",
        ),
        (
            ("block_ratio_proxy_mean", "block ratio proxy mean", "o"),
            ("block_ratio_proxy_std", "block ratio proxy std", "s"),
            "value",
            "Block Ratio Proxy",
        ),
        (
            ("sum_to_blockmean_scale_ratio", "sum/blockmean", "o"),
            ("mean_to_blockmean_scale_ratio", "mean/blockmean", "s"),
            "ratio",
            "Block Scale Ratios",
        ),
    ]
    for axis, (primary_spec, secondary_spec, ylabel, title) in zip(axes, panel_specs, strict=True):
        for policy_ratio_mode, logs in mode_logs.items():
            primary_column, primary_label, primary_marker = primary_spec
            primary_values = (
                logs[primary_column] if primary_column in logs.columns else np.zeros(len(logs))
            )
            axis.plot(
                logs["epoch"],
                primary_values,
                marker=primary_marker,
                label=f"{policy_ratio_mode} {primary_label}",
            )
            secondary_column, secondary_label, secondary_marker = secondary_spec
            secondary_values = (
                logs[secondary_column]
                if secondary_column in logs.columns
                else np.zeros(len(logs))
            )
            axis.plot(
                logs["epoch"],
                secondary_values,
                marker=secondary_marker,
                linestyle="--",
                label=f"{policy_ratio_mode} {secondary_label}",
            )
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_block_clip_fraction_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("block_clip_fraction_mean", "Block Clip Fraction Mean"),
            ("block_clip_fraction_std", "Block Clip Fraction Std"),
            (
                "block_positive_adv_clip_fraction_mean",
                "Block Positive-Adv Clip Fraction Mean",
            ),
            (
                "block_negative_adv_clip_fraction_mean",
                "Block Negative-Adv Clip Fraction Mean",
            ),
        ],
        output_path=output_path,
    )


def plot_block_surrogate_scale_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("block_ratio_mean", "Block Ratio Mean"),
            ("block_ratio_std", "Block Ratio Std"),
            ("block_surrogate_mean", "Block Surrogate Mean"),
            ("block_surrogate_std", "Block Surrogate Std"),
            (
                "per_block_selected_action_prob_gain_mean",
                "Per-Block Selected-Action Prob Gain Mean",
            ),
            (
                "per_block_selected_action_prob_gain_std",
                "Per-Block Selected-Action Prob Gain Std",
            ),
        ],
        output_path=output_path,
    )


def plot_block_weight_metric_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("block_weight_entropy", "Block Weight Entropy"),
            ("block_weight_max_mean", "Block Weight Max Mean"),
            ("top_k_block_weight_share", "Top-K Block Weight Share"),
            ("active_block_count_mean", "Active Block Count Mean"),
        ],
        output_path=output_path,
    )


def plot_bucket_metric_compare(
    mode_bucket_stats: dict[str, pd.DataFrame],
    value_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    bucket_names = sorted(
        {
            str(bucket_name)
            for bucket_stats in mode_bucket_stats.values()
            if "bucket_name" in bucket_stats.columns
            for bucket_name in bucket_stats["bucket_name"].dropna().unique().tolist()
        }
    )
    if not bucket_names:
        return

    fig, axes = plt.subplots(len(bucket_names), 1, figsize=(9, 3.5 * len(bucket_names)), sharex=True)
    if len(bucket_names) == 1:
        axes = [axes]
    for axis, bucket_name in zip(axes, bucket_names, strict=True):
        for policy_ratio_mode, bucket_stats in mode_bucket_stats.items():
            if value_column not in bucket_stats.columns or "bucket_name" not in bucket_stats.columns:
                continue
            bucket_df = bucket_stats[bucket_stats["bucket_name"] == bucket_name]
            if bucket_df.empty:
                continue
            axis.plot(
                bucket_df["epoch"],
                bucket_df[value_column],
                marker="o",
                label=policy_ratio_mode,
            )
        axis.set_ylabel(ylabel)
        axis.set_title(f"{title} [{bucket_name}]")
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_comparisons(root_dir: Path, summary_df: pd.DataFrame) -> None:
    plots_dir = root_dir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    mode_logs = load_mode_logs(summary_df)
    mode_bucket_stats = load_mode_bucket_stats(summary_df)

    plot_metric_compare(
        mode_logs,
        column="episode_reward",
        title="Reward Curve by Policy Ratio Mode",
        ylabel="episode_reward",
        output_path=plots_dir / "reward_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="policy_loss",
        title="Policy Loss by Policy Ratio Mode",
        ylabel="policy_loss",
        output_path=plots_dir / "policy_loss_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="clip_fraction",
        title="Clip Fraction by Policy Ratio Mode",
        ylabel="clip_fraction",
        output_path=plots_dir / "clip_fraction_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="approx_kl",
        title="Approx KL by Policy Ratio Mode",
        ylabel="approx_kl",
        output_path=plots_dir / "approx_kl_curve_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("ratio_mean", "Ratio Mean"),
            ("ratio_std", "Ratio Std"),
            ("ratio_min", "Ratio Min"),
            ("ratio_max", "Ratio Max"),
        ],
        output_path=plots_dir / "ratio_stats_curve_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("advantage_action_alignment", "Advantage / Log-Prob Alignment"),
            ("delta_log_prob_selected_action_mean", "Selected-Action Log-Prob Delta Mean"),
            ("delta_log_prob_selected_action_std", "Selected-Action Log-Prob Delta Std"),
        ],
        output_path=plots_dir / "advantage_alignment_curve_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("probe_policy_pairwise_kl_mean", "Probe Policy Pairwise KL Mean"),
            ("probe_policy_pairwise_l1_mean", "Probe Policy Pairwise L1 Mean"),
        ],
        output_path=plots_dir / "probe_policy_kl_curve_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("selected_action_change_rate", "Selected Action Change Rate"),
            ("top1_action_change_rate", "Top-1 Action Change Rate"),
            ("top1_prob_delta_mean", "Top-1 Probability Delta Mean"),
        ],
        output_path=plots_dir / "selected_action_change_rate_curve_compare.png",
    )
    plot_logprob_scale_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "logprob_scale_curve_compare.png",
    )
    plot_block_logprob_scale_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "block_logprob_scale_curve_compare.png",
    )
    plot_block_clip_fraction_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "block_clip_fraction_curve_compare.png",
    )
    plot_block_surrogate_scale_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "block_surrogate_scale_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="block_weight_entropy",
        title="Block Weight Entropy by Policy Ratio Mode",
        ylabel="block_weight_entropy",
        output_path=plots_dir / "block_weight_entropy_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="top_k_block_weight_share",
        title="Top-K Block Weight Share by Policy Ratio Mode",
        ylabel="top_k_block_weight_share",
        output_path=plots_dir / "top_k_block_weight_share_curve_compare.png",
    )
    plot_bucket_metric_compare(
        mode_bucket_stats=mode_bucket_stats,
        value_column="bucket_mean_delta_log_prob_selected_action",
        title="Advantage Bucket Mean Delta Log-Prob",
        ylabel="bucket_mean_delta_log_prob_selected_action",
        output_path=plots_dir / "advantage_bucket_update_compare.png",
    )
    plot_bucket_metric_compare(
        mode_bucket_stats=mode_bucket_stats,
        value_column="bucket_mean_selected_action_prob_gain",
        title="Selected Action Prob Gain by Advantage Bucket",
        ylabel="bucket_mean_selected_action_prob_gain",
        output_path=plots_dir / "selected_action_prob_gain_by_adv_bucket_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("value_explained_variance", "Value Explained Variance"),
            ("prediction_target_corr", "Prediction Target Correlation"),
            ("prediction_std_over_target_std", "Prediction Std / Target Std"),
        ],
        output_path=plots_dir / "critic_health_compare.png",
    )


def run_experiment(policy_ratio_modes: list[str], seed: int, output_root: Path) -> Path:
    root_dir = output_root / datetime.now().strftime("dense_policy_ratio_block_experiment_%Y%m%d_%H%M%S")
    root_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStoppingConfig(
        patience=FIXED_PATIENCE,
        min_delta=FIXED_MIN_DELTA,
        monitor_start_epoch=FIXED_START_CHECK_EPOCH,
    )
    probe_config = build_policy_ratio_mode_config(policy_ratio_mode=policy_ratio_modes[0], seed=seed)
    probe_states = collect_probe_states(config=probe_config, seed=seed, probe_count=PROBE_STATE_COUNT)
    np.save(root_dir / "probe_states.npy", probe_states)

    manifest: dict[str, object] = {
        "root_dir": str(root_dir),
        "fixed_params": {
            "topology": FIXED_TOPOLOGY,
            "seed": seed,
            "reward_mode": FIXED_REWARD_MODE,
            "value_target_mode": FIXED_VALUE_TARGET_MODE,
            "value_loss_mode": FIXED_VALUE_LOSS_MODE,
            "critic_arch_mode": FIXED_CRITIC_ARCH_MODE,
            "critic_input_mode": FIXED_CRITIC_INPUT_MODE,
            "actor_input_mode": FIXED_ACTOR_INPUT_MODE,
            "actor_raw_input_mode": FIXED_ACTOR_RAW_INPUT_MODE,
            "actor_learning_rate": FIXED_ACTOR_LEARNING_RATE,
            "critic_learning_rate": FIXED_CRITIC_LEARNING_RATE,
            "entropy_coeff": FIXED_ENTROPY_COEFF,
            "value_coeff": FIXED_VALUE_COEFF,
            "update_epochs": FIXED_UPDATE_EPOCHS,
            "time_steps": FIXED_TIME_STEPS,
            "dt_history_window": FIXED_DT_HISTORY_WINDOW,
            "dt_prediction_horizon": FIXED_DT_PREDICTION_HORIZON,
            "dt_hidden_size": FIXED_DT_HIDDEN_SIZE,
            "dt_retrain_interval": FIXED_DT_RETRAIN_INTERVAL,
            "dt_train_epochs": FIXED_DT_TRAIN_EPOCHS,
            "dt_num_layers": FIXED_DT_NUM_LAYERS,
            "dt_learning_rate": FIXED_DT_LEARNING_RATE,
            "dt_batch_size": FIXED_DT_BATCH_SIZE,
            "dt_min_history_to_train": FIXED_DT_MIN_HISTORY_TO_TRAIN,
            "q": FIXED_NOMA_QUANTILE,
            "Umax": FIXED_MAX_CLUSTER_SIZE,
            "alpha_p": FIXED_ALPHA_P,
            "rho0": FIXED_RHO0,
            "gamma_q": FIXED_GAMMA_Q,
            "gamma_e": FIXED_GAMMA_E,
            "start_check_epoch": FIXED_START_CHECK_EPOCH,
            "max_epochs": FIXED_MAX_EPOCHS,
            "patience": FIXED_PATIENCE,
            "min_delta": FIXED_MIN_DELTA,
            "probe_state_count": PROBE_STATE_COUNT,
        },
        "policy_ratio_modes": policy_ratio_modes,
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []
    for policy_ratio_mode in policy_ratio_modes:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root_dir / f"policy_ratio_{policy_ratio_mode}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = build_policy_ratio_mode_config(policy_ratio_mode=policy_ratio_mode, seed=seed)
        state_layout = build_state_layout(config)
        agent_preview = PPOAgent(
            config=config.ppo,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            critic_state_layout=state_layout,
        )
        (run_dir / "experiment_config.json").write_text(
            json.dumps(
                {
                    "training": asdict(config.training),
                    "ppo": asdict(config.ppo),
                    "dt": asdict(config.dt),
                    "system": asdict(config.system),
                    "critic_state_layout": state_layout,
                    "action_block_layout": build_action_block_layout(config),
                    "block_weight_definition": (
                        "For blockwise_weighted_surrogate_mean only: "
                        "normalized per-block activity weight built from "
                        "workload + virtual_queue + local_queue + sensor-shared access_queue."
                    ),
                    "actor_raw_layout": agent_preview.describe_actor_raw_layout(),
                    "early_stopping": asdict(early_stopping),
                    "probe_state_count": PROBE_STATE_COUNT,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[train] policy_ratio_mode={policy_ratio_mode}, seed={seed} -> {run_dir}")
        set_global_seeds(seed)
        train_with_diagnosis(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
            probe_states=probe_states,
            agent_kwargs={"critic_state_layout": state_layout},
        )
        plot_single_run_outputs(run_dir)

        metrics = extract_metrics(run_dir=run_dir)
        metrics["final_critic_backbone_to_head_grad_ratio"] = float(
            metrics["final_critic_backbone_grad_norm"]
        ) / (float(metrics["final_critic_head_grad_norm"]) + 1e-8)
        metrics["final_target_bucket_prediction_slope"] = compute_bucket_slope(run_dir)
        (run_dir / "analysis_summary.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_metrics.append(metrics)
        manifest["runs"].append(
            {
                "policy_ratio_mode": policy_ratio_mode,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    summary_df = pd.DataFrame(all_metrics)
    summary_block_csv = root_dir / "policy_surrogate_weighted_summary.csv"
    summary_block_json = root_dir / "policy_surrogate_weighted_summary.json"
    summary_block_md = root_dir / "policy_surrogate_weighted_summary.md"
    summary_csv = root_dir / "policy_ratio_mode_summary.csv"
    summary_json = root_dir / "policy_ratio_mode_summary.json"
    summary_md = root_dir / "policy_ratio_mode_summary.md"
    legacy_summary_block_csv = root_dir / "policy_ratio_block_summary.csv"
    legacy_summary_block_json = root_dir / "policy_ratio_block_summary.json"
    legacy_summary_block_md = root_dir / "policy_ratio_block_summary.md"
    summary_df.to_csv(summary_block_csv, index=False, encoding="utf-8-sig")
    summary_block_json.write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_block_md.write_text(dataframe_to_markdown(summary_df), encoding="utf-8")
    summary_df.to_csv(legacy_summary_block_csv, index=False, encoding="utf-8-sig")
    legacy_summary_block_json.write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    legacy_summary_block_md.write_text(dataframe_to_markdown(summary_df), encoding="utf-8")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(dataframe_to_markdown(summary_df), encoding="utf-8")

    plot_comparisons(root_dir=root_dir, summary_df=summary_df)

    manifest["summary"] = {
        "summary_csv": str(summary_block_csv),
        "summary_json": str(summary_block_json),
        "summary_md": str(summary_block_md),
        "summary_csv_legacy_block": str(legacy_summary_block_csv),
        "summary_json_legacy_block": str(legacy_summary_block_json),
        "summary_md_legacy_block": str(legacy_summary_block_md),
        "summary_csv_legacy": str(summary_csv),
        "summary_json_legacy": str(summary_json),
        "summary_md_legacy": str(summary_md),
        "reward_curve_compare": str(root_dir / "comparison_plots" / "reward_curve_compare.png"),
        "policy_loss_curve_compare": str(
            root_dir / "comparison_plots" / "policy_loss_curve_compare.png"
        ),
        "approx_kl_curve_compare": str(
            root_dir / "comparison_plots" / "approx_kl_curve_compare.png"
        ),
        "clip_fraction_curve_compare": str(
            root_dir / "comparison_plots" / "clip_fraction_curve_compare.png"
        ),
        "ratio_stats_curve_compare": str(
            root_dir / "comparison_plots" / "ratio_stats_curve_compare.png"
        ),
        "advantage_alignment_curve_compare": str(
            root_dir / "comparison_plots" / "advantage_alignment_curve_compare.png"
        ),
        "advantage_bucket_update_compare": str(
            root_dir / "comparison_plots" / "advantage_bucket_update_compare.png"
        ),
        "selected_action_prob_gain_by_adv_bucket_compare": str(
            root_dir
            / "comparison_plots"
            / "selected_action_prob_gain_by_adv_bucket_compare.png"
        ),
        "probe_policy_kl_curve_compare": str(
            root_dir / "comparison_plots" / "probe_policy_kl_curve_compare.png"
        ),
        "selected_action_change_rate_curve_compare": str(
            root_dir
            / "comparison_plots"
            / "selected_action_change_rate_curve_compare.png"
        ),
        "logprob_scale_curve_compare": str(
            root_dir / "comparison_plots" / "logprob_scale_curve_compare.png"
        ),
        "block_logprob_scale_curve_compare": str(
            root_dir / "comparison_plots" / "block_logprob_scale_curve_compare.png"
        ),
        "block_clip_fraction_curve_compare": str(
            root_dir / "comparison_plots" / "block_clip_fraction_curve_compare.png"
        ),
        "block_surrogate_scale_curve_compare": str(
            root_dir / "comparison_plots" / "block_surrogate_scale_curve_compare.png"
        ),
        "block_weight_entropy_curve_compare": str(
            root_dir / "comparison_plots" / "block_weight_entropy_curve_compare.png"
        ),
        "top_k_block_weight_share_curve_compare": str(
            root_dir / "comparison_plots" / "top_k_block_weight_share_curve_compare.png"
        ),
        "critic_health_compare": str(root_dir / "comparison_plots" / "critic_health_compare.png"),
    }
    (root_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root_dir


def main() -> None:
    args = parse_args()
    root_dir = run_experiment(
        policy_ratio_modes=args.policy_ratio_modes,
        seed=args.seed,
        output_root=Path(args.output_root),
    )
    print(f"Dense policy ratio mode experiment completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
