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

DEFAULT_ACTOR_RAW_INPUT_MODES = (
    "baseline_actor_raw",
    "prune_compute18_actor_raw",
    "prune_all_static42_actor_raw",
)
ACTION_HISTOGRAM_LABELS = [
    "lt_neg_2",
    "neg_2_to_neg_1",
    "neg_1_to_neg_0p5",
    "neg_0p5_to_0",
    "zero_to_0p5",
    "0p5_to_1",
    "1_to_2",
    "gt_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled dense actor raw-state pruning experiment on the fixed actor/critic mainline."
    )
    parser.add_argument(
        "--actor-raw-input-modes",
        nargs="+",
        default=list(DEFAULT_ACTOR_RAW_INPUT_MODES),
        help="Actor raw-state composition modes to compare.",
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


def build_actor_raw_prune_config(actor_raw_input_mode: str, seed: int) -> MECConfig:
    config = build_config(topology_mode=FIXED_TOPOLOGY)
    training = replace(
        config.training,
        num_epochs=FIXED_MAX_EPOCHS,
        time_steps=FIXED_TIME_STEPS,
        seed=seed,
    )
    ppo = replace(
        config.ppo,
        actor_input_mode=FIXED_ACTOR_INPUT_MODE,
        actor_raw_input_mode=actor_raw_input_mode,
        critic_arch_mode=FIXED_CRITIC_ARCH_MODE,
        critic_input_mode=FIXED_CRITIC_INPUT_MODE,
        actor_learning_rate=FIXED_ACTOR_LEARNING_RATE,
        critic_learning_rate=FIXED_CRITIC_LEARNING_RATE,
        entropy_coeff=FIXED_ENTROPY_COEFF,
        value_coeff=FIXED_VALUE_COEFF,
        value_target_mode=FIXED_VALUE_TARGET_MODE,
        value_loss_mode=FIXED_VALUE_LOSS_MODE,
        update_epochs=FIXED_UPDATE_EPOCHS,
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
    logs = pd.read_csv(run_dir / "train_logs.csv")
    payload_path = Path(str(logs["prediction_target_payload_path"].iloc[-1]))
    payload = pd.read_csv(payload_path)
    bucket_df = payload.copy()
    bucket_df["target_bucket"] = pd.qcut(
        bucket_df["value_target"],
        q=min(10, len(bucket_df)),
        duplicates="drop",
    )
    bucket_summary = (
        bucket_df.groupby("target_bucket", observed=True)
        .agg(
            bucket_target_mean=("value_target", "mean"),
            bucket_prediction_mean=("critic_prediction", "mean"),
            bucket_count=("critic_prediction", "size"),
        )
        .reset_index(drop=True)
    )
    bucket_summary.to_csv(run_dir / "target_bucket_prediction_mean.csv", index=False, encoding="utf-8-sig")
    if len(bucket_summary) < 2:
        return 0.0
    slope, _ = np.polyfit(
        bucket_summary["bucket_target_mean"].to_numpy(),
        bucket_summary["bucket_prediction_mean"].to_numpy(),
        deg=1,
    )
    return float(slope)


def rank_runs(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.copy()
    ranked = ranked.sort_values(
        by=[
            "best_epoch_is_one",
            "best_epoch",
            "best_reward",
            "final_reward",
            "reward_gap_abs",
            "final_actor_raw_zero_var_dim_count",
            "final_actor_raw_dim_std_mean",
            "final_policy_entropy",
            "final_prediction_target_corr",
            "final_value_explained_variance",
        ],
        ascending=[True, False, False, False, True, True, False, False, False, False],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def load_mode_logs(ranked_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mode_logs: dict[str, pd.DataFrame] = {}
    for row in ranked_df.itertuples(index=False):
        mode_logs[str(row.actor_raw_input_mode)] = pd.read_csv(Path(row.run_dir) / "train_logs.csv")
    return mode_logs


def plot_metric_compare(
    mode_logs: dict[str, pd.DataFrame],
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for actor_raw_input_mode, logs in mode_logs.items():
        values = logs[column] if column in logs.columns else np.zeros(len(logs))
        plt.plot(logs["epoch"], values, marker="o", label=actor_raw_input_mode)
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
        for actor_raw_input_mode, logs in mode_logs.items():
            values = logs[column] if column in logs.columns else np.zeros(len(logs))
            axis.plot(logs["epoch"], values, marker="o", label=actor_raw_input_mode)
        axis.set_ylabel(column)
        axis.set_title(title)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_selected_action_histogram(value: object) -> dict[str, int]:
    histogram = {label: 0 for label in ACTION_HISTOGRAM_LABELS}
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(value, dict):
        parsed = value
    else:
        parsed = {}
    for label in ACTION_HISTOGRAM_LABELS:
        if label in parsed:
            histogram[label] = int(parsed[label])
    return histogram


def plot_action_distribution_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=False)
    for actor_raw_input_mode, logs in mode_logs.items():
        action_prob_std = (
            logs["action_prob_std"] if "action_prob_std" in logs.columns else np.zeros(len(logs))
        )
        policy_confidence = (
            logs["policy_confidence_mean"]
            if "policy_confidence_mean" in logs.columns
            else np.zeros(len(logs))
        )
        axes[0].plot(logs["epoch"], action_prob_std, marker="o", label=actor_raw_input_mode)
        axes[1].plot(logs["epoch"], policy_confidence, marker="o", label=actor_raw_input_mode)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("action_prob_std")
    axes[0].set_title("Action Distribution Std")
    axes[0].legend()
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("policy_confidence_mean")
    axes[1].set_title("Policy Confidence Mean")
    axes[1].legend()

    x = np.arange(len(ACTION_HISTOGRAM_LABELS))
    width = 0.8 / max(len(mode_logs), 1)
    for idx, (actor_raw_input_mode, logs) in enumerate(mode_logs.items()):
        serialized = (
            logs["selected_action_histogram"].iloc[-1]
            if "selected_action_histogram" in logs.columns and len(logs) > 0
            else "{}"
        )
        histogram = parse_selected_action_histogram(serialized)
        offsets = x + (idx - (len(mode_logs) - 1) / 2.0) * width
        axes[2].bar(
            offsets,
            [histogram[label] for label in ACTION_HISTOGRAM_LABELS],
            width=width,
            label=actor_raw_input_mode,
        )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(ACTION_HISTOGRAM_LABELS, rotation=20, ha="right")
    axes[2].set_ylabel("count")
    axes[2].set_title("Final Selected Action Histogram")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scatter_compare(ranked_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(ranked_df), figsize=(7 * len(ranked_df), 6), squeeze=False)
    for axis, row in zip(axes[0], ranked_df.itertuples(index=False), strict=True):
        logs = pd.read_csv(Path(row.run_dir) / "train_logs.csv")
        payload_path = Path(str(logs["prediction_target_payload_path"].iloc[-1]))
        payload = pd.read_csv(payload_path)
        axis.scatter(payload["value_target"], payload["critic_prediction"], s=12, alpha=0.6)
        lower = min(payload["value_target"].min(), payload["critic_prediction"].min())
        upper = max(payload["value_target"].max(), payload["critic_prediction"].max())
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#d62728", linewidth=1.2)
        axis.set_xlabel("value_target")
        axis.set_ylabel("critic_prediction")
        axis.set_title(str(row.actor_raw_input_mode))
    fig.suptitle("Prediction vs Target Scatter", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_bucket_compare(root_dir: Path, ranked_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(ranked_df), figsize=(8 * len(ranked_df), 5), squeeze=False)
    bucket_frames: list[pd.DataFrame] = []
    for axis, row in zip(axes[0], ranked_df.itertuples(index=False), strict=True):
        bucket_summary = pd.read_csv(Path(row.run_dir) / "target_bucket_prediction_mean.csv")
        bucket_summary.insert(0, "actor_raw_input_mode", str(row.actor_raw_input_mode))
        bucket_frames.append(bucket_summary)

        x = np.arange(len(bucket_summary))
        axis.plot(x, bucket_summary["bucket_target_mean"], marker="o", label="target mean")
        axis.plot(
            x,
            bucket_summary["bucket_prediction_mean"],
            marker="s",
            label="prediction mean",
        )
        axis.set_xlabel("target quantile bucket")
        axis.set_ylabel("mean")
        axis.set_title(str(row.actor_raw_input_mode))
        axis.legend()

    pd.concat(bucket_frames, ignore_index=True).to_csv(
        root_dir / "target_bucket_prediction_mean.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fig.suptitle("Target Bucket Prediction Mean", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_comparisons(root_dir: Path, ranked_df: pd.DataFrame) -> None:
    plots_dir = root_dir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    mode_logs = load_mode_logs(ranked_df)

    plot_metric_compare(
        mode_logs,
        column="episode_reward",
        title="Reward Curve by Actor Raw Input Mode",
        ylabel="episode_reward",
        output_path=plots_dir / "reward_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="actor_input_std",
        title="Actor Input Std by Actor Raw Input Mode",
        ylabel="actor_input_std",
        output_path=plots_dir / "actor_input_std_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("actor_input_dim_std_mean", "Actor Input Dim Std Mean"),
            ("actor_input_dim_std_min", "Actor Input Dim Std Min"),
        ],
        output_path=plots_dir / "actor_input_dim_std_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="actor_grad_norm",
        title="Actor Grad Norm by Actor Raw Input Mode",
        ylabel="actor_grad_norm",
        output_path=plots_dir / "actor_grad_norm_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="policy_entropy",
        title="Policy Entropy by Actor Raw Input Mode",
        ylabel="policy_entropy",
        output_path=plots_dir / "policy_entropy_compare.png",
    )
    plot_action_distribution_compare(
        mode_logs,
        output_path=plots_dir / "action_distribution_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("advantage_mean", "Advantage Mean"),
            ("advantage_std", "Advantage Std"),
        ],
        output_path=plots_dir / "advantage_stats_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="critic_loss",
        title="Critic Loss by Actor Raw Input Mode",
        ylabel="critic_loss",
        output_path=plots_dir / "critic_loss_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="value_explained_variance",
        title="Explained Variance by Actor Raw Input Mode",
        ylabel="value_explained_variance",
        output_path=plots_dir / "explained_variance_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="prediction_target_corr",
        title="Prediction/Target Correlation by Actor Raw Input Mode",
        ylabel="prediction_target_corr",
        output_path=plots_dir / "prediction_target_corr_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="prediction_std_over_target_std",
        title="Prediction Std over Target Std by Actor Raw Input Mode",
        ylabel="prediction_std_over_target_std",
        output_path=plots_dir / "prediction_std_ratio_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="actor_raw_zero_var_dim_count",
        title="Actor Raw Zero-Var Dim Count by Actor Raw Input Mode",
        ylabel="actor_raw_zero_var_dim_count",
        output_path=plots_dir / "actor_raw_zero_var_dim_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("actor_raw_dim_std_mean", "Actor Raw Dim Std Mean"),
            ("actor_raw_dim_std_min", "Actor Raw Dim Std Min"),
        ],
        output_path=plots_dir / "actor_raw_dim_std_compare.png",
    )
    plot_scatter_compare(
        ranked_df=ranked_df,
        output_path=plots_dir / "prediction_vs_target_scatter.png",
    )
    plot_bucket_compare(
        root_dir=root_dir,
        ranked_df=ranked_df,
        output_path=plots_dir / "target_bucket_prediction_mean.png",
    )


def build_static_block_check(
    actor_raw_input_modes: list[str],
    seed: int,
) -> dict[str, object]:
    check: dict[str, object] = {
        "topology": FIXED_TOPOLOGY,
        "seed": seed,
        "actor_input_mode": FIXED_ACTOR_INPUT_MODE,
        "critic_input_mode": FIXED_CRITIC_INPUT_MODE,
        "modes": {},
    }

    reference_config = build_actor_raw_prune_config(
        actor_raw_input_mode=actor_raw_input_modes[0],
        seed=seed,
    )
    state_layout = build_state_layout(reference_config)
    raw_state_blocks: dict[str, object] | None = None
    for actor_raw_input_mode in actor_raw_input_modes:
        config = build_actor_raw_prune_config(actor_raw_input_mode=actor_raw_input_mode, seed=seed)
        agent = PPOAgent(
            config=config.ppo,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            critic_state_layout=state_layout,
        )
        description = agent.describe_actor_raw_layout()
        if raw_state_blocks is None:
            raw_state_blocks = description["raw_state_blocks"]
        check["modes"][actor_raw_input_mode] = {
            "actor_raw_dim": description["actor_raw_dim"],
            "actor_input_total_dim": description["actor_input_total_dim"],
            "actor_input_derived_dim": description["actor_input_derived_dim"],
            "actor_raw_pruned_dim_count": description["actor_raw_pruned_dim_count"],
        }
    check["state_dim"] = reference_config.state_dim
    check["raw_state_blocks"] = raw_state_blocks
    check["static_blocks"] = {
        "reachable_mask": raw_state_blocks["reachable_mask"] if raw_state_blocks is not None else {},
        "access_gain": raw_state_blocks["access_gain"] if raw_state_blocks is not None else {},
        "compute_rate": raw_state_blocks["compute_rate"] if raw_state_blocks is not None else {},
    }
    return check


def run_experiment(actor_raw_input_modes: list[str], seed: int, output_root: Path) -> Path:
    root_dir = output_root / datetime.now().strftime("dense_actor_raw_prune_experiment_%Y%m%d_%H%M%S")
    root_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStoppingConfig(
        patience=FIXED_PATIENCE,
        min_delta=FIXED_MIN_DELTA,
        monitor_start_epoch=FIXED_START_CHECK_EPOCH,
    )
    probe_config = build_actor_raw_prune_config(
        actor_raw_input_mode=actor_raw_input_modes[0],
        seed=seed,
    )
    probe_states = collect_probe_states(config=probe_config, seed=seed, probe_count=PROBE_STATE_COUNT)
    np.save(root_dir / "probe_states.npy", probe_states)

    static_block_check = build_static_block_check(actor_raw_input_modes=actor_raw_input_modes, seed=seed)
    static_block_check_path = root_dir / "actor_raw_static_block_check.json"
    static_block_check_path.write_text(
        json.dumps(static_block_check, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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
        "actor_raw_input_modes": actor_raw_input_modes,
        "actor_raw_static_block_check": str(static_block_check_path),
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []
    for actor_raw_input_mode in actor_raw_input_modes:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root_dir / f"actor_raw_{actor_raw_input_mode}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = build_actor_raw_prune_config(actor_raw_input_mode=actor_raw_input_mode, seed=seed)
        state_layout = build_state_layout(config)
        agent = PPOAgent(
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
                    "actor_raw_layout": agent.describe_actor_raw_layout(),
                    "early_stopping": asdict(early_stopping),
                    "probe_state_count": PROBE_STATE_COUNT,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[train] actor_raw_input_mode={actor_raw_input_mode}, seed={seed} -> {run_dir}")
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
                "actor_raw_input_mode": actor_raw_input_mode,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    ranked_df = rank_runs(pd.DataFrame(all_metrics))
    summary_csv = root_dir / "actor_raw_prune_summary.csv"
    summary_json = root_dir / "actor_raw_prune_summary.json"
    summary_md = root_dir / "actor_raw_prune_summary.md"
    ranked_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        ranked_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(dataframe_to_markdown(ranked_df), encoding="utf-8")

    plot_comparisons(root_dir=root_dir, ranked_df=ranked_df)

    manifest["summary"] = {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "actor_raw_static_block_check": str(static_block_check_path),
        "reward_curve_plot": str(root_dir / "comparison_plots" / "reward_curve_compare.png"),
        "actor_input_std_plot": str(root_dir / "comparison_plots" / "actor_input_std_compare.png"),
        "actor_input_dim_std_plot": str(
            root_dir / "comparison_plots" / "actor_input_dim_std_compare.png"
        ),
        "actor_grad_norm_plot": str(root_dir / "comparison_plots" / "actor_grad_norm_compare.png"),
        "policy_entropy_plot": str(root_dir / "comparison_plots" / "policy_entropy_compare.png"),
        "action_distribution_plot": str(
            root_dir / "comparison_plots" / "action_distribution_compare.png"
        ),
        "advantage_stats_plot": str(
            root_dir / "comparison_plots" / "advantage_stats_compare.png"
        ),
        "critic_loss_plot": str(root_dir / "comparison_plots" / "critic_loss_compare.png"),
        "explained_variance_plot": str(
            root_dir / "comparison_plots" / "explained_variance_compare.png"
        ),
        "prediction_target_corr_plot": str(
            root_dir / "comparison_plots" / "prediction_target_corr_compare.png"
        ),
        "prediction_std_ratio_plot": str(
            root_dir / "comparison_plots" / "prediction_std_ratio_compare.png"
        ),
        "actor_raw_zero_var_dim_plot": str(
            root_dir / "comparison_plots" / "actor_raw_zero_var_dim_compare.png"
        ),
        "actor_raw_dim_std_plot": str(
            root_dir / "comparison_plots" / "actor_raw_dim_std_compare.png"
        ),
        "prediction_vs_target_scatter": str(
            root_dir / "comparison_plots" / "prediction_vs_target_scatter.png"
        ),
        "target_bucket_prediction_mean_plot": str(
            root_dir / "comparison_plots" / "target_bucket_prediction_mean.png"
        ),
    }
    (root_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root_dir


def main() -> None:
    args = parse_args()
    root_dir = run_experiment(
        actor_raw_input_modes=args.actor_raw_input_modes,
        seed=args.seed,
        output_root=Path(args.output_root),
    )
    print(f"Dense actor raw prune experiment completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
