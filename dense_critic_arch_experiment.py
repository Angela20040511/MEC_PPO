from __future__ import annotations

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
from dense_critic_diagnosis import (
    PROBE_STATE_COUNT,
    collect_probe_states,
    dataframe_to_markdown,
    extract_metrics,
    plot_single_run_outputs,
    train_with_diagnosis,
)
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds

FIXED_SEED = 2025
FIXED_TOPOLOGY = "dense"
FIXED_REWARD_MODE = "per_sensor"
FIXED_VALUE_TARGET_MODE = "popart_return_norm"
FIXED_VALUE_LOSS_MODE = "huber"

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

DEFAULT_CRITIC_ARCH_MODES = ("baseline_critic", "stronger_critic_backbone")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled dense critic architecture comparison."
    )
    parser.add_argument(
        "--critic-arch-modes",
        nargs="+",
        default=list(DEFAULT_CRITIC_ARCH_MODES),
        help="Critic architecture modes to compare.",
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


def build_critic_arch_config(critic_arch_mode: str, seed: int) -> MECConfig:
    config = build_config(topology_mode=FIXED_TOPOLOGY)
    training = replace(
        config.training,
        num_epochs=FIXED_MAX_EPOCHS,
        time_steps=FIXED_TIME_STEPS,
        seed=seed,
    )
    ppo = replace(
        config.ppo,
        critic_arch_mode=critic_arch_mode,
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


def rank_runs(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.copy()
    ranked["final_prediction_std_ratio_gap"] = np.abs(
        ranked["final_prediction_std_over_target_std"] - 1.0
    )
    ranked = ranked.sort_values(
        by=[
            "final_prediction_std_over_target_std",
            "final_value_explained_variance",
            "final_prediction_target_corr",
            "final_critic_vs_constant_mse_gain",
            "final_critic_hidden_feature_dim_std_mean",
            "final_critic_backbone_to_head_grad_ratio",
            "best_epoch_is_one",
            "final_reward",
        ],
        ascending=[False, False, False, False, False, False, True, False],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def load_mode_logs(ranked_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mode_logs: dict[str, pd.DataFrame] = {}
    for row in ranked_df.itertuples(index=False):
        mode_logs[str(row.critic_arch_mode)] = pd.read_csv(Path(row.run_dir) / "train_logs.csv")
    return mode_logs


def plot_metric_compare(
    mode_logs: dict[str, pd.DataFrame],
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for critic_arch_mode, logs in mode_logs.items():
        plt.plot(logs["epoch"], logs[column], marker="o", label=critic_arch_mode)
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
        for critic_arch_mode, logs in mode_logs.items():
            axis.plot(logs["epoch"], logs[column], marker="o", label=critic_arch_mode)
        axis.set_ylabel(column)
        axis.set_title(title)
        axis.legend()
    axes[-1].set_xlabel("epoch")
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
        axis.set_title(str(row.critic_arch_mode))
    fig.suptitle("Prediction vs Target Scatter", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_bucket_compare(root_dir: Path, ranked_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(ranked_df), figsize=(8 * len(ranked_df), 5), squeeze=False)
    bucket_frames: list[pd.DataFrame] = []
    for axis, row in zip(axes[0], ranked_df.itertuples(index=False), strict=True):
        logs = pd.read_csv(Path(row.run_dir) / "train_logs.csv")
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
        bucket_summary.insert(0, "critic_arch_mode", str(row.critic_arch_mode))
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
        axis.set_title(str(row.critic_arch_mode))
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
        column="critic_loss",
        title="Critic Loss by Critic Architecture",
        ylabel="critic_loss",
        output_path=plots_dir / "critic_loss_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="episode_reward",
        title="Reward Curve by Critic Architecture",
        ylabel="episode_reward",
        output_path=plots_dir / "reward_curve_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("value_target_mean", "Value Target Mean"),
            ("value_target_std", "Value Target Std"),
        ],
        output_path=plots_dir / "target_stats_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("critic_raw_prediction_mean", "Critic Raw Prediction Mean"),
            ("critic_normalized_prediction_mean", "Critic Normalized Prediction Mean"),
        ],
        output_path=plots_dir / "prediction_mean_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("critic_raw_prediction_std", "Critic Raw Prediction Std"),
            ("critic_normalized_prediction_std", "Critic Normalized Prediction Std"),
        ],
        output_path=plots_dir / "prediction_std_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="value_explained_variance",
        title="Explained Variance by Critic Architecture",
        ylabel="value_explained_variance",
        output_path=plots_dir / "explained_variance_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="prediction_target_corr",
        title="Prediction/Target Correlation by Critic Architecture",
        ylabel="prediction_target_corr",
        output_path=plots_dir / "prediction_target_corr_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="prediction_std_over_target_std",
        title="Prediction Std over Target Std by Critic Architecture",
        ylabel="prediction_std_over_target_std",
        output_path=plots_dir / "prediction_std_ratio_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="probe_prediction_std",
        title="Probe Prediction Std by Critic Architecture",
        ylabel="probe_prediction_std",
        output_path=plots_dir / "probe_prediction_std_curve.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("critic_vs_constant_mse_gain", "Critic vs Constant Baseline MSE Gain"),
            ("critic_vs_constant_huber_gain", "Critic vs Constant Baseline Huber Gain"),
        ],
        output_path=plots_dir / "critic_vs_constant_gain_compare.png",
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
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("critic_hidden_feature_std", "Critic Hidden Feature Std"),
            ("critic_hidden_feature_dim_std_mean", "Critic Hidden Feature Dim Std Mean"),
        ],
        output_path=plots_dir / "critic_hidden_feature_std_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("critic_backbone_grad_norm", "Critic Backbone Grad Norm"),
            ("critic_head_grad_norm", "Critic Head Grad Norm"),
        ],
        output_path=plots_dir / "critic_grad_norm_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="critic_head_bias_mean",
        title="Critic Head Bias Mean by Critic Architecture",
        ylabel="critic_head_bias_mean",
        output_path=plots_dir / "critic_head_bias_compare.png",
    )


def run_experiment(critic_arch_modes: list[str], seed: int, output_root: Path) -> Path:
    root_dir = output_root / datetime.now().strftime("dense_critic_arch_experiment_%Y%m%d_%H%M%S")
    root_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStoppingConfig(
        patience=FIXED_PATIENCE,
        min_delta=FIXED_MIN_DELTA,
        monitor_start_epoch=FIXED_START_CHECK_EPOCH,
    )
    probe_config = build_critic_arch_config(critic_arch_mode=critic_arch_modes[0], seed=seed)
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
        "critic_arch_modes": critic_arch_modes,
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []
    for critic_arch_mode in critic_arch_modes:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root_dir / f"critic_arch_{critic_arch_mode}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = build_critic_arch_config(critic_arch_mode=critic_arch_mode, seed=seed)
        (run_dir / "experiment_config.json").write_text(
            json.dumps(
                {
                    "training": asdict(config.training),
                    "ppo": asdict(config.ppo),
                    "dt": asdict(config.dt),
                    "system": asdict(config.system),
                    "early_stopping": asdict(early_stopping),
                    "probe_state_count": PROBE_STATE_COUNT,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[train] critic_arch_mode={critic_arch_mode}, seed={seed} -> {run_dir}")
        set_global_seeds(seed)
        train_with_diagnosis(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
            probe_states=probe_states,
        )
        plot_single_run_outputs(run_dir)

        metrics = extract_metrics(run_dir=run_dir)
        metrics["final_critic_backbone_to_head_grad_ratio"] = float(
            metrics["final_critic_backbone_grad_norm"]
        ) / (float(metrics["final_critic_head_grad_norm"]) + 1e-8)
        (run_dir / "analysis_summary.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_metrics.append(metrics)
        manifest["runs"].append(
            {
                "critic_arch_mode": critic_arch_mode,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    ranked_df = rank_runs(pd.DataFrame(all_metrics))
    summary_csv = root_dir / "critic_arch_summary.csv"
    summary_json = root_dir / "critic_arch_summary.json"
    summary_md = root_dir / "critic_arch_summary.md"
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
        "critic_loss_plot": str(root_dir / "comparison_plots" / "critic_loss_compare.png"),
        "reward_curve_plot": str(root_dir / "comparison_plots" / "reward_curve_compare.png"),
        "target_stats_plot": str(root_dir / "comparison_plots" / "target_stats_compare.png"),
        "prediction_mean_plot": str(root_dir / "comparison_plots" / "prediction_mean_compare.png"),
        "prediction_std_plot": str(root_dir / "comparison_plots" / "prediction_std_compare.png"),
        "explained_variance_plot": str(
            root_dir / "comparison_plots" / "explained_variance_compare.png"
        ),
        "prediction_target_corr_plot": str(
            root_dir / "comparison_plots" / "prediction_target_corr_compare.png"
        ),
        "prediction_std_ratio_plot": str(
            root_dir / "comparison_plots" / "prediction_std_ratio_compare.png"
        ),
        "probe_prediction_std_plot": str(
            root_dir / "comparison_plots" / "probe_prediction_std_curve.png"
        ),
        "critic_vs_constant_gain_plot": str(
            root_dir / "comparison_plots" / "critic_vs_constant_gain_compare.png"
        ),
        "prediction_vs_target_scatter": str(
            root_dir / "comparison_plots" / "prediction_vs_target_scatter.png"
        ),
        "target_bucket_prediction_mean_plot": str(
            root_dir / "comparison_plots" / "target_bucket_prediction_mean.png"
        ),
        "critic_hidden_feature_std_plot": str(
            root_dir / "comparison_plots" / "critic_hidden_feature_std_compare.png"
        ),
        "critic_grad_norm_plot": str(
            root_dir / "comparison_plots" / "critic_grad_norm_compare.png"
        ),
        "critic_head_bias_plot": str(
            root_dir / "comparison_plots" / "critic_head_bias_compare.png"
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
        critic_arch_modes=args.critic_arch_modes,
        seed=args.seed,
        output_root=Path(args.output_root),
    )
    print(f"Dense critic architecture experiment completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
