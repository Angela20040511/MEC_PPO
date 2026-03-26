from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import MECConfig, build_default_config
from train import train_agent

FIXED_ACTOR_LEARNING_RATE = 5e-4
FIXED_ENTROPY_COEFF = 1e-3
FIXED_UPDATE_EPOCHS = 10
FIXED_TIME_STEPS = 100
DEFAULT_EPOCHS_GRID = (40, 60, 80, 100, 120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled epochs-only search for PPO training.")
    parser.add_argument(
        "--epochs-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_EPOCHS_GRID),
        help="Epoch values to search. Defaults to 40 60 80 100 120.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="checkpoints",
        help="Root directory where the search folder will be created.",
    )
    return parser.parse_args()


def build_search_config(num_epochs: int) -> MECConfig:
    config = build_default_config()
    training = replace(
        config.training,
        num_epochs=num_epochs,
        time_steps=FIXED_TIME_STEPS,
    )
    ppo = replace(
        config.ppo,
        actor_learning_rate=FIXED_ACTOR_LEARNING_RATE,
        entropy_coeff=FIXED_ENTROPY_COEFF,
        update_epochs=FIXED_UPDATE_EPOCHS,
    )
    return replace(config, training=training, ppo=ppo)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_series(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    for col in y_cols:
        if col in df.columns:
            plt.plot(df[x_col], df[col], label=col, linewidth=2)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_run_plots(run_dir: Path) -> list[str]:
    csv_path = run_dir / "train_logs.csv"
    df = pd.read_csv(csv_path)
    output_dir = run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_series(
        df,
        x_col="epoch",
        y_cols=["episode_reward"],
        title="Training Reward",
        ylabel="Reward",
        output_path=output_dir / "reward.png",
    )
    plot_series(
        df,
        x_col="epoch",
        y_cols=["episode_cost", "episode_queue_penalty"],
        title="Cost and Queue Penalty",
        ylabel="Value",
        output_path=output_dir / "cost_queue_penalty.png",
    )
    plot_series(
        df,
        x_col="epoch",
        y_cols=["episode_delay", "episode_energy"],
        title="Delay and Energy",
        ylabel="Value",
        output_path=output_dir / "delay_energy.png",
    )
    plot_series(
        df,
        x_col="epoch",
        y_cols=["avg_total_backlog", "avg_theta"],
        title="Backlog and Avg Theta",
        ylabel="Value",
        output_path=output_dir / "backlog_theta.png",
    )
    plot_series(
        df,
        x_col="epoch",
        y_cols=["actor_loss", "critic_loss", "entropy"],
        title="Actor Loss, Critic Loss, and Entropy",
        ylabel="Value",
        output_path=output_dir / "losses_entropy.png",
    )
    plot_series(
        df,
        x_col="epoch",
        y_cols=[
            "episode_local_delay_cost",
            "episode_uplink_delay_cost",
            "episode_backhaul_delay_cost",
            "episode_bs_compute_delay_cost",
        ],
        title="Delay Breakdown",
        ylabel="Delay",
        output_path=output_dir / "delay_breakdown.png",
    )
    plot_series(
        df,
        x_col="epoch",
        y_cols=[
            "episode_uplink_energy",
            "episode_local_compute_energy",
            "episode_bs_compute_energy",
        ],
        title="Energy Breakdown",
        ylabel="Energy",
        output_path=output_dir / "energy_breakdown.png",
    )
    return sorted(str(path) for path in output_dir.glob("*.png"))


def avg_theta_trend(series: pd.Series) -> tuple[str, float, float, float]:
    values = series.astype(float).to_numpy()
    if values.size == 0:
        return "insufficient data", 0.0, 0.0, 0.0

    window = max(1, values.size // 4)
    first_mean = float(values[:window].mean())
    last_mean = float(values[-window:].mean())
    if values.size == 1:
        slope = 0.0
    else:
        slope = float(np.polyfit(np.arange(values.size), values, deg=1)[0])

    distance = abs(last_mean - 0.5)
    if distance < 0.02 and np.std(values) < 0.03:
        label = "长期卡在0.5附近"
    elif slope > 5e-4:
        label = f"整体上升，末段均值={last_mean:.3f}"
    elif slope < -5e-4:
        label = f"整体下降，末段均值={last_mean:.3f}"
    else:
        label = f"基本平稳，末段均值={last_mean:.3f}"

    if distance >= 0.03 and "0.5附近" not in label:
        label += "，已明显偏离0.5"
    return label, first_mean, last_mean, slope


def degradation_flag(best_reward: float, final_reward: float) -> tuple[str, float]:
    gap = float(best_reward - final_reward)
    threshold = max(10.0, 0.03 * max(abs(best_reward), 1.0))
    if gap > 2 * threshold:
        return "明显", gap
    if gap > threshold:
        return "轻微", gap
    return "否", gap


def critic_divergence_flag(series: pd.Series) -> tuple[str, float]:
    values = series.astype(float).to_numpy()
    if values.size == 0:
        return "unknown", 1.0
    min_value = float(np.min(values))
    final_value = float(values[-1])
    ratio = final_value / max(min_value, 1e-8)
    if ratio > 5.0:
        return "明显发散", ratio
    if ratio > 2.0:
        return "有发散迹象", ratio
    return "基本稳定", ratio


def extract_metrics(run_dir: Path, epochs: int) -> dict[str, float | int | str]:
    df = pd.read_csv(run_dir / "train_logs.csv")
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))

    best_reward = float(summary["best_reward"])
    final_reward = float(summary["final_reward"])
    best_epoch = int(best_model_info["best_epoch"])
    reward_degradation, reward_gap = degradation_flag(best_reward, final_reward)
    theta_label, theta_first_mean, theta_last_mean, theta_slope = avg_theta_trend(df["avg_theta"])
    critic_flag, critic_ratio = critic_divergence_flag(df["critic_loss"])

    metrics: dict[str, float | int | str] = {
        "epochs": epochs,
        "run_dir": str(run_dir),
        "best_reward": best_reward,
        "final_reward": final_reward,
        "best_epoch": best_epoch,
        "best_delay": float(df["episode_delay"].min()),
        "best_cost": float(df["episode_cost"].min()),
        "delay_at_best_reward": float(df.loc[df["episode_reward"].idxmax(), "episode_delay"]),
        "cost_at_best_reward": float(df.loc[df["episode_reward"].idxmax(), "episode_cost"]),
        "final_critic_loss": float(df["critic_loss"].iloc[-1]),
        "avg_theta_trend": theta_label,
        "avg_theta_first_mean": theta_first_mean,
        "avg_theta_last_mean": theta_last_mean,
        "avg_theta_slope": theta_slope,
        "reward_degradation": reward_degradation,
        "reward_gap": reward_gap,
        "critic_loss_status": critic_flag,
        "critic_loss_ratio_vs_min": critic_ratio,
    }
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def plot_comparison_lines(
    experiment_frames: dict[int, pd.DataFrame],
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(11, 6))
    for epochs, df in sorted(experiment_frames.items()):
        if y_col not in df.columns:
            continue
        plt.plot(df["epoch"], df[y_col], label=f"epochs={epochs}", linewidth=2)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_reward_bar(summary_df: pd.DataFrame, output_path: Path) -> None:
    positions = np.arange(len(summary_df))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(positions - width / 2, summary_df["best_reward"], width=width, label="best_reward")
    plt.bar(positions + width / 2, summary_df["final_reward"], width=width, label="final_reward")
    plt.xticks(positions, [str(value) for value in summary_df["epochs"]])
    plt.xlabel("epochs")
    plt.ylabel("reward")
    plt.title("Best Reward vs Final Reward")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def rank_experiments(summary_df: pd.DataFrame) -> pd.DataFrame:
    working = summary_df.copy()
    working["degradation_score"] = working["reward_degradation"].map({"否": 0, "轻微": 1, "明显": 2}).fillna(3)
    working["theta_stuck_penalty"] = np.where(
        (working["avg_theta_last_mean"] - 0.5).abs() < 0.02,
        1,
        0,
    )
    working["critic_penalty"] = working["critic_loss_status"].map(
        {"基本稳定": 0, "有发散迹象": 1, "明显发散": 2}
    ).fillna(3)
    working = working.sort_values(
        by=[
            "best_reward",
            "reward_gap",
            "degradation_score",
            "theta_stuck_penalty",
            "critic_penalty",
        ],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    working.insert(0, "rank", np.arange(1, len(working) + 1))
    return working


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_comparison_outputs(root_dir: Path, metrics: list[dict[str, float | int | str]]) -> dict[str, object]:
    comparison_dir = root_dir / "comparison_plots"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    experiment_frames = {
        int(item["epochs"]): pd.read_csv(Path(str(item["run_dir"])) / "train_logs.csv")
        for item in metrics
    }
    plot_comparison_lines(
        experiment_frames,
        y_col="episode_reward",
        title="Reward Comparison Across Epoch Settings",
        ylabel="Reward",
        output_path=comparison_dir / "reward_comparison.png",
    )
    plot_comparison_lines(
        experiment_frames,
        y_col="critic_loss",
        title="Critic Loss Comparison Across Epoch Settings",
        ylabel="Critic Loss",
        output_path=comparison_dir / "critic_loss_comparison.png",
    )
    plot_comparison_lines(
        experiment_frames,
        y_col="avg_theta",
        title="Avg Theta Comparison Across Epoch Settings",
        ylabel="Avg Theta",
        output_path=comparison_dir / "avg_theta_comparison.png",
    )
    plot_comparison_lines(
        experiment_frames,
        y_col="avg_total_backlog",
        title="Backlog Comparison Across Epoch Settings",
        ylabel="Normalized Avg Backlog",
        output_path=comparison_dir / "backlog_comparison.png",
    )

    summary_df = pd.DataFrame(metrics)
    summary_df = rank_experiments(summary_df)
    plot_reward_bar(
        summary_df,
        output_path=comparison_dir / "best_final_reward_comparison.png",
    )

    summary_csv = root_dir / "comparison_summary.csv"
    summary_json = root_dir / "comparison_summary.json"
    summary_md = root_dir / "comparison_summary.md"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(summary_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(dataframe_to_markdown(summary_df), encoding="utf-8")

    return {
        "comparison_dir": str(comparison_dir),
        "comparison_plots": sorted(str(path) for path in comparison_dir.glob("*.png")),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }


def run_search(epochs_grid: list[int], output_root: Path) -> Path:
    search_tag = datetime.now().strftime("controlled_epochs_search_%Y%m%d_%H%M%S")
    root_dir = output_root / search_tag
    root_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "root_dir": str(root_dir),
        "fixed_params": {
            "actor_learning_rate": FIXED_ACTOR_LEARNING_RATE,
            "entropy_coeff": FIXED_ENTROPY_COEFF,
            "update_epochs": FIXED_UPDATE_EPOCHS,
            "time_steps": FIXED_TIME_STEPS,
        },
        "epochs_grid": epochs_grid,
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []

    for epochs in epochs_grid:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root_dir / f"epochs_{epochs:03d}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = build_search_config(epochs)
        (run_dir / "experiment_config.json").write_text(
            json.dumps(
                {
                    "epochs": epochs,
                    "training": asdict(config.training),
                    "ppo": asdict(config.ppo),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[train] epochs={epochs} -> {run_dir}")
        set_global_seeds(config.training.seed)
        train_agent(config=config, checkpoint_dir=str(run_dir))
        plot_paths = generate_run_plots(run_dir)
        metrics = extract_metrics(run_dir, epochs)
        metrics["plots_dir"] = str(run_dir / "plots")
        all_metrics.append(metrics)
        manifest["runs"].append(
            {
                "epochs": epochs,
                "run_dir": str(run_dir),
                "plots": plot_paths,
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    comparison_info = save_comparison_outputs(root_dir, all_metrics)
    manifest["comparison"] = comparison_info
    (root_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root_dir


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    root_dir = run_search(args.epochs_grid, output_root)
    print(f"Search completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
