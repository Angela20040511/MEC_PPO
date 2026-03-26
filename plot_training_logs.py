from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
            plt.plot(df[x_col], df[col], label=col)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from train_logs.csv")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to train_logs.csv",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "epoch" not in df.columns:
        raise ValueError("train_logs.csv must contain an 'epoch' column")

    output_dir = csv_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. reward
    plot_series(
        df,
        x_col="epoch",
        y_cols=["episode_reward"],
        title="Training Reward",
        ylabel="Reward",
        output_path=output_dir / "reward.png",
    )

    # 2. cost and queue penalty
    plot_series(
        df,
        x_col="epoch",
        y_cols=["episode_cost", "episode_queue_penalty"],
        title="Cost and Queue Penalty",
        ylabel="Value",
        output_path=output_dir / "cost_queue_penalty.png",
    )

    # 3. delay and energy
    plot_series(
        df,
        x_col="epoch",
        y_cols=["episode_delay", "episode_energy"],
        title="Delay and Energy",
        ylabel="Value",
        output_path=output_dir / "delay_energy.png",
    )

    # 4. backlog and avg_theta
    plot_series(
        df,
        x_col="epoch",
        y_cols=["avg_total_backlog", "avg_theta"],
        title="Backlog and Avg Theta",
        ylabel="Value",
        output_path=output_dir / "backlog_theta.png",
    )

    # 5. losses
    plot_series(
        df,
        x_col="epoch",
        y_cols=["critic_loss", "actor_loss", "entropy"],
        title="Training Losses and Entropy",
        ylabel="Value",
        output_path=output_dir / "losses_entropy.png",
    )

    # 6. delay breakdown
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

    # 7. energy breakdown
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

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()