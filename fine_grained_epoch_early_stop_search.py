from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import MECConfig, build_default_config
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator

FIXED_ACTOR_LEARNING_RATE = 5e-4
FIXED_ENTROPY_COEFF = 1e-3
FIXED_UPDATE_EPOCHS = 10
FIXED_TIME_STEPS = 100
DEFAULT_MAX_EPOCHS_GRID = (30, 33, 35, 37, 40, 45)
DEFAULT_PATIENCE_GRID = (5, 8, 10)
DEFAULT_MIN_DELTA_GRID = (0.5, 1.0, 2.0)
EARLY_STOP_MONITOR_START_EPOCH = 25


@dataclass(frozen=True)
class EarlyStoppingConfig:
    patience: int
    min_delta: float
    monitor_start_epoch: int = EARLY_STOP_MONITOR_START_EPOCH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-grained epoch search with early stopping for PPO."
    )
    parser.add_argument(
        "--max-epochs-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_MAX_EPOCHS_GRID),
        help="Candidate max_epochs values.",
    )
    parser.add_argument(
        "--patience-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_PATIENCE_GRID),
        help="Candidate early stopping patience values.",
    )
    parser.add_argument(
        "--min-delta-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_MIN_DELTA_GRID),
        help="Candidate early stopping min_delta values.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="checkpoints",
        help="Root directory for experiment outputs.",
    )
    return parser.parse_args()


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_search_config(max_epochs: int) -> MECConfig:
    config = build_default_config()
    training = replace(
        config.training,
        num_epochs=max_epochs,
        time_steps=FIXED_TIME_STEPS,
    )
    ppo = replace(
        config.ppo,
        actor_learning_rate=FIXED_ACTOR_LEARNING_RATE,
        entropy_coeff=FIXED_ENTROPY_COEFF,
        update_epochs=FIXED_UPDATE_EPOCHS,
    )
    return replace(config, training=training, ppo=ppo)


def run_training_epoch(
    agent: PPOAgent,
    simulator: Simulator,
    config: MECConfig,
    epoch: int,
) -> dict[str, float]:
    state = simulator.reset(seed=config.training.seed + epoch)

    episode_reward = 0.0
    episode_cost = 0.0
    episode_queue_penalty = 0.0

    episode_delay = 0.0
    episode_local_delay_cost = 0.0
    episode_uplink_delay_cost = 0.0
    episode_backhaul_delay_cost = 0.0
    episode_bs_compute_delay_cost = 0.0

    episode_energy = 0.0
    episode_uplink_energy = 0.0
    episode_local_compute_energy = 0.0
    episode_bs_compute_energy = 0.0

    episode_normalized_avg_queue_len = 0.0
    episode_avg_theta = 0.0
    step_count = 0

    done = False
    for _ in range(config.training.time_steps):
        action, log_prob, value, policy_cache = agent.select_action_with_info(state)
        next_state, reward, done, info = simulator.step(action)

        agent.store_transition(
            state,
            action,
            log_prob,
            reward,
            done,
            value,
            policy_cache=policy_cache,
        )
        state = next_state

        episode_reward += reward
        episode_cost += float(info["cost"])
        episode_queue_penalty += float(info["queue_penalty"])

        episode_delay += float(info["total_delay"])
        episode_local_delay_cost += float(info["local_delay_cost"])
        episode_uplink_delay_cost += float(info["uplink_delay_cost"])
        episode_backhaul_delay_cost += float(info["backhaul_delay_cost"])
        episode_bs_compute_delay_cost += float(info["bs_compute_delay_cost"])

        episode_energy += float(info["total_energy"])
        episode_uplink_energy += float(info["uplink_energy"])
        episode_local_compute_energy += float(info["local_compute_energy"])
        episode_bs_compute_energy += float(info["bs_compute_energy"])

        episode_normalized_avg_queue_len += float(info["total_backlog"])
        episode_avg_theta += float(info["avg_theta"])

        step_count += 1
        if done:
            break

    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)
    losses = agent.train()

    avg_backlog = episode_normalized_avg_queue_len / max(step_count, 1)
    avg_theta = episode_avg_theta / max(step_count, 1)
    return {
        "epoch": float(epoch),
        "episode_reward": episode_reward,
        "episode_cost": episode_cost,
        "episode_queue_penalty": episode_queue_penalty,
        "episode_delay": episode_delay,
        "episode_local_delay_cost": episode_local_delay_cost,
        "episode_uplink_delay_cost": episode_uplink_delay_cost,
        "episode_backhaul_delay_cost": episode_backhaul_delay_cost,
        "episode_bs_compute_delay_cost": episode_bs_compute_delay_cost,
        "episode_energy": episode_energy,
        "episode_uplink_energy": episode_uplink_energy,
        "episode_local_compute_energy": episode_local_compute_energy,
        "episode_bs_compute_energy": episode_bs_compute_energy,
        "avg_total_backlog": avg_backlog,
        "avg_normalized_avg_queue_len": avg_backlog,
        "avg_theta": avg_theta,
        "num_steps": float(step_count),
        "actor_loss": losses["actor_loss"],
        "critic_loss": losses["critic_loss"],
        "entropy": losses["entropy"],
    }


def save_logs(run_dir: Path, logs: list[dict[str, float]]) -> None:
    with (run_dir / "train_logs.json").open("w", encoding="utf-8") as handle:
        json.dump(logs, handle, ensure_ascii=False, indent=2)

    if not logs:
        return

    fieldnames = list(logs[0].keys())
    with (run_dir / "train_logs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)


def train_agent_with_early_stopping(
    config: MECConfig,
    checkpoint_dir: str,
    early_stopping: EarlyStoppingConfig,
) -> tuple[PPOAgent, list[dict[str, float]], dict[str, object]]:
    simulator = Simulator(config)
    agent = PPOAgent(config=config.ppo, state_dim=config.state_dim, action_dim=config.action_dim)
    logs: list[dict[str, float]] = []

    run_dir = Path(checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_reward = float("-inf")
    best_epoch = -1
    monitor_reference = float("-inf")
    epochs_without_significant_improvement = 0
    stopped_early = False
    stop_reason = "reached_max_epochs"

    for epoch in range(config.training.num_epochs):
        log = run_training_epoch(agent, simulator, config, epoch)
        logs.append(log)
        episode_reward = float(log["episode_reward"])

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_epoch = epoch
            agent.save(str(run_dir / "best_model.pt"))
            (run_dir / "best_model_info.json").write_text(
                json.dumps(
                    {
                        "best_epoch": best_epoch,
                        "best_reward": best_reward,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        if epoch == early_stopping.monitor_start_epoch - 1:
            monitor_reference = best_reward
            epochs_without_significant_improvement = 0
        elif epoch >= early_stopping.monitor_start_epoch:
            if episode_reward >= monitor_reference + early_stopping.min_delta:
                monitor_reference = episode_reward
                epochs_without_significant_improvement = 0
            else:
                epochs_without_significant_improvement += 1

            if epochs_without_significant_improvement >= early_stopping.patience:
                stopped_early = True
                stop_reason = (
                    "no significant reward improvement "
                    f"for {early_stopping.patience} epochs after epoch "
                    f"{early_stopping.monitor_start_epoch}"
                )
                break

    if (run_dir / "best_model.pt").exists():
        agent.load(str(run_dir / "best_model.pt"), load_optimizer=False)

    save_logs(run_dir, logs)

    final_epoch = len(logs) - 1 if logs else -1
    final_reward = logs[-1]["episode_reward"] if logs else None
    summary: dict[str, object] = {
        "max_epochs": config.training.num_epochs,
        "epochs_completed": len(logs),
        "best_epoch": best_epoch,
        "best_reward": best_reward,
        "final_epoch": final_epoch,
        "final_reward": final_reward,
        "reward_gap": (final_reward - best_reward) if final_reward is not None else None,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "monitor_start_epoch": early_stopping.monitor_start_epoch,
        "early_stopping_patience": early_stopping.patience,
        "min_delta": early_stopping.min_delta,
        "restored_best_model": True,
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return agent, logs, summary


def plot_metric(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    best_epoch: int,
    stop_epoch: int,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df[y_col], linewidth=2, color="#1f77b4")
    plt.axvline(best_epoch, color="#d62728", linestyle="--", linewidth=1.5, label=f"best_epoch={best_epoch}")
    plt.axvline(stop_epoch, color="#2ca02c", linestyle=":", linewidth=1.5, label=f"stop_epoch={stop_epoch}")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_run_plots(run_dir: Path, best_epoch: int, stop_epoch: int) -> list[str]:
    df = pd.read_csv(run_dir / "train_logs.csv")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(
        df,
        y_col="episode_reward",
        title="Reward Curve",
        ylabel="Reward",
        output_path=plots_dir / "reward_curve.png",
        best_epoch=best_epoch,
        stop_epoch=stop_epoch,
    )
    plot_metric(
        df,
        y_col="critic_loss",
        title="Critic Loss Curve",
        ylabel="Critic Loss",
        output_path=plots_dir / "critic_loss_curve.png",
        best_epoch=best_epoch,
        stop_epoch=stop_epoch,
    )
    plot_metric(
        df,
        y_col="avg_theta",
        title="Avg Theta Curve",
        ylabel="Avg Theta",
        output_path=plots_dir / "avg_theta_curve.png",
        best_epoch=best_epoch,
        stop_epoch=stop_epoch,
    )
    plot_metric(
        df,
        y_col="avg_total_backlog",
        title="Backlog Curve",
        ylabel="Normalized Avg Backlog",
        output_path=plots_dir / "backlog_curve.png",
        best_epoch=best_epoch,
        stop_epoch=stop_epoch,
    )
    return sorted(str(path) for path in plots_dir.glob("*.png"))


def stability_score(df: pd.DataFrame) -> float:
    reward_tail = df["episode_reward"].tail(min(10, len(df))).astype(float)
    critic_tail = df["critic_loss"].tail(min(10, len(df))).astype(float)
    reward_std = float(reward_tail.std(ddof=0)) if len(reward_tail) > 1 else 0.0
    critic_std = float(critic_tail.std(ddof=0)) if len(critic_tail) > 1 else 0.0
    return reward_std + 0.01 * critic_std


def extract_metrics(
    run_dir: Path,
    max_epochs: int,
    patience: int,
    min_delta: float,
) -> dict[str, float | int | str | bool]:
    df = pd.read_csv(run_dir / "train_logs.csv")
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))

    best_reward = float(summary["best_reward"])
    final_reward = float(summary["final_reward"])
    reward_gap = float(final_reward - best_reward)
    best_epoch = int(best_model_info["best_epoch"])
    avg_theta_last10_mean = float(df["avg_theta"].tail(min(10, len(df))).mean())
    final_critic_loss = float(df["critic_loss"].iloc[-1])
    post_peak_degradation = reward_gap < 0.0
    metrics: dict[str, float | int | str | bool] = {
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "run_dir": str(run_dir),
        "epochs_completed": int(summary["epochs_completed"]),
        "stopped_early": bool(summary["stopped_early"]),
        "best_reward": best_reward,
        "final_reward": final_reward,
        "best_epoch": best_epoch,
        "reward_gap": reward_gap,
        "reward_gap_abs": abs(reward_gap),
        "final_critic_loss": final_critic_loss,
        "avg_theta_last10_mean": avg_theta_last10_mean,
        "post_peak_degradation": post_peak_degradation,
        "stability_score": stability_score(df),
    }
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in df.to_numpy():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def rank_experiments(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.copy()
    ranked = ranked.sort_values(
        by=[
            "best_reward",
            "reward_gap_abs",
            "final_reward",
            "stability_score",
        ],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def plot_top20_best_reward(summary_df: pd.DataFrame, output_path: Path) -> None:
    top20 = summary_df.head(20).copy()
    labels = [
        f"E{int(row.max_epochs)}-P{int(row.patience)}-D{row.min_delta:g}"
        for row in top20.itertuples(index=False)
    ]
    plt.figure(figsize=(14, 7))
    plt.bar(labels, top20["best_reward"], color="#1f77b4")
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Best Reward")
    plt.title("Top 20 Configurations by Best Reward")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_reward_gap_scatter(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        summary_df["best_reward"],
        summary_df["reward_gap_abs"],
        c=summary_df["max_epochs"],
        cmap="viridis",
        alpha=0.85,
    )
    plt.colorbar(scatter, label="max_epochs")
    plt.xlabel("best_reward")
    plt.ylabel("|final_reward - best_reward|")
    plt.title("Best Reward vs Late-stage Degradation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_comparison_outputs(root_dir: Path, metrics: list[dict[str, float | int | str | bool]]) -> dict[str, str]:
    summary_df = pd.DataFrame(metrics)
    ranked_df = rank_experiments(summary_df)

    comparison_dir = root_dir / "comparison_plots"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plot_top20_best_reward(ranked_df, comparison_dir / "top20_best_reward.png")
    plot_reward_gap_scatter(ranked_df, comparison_dir / "best_reward_vs_gap.png")

    summary_csv = root_dir / "comparison_summary.csv"
    summary_json = root_dir / "comparison_summary.json"
    summary_md = root_dir / "comparison_summary.md"
    ranked_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        ranked_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(dataframe_to_markdown(ranked_df), encoding="utf-8")

    return {
        "comparison_dir": str(comparison_dir),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }


def run_search(
    max_epochs_grid: list[int],
    patience_grid: list[int],
    min_delta_grid: list[float],
    output_root: Path,
) -> Path:
    search_tag = datetime.now().strftime("fine_grained_epoch_early_stop_%Y%m%d_%H%M%S")
    root_dir = output_root / search_tag
    root_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "root_dir": str(root_dir),
        "fixed_params": {
            "actor_learning_rate": FIXED_ACTOR_LEARNING_RATE,
            "entropy_coeff": FIXED_ENTROPY_COEFF,
            "update_epochs": FIXED_UPDATE_EPOCHS,
            "time_steps": FIXED_TIME_STEPS,
            "monitor_start_epoch": EARLY_STOP_MONITOR_START_EPOCH,
        },
        "grid": {
            "max_epochs": max_epochs_grid,
            "patience": patience_grid,
            "min_delta": min_delta_grid,
        },
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str | bool]] = []

    for max_epochs in max_epochs_grid:
        for patience in patience_grid:
            for min_delta in min_delta_grid:
                run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                run_dir = root_dir / (
                    f"max_{max_epochs:03d}_pat_{patience:02d}_delta_{str(min_delta).replace('.', 'p')}_{run_tag}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)

                config = build_search_config(max_epochs)
                early_stopping = EarlyStoppingConfig(patience=patience, min_delta=min_delta)
                (run_dir / "experiment_config.json").write_text(
                    json.dumps(
                        {
                            "max_epochs": max_epochs,
                            "training": asdict(config.training),
                            "ppo": asdict(config.ppo),
                            "early_stopping": asdict(early_stopping),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                print(
                    f"[train] max_epochs={max_epochs}, patience={patience}, min_delta={min_delta:g} -> {run_dir}"
                )
                set_global_seeds(config.training.seed)
                _, _, summary = train_agent_with_early_stopping(
                    config=config,
                    checkpoint_dir=str(run_dir),
                    early_stopping=early_stopping,
                )
                plots = generate_run_plots(
                    run_dir,
                    best_epoch=int(summary["best_epoch"]),
                    stop_epoch=int(summary["final_epoch"]),
                )
                metrics = extract_metrics(run_dir, max_epochs, patience, min_delta)
                all_metrics.append(metrics)
                manifest["runs"].append(
                    {
                        "max_epochs": max_epochs,
                        "patience": patience,
                        "min_delta": min_delta,
                        "run_dir": str(run_dir),
                        "analysis_summary": str(run_dir / "analysis_summary.json"),
                        "plots": plots,
                    }
                )

    manifest["comparison"] = save_comparison_outputs(root_dir, all_metrics)
    (root_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root_dir


def main() -> None:
    args = parse_args()
    root_dir = run_search(
        max_epochs_grid=args.max_epochs_grid,
        patience_grid=args.patience_grid,
        min_delta_grid=args.min_delta_grid,
        output_root=Path(args.output_root),
    )
    print(f"Fine-grained search completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
