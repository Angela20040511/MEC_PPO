from __future__ import annotations

import argparse
import csv
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
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator

FIXED_SEED = 2025
FIXED_TOPOLOGY = "dense"
FIXED_REWARD_MODE = "per_sensor"

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

DEFAULT_VALUE_TARGET_MODES = ("normalized_return", "popart_return_norm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled dense value-target normalization experiment."
    )
    parser.add_argument(
        "--value-target-modes",
        nargs="+",
        default=list(DEFAULT_VALUE_TARGET_MODES),
        help="Value target modes to compare.",
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


def build_value_target_config(value_target_mode: str, seed: int) -> MECConfig:
    config = build_config(topology_mode=FIXED_TOPOLOGY)
    training = replace(
        config.training,
        num_epochs=FIXED_MAX_EPOCHS,
        time_steps=FIXED_TIME_STEPS,
        seed=seed,
    )
    ppo = replace(
        config.ppo,
        actor_learning_rate=FIXED_ACTOR_LEARNING_RATE,
        critic_learning_rate=FIXED_CRITIC_LEARNING_RATE,
        entropy_coeff=FIXED_ENTROPY_COEFF,
        value_coeff=FIXED_VALUE_COEFF,
        value_target_mode=value_target_mode,
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


def save_logs(run_dir: Path, logs: list[dict[str, float | int | str]]) -> None:
    if not logs:
        return
    with (run_dir / "train_logs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)


def run_training_epoch(
    agent: PPOAgent,
    simulator: Simulator,
    config: MECConfig,
    epoch: int,
) -> dict[str, float | int | str]:
    state = simulator.reset(seed=config.training.seed + epoch)

    episode_reward = 0.0
    episode_delay = 0.0
    episode_energy = 0.0
    episode_raw_total_backlog = 0.0
    episode_normalized_total_backlog = 0.0
    episode_normalized_avg_queue_len = 0.0
    episode_reward_delay_term = 0.0
    episode_reward_energy_term = 0.0
    episode_reward_backlog_term = 0.0
    episode_avg_theta = 0.0

    step_count = 0
    done = False
    for _ in range(config.training.time_steps):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, info = simulator.step(action)

        agent.store_transition(state, action, log_prob, reward, done, value)
        state = next_state

        episode_reward += float(reward)
        episode_delay += float(info["raw_total_delay"])
        episode_energy += float(info["raw_total_energy"])
        episode_raw_total_backlog += float(info["raw_total_backlog"])
        episode_normalized_total_backlog += float(info["normalized_total_backlog"])
        episode_normalized_avg_queue_len += float(info["avg_queue_len"])
        episode_reward_delay_term += float(info["reward_delay_term"])
        episode_reward_energy_term += float(info["reward_energy_term"])
        episode_reward_backlog_term += float(info["reward_backlog_term"])
        episode_avg_theta += float(info["avg_theta"])

        step_count += 1
        if done:
            break

    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)
    losses = agent.train()

    return {
        "epoch": int(epoch),
        "reward_mode": config.system.reward_mode,
        "value_target_mode": config.ppo.value_target_mode,
        "num_sensors": int(len(config.sensors)),
        "episode_reward": episode_reward,
        "episode_delay": episode_delay,
        "episode_energy": episode_energy,
        "avg_raw_total_backlog": episode_raw_total_backlog / max(step_count, 1),
        "avg_normalized_total_backlog": episode_normalized_total_backlog / max(step_count, 1),
        "avg_normalized_avg_queue_len": episode_normalized_avg_queue_len / max(step_count, 1),
        "episode_reward_delay_term": episode_reward_delay_term,
        "episode_reward_energy_term": episode_reward_energy_term,
        "episode_reward_backlog_term": episode_reward_backlog_term,
        "avg_theta": episode_avg_theta / max(step_count, 1),
        "num_steps": int(step_count),
        "actor_loss": losses["actor_loss"],
        "critic_loss": losses["critic_loss"],
        "entropy": losses["entropy"],
        "raw_return_mean": losses["raw_return_mean"],
        "raw_return_std": losses["raw_return_std"],
        "value_target_mean": losses["value_target_mean"],
        "value_target_std": losses["value_target_std"],
        "running_return_mean": losses["running_return_mean"],
        "running_return_std": losses["running_return_std"],
        "popart_mean": losses["popart_mean"],
        "popart_std": losses["popart_std"],
        "critic_raw_prediction_mean": losses["critic_raw_prediction_mean"],
        "critic_raw_prediction_std": losses["critic_raw_prediction_std"],
        "critic_normalized_prediction_mean": losses["critic_normalized_prediction_mean"],
        "critic_normalized_prediction_std": losses["critic_normalized_prediction_std"],
    }


def train_agent_with_early_stopping(
    config: MECConfig,
    checkpoint_dir: str,
    early_stopping: EarlyStoppingConfig,
) -> tuple[PPOAgent, list[dict[str, float | int | str]], dict[str, object]]:
    simulator = Simulator(config)
    agent = PPOAgent(config=config.ppo, state_dim=config.state_dim, action_dim=config.action_dim)
    logs: list[dict[str, float | int | str]] = []

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
    final_reward = float(logs[-1]["episode_reward"]) if logs else None
    summary: dict[str, object] = {
        "reward_mode": config.system.reward_mode,
        "value_target_mode": config.ppo.value_target_mode,
        "topology": FIXED_TOPOLOGY,
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


def extract_metrics(run_dir: Path, value_target_mode: str) -> dict[str, float | int | str]:
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    best_reward = float(summary["best_reward"])
    final_reward = float(summary["final_reward"])
    reward_gap = final_reward - best_reward
    best_epoch = int(best_model_info["best_epoch"])
    metrics: dict[str, float | int | str] = {
        "value_target_mode": value_target_mode,
        "run_dir": str(run_dir),
        "epochs_completed": int(summary["epochs_completed"]),
        "best_reward": best_reward,
        "final_reward": final_reward,
        "best_epoch": best_epoch,
        "reward_gap": reward_gap,
        "reward_gap_abs": abs(reward_gap),
        "final_critic_loss": float(logs["critic_loss"].iloc[-1]),
        "final_raw_return_mean": float(logs["raw_return_mean"].iloc[-1]),
        "final_raw_return_std": float(logs["raw_return_std"].iloc[-1]),
        "final_value_target_mean": float(logs["value_target_mean"].iloc[-1]),
        "final_value_target_std": float(logs["value_target_std"].iloc[-1]),
        "final_running_return_mean": float(logs["running_return_mean"].iloc[-1]),
        "final_running_return_std": float(logs["running_return_std"].iloc[-1]),
        "final_popart_mean": float(logs["popart_mean"].iloc[-1]),
        "final_popart_std": float(logs["popart_std"].iloc[-1]),
        "final_critic_raw_prediction_mean": float(logs["critic_raw_prediction_mean"].iloc[-1]),
        "final_critic_raw_prediction_std": float(logs["critic_raw_prediction_std"].iloc[-1]),
        "final_critic_normalized_prediction_mean": float(
            logs["critic_normalized_prediction_mean"].iloc[-1]
        ),
        "final_critic_normalized_prediction_std": float(
            logs["critic_normalized_prediction_std"].iloc[-1]
        ),
        "best_epoch_is_one": int(best_epoch == 1),
    }
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def rank_runs(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.copy()
    ranked = ranked.sort_values(
        by=[
            "best_epoch_is_one",
            "reward_gap_abs",
            "final_reward",
            "final_critic_loss",
            "best_reward",
        ],
        ascending=[True, True, False, True, False],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def plot_comparisons(root_dir: Path, ranked_df: pd.DataFrame) -> None:
    plots_dir = root_dir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mode_logs: dict[str, pd.DataFrame] = {}
    for row in ranked_df.itertuples(index=False):
        mode_logs[str(row.value_target_mode)] = pd.read_csv(Path(row.run_dir) / "train_logs.csv")

    plt.figure(figsize=(8, 5))
    for value_target_mode, logs in mode_logs.items():
        plt.plot(logs["epoch"], logs["critic_loss"], marker="o", label=value_target_mode)
    plt.xlabel("epoch")
    plt.ylabel("critic_loss")
    plt.title("Critic Loss by Value Target Mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "critic_loss_compare.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for value_target_mode, logs in mode_logs.items():
        plt.plot(logs["epoch"], logs["episode_reward"], marker="o", label=value_target_mode)
    plt.xlabel("epoch")
    plt.ylabel("episode_reward")
    plt.title("Reward Curve by Value Target Mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "reward_curve_compare.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    target_columns = [
        ("value_target_mean", "Value Target Mean"),
        ("value_target_std", "Value Target Std"),
    ]
    for axis, (column, title) in zip(axes, target_columns):
        for value_target_mode, logs in mode_logs.items():
            axis.plot(logs["epoch"], logs[column], marker="o", label=value_target_mode)
        axis.set_ylabel(column)
        axis.set_title(title)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "target_stats_compare.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    running_columns = [
        ("running_return_mean", "Running Return Mean"),
        ("running_return_std", "Running Return Std"),
    ]
    for axis, (column, title) in zip(axes, running_columns):
        for value_target_mode, logs in mode_logs.items():
            axis.plot(logs["epoch"], logs[column], marker="o", label=value_target_mode)
        axis.set_ylabel(column)
        axis.set_title(title)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "running_stats_compare.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    prediction_columns = [
        ("critic_raw_prediction_mean", "Critic Raw Prediction Mean"),
        ("critic_normalized_prediction_mean", "Critic Normalized Prediction Mean"),
    ]
    for axis, (column, title) in zip(axes, prediction_columns):
        for value_target_mode, logs in mode_logs.items():
            axis.plot(logs["epoch"], logs[column], marker="o", label=value_target_mode)
        axis.set_ylabel(column)
        axis.set_title(title)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "prediction_mean_compare.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    prediction_std_columns = [
        ("critic_raw_prediction_std", "Critic Raw Prediction Std"),
        ("critic_normalized_prediction_std", "Critic Normalized Prediction Std"),
    ]
    for axis, (column, title) in zip(axes, prediction_std_columns):
        for value_target_mode, logs in mode_logs.items():
            axis.plot(logs["epoch"], logs[column], marker="o", label=value_target_mode)
        axis.set_ylabel(column)
        axis.set_title(title)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "prediction_std_compare.png", dpi=150)
    plt.close(fig)


def run_experiment(value_target_modes: list[str], seed: int, output_root: Path) -> Path:
    root_dir = output_root / datetime.now().strftime("dense_value_target_experiment_%Y%m%d_%H%M%S")
    root_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStoppingConfig(
        patience=FIXED_PATIENCE,
        min_delta=FIXED_MIN_DELTA,
        monitor_start_epoch=FIXED_START_CHECK_EPOCH,
    )
    manifest: dict[str, object] = {
        "root_dir": str(root_dir),
        "fixed_params": {
            "topology": FIXED_TOPOLOGY,
            "seed": seed,
            "reward_mode": FIXED_REWARD_MODE,
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
        },
        "value_target_modes": value_target_modes,
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []
    for value_target_mode in value_target_modes:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root_dir / f"value_target_{value_target_mode}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = build_value_target_config(value_target_mode=value_target_mode, seed=seed)
        (run_dir / "experiment_config.json").write_text(
            json.dumps(
                {
                    "training": asdict(config.training),
                    "ppo": asdict(config.ppo),
                    "dt": asdict(config.dt),
                    "system": asdict(config.system),
                    "early_stopping": asdict(early_stopping),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[train] value_target_mode={value_target_mode}, seed={seed} -> {run_dir}")
        set_global_seeds(seed)
        train_agent_with_early_stopping(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
        )
        metrics = extract_metrics(run_dir=run_dir, value_target_mode=value_target_mode)
        all_metrics.append(metrics)
        manifest["runs"].append(
            {
                "value_target_mode": value_target_mode,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    ranked_df = rank_runs(pd.DataFrame(all_metrics))
    summary_csv = root_dir / "value_target_summary.csv"
    summary_json = root_dir / "value_target_summary.json"
    summary_md = root_dir / "value_target_summary.md"
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
        "running_stats_plot": str(root_dir / "comparison_plots" / "running_stats_compare.png"),
        "prediction_mean_plot": str(root_dir / "comparison_plots" / "prediction_mean_compare.png"),
        "prediction_std_plot": str(root_dir / "comparison_plots" / "prediction_std_compare.png"),
    }
    (root_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root_dir


def main() -> None:
    args = parse_args()
    root_dir = run_experiment(
        value_target_modes=args.value_target_modes,
        seed=args.seed,
        output_root=Path(args.output_root),
    )
    print(f"Dense value-target experiment completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
