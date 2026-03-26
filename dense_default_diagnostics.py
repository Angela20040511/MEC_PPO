from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import MECConfig, build_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator

FIXED_SEED = 2025

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

FIXED_DIAGNOSTIC_EPOCHS = 3
FIXED_LOG_STEPS_PER_EPOCH = 20

TOPOLOGIES = ("default", "dense")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare early-training diagnostics between default and dense topologies."
    )
    parser.add_argument(
        "--diagnostic-epochs",
        type=int,
        default=FIXED_DIAGNOSTIC_EPOCHS,
        help="Number of early epochs to inspect.",
    )
    parser.add_argument(
        "--log-steps-per-epoch",
        type=int,
        default=FIXED_LOG_STEPS_PER_EPOCH,
        help="How many early steps per epoch to save in the step-level CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FIXED_SEED,
        help="Fixed seed for the diagnostics run.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="checkpoints",
        help="Root directory for experiment outputs.",
    )
    return parser.parse_args()


def build_fixed_config(topology_mode: str, seed: int, diagnostic_epochs: int) -> MECConfig:
    config = build_config(topology_mode=topology_mode)
    training = replace(
        config.training,
        num_epochs=diagnostic_epochs,
        time_steps=FIXED_TIME_STEPS,
        seed=seed,
    )
    ppo = replace(
        config.ppo,
        actor_learning_rate=FIXED_ACTOR_LEARNING_RATE,
        critic_learning_rate=FIXED_CRITIC_LEARNING_RATE,
        entropy_coeff=FIXED_ENTROPY_COEFF,
        value_coeff=FIXED_VALUE_COEFF,
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
        noma_quantile=FIXED_NOMA_QUANTILE,
        max_cluster_size=FIXED_MAX_CLUSTER_SIZE,
        alpha_p=FIXED_ALPHA_P,
        rho0=FIXED_RHO0,
        gamma_q=FIXED_GAMMA_Q,
        gamma_e=FIXED_GAMMA_E,
    )
    return replace(config, training=training, ppo=ppo, dt=dt, system=system)


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def state_summary_row(state: np.ndarray) -> dict[str, float]:
    state_array = np.asarray(state, dtype=np.float64)
    return {
        "state_mean": float(state_array.mean()),
        "state_max": float(state_array.max(initial=0.0)),
        "state_var": float(state_array.var()),
    }


def run_topology_diagnostic(
    *,
    topology_mode: str,
    seed: int,
    diagnostic_epochs: int,
    log_steps_per_epoch: int,
    root_dir: Path,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    config = build_fixed_config(topology_mode=topology_mode, seed=seed, diagnostic_epochs=diagnostic_epochs)
    topology_dir = root_dir / topology_mode
    topology_dir.mkdir(parents=True, exist_ok=True)
    (topology_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "topology": topology_mode,
                "seed": seed,
                "diagnostic_epochs": diagnostic_epochs,
                "log_steps_per_epoch": log_steps_per_epoch,
                "training": asdict(config.training),
                "ppo": asdict(config.ppo),
                "dt": asdict(config.dt),
                "system": asdict(config.system),
                "state_dim": config.state_dim,
                "action_dim": config.action_dim,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    simulator = Simulator(config)
    agent = PPOAgent(config=config.ppo, state_dim=config.state_dim, action_dim=config.action_dim)

    state_sum = np.zeros(config.state_dim, dtype=np.float64)
    state_sq_sum = np.zeros(config.state_dim, dtype=np.float64)
    state_max = np.full(config.state_dim, -np.inf, dtype=np.float64)
    observed_state_count = 0

    step_rows: list[dict[str, float | int | str]] = []
    epoch_rows: list[dict[str, float | int | str]] = []

    for epoch in range(config.training.num_epochs):
        state = simulator.reset(seed=config.training.seed + epoch)

        epoch_metrics = {
            "state_mean": 0.0,
            "state_max": 0.0,
            "state_var": 0.0,
            "raw_total_backlog": 0.0,
            "normalized_total_backlog": 0.0,
            "total_backlog": 0.0,
            "total_delay": 0.0,
            "total_energy": 0.0,
            "queue_penalty": 0.0,
            "reward_delay_term": 0.0,
            "reward_energy_term": 0.0,
            "reward_backlog_term": 0.0,
            "avg_theta": 0.0,
            "reward": 0.0,
        }

        step_count = 0
        done = False
        for step in range(config.training.time_steps):
            summary = state_summary_row(state)
            state_sum += state
            state_sq_sum += np.square(state)
            state_max = np.maximum(state_max, state)
            observed_state_count += 1

            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = simulator.step(action)
            agent.store_transition(state, action, log_prob, reward, done, value)

            epoch_metrics["state_mean"] += summary["state_mean"]
            epoch_metrics["state_max"] += summary["state_max"]
            epoch_metrics["state_var"] += summary["state_var"]
            epoch_metrics["raw_total_backlog"] += float(info["raw_total_backlog"])
            epoch_metrics["normalized_total_backlog"] += float(info["normalized_total_backlog"])
            epoch_metrics["total_backlog"] += float(info["total_backlog"])
            epoch_metrics["total_delay"] += float(info["total_delay"])
            epoch_metrics["total_energy"] += float(info["total_energy"])
            epoch_metrics["queue_penalty"] += float(info["queue_penalty"])
            epoch_metrics["reward_delay_term"] += float(info["reward_delay_term"])
            epoch_metrics["reward_energy_term"] += float(info["reward_energy_term"])
            epoch_metrics["reward_backlog_term"] += float(info["reward_backlog_term"])
            epoch_metrics["avg_theta"] += float(info["avg_theta"])
            epoch_metrics["reward"] += float(reward)

            if step < log_steps_per_epoch:
                step_rows.append(
                    {
                        "topology": topology_mode,
                        "epoch": epoch,
                        "step": step,
                        "state_mean": summary["state_mean"],
                        "state_max": summary["state_max"],
                        "state_var": summary["state_var"],
                        "raw_total_backlog": float(info["raw_total_backlog"]),
                        "normalized_total_backlog": float(info["normalized_total_backlog"]),
                        "total_backlog": float(info["total_backlog"]),
                        "total_delay": float(info["total_delay"]),
                        "total_energy": float(info["total_energy"]),
                        "queue_penalty": float(info["queue_penalty"]),
                        "reward_delay_term": float(info["reward_delay_term"]),
                        "reward_energy_term": float(info["reward_energy_term"]),
                        "reward_backlog_term": float(info["reward_backlog_term"]),
                        "avg_theta": float(info["avg_theta"]),
                        "reward": float(reward),
                    }
                )

            state = next_state
            step_count += 1
            if done:
                break

        last_value = 0.0 if done else agent.evaluate_value(state)
        agent.finish_trajectory(last_value)
        losses = agent.train()

        epoch_rows.append(
            {
                "topology": topology_mode,
                "epoch": epoch,
                "state_mean": epoch_metrics["state_mean"] / max(step_count, 1),
                "state_max": epoch_metrics["state_max"] / max(step_count, 1),
                "state_var": epoch_metrics["state_var"] / max(step_count, 1),
                "raw_total_backlog": epoch_metrics["raw_total_backlog"] / max(step_count, 1),
                "normalized_total_backlog": epoch_metrics["normalized_total_backlog"] / max(step_count, 1),
                "total_backlog": epoch_metrics["total_backlog"] / max(step_count, 1),
                "total_delay": epoch_metrics["total_delay"] / max(step_count, 1),
                "total_energy": epoch_metrics["total_energy"] / max(step_count, 1),
                "queue_penalty": epoch_metrics["queue_penalty"] / max(step_count, 1),
                "reward_delay_term": epoch_metrics["reward_delay_term"] / max(step_count, 1),
                "reward_energy_term": epoch_metrics["reward_energy_term"] / max(step_count, 1),
                "reward_backlog_term": epoch_metrics["reward_backlog_term"] / max(step_count, 1),
                "avg_theta": epoch_metrics["avg_theta"] / max(step_count, 1),
                "avg_reward": epoch_metrics["reward"] / max(step_count, 1),
                "critic_loss": losses["critic_loss"],
                "actor_loss": losses["actor_loss"],
                "entropy": losses["entropy"],
            }
        )

    dim_mean = state_sum / max(observed_state_count, 1)
    dim_var = state_sq_sum / max(observed_state_count, 1) - np.square(dim_mean)
    dim_rows = [
        {
            "dim_index": index,
            "mean": float(dim_mean[index]),
            "max": float(state_max[index]),
            "var": float(dim_var[index]),
        }
        for index in range(config.state_dim)
    ]

    write_csv(topology_dir / "step_diagnostics.csv", step_rows)
    write_csv(topology_dir / "epoch_diagnostics.csv", epoch_rows)
    write_csv(topology_dir / "state_dimension_summary.csv", dim_rows)

    step_df = pd.DataFrame(step_rows)
    epoch_df = pd.DataFrame(epoch_rows)
    summary = {
        "topology": topology_mode,
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
        "mean_state_mean": float(step_df["state_mean"].mean()),
        "mean_state_max": float(step_df["state_max"].mean()),
        "mean_state_var": float(step_df["state_var"].mean()),
        "mean_raw_total_backlog": float(step_df["raw_total_backlog"].mean()),
        "mean_normalized_total_backlog": float(step_df["normalized_total_backlog"].mean()),
        "mean_total_backlog": float(step_df["total_backlog"].mean()),
        "mean_total_delay": float(step_df["total_delay"].mean()),
        "mean_total_energy": float(step_df["total_energy"].mean()),
        "mean_queue_penalty": float(step_df["queue_penalty"].mean()),
        "mean_reward_delay_term": float(step_df["reward_delay_term"].mean()),
        "mean_reward_energy_term": float(step_df["reward_energy_term"].mean()),
        "mean_reward_backlog_term": float(step_df["reward_backlog_term"].mean()),
        "mean_avg_theta": float(step_df["avg_theta"].mean()),
        "mean_reward": float(step_df["reward"].mean()),
        "final_epoch_critic_loss": float(epoch_df["critic_loss"].iloc[-1]),
        "max_epoch_critic_loss": float(epoch_df["critic_loss"].max()),
    }
    (topology_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary, epoch_df


def main() -> None:
    args = parse_args()
    root_dir = Path(args.output_root) / datetime.now().strftime("default_dense_diagnostics_%Y%m%d_%H%M%S")
    root_dir.mkdir(parents=True, exist_ok=True)

    set_global_seeds(args.seed)
    summaries: list[dict[str, float | int | str]] = []
    epoch_frames: list[pd.DataFrame] = []
    for topology in TOPOLOGIES:
        summary, epoch_df = run_topology_diagnostic(
            topology_mode=topology,
            seed=args.seed,
            diagnostic_epochs=args.diagnostic_epochs,
            log_steps_per_epoch=args.log_steps_per_epoch,
            root_dir=root_dir,
        )
        summaries.append(summary)
        epoch_frames.append(epoch_df)

    summary_df = pd.DataFrame(summaries).sort_values("topology").reset_index(drop=True)
    summary_df.to_csv(root_dir / "topology_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(epoch_frames, ignore_index=True).to_csv(
        root_dir / "combined_epoch_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if {"default", "dense"} == set(summary_df["topology"].tolist()):
        default_row = summary_df[summary_df["topology"] == "default"].iloc[0]
        dense_row = summary_df[summary_df["topology"] == "dense"].iloc[0]
        comparison = {
            "state_dim_default": int(default_row["state_dim"]),
            "state_dim_dense": int(dense_row["state_dim"]),
            "mean_state_mean_ratio_dense_vs_default": float(
                dense_row["mean_state_mean"] / max(default_row["mean_state_mean"], 1e-9)
            ),
            "mean_state_max_ratio_dense_vs_default": float(
                dense_row["mean_state_max"] / max(default_row["mean_state_max"], 1e-9)
            ),
            "mean_state_var_ratio_dense_vs_default": float(
                dense_row["mean_state_var"] / max(default_row["mean_state_var"], 1e-9)
            ),
            "mean_raw_backlog_ratio_dense_vs_default": float(
                dense_row["mean_raw_total_backlog"] / max(default_row["mean_raw_total_backlog"], 1e-9)
            ),
            "mean_delay_ratio_dense_vs_default": float(
                dense_row["mean_total_delay"] / max(default_row["mean_total_delay"], 1e-9)
            ),
            "mean_energy_ratio_dense_vs_default": float(
                dense_row["mean_total_energy"] / max(default_row["mean_total_energy"], 1e-9)
            ),
            "mean_reward_delay_term_ratio_dense_vs_default": float(
                dense_row["mean_reward_delay_term"] / max(default_row["mean_reward_delay_term"], 1e-9)
            ),
            "mean_reward_energy_term_ratio_dense_vs_default": float(
                dense_row["mean_reward_energy_term"] / max(default_row["mean_reward_energy_term"], 1e-9)
            ),
            "mean_reward_backlog_term_ratio_dense_vs_default": float(
                dense_row["mean_reward_backlog_term"] / max(default_row["mean_reward_backlog_term"], 1e-9)
            ),
            "final_epoch_critic_loss_ratio_dense_vs_default": float(
                dense_row["final_epoch_critic_loss"] / max(default_row["final_epoch_critic_loss"], 1e-9)
            ),
        }
    else:
        comparison = {}

    (root_dir / "comparison_summary.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Default-vs-dense diagnostics completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
