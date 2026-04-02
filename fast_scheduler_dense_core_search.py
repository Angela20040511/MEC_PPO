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
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator

FIXED_SEED = 2025
FIXED_TOPOLOGY = "dense"

FIXED_ACTOR_LEARNING_RATE = 5e-4
FIXED_ENTROPY_COEFF = 1e-3
FIXED_UPDATE_EPOCHS = 10
FIXED_START_CHECK_EPOCH = 3
FIXED_MAX_EPOCHS = 12
FIXED_PATIENCE = 3
FIXED_MIN_DELTA = 1.0
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

FIXED_ALPHA_P = 0.4

DEFAULT_NOMA_QUANTILE_GRID = (0.80, 0.85, 0.90, 0.95)
DEFAULT_MAX_CLUSTER_SIZE_GRID = (2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense-topology fast scheduler search for q and Umax."
    )
    parser.add_argument(
        "--noma-quantile-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_NOMA_QUANTILE_GRID),
        help="Candidate OMA quantile thresholds q.",
    )
    parser.add_argument(
        "--max-cluster-size-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_MAX_CLUSTER_SIZE_GRID),
        help="Candidate NOMA max cluster sizes Umax.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FIXED_SEED,
        help="Fixed seed for this coarse search.",
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


def build_dense_scheduler_config(noma_quantile: float, max_cluster_size: int, seed: int) -> MECConfig:
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
        entropy_coeff=FIXED_ENTROPY_COEFF,
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
        alpha_p=FIXED_ALPHA_P,
        noma_quantile=noma_quantile,
        max_cluster_size=max_cluster_size,
    )
    return replace(config, training=training, ppo=ppo, dt=dt, system=system)


def save_logs(run_dir: Path, logs: list[dict[str, float | int | str]]) -> None:
    with (run_dir / "train_logs.json").open("w", encoding="utf-8") as handle:
        json.dump(logs, handle, ensure_ascii=False, indent=2)

    if not logs:
        return

    fieldnames = list(logs[0].keys())
    with (run_dir / "train_logs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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

    episode_backlog = 0.0
    episode_avg_theta = 0.0

    episode_oma_cluster_count = 0.0
    episode_noma_cluster_count = 0.0
    episode_multi_user_cluster_count = 0.0
    episode_total_cluster_count = 0.0
    episode_users_in_noma_clusters = 0.0
    episode_avg_cluster_size = 0.0
    multi_user_steps = 0
    max_cluster_size_observed = 0
    cluster_histogram: dict[str, int] = {}

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

        episode_backlog += float(info["total_backlog"])
        episode_avg_theta += float(info["avg_theta"])

        episode_oma_cluster_count += float(info["oma_cluster_count"])
        episode_noma_cluster_count += float(info["noma_cluster_count"])
        episode_multi_user_cluster_count += float(info["multi_user_cluster_count"])
        episode_total_cluster_count += float(info["total_cluster_count"])
        episode_users_in_noma_clusters += float(info["users_in_noma_clusters"])
        episode_avg_cluster_size += float(info["avg_cluster_size"])
        max_cluster_size_observed = max(max_cluster_size_observed, int(info["max_cluster_size_observed"]))
        if int(info["multi_user_cluster_count"]) > 0:
            multi_user_steps += 1
        for size, count in dict(info["cluster_size_histogram"]).items():
            cluster_histogram[size] = cluster_histogram.get(size, 0) + int(count)

        step_count += 1
        if done:
            break

    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)
    losses = agent.train()

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
        "avg_total_backlog": episode_backlog / max(step_count, 1),
        "avg_theta": episode_avg_theta / max(step_count, 1),
        "num_steps": float(step_count),
        "actor_loss": losses["actor_loss"],
        "critic_loss": losses["critic_loss"],
        "entropy": losses["entropy"],
        "avg_oma_cluster_count": episode_oma_cluster_count / max(step_count, 1),
        "avg_noma_cluster_count": episode_noma_cluster_count / max(step_count, 1),
        "avg_multi_user_cluster_count": episode_multi_user_cluster_count / max(step_count, 1),
        "avg_total_cluster_count": episode_total_cluster_count / max(step_count, 1),
        "avg_users_in_noma_clusters": episode_users_in_noma_clusters / max(step_count, 1),
        "avg_cluster_size": episode_avg_cluster_size / max(step_count, 1),
        "multi_user_step_fraction": multi_user_steps / max(step_count, 1),
        "max_cluster_size_observed": float(max_cluster_size_observed),
        "cluster_size_histogram": json.dumps(cluster_histogram, ensure_ascii=False, sort_keys=True),
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


def extract_run_metrics(run_dir: Path, noma_quantile: float, max_cluster_size: int) -> dict[str, float | int | str]:
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    metrics: dict[str, float | int | str] = {
        "noma_quantile": noma_quantile,
        "max_cluster_size": max_cluster_size,
        "run_dir": str(run_dir),
        "epochs_completed": int(summary["epochs_completed"]),
        "best_reward": float(summary["best_reward"]),
        "final_reward": float(summary["final_reward"]),
        "best_epoch": int(best_model_info["best_epoch"]),
        "reward_gap": float(summary["final_reward"] - summary["best_reward"]),
        "reward_gap_abs": abs(float(summary["final_reward"] - summary["best_reward"])),
        "final_critic_loss": float(logs["critic_loss"].iloc[-1]),
        "avg_theta_last10": float(logs["avg_theta"].tail(min(10, len(logs))).mean()),
        "mean_oma_cluster_count": float(logs["avg_oma_cluster_count"].mean()),
        "mean_noma_cluster_count": float(logs["avg_noma_cluster_count"].mean()),
        "mean_multi_user_cluster_count": float(logs["avg_multi_user_cluster_count"].mean()),
        "mean_users_in_noma_clusters": float(logs["avg_users_in_noma_clusters"].mean()),
        "mean_cluster_size": float(logs["avg_cluster_size"].mean()),
        "mean_multi_user_step_fraction": float(logs["multi_user_step_fraction"].mean()),
        "max_cluster_size_seen": int(logs["max_cluster_size_observed"].max()),
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
            "final_reward",
            "reward_gap_abs",
            "best_reward",
            "final_critic_loss",
        ],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def run_search(
    noma_quantile_grid: list[float],
    max_cluster_size_grid: list[int],
    seed: int,
    output_root: Path,
) -> Path:
    search_tag = datetime.now().strftime("fast_scheduler_dense_core_search_%Y%m%d_%H%M%S")
    root_dir = output_root / search_tag
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
            "actor_learning_rate": FIXED_ACTOR_LEARNING_RATE,
            "entropy_coeff": FIXED_ENTROPY_COEFF,
            "update_epochs": FIXED_UPDATE_EPOCHS,
            "start_check_epoch": FIXED_START_CHECK_EPOCH,
            "max_epochs": FIXED_MAX_EPOCHS,
            "patience": FIXED_PATIENCE,
            "min_delta": FIXED_MIN_DELTA,
            "time_steps": FIXED_TIME_STEPS,
            "alpha_p": FIXED_ALPHA_P,
            "dt_history_window": FIXED_DT_HISTORY_WINDOW,
            "dt_prediction_horizon": FIXED_DT_PREDICTION_HORIZON,
            "dt_hidden_size": FIXED_DT_HIDDEN_SIZE,
            "dt_retrain_interval": FIXED_DT_RETRAIN_INTERVAL,
            "dt_train_epochs": FIXED_DT_TRAIN_EPOCHS,
            "dt_num_layers": FIXED_DT_NUM_LAYERS,
            "dt_learning_rate": FIXED_DT_LEARNING_RATE,
            "dt_batch_size": FIXED_DT_BATCH_SIZE,
            "dt_min_history_to_train": FIXED_DT_MIN_HISTORY_TO_TRAIN,
        },
        "grid": {
            "noma_quantile": noma_quantile_grid,
            "max_cluster_size": max_cluster_size_grid,
        },
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []

    for noma_quantile in noma_quantile_grid:
        for max_cluster_size in max_cluster_size_grid:
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_dir = root_dir / (
                f"q_{str(f'{noma_quantile:.2f}').replace('.', 'p')}_"
                f"umax_{max_cluster_size:02d}_{run_tag}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            config = build_dense_scheduler_config(
                noma_quantile=noma_quantile,
                max_cluster_size=max_cluster_size,
                seed=seed,
            )
            (run_dir / "experiment_config.json").write_text(
                json.dumps(
                    {
                        "topology": FIXED_TOPOLOGY,
                        "seed": seed,
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

            print(
                "[train] "
                f"topology={FIXED_TOPOLOGY}, q={noma_quantile:.2f}, "
                f"Umax={max_cluster_size}, seed={seed} -> {run_dir}"
            )
            set_global_seeds(seed)
            train_agent_with_early_stopping(
                config=config,
                checkpoint_dir=str(run_dir),
                early_stopping=early_stopping,
            )
            metrics = extract_run_metrics(
                run_dir=run_dir,
                noma_quantile=noma_quantile,
                max_cluster_size=max_cluster_size,
            )
            all_metrics.append(metrics)
            manifest["runs"].append(
                {
                    "noma_quantile": noma_quantile,
                    "max_cluster_size": max_cluster_size,
                    "run_dir": str(run_dir),
                    "analysis_summary": str(run_dir / "analysis_summary.json"),
                }
            )

    summary_df = pd.DataFrame(all_metrics)
    ranked_df = rank_runs(summary_df)

    summary_csv = root_dir / "fast_scheduler_dense_summary.csv"
    summary_json = root_dir / "fast_scheduler_dense_summary.json"
    summary_md = root_dir / "fast_scheduler_dense_summary.md"
    ranked_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        ranked_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(dataframe_to_markdown(ranked_df), encoding="utf-8")

    manifest["summary"] = {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }
    (root_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root_dir


def main() -> None:
    args = parse_args()
    root_dir = run_search(
        noma_quantile_grid=args.noma_quantile_grid,
        max_cluster_size_grid=args.max_cluster_size_grid,
        seed=args.seed,
        output_root=Path(args.output_root),
    )
    print(f"Dense fast scheduler search completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
