from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import MECConfig, build_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from rl.ppo_agent import PPOAgent
from scheduler.fast_scheduler import FastScheduler
from simulator.simulator import Simulator

FIXED_SEED = 2025
FIXED_TOPOLOGY = "dense"
FIXED_ALPHA_P = 0.4
FIXED_MAX_CLUSTER_SIZE = 2
FIXED_BASE_ROLLOUT_Q = 0.85

FIXED_ACTOR_LEARNING_RATE = 5e-4
FIXED_ENTROPY_COEFF = 1e-3
FIXED_UPDATE_EPOCHS = 10
FIXED_DEBUG_STEPS = 20

FIXED_DT_HISTORY_WINDOW = 8
FIXED_DT_PREDICTION_HORIZON = 2
FIXED_DT_HIDDEN_SIZE = 128
FIXED_DT_RETRAIN_INTERVAL = 5
FIXED_DT_TRAIN_EPOCHS = 5
FIXED_DT_NUM_LAYERS = 1
FIXED_DT_LEARNING_RATE = 1e-3
FIXED_DT_BATCH_SIZE = 32
FIXED_DT_MIN_HISTORY_TO_TRAIN = 20

DEFAULT_Q_GRID = (0.80, 0.85, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug q sensitivity of cluster_users() on dense topology."
    )
    parser.add_argument(
        "--q-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_Q_GRID),
        help="Candidate q values to compare on the same state snapshots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FIXED_SEED,
        help="Random seed for the debug rollout.",
    )
    parser.add_argument(
        "--debug-steps",
        type=int,
        default=FIXED_DEBUG_STEPS,
        help="Number of rollout steps to inspect.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="checkpoints",
        help="Root directory for debug outputs.",
    )
    return parser.parse_args()


def build_dense_debug_config(seed: int, debug_steps: int) -> MECConfig:
    config = build_config(topology_mode=FIXED_TOPOLOGY)
    training = replace(
        config.training,
        num_epochs=1,
        time_steps=debug_steps,
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
        noma_quantile=FIXED_BASE_ROLLOUT_Q,
        max_cluster_size=FIXED_MAX_CLUSTER_SIZE,
    )
    return replace(config, training=training, ppo=ppo, dt=dt, system=system)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_q_schedulers(base_config: MECConfig, channel_model: Any, q_grid: list[float]) -> dict[float, FastScheduler]:
    schedulers: dict[float, FastScheduler] = {}
    for q in q_grid:
        q_config = replace(
            base_config,
            system=replace(
                base_config.system,
                noma_quantile=q,
                max_cluster_size=FIXED_MAX_CLUSTER_SIZE,
            ),
        )
        schedulers[q] = FastScheduler(config=q_config, channel_model=channel_model)
    return schedulers


def extract_sorted_pressures(sorted_candidates: list[dict[str, float]], limit: int = 4) -> list[float]:
    pressures = [float(candidate["pressure"]) for candidate in sorted_candidates[:limit]]
    while len(pressures) < limit:
        pressures.append(np.nan)
    return pressures


def collect_snapshot_rows(
    step: int,
    q: float,
    cluster_debug: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bs_id, bs_debug in cluster_debug.items():
        sorted_candidates = list(bs_debug["sorted_candidates"])
        top_pressures = extract_sorted_pressures(sorted_candidates)
        rows.append(
            {
                "step": step,
                "bs_id": bs_id,
                "q": q,
                "candidate_user_count": int(bs_debug["candidate_user_count"]),
                "candidate_users": stable_json(bs_debug["candidate_users"]),
                "user_pressures": stable_json(bs_debug["user_pressures"]),
                "sorted_candidates": stable_json(sorted_candidates),
                "quantile_threshold": float(bs_debug["quantile_threshold"]),
                "oma_users": stable_json(bs_debug["oma_users"]),
                "noma_users": stable_json(bs_debug["noma_users"]),
                "noma_clusters": stable_json(bs_debug["noma_clusters"]),
                "cluster_sizes": stable_json(bs_debug["cluster_sizes"]),
                "oma_cluster_count": int(bs_debug["oma_cluster_count"]),
                "noma_cluster_count": int(bs_debug["noma_cluster_count"]),
                "users_in_noma_clusters": int(bs_debug["users_in_noma_clusters"]),
                "max_cluster_size_seen": int(bs_debug["max_cluster_size_seen"]),
                "top_pressure_1": top_pressures[0],
                "top_pressure_2": top_pressures[1],
                "top_pressure_3": top_pressures[2],
                "top_pressure_4": top_pressures[3],
            }
        )
    return rows


def summarize_by_q(records: pd.DataFrame) -> pd.DataFrame:
    per_step = (
        records.groupby(["q", "step"], as_index=False)
        .agg(
            total_oma_clusters=("oma_cluster_count", "sum"),
            total_noma_clusters=("noma_cluster_count", "sum"),
            total_users_in_noma_clusters=("users_in_noma_clusters", "sum"),
            max_cluster_size_seen=("max_cluster_size_seen", "max"),
        )
    )
    summary = (
        per_step.groupby("q", as_index=False)
        .agg(
            mean_oma_clusters=("total_oma_clusters", "mean"),
            mean_noma_clusters=("total_noma_clusters", "mean"),
            mean_users_in_noma_clusters=("total_users_in_noma_clusters", "mean"),
            max_cluster_size_seen=("max_cluster_size_seen", "max"),
        )
    )
    return summary.sort_values("q").reset_index(drop=True)


def summarize_cross_q(records: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    comparison_rows: list[dict[str, Any]] = []
    for (step, bs_id), group in records.groupby(["step", "bs_id"], sort=True):
        threshold_values = list(group["quantile_threshold"].astype(float))
        oma_signatures = list(group["oma_users"])
        noma_signatures = list(group["noma_users"])
        cluster_signatures = list(group["noma_clusters"])
        top_1 = float(group["top_pressure_1"].iloc[0])
        top_2 = float(group["top_pressure_2"].iloc[0])
        comparison_rows.append(
            {
                "step": int(step),
                "bs_id": bs_id,
                "candidate_user_count": int(group["candidate_user_count"].iloc[0]),
                "threshold_changed": len({round(value, 10) for value in threshold_values}) > 1,
                "oma_changed": len(set(oma_signatures)) > 1,
                "noma_changed": len(set(noma_signatures)) > 1,
                "cluster_changed": len(set(cluster_signatures)) > 1,
                "top_two_tie": bool(np.isclose(top_1, top_2, rtol=1e-8, atol=1e-8)),
                "threshold_map": stable_json(
                    {
                        f"{float(row.q):.2f}": float(row.quantile_threshold)
                        for row in group.itertuples(index=False)
                    }
                ),
                "oma_map": stable_json(
                    {
                        f"{float(row.q):.2f}": json.loads(row.oma_users)
                        for row in group.itertuples(index=False)
                    }
                ),
                "noma_map": stable_json(
                    {
                        f"{float(row.q):.2f}": json.loads(row.noma_users)
                        for row in group.itertuples(index=False)
                    }
                ),
                "cluster_map": stable_json(
                    {
                        f"{float(row.q):.2f}": json.loads(row.noma_clusters)
                        for row in group.itertuples(index=False)
                    }
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(["step", "bs_id"]).reset_index(drop=True)
    totals = len(comparison)
    summary = {
        "total_bs_step_snapshots": int(totals),
        "threshold_changed_snapshots": int(comparison["threshold_changed"].sum()),
        "threshold_constant_snapshots": int((~comparison["threshold_changed"]).sum()),
        "oma_changed_snapshots": int(comparison["oma_changed"].sum()),
        "noma_changed_snapshots": int(comparison["noma_changed"].sum()),
        "cluster_changed_snapshots": int(comparison["cluster_changed"].sum()),
        "threshold_changed_but_cluster_unchanged": int(
            (comparison["threshold_changed"] & ~comparison["cluster_changed"]).sum()
        ),
        "top_two_tie_snapshots": int(comparison["top_two_tie"].sum()),
        "candidate_user_count_values": sorted(
            int(value) for value in comparison["candidate_user_count"].unique().tolist()
        ),
    }
    return comparison, summary


def build_examples(comparison: pd.DataFrame) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    threshold_changed_same_cluster = comparison[
        comparison["threshold_changed"] & ~comparison["cluster_changed"]
    ]
    if not threshold_changed_same_cluster.empty:
        examples["threshold_changed_but_cluster_same"] = json.loads(
            threshold_changed_same_cluster.iloc[0].to_json(force_ascii=False)
        )

    threshold_constant_same_cluster = comparison[
        ~comparison["threshold_changed"] & ~comparison["cluster_changed"]
    ]
    if not threshold_constant_same_cluster.empty:
        examples["threshold_constant_and_cluster_same"] = json.loads(
            threshold_constant_same_cluster.iloc[0].to_json(force_ascii=False)
        )
    return examples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    q_grid = [float(q) for q in args.q_grid]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = Path(args.output_root) / f"dense_cluster_users_debug_{timestamp}"
    root_dir.mkdir(parents=True, exist_ok=True)

    config = build_dense_debug_config(seed=args.seed, debug_steps=args.debug_steps)
    (root_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "debug_steps": args.debug_steps,
                "base_rollout_q": FIXED_BASE_ROLLOUT_Q,
                "q_grid": q_grid,
                "topology": FIXED_TOPOLOGY,
                "max_cluster_size": FIXED_MAX_CLUSTER_SIZE,
                "alpha_p": FIXED_ALPHA_P,
                "config": asdict(config),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    set_global_seeds(args.seed)
    simulator = Simulator(config)
    agent = PPOAgent(config=config.ppo, state_dim=config.state_dim, action_dim=config.action_dim)
    observation = simulator.reset(seed=args.seed)
    q_schedulers = build_q_schedulers(config, simulator.channel_model, q_grid)

    snapshot_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    for step in range(args.debug_steps):
        effective_rates = simulator.compute_model.effective_compute_rates()
        lyapunov = simulator.lyapunov_scheduler.calculate_lyapunov_weights(
            simulator.state,
            effective_rates,
        )

        for q, scheduler in q_schedulers.items():
            fast_result = scheduler.schedule_access(
                state=simulator.state,
                alpha=lyapunov.communication_shares,
            )
            snapshot_rows.extend(
                collect_snapshot_rows(
                    step=int(simulator.state.slot),
                    q=q,
                    cluster_debug=fast_result.cluster_debug,
                )
            )

        action, _, _ = agent.select_action(observation, deterministic=True)
        observation, reward, done, info = simulator.step(action)
        rollout_rows.append(
            {
                "step": step,
                "reward": float(reward),
                "avg_theta": float(info["avg_theta"]),
                "queue_penalty": float(info["queue_penalty"]),
                "oma_cluster_count": int(info["oma_cluster_count"]),
                "noma_cluster_count": int(info["noma_cluster_count"]),
                "users_in_noma_clusters": int(info["users_in_noma_clusters"]),
                "max_cluster_size_observed": int(info["max_cluster_size_observed"]),
            }
        )
        if done:
            break

    write_csv(root_dir / "cluster_debug.csv", snapshot_rows)
    write_csv(root_dir / "rollout_metrics.csv", rollout_rows)

    records = pd.DataFrame(snapshot_rows)
    q_summary = summarize_by_q(records)
    q_summary.to_csv(root_dir / "q_summary.csv", index=False)

    comparison, cross_q_summary = summarize_cross_q(records)
    comparison.to_csv(root_dir / "cross_q_comparison.csv", index=False)
    examples = build_examples(comparison)

    final_summary = {
        "q_summary": json.loads(q_summary.to_json(orient="records", force_ascii=False)),
        "cross_q_summary": cross_q_summary,
        "examples": examples,
    }
    (root_dir / "cluster_debug_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved dense cluster debug artifacts to: {root_dir}")


if __name__ == "__main__":
    main()
