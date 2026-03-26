from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fine_grained_epoch_early_stop_search import (
    EarlyStoppingConfig,
    build_search_config,
    set_global_seeds,
    train_agent_with_early_stopping,
)

FIXED_SEED = 2025

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

DEFAULT_NOMA_QUANTILE_GRID = (0.80, 0.85, 0.90, 0.95)
DEFAULT_MAX_CLUSTER_SIZE_GRID = (2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="First-round fast scheduler search under fixed PPO and DT baselines."
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


def build_scheduler_search_config(noma_quantile: float, max_cluster_size: int, seed: int):
    config = build_search_config(FIXED_MAX_EPOCHS)
    training = replace(
        config.training,
        seed=seed,
        time_steps=FIXED_TIME_STEPS,
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
        noma_quantile=noma_quantile,
        max_cluster_size=max_cluster_size,
    )
    return replace(config, training=training, ppo=ppo, dt=dt, system=system)


def extract_metrics(run_dir: Path, noma_quantile: float, max_cluster_size: int, seed: int) -> dict[str, float | int | str]:
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    best_reward = float(summary["best_reward"])
    final_reward = float(summary["final_reward"])
    reward_gap = float(final_reward - best_reward)
    metrics: dict[str, float | int | str] = {
        "noma_quantile": noma_quantile,
        "max_cluster_size": max_cluster_size,
        "seed": seed,
        "run_dir": str(run_dir),
        "epochs_completed": int(summary["epochs_completed"]),
        "best_reward": best_reward,
        "final_reward": final_reward,
        "best_epoch": int(best_model_info["best_epoch"]),
        "reward_gap": reward_gap,
        "reward_gap_abs": abs(reward_gap),
        "final_critic_loss": float(logs["critic_loss"].iloc[-1]),
        "avg_theta_last10": float(logs["avg_theta"].tail(min(10, len(logs))).mean()),
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
    search_tag = datetime.now().strftime("fast_scheduler_core_search_%Y%m%d_%H%M%S")
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
            "seed": seed,
            "actor_learning_rate": FIXED_ACTOR_LEARNING_RATE,
            "entropy_coeff": FIXED_ENTROPY_COEFF,
            "update_epochs": FIXED_UPDATE_EPOCHS,
            "start_check_epoch": FIXED_START_CHECK_EPOCH,
            "max_epochs": FIXED_MAX_EPOCHS,
            "patience": FIXED_PATIENCE,
            "min_delta": FIXED_MIN_DELTA,
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

            config = build_scheduler_search_config(
                noma_quantile=noma_quantile,
                max_cluster_size=max_cluster_size,
                seed=seed,
            )
            (run_dir / "experiment_config.json").write_text(
                json.dumps(
                    {
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
                f"q={noma_quantile:.2f}, Umax={max_cluster_size}, seed={seed} -> {run_dir}"
            )
            set_global_seeds(seed)
            train_agent_with_early_stopping(
                config=config,
                checkpoint_dir=str(run_dir),
                early_stopping=early_stopping,
            )
            metrics = extract_metrics(
                run_dir=run_dir,
                noma_quantile=noma_quantile,
                max_cluster_size=max_cluster_size,
                seed=seed,
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

    summary_csv = root_dir / "fast_scheduler_summary.csv"
    summary_json = root_dir / "fast_scheduler_summary.json"
    summary_md = root_dir / "fast_scheduler_summary.md"
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
    print(f"Fast scheduler search completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
