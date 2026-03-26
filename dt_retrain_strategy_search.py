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

FIXED_NUM_LAYERS = 1
FIXED_DT_LEARNING_RATE = 1e-3
FIXED_BATCH_SIZE = 32
FIXED_MIN_HISTORY_TO_TRAIN = 20

DEFAULT_RETRAIN_INTERVAL_GRID = (3, 5, 8)
DEFAULT_TRAIN_EPOCHS_GRID = (3, 5, 8)

CANDIDATE_STRUCTURES = {
    "A": {
        "history_window": 4,
        "prediction_horizon": 6,
        "hidden_size": 32,
    },
    "B": {
        "history_window": 8,
        "prediction_horizon": 2,
        "hidden_size": 128,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Second-round DT search for retraining interval and train epochs."
    )
    parser.add_argument(
        "--retrain-interval-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_RETRAIN_INTERVAL_GRID),
        help="Candidate DT retrain_interval values.",
    )
    parser.add_argument(
        "--train-epochs-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAIN_EPOCHS_GRID),
        help="Candidate DT train_epochs values.",
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


def build_dt_config(
    candidate_name: str,
    retrain_interval: int,
    train_epochs: int,
    seed: int,
):
    structure = CANDIDATE_STRUCTURES[candidate_name]
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
        history_window=structure["history_window"],
        prediction_horizon=structure["prediction_horizon"],
        hidden_size=structure["hidden_size"],
        num_layers=FIXED_NUM_LAYERS,
        learning_rate=FIXED_DT_LEARNING_RATE,
        batch_size=FIXED_BATCH_SIZE,
        train_epochs=train_epochs,
        min_history_to_train=FIXED_MIN_HISTORY_TO_TRAIN,
        retrain_interval=retrain_interval,
    )
    return replace(config, training=training, ppo=ppo, dt=dt)


def extract_metrics(
    run_dir: Path,
    candidate_name: str,
    retrain_interval: int,
    train_epochs: int,
    seed: int,
) -> dict[str, float | int | str]:
    structure = CANDIDATE_STRUCTURES[candidate_name]
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    best_reward = float(summary["best_reward"])
    final_reward = float(summary["final_reward"])
    reward_gap = float(final_reward - best_reward)
    metrics: dict[str, float | int | str] = {
        "candidate": candidate_name,
        "history_window": structure["history_window"],
        "prediction_horizon": structure["prediction_horizon"],
        "hidden_size": structure["hidden_size"],
        "retrain_interval": retrain_interval,
        "train_epochs": train_epochs,
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
            "best_reward",
            "reward_gap_abs",
            "final_reward",
            "final_critic_loss",
        ],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def build_candidate_best_summary(ranked_df: pd.DataFrame) -> pd.DataFrame:
    best_rows = (
        ranked_df.sort_values("rank")
        .groupby("candidate", as_index=False)
        .first()
        .sort_values("candidate")
        .reset_index(drop=True)
    )
    return best_rows


def run_search(
    retrain_interval_grid: list[int],
    train_epochs_grid: list[int],
    seed: int,
    output_root: Path,
) -> Path:
    search_tag = datetime.now().strftime("dt_retrain_strategy_search_%Y%m%d_%H%M%S")
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
            "num_layers": FIXED_NUM_LAYERS,
            "learning_rate": FIXED_DT_LEARNING_RATE,
            "batch_size": FIXED_BATCH_SIZE,
            "min_history_to_train": FIXED_MIN_HISTORY_TO_TRAIN,
        },
        "candidates": CANDIDATE_STRUCTURES,
        "grid": {
            "retrain_interval": retrain_interval_grid,
            "train_epochs": train_epochs_grid,
        },
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []

    for candidate_name in ("A", "B"):
        for retrain_interval in retrain_interval_grid:
            for train_epochs in train_epochs_grid:
                run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                run_dir = root_dir / (
                    f"candidate_{candidate_name}_ri_{retrain_interval:02d}_"
                    f"te_{train_epochs:02d}_{run_tag}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)

                config = build_dt_config(
                    candidate_name=candidate_name,
                    retrain_interval=retrain_interval,
                    train_epochs=train_epochs,
                    seed=seed,
                )
                (run_dir / "experiment_config.json").write_text(
                    json.dumps(
                        {
                            "candidate": candidate_name,
                            "training": asdict(config.training),
                            "ppo": asdict(config.ppo),
                            "dt": asdict(config.dt),
                            "early_stopping": asdict(early_stopping),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                print(
                    "[train] "
                    f"candidate={candidate_name}, retrain_interval={retrain_interval}, "
                    f"train_epochs={train_epochs}, seed={seed} -> {run_dir}"
                )
                set_global_seeds(seed)
                train_agent_with_early_stopping(
                    config=config,
                    checkpoint_dir=str(run_dir),
                    early_stopping=early_stopping,
                )
                metrics = extract_metrics(
                    run_dir=run_dir,
                    candidate_name=candidate_name,
                    retrain_interval=retrain_interval,
                    train_epochs=train_epochs,
                    seed=seed,
                )
                all_metrics.append(metrics)
                manifest["runs"].append(
                    {
                        "candidate": candidate_name,
                        "retrain_interval": retrain_interval,
                        "train_epochs": train_epochs,
                        "run_dir": str(run_dir),
                        "analysis_summary": str(run_dir / "analysis_summary.json"),
                    }
                )

    summary_df = pd.DataFrame(all_metrics)
    ranked_df = rank_runs(summary_df)
    candidate_best_df = build_candidate_best_summary(ranked_df)

    summary_csv = root_dir / "dt_retrain_strategy_summary.csv"
    summary_json = root_dir / "dt_retrain_strategy_summary.json"
    summary_md = root_dir / "dt_retrain_strategy_summary.md"
    ranked_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        ranked_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(dataframe_to_markdown(ranked_df), encoding="utf-8")

    candidate_best_csv = root_dir / "candidate_best_summary.csv"
    candidate_best_json = root_dir / "candidate_best_summary.json"
    candidate_best_df.to_csv(candidate_best_csv, index=False, encoding="utf-8-sig")
    candidate_best_json.write_text(
        candidate_best_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest["summary"] = {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "candidate_best_csv": str(candidate_best_csv),
        "candidate_best_json": str(candidate_best_json),
    }
    (root_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root_dir


def main() -> None:
    args = parse_args()
    root_dir = run_search(
        retrain_interval_grid=args.retrain_interval_grid,
        train_epochs_grid=args.train_epochs_grid,
        seed=args.seed,
        output_root=Path(args.output_root),
    )
    print(f"DT retrain strategy search completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
