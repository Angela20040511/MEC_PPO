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

FIXED_START_CHECK_EPOCH = 3
FIXED_MAX_EPOCHS = 12
FIXED_PATIENCE = 3
FIXED_MIN_DELTA = 1.0
FIXED_ACTOR_LEARNING_RATE = 5e-4
FIXED_ENTROPY_COEFF = 1e-3
FIXED_UPDATE_EPOCHS = 10
FIXED_TIME_STEPS = 100

FIXED_NUM_LAYERS = 1
FIXED_DT_LEARNING_RATE = 1e-3
FIXED_BATCH_SIZE = 32
FIXED_MIN_HISTORY_TO_TRAIN = 20

DEFAULT_SEEDS = (2025, 2026, 2027, 2028, 2029)

CANDIDATE_STRUCTURES = {
    "A": {
        "history_window": 4,
        "prediction_horizon": 6,
        "hidden_size": 32,
        "retrain_interval": 5,
        "train_epochs": 5,
    },
    "B": {
        "history_window": 8,
        "prediction_horizon": 2,
        "hidden_size": 128,
        "retrain_interval": 5,
        "train_epochs": 5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed robustness validation for the final DT candidate structures."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Seeds for robustness validation.",
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


def build_candidate_config(candidate_name: str, seed: int):
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
        train_epochs=structure["train_epochs"],
        min_history_to_train=FIXED_MIN_HISTORY_TO_TRAIN,
        retrain_interval=structure["retrain_interval"],
    )
    return replace(config, training=training, ppo=ppo, dt=dt)


def extract_seed_metrics(run_dir: Path, candidate_name: str, seed: int) -> dict[str, float | int | str]:
    structure = CANDIDATE_STRUCTURES[candidate_name]
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    return {
        "candidate": candidate_name,
        "history_window": structure["history_window"],
        "prediction_horizon": structure["prediction_horizon"],
        "hidden_size": structure["hidden_size"],
        "retrain_interval": structure["retrain_interval"],
        "train_epochs": structure["train_epochs"],
        "seed": seed,
        "epochs_completed": int(summary["epochs_completed"]),
        "best_reward": float(summary["best_reward"]),
        "final_reward": float(summary["final_reward"]),
        "best_epoch": int(best_model_info["best_epoch"]),
        "reward_gap": float(summary["final_reward"] - summary["best_reward"]),
        "final_critic_loss": float(logs["critic_loss"].iloc[-1]),
        "avg_theta_last10": float(logs["avg_theta"].tail(min(10, len(logs))).mean()),
    }


def aggregate_candidate_metrics(candidate_df: pd.DataFrame, candidate_name: str, candidate_dir: Path) -> dict[str, float | str]:
    structure = CANDIDATE_STRUCTURES[candidate_name]
    metrics = {
        "candidate": candidate_name,
        "candidate_dir": str(candidate_dir),
        "history_window": structure["history_window"],
        "prediction_horizon": structure["prediction_horizon"],
        "hidden_size": structure["hidden_size"],
        "retrain_interval": structure["retrain_interval"],
        "train_epochs": structure["train_epochs"],
        "mean_best_reward": float(candidate_df["best_reward"].mean()),
        "std_best_reward": float(candidate_df["best_reward"].std(ddof=0)),
        "mean_final_reward": float(candidate_df["final_reward"].mean()),
        "std_final_reward": float(candidate_df["final_reward"].std(ddof=0)),
        "mean_reward_gap": float(candidate_df["reward_gap"].mean()),
        "std_reward_gap": float(candidate_df["reward_gap"].std(ddof=0)),
        "mean_best_epoch": float(candidate_df["best_epoch"].mean()),
        "mean_final_critic_loss": float(candidate_df["final_critic_loss"].mean()),
        "mean_avg_theta_last10": float(candidate_df["avg_theta_last10"].mean()),
    }
    candidate_df.to_csv(candidate_dir / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    (candidate_dir / "seed_metrics.json").write_text(
        candidate_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    (candidate_dir / "candidate_statistics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def rank_candidates(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.copy()
    ranked["abs_mean_reward_gap"] = ranked["mean_reward_gap"].abs()
    ranked = ranked.sort_values(
        by=[
            "mean_best_reward",
            "abs_mean_reward_gap",
            "mean_final_reward",
            "std_best_reward",
            "mean_final_critic_loss",
        ],
        ascending=[False, True, False, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def run_validation(seeds: list[int], output_root: Path) -> Path:
    search_tag = datetime.now().strftime("dt_candidate_seed_validation_%Y%m%d_%H%M%S")
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
        "seeds": seeds,
        "groups": [],
    }

    all_candidate_metrics: list[dict[str, float | str]] = []

    for candidate_name in ("A", "B"):
        candidate_dir = root_dir / f"candidate_{candidate_name}"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        group_manifest: dict[str, object] = {
            "candidate": candidate_name,
            "candidate_dir": str(candidate_dir),
            "seed_runs": [],
        }
        seed_metrics: list[dict[str, float | int | str]] = []

        for seed in seeds:
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_dir = candidate_dir / f"seed_{seed}_{run_tag}"
            run_dir.mkdir(parents=True, exist_ok=True)

            config = build_candidate_config(candidate_name, seed)
            (run_dir / "experiment_config.json").write_text(
                json.dumps(
                    {
                        "candidate": candidate_name,
                        "seed": seed,
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

            print(f"[train] candidate={candidate_name}, seed={seed} -> {run_dir}")
            set_global_seeds(seed)
            train_agent_with_early_stopping(
                config=config,
                checkpoint_dir=str(run_dir),
                early_stopping=early_stopping,
            )
            metrics = extract_seed_metrics(run_dir, candidate_name, seed)
            seed_metrics.append(metrics)
            group_manifest["seed_runs"].append(
                {
                    "seed": seed,
                    "run_dir": str(run_dir),
                }
            )

        candidate_df = pd.DataFrame(seed_metrics).sort_values("seed").reset_index(drop=True)
        candidate_metrics = aggregate_candidate_metrics(candidate_df, candidate_name, candidate_dir)
        all_candidate_metrics.append(candidate_metrics)
        group_manifest["candidate_statistics"] = candidate_metrics
        manifest["groups"].append(group_manifest)

    summary_df = pd.DataFrame(all_candidate_metrics)
    ranked_df = rank_candidates(summary_df)

    summary_csv = root_dir / "dt_candidate_seed_summary.csv"
    summary_json = root_dir / "dt_candidate_seed_summary.json"
    summary_md = root_dir / "dt_candidate_seed_summary.md"
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
    root_dir = run_validation(seeds=args.seeds, output_root=Path(args.output_root))
    print(f"DT candidate seed validation completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
