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
FIXED_ENTROPY_COEFF = 1e-3
FIXED_UPDATE_EPOCHS = 10
FIXED_TIME_STEPS = 100

DEFAULT_ACTOR_LR_GRID = (3e-4, 4e-4, 5e-4, 6e-4)
DEFAULT_SEEDS = (2025, 2026, 2027, 2028, 2029)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed actor learning rate search under a fixed early stopping policy."
    )
    parser.add_argument(
        "--actor-lr-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_ACTOR_LR_GRID),
        help="Candidate actor learning rates.",
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


def build_actor_lr_config(actor_learning_rate: float, seed: int):
    config = build_search_config(FIXED_MAX_EPOCHS)
    training = replace(
        config.training,
        seed=seed,
        time_steps=FIXED_TIME_STEPS,
    )
    ppo = replace(
        config.ppo,
        actor_learning_rate=actor_learning_rate,
        entropy_coeff=FIXED_ENTROPY_COEFF,
        update_epochs=FIXED_UPDATE_EPOCHS,
    )
    return replace(config, training=training, ppo=ppo)


def extract_seed_metrics(run_dir: Path, seed: int) -> dict[str, float | int]:
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")
    return {
        "seed": seed,
        "epochs_completed": int(summary["epochs_completed"]),
        "best_reward": float(summary["best_reward"]),
        "final_reward": float(summary["final_reward"]),
        "best_epoch": int(best_model_info["best_epoch"]),
        "reward_gap": float(summary["final_reward"] - summary["best_reward"]),
        "final_critic_loss": float(logs["critic_loss"].iloc[-1]),
        "avg_theta_last10_mean": float(logs["avg_theta"].tail(min(10, len(logs))).mean()),
    }


def aggregate_lr_metrics(
    lr_df: pd.DataFrame,
    actor_learning_rate: float,
    lr_dir: Path,
) -> dict[str, float | str]:
    metrics = {
        "actor_learning_rate": actor_learning_rate,
        "lr_dir": str(lr_dir),
        "mean_best_reward": float(lr_df["best_reward"].mean()),
        "std_best_reward": float(lr_df["best_reward"].std(ddof=0)),
        "mean_final_reward": float(lr_df["final_reward"].mean()),
        "std_final_reward": float(lr_df["final_reward"].std(ddof=0)),
        "mean_reward_gap": float(lr_df["reward_gap"].mean()),
        "std_reward_gap": float(lr_df["reward_gap"].std(ddof=0)),
        "mean_best_epoch": float(lr_df["best_epoch"].mean()),
        "mean_final_critic_loss": float(lr_df["final_critic_loss"].mean()),
        "mean_avg_theta_last10": float(lr_df["avg_theta_last10_mean"].mean()),
    }
    (lr_dir / "lr_statistics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lr_df.to_csv(lr_dir / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    (lr_dir / "seed_metrics.json").write_text(
        lr_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def rank_learning_rates(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.copy()
    ranked["abs_mean_reward_gap"] = ranked["mean_reward_gap"].abs()
    ranked = ranked.sort_values(
        by=[
            "mean_best_reward",
            "abs_mean_reward_gap",
            "std_best_reward",
            "mean_final_critic_loss",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def run_search(actor_lr_grid: list[float], seeds: list[int], output_root: Path) -> Path:
    search_tag = datetime.now().strftime("actor_lr_seed_search_%Y%m%d_%H%M%S")
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
            "start_check_epoch": FIXED_START_CHECK_EPOCH,
            "max_epochs": FIXED_MAX_EPOCHS,
            "patience": FIXED_PATIENCE,
            "min_delta": FIXED_MIN_DELTA,
            "entropy_coeff": FIXED_ENTROPY_COEFF,
            "update_epochs": FIXED_UPDATE_EPOCHS,
            "time_steps": FIXED_TIME_STEPS,
        },
        "actor_lr_grid": actor_lr_grid,
        "seeds": seeds,
        "groups": [],
    }

    all_lr_metrics: list[dict[str, float | str]] = []

    for actor_lr in actor_lr_grid:
        lr_name = f"actor_lr_{actor_lr:.0e}".replace("-", "m")
        lr_dir = root_dir / lr_name
        lr_dir.mkdir(parents=True, exist_ok=True)

        group_manifest: dict[str, object] = {
            "actor_learning_rate": actor_lr,
            "lr_dir": str(lr_dir),
            "seed_runs": [],
        }
        seed_metrics: list[dict[str, float | int]] = []

        for seed in seeds:
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_dir = lr_dir / f"seed_{seed}_{run_tag}"
            run_dir.mkdir(parents=True, exist_ok=True)

            config = build_actor_lr_config(actor_lr, seed)
            (run_dir / "experiment_config.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "training": asdict(config.training),
                        "ppo": asdict(config.ppo),
                        "early_stopping": asdict(early_stopping),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(f"[train] actor_lr={actor_lr:g}, seed={seed} -> {run_dir}")
            set_global_seeds(seed)
            train_agent_with_early_stopping(
                config=config,
                checkpoint_dir=str(run_dir),
                early_stopping=early_stopping,
            )
            metrics = extract_seed_metrics(run_dir, seed)
            seed_metrics.append(metrics)
            group_manifest["seed_runs"].append(
                {
                    "seed": seed,
                    "run_dir": str(run_dir),
                }
            )

        lr_df = pd.DataFrame(seed_metrics).sort_values("seed").reset_index(drop=True)
        lr_metrics = aggregate_lr_metrics(
            lr_df=lr_df,
            actor_learning_rate=actor_lr,
            lr_dir=lr_dir,
        )
        all_lr_metrics.append(lr_metrics)
        group_manifest["lr_statistics"] = lr_metrics
        manifest["groups"].append(group_manifest)

    summary_df = pd.DataFrame(all_lr_metrics)
    ranked_df = rank_learning_rates(summary_df)

    summary_csv = root_dir / "actor_lr_summary.csv"
    summary_json = root_dir / "actor_lr_summary.json"
    summary_md = root_dir / "actor_lr_summary.md"
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
        actor_lr_grid=args.actor_lr_grid,
        seeds=args.seeds,
        output_root=Path(args.output_root),
    )
    print(f"Actor learning rate search completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
