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

FIXED_MAX_EPOCHS = 37
FIXED_PATIENCE = 8
FIXED_MIN_DELTA = 1.0
DEFAULT_SEEDS = (2025, 2026, 2027, 2028, 2029)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate PPO robustness across multiple random seeds.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds to evaluate.",
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


def extract_metrics(run_dir: Path, seed: int) -> dict[str, float | int | str | bool]:
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    best_reward = float(summary["best_reward"])
    final_reward = float(summary["final_reward"])
    final_critic_loss = float(logs["critic_loss"].iloc[-1])
    avg_theta_last10_mean = float(logs["avg_theta"].tail(min(10, len(logs))).mean())

    metrics: dict[str, float | int | str | bool] = {
        "seed": seed,
        "run_dir": str(run_dir),
        "epochs_completed": int(summary["epochs_completed"]),
        "stopped_early": bool(summary["stopped_early"]),
        "best_reward": best_reward,
        "final_reward": final_reward,
        "best_epoch": int(best_model_info["best_epoch"]),
        "reward_gap": float(final_reward - best_reward),
        "final_critic_loss": final_critic_loss,
        "avg_theta_last10_mean": avg_theta_last10_mean,
    }
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def aggregate_statistics(df: pd.DataFrame) -> dict[str, float]:
    stats = {
        "mean_best_reward": float(df["best_reward"].mean()),
        "std_best_reward": float(df["best_reward"].std(ddof=0)),
        "mean_final_reward": float(df["final_reward"].mean()),
        "std_final_reward": float(df["final_reward"].std(ddof=0)),
        "mean_reward_gap": float(df["reward_gap"].mean()),
        "std_reward_gap": float(df["reward_gap"].std(ddof=0)),
        "mean_best_epoch": float(df["best_epoch"].mean()),
    }
    return stats


def run_validation(seeds: list[int], output_root: Path) -> Path:
    validation_tag = datetime.now().strftime("seed_robustness_%Y%m%d_%H%M%S")
    root_dir = output_root / validation_tag
    root_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStoppingConfig(patience=FIXED_PATIENCE, min_delta=FIXED_MIN_DELTA)
    manifest: dict[str, object] = {
        "root_dir": str(root_dir),
        "fixed_params": {
            "max_epochs": FIXED_MAX_EPOCHS,
            "patience": FIXED_PATIENCE,
            "min_delta": FIXED_MIN_DELTA,
        },
        "seeds": seeds,
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str | bool]] = []

    for seed in seeds:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root_dir / f"seed_{seed}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        base_config = build_search_config(FIXED_MAX_EPOCHS)
        training = replace(base_config.training, seed=seed)
        config = replace(base_config, training=training)
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

        print(f"[train] seed={seed} -> {run_dir}")
        set_global_seeds(seed)
        train_agent_with_early_stopping(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
        )
        metrics = extract_metrics(run_dir, seed)
        all_metrics.append(metrics)
        manifest["runs"].append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    summary_df = pd.DataFrame(all_metrics).sort_values("seed").reset_index(drop=True)
    stats = aggregate_statistics(summary_df)

    summary_csv = root_dir / "seed_summary.csv"
    summary_json = root_dir / "seed_summary.json"
    summary_md = root_dir / "seed_summary.md"
    stats_json = root_dir / "seed_statistics.json"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(summary_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(dataframe_to_markdown(summary_df), encoding="utf-8")
    stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest["summary"] = {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "stats_json": str(stats_json),
        "statistics": stats,
    }
    (root_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root_dir


def main() -> None:
    args = parse_args()
    root_dir = run_validation(args.seeds, Path(args.output_root))
    print(f"Seed robustness validation completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
