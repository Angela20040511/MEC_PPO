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

FIXED_MIN_DELTA = 1.0
DEFAULT_START_CHECK_GRID = (3, 5, 8)
DEFAULT_MAX_EPOCHS_GRID = (12, 15, 20)
DEFAULT_PATIENCE_GRID = (3, 5)
DEFAULT_SEEDS = (2025, 2026, 2027, 2028, 2029)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search earlier PPO stopping policies with multi-seed validation."
    )
    parser.add_argument(
        "--start-check-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_START_CHECK_GRID),
        help="Candidate start_check_epoch values.",
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
        help="Candidate patience values.",
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


def extract_seed_metrics(run_dir: Path, seed: int) -> dict[str, float | int | bool]:
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    return {
        "seed": seed,
        "epochs_completed": int(summary["epochs_completed"]),
        "stopped_early": bool(summary["stopped_early"]),
        "best_reward": float(summary["best_reward"]),
        "final_reward": float(summary["final_reward"]),
        "best_epoch": int(best_model_info["best_epoch"]),
        "reward_gap": float(summary["final_reward"] - summary["best_reward"]),
        "final_critic_loss": float(logs["critic_loss"].iloc[-1]),
        "avg_theta_last10_mean": float(logs["avg_theta"].tail(min(10, len(logs))).mean()),
    }


def aggregate_combo_metrics(
    combo_df: pd.DataFrame,
    start_check_epoch: int,
    max_epochs: int,
    patience: int,
    combo_dir: Path,
) -> dict[str, float | int | str]:
    metrics = {
        "start_check_epoch": start_check_epoch,
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": FIXED_MIN_DELTA,
        "combo_dir": str(combo_dir),
        "mean_best_reward": float(combo_df["best_reward"].mean()),
        "std_best_reward": float(combo_df["best_reward"].std(ddof=0)),
        "mean_final_reward": float(combo_df["final_reward"].mean()),
        "std_final_reward": float(combo_df["final_reward"].std(ddof=0)),
        "mean_reward_gap": float(combo_df["reward_gap"].mean()),
        "std_reward_gap": float(combo_df["reward_gap"].std(ddof=0)),
        "mean_best_epoch": float(combo_df["best_epoch"].mean()),
        "mean_final_critic_loss": float(combo_df["final_critic_loss"].mean()),
        "mean_avg_theta_last10": float(combo_df["avg_theta_last10_mean"].mean()),
        "early_stop_trigger_rate": float(combo_df["stopped_early"].mean()),
        "mean_epochs_completed": float(combo_df["epochs_completed"].mean()),
    }
    (combo_dir / "combo_statistics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    combo_df.to_csv(combo_dir / "seed_metrics.csv", index=False, encoding="utf-8-sig")
    (combo_dir / "seed_metrics.json").write_text(
        combo_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def rank_combos(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.copy()
    ranked["abs_mean_reward_gap"] = ranked["mean_reward_gap"].abs()
    ranked = ranked.sort_values(
        by=[
            "mean_best_reward",
            "abs_mean_reward_gap",
            "std_best_reward",
            "mean_best_epoch",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def run_search(
    start_check_grid: list[int],
    max_epochs_grid: list[int],
    patience_grid: list[int],
    seeds: list[int],
    output_root: Path,
) -> Path:
    search_tag = datetime.now().strftime("early_stop_policy_seed_search_%Y%m%d_%H%M%S")
    root_dir = output_root / search_tag
    root_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "root_dir": str(root_dir),
        "fixed_params": {
            "min_delta": FIXED_MIN_DELTA,
        },
        "grid": {
            "start_check_epoch": start_check_grid,
            "max_epochs": max_epochs_grid,
            "patience": patience_grid,
            "seeds": seeds,
        },
        "combos": [],
    }

    combo_summaries: list[dict[str, float | int | str]] = []

    for start_check_epoch in start_check_grid:
        for max_epochs in max_epochs_grid:
            for patience in patience_grid:
                combo_name = (
                    f"start_{start_check_epoch:02d}_max_{max_epochs:02d}_"
                    f"pat_{patience:02d}_delta_1p0"
                )
                combo_dir = root_dir / combo_name
                combo_dir.mkdir(parents=True, exist_ok=True)

                combo_manifest: dict[str, object] = {
                    "start_check_epoch": start_check_epoch,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "min_delta": FIXED_MIN_DELTA,
                    "combo_dir": str(combo_dir),
                    "seed_runs": [],
                }
                seed_metrics: list[dict[str, float | int | bool]] = []

                for seed in seeds:
                    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    run_dir = combo_dir / f"seed_{seed}_{run_tag}"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    config = build_search_config(max_epochs)
                    training = replace(config.training, seed=seed)
                    config = replace(config, training=training)
                    early_stopping = EarlyStoppingConfig(
                        patience=patience,
                        min_delta=FIXED_MIN_DELTA,
                        monitor_start_epoch=start_check_epoch,
                    )
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

                    print(
                        "[train] "
                        f"start_check_epoch={start_check_epoch}, max_epochs={max_epochs}, "
                        f"patience={patience}, seed={seed} -> {run_dir}"
                    )
                    set_global_seeds(seed)
                    train_agent_with_early_stopping(
                        config=config,
                        checkpoint_dir=str(run_dir),
                        early_stopping=early_stopping,
                    )
                    metrics = extract_seed_metrics(run_dir, seed)
                    seed_metrics.append(metrics)
                    combo_manifest["seed_runs"].append(
                        {
                            "seed": seed,
                            "run_dir": str(run_dir),
                        }
                    )

                combo_df = pd.DataFrame(seed_metrics).sort_values("seed").reset_index(drop=True)
                combo_summary = aggregate_combo_metrics(
                    combo_df=combo_df,
                    start_check_epoch=start_check_epoch,
                    max_epochs=max_epochs,
                    patience=patience,
                    combo_dir=combo_dir,
                )
                combo_summaries.append(combo_summary)
                combo_manifest["combo_statistics"] = combo_summary
                manifest["combos"].append(combo_manifest)

    summary_df = pd.DataFrame(combo_summaries)
    ranked_df = rank_combos(summary_df)

    summary_csv = root_dir / "combo_summary.csv"
    summary_json = root_dir / "combo_summary.json"
    summary_md = root_dir / "combo_summary.md"
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
        start_check_grid=args.start_check_grid,
        max_epochs_grid=args.max_epochs_grid,
        patience_grid=args.patience_grid,
        seeds=args.seeds,
        output_root=Path(args.output_root),
    )
    print(f"Early stop policy search completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
