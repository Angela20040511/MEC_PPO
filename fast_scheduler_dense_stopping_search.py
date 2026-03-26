from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import MECConfig, build_config
from fast_scheduler_dense_core_search import train_agent_with_early_stopping
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds

FIXED_SEED = 2025
FIXED_TOPOLOGY = "dense"

FIXED_ACTOR_LEARNING_RATE = 5e-4
FIXED_ENTROPY_COEFF = 1e-3
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
FIXED_ALPHA_P = 0.2
FIXED_MAX_CLUSTER_SIZE = 2
FIXED_MIN_DELTA = 1.0

DEFAULT_START_CHECK_GRID = (1, 2, 3)
DEFAULT_MAX_EPOCHS_GRID = (4, 6, 8)
DEFAULT_PATIENCE_GRID = (2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense-topology stopping-policy search under fixed PPO, DT, q, Umax, and alpha_p."
    )
    parser.add_argument(
        "--start-check-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_START_CHECK_GRID),
        help="Candidate early-stop monitor start epochs.",
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
        help="Candidate early-stop patience values.",
    )
    parser.add_argument(
        "--noma-quantile",
        type=float,
        default=FIXED_NOMA_QUANTILE,
        help="Fixed q selected from stage 1.",
    )
    parser.add_argument(
        "--alpha-p",
        type=float,
        default=FIXED_ALPHA_P,
        help="Fixed alpha_p selected from stage 2.",
    )
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=FIXED_MAX_CLUSTER_SIZE,
        help="Fixed Umax value.",
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


def build_dense_stopping_config(
    max_epochs: int,
    noma_quantile: float,
    alpha_p: float,
    max_cluster_size: int,
    seed: int,
) -> MECConfig:
    config = build_config(topology_mode=FIXED_TOPOLOGY)
    training = replace(
        config.training,
        num_epochs=max_epochs,
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
        alpha_p=alpha_p,
        noma_quantile=noma_quantile,
        max_cluster_size=max_cluster_size,
    )
    return replace(config, training=training, ppo=ppo, dt=dt, system=system)


def extract_metrics(
    run_dir: Path,
    start_check_epoch: int,
    max_epochs: int,
    patience: int,
) -> dict[str, float | int | str]:
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    best_reward = float(summary["best_reward"])
    final_reward = float(summary["final_reward"])
    reward_gap = final_reward - best_reward
    metrics: dict[str, float | int | str] = {
        "start_check_epoch": start_check_epoch,
        "max_epochs": max_epochs,
        "patience": patience,
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
    start_check_grid: list[int],
    max_epochs_grid: list[int],
    patience_grid: list[int],
    noma_quantile: float,
    alpha_p: float,
    max_cluster_size: int,
    seed: int,
    output_root: Path,
) -> Path:
    search_tag = datetime.now().strftime("fast_scheduler_dense_stopping_search_%Y%m%d_%H%M%S")
    root_dir = output_root / search_tag
    root_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "root_dir": str(root_dir),
        "fixed_params": {
            "topology": FIXED_TOPOLOGY,
            "seed": seed,
            "actor_learning_rate": FIXED_ACTOR_LEARNING_RATE,
            "entropy_coeff": FIXED_ENTROPY_COEFF,
            "update_epochs": FIXED_UPDATE_EPOCHS,
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
            "noma_quantile": noma_quantile,
            "alpha_p": alpha_p,
            "max_cluster_size": max_cluster_size,
            "min_delta": FIXED_MIN_DELTA,
        },
        "grid": {
            "start_check_epoch": start_check_grid,
            "max_epochs": max_epochs_grid,
            "patience": patience_grid,
        },
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []

    for start_check_epoch, max_epochs, patience in itertools.product(
        start_check_grid,
        max_epochs_grid,
        patience_grid,
    ):
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root_dir / (
            f"start_{start_check_epoch:02d}_"
            f"max_{max_epochs:02d}_"
            f"pat_{patience:02d}_{run_tag}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        config = build_dense_stopping_config(
            max_epochs=max_epochs,
            noma_quantile=noma_quantile,
            alpha_p=alpha_p,
            max_cluster_size=max_cluster_size,
            seed=seed,
        )
        early_stopping = EarlyStoppingConfig(
            patience=patience,
            min_delta=FIXED_MIN_DELTA,
            monitor_start_epoch=start_check_epoch,
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
            f"topology={FIXED_TOPOLOGY}, q={noma_quantile:.2f}, alpha_p={alpha_p:.2f}, "
            f"Umax={max_cluster_size}, start={start_check_epoch}, max_epochs={max_epochs}, "
            f"patience={patience}, seed={seed} -> {run_dir}"
        )
        set_global_seeds(seed)
        train_agent_with_early_stopping(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
        )
        metrics = extract_metrics(
            run_dir=run_dir,
            start_check_epoch=start_check_epoch,
            max_epochs=max_epochs,
            patience=patience,
        )
        all_metrics.append(metrics)
        manifest["runs"].append(
            {
                "start_check_epoch": start_check_epoch,
                "max_epochs": max_epochs,
                "patience": patience,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    ranked_df = rank_runs(pd.DataFrame(all_metrics))
    summary_csv = root_dir / "fast_scheduler_dense_stopping_summary.csv"
    summary_json = root_dir / "fast_scheduler_dense_stopping_summary.json"
    summary_md = root_dir / "fast_scheduler_dense_stopping_summary.md"
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
        noma_quantile=args.noma_quantile,
        alpha_p=args.alpha_p,
        max_cluster_size=args.max_cluster_size,
        seed=args.seed,
        output_root=Path(args.output_root),
    )
    print(f"Dense stopping-policy search completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
