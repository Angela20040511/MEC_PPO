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
FIXED_TRAIN_EPOCHS = 5
FIXED_MIN_HISTORY_TO_TRAIN = 20
FIXED_RETRAIN_INTERVAL = 5

DEFAULT_HISTORY_WINDOW_GRID = (4, 8, 12, 16)
DEFAULT_PREDICTION_HORIZON_GRID = (2, 4, 6)
DEFAULT_HIDDEN_SIZE_GRID = (32, 64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coarse search for DT/LSTM core hyperparameters under a fixed PPO setup."
    )
    parser.add_argument(
        "--history-window-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_HISTORY_WINDOW_GRID),
        help="Candidate DT history_window values.",
    )
    parser.add_argument(
        "--prediction-horizon-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_PREDICTION_HORIZON_GRID),
        help="Candidate DT prediction_horizon values.",
    )
    parser.add_argument(
        "--hidden-size-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_HIDDEN_SIZE_GRID),
        help="Candidate DT hidden_size values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FIXED_SEED,
        help="Fixed seed for the first-round coarse search.",
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


def build_dt_search_config(
    history_window: int,
    prediction_horizon: int,
    hidden_size: int,
    seed: int,
):
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
        history_window=history_window,
        prediction_horizon=prediction_horizon,
        hidden_size=hidden_size,
        num_layers=FIXED_NUM_LAYERS,
        learning_rate=FIXED_DT_LEARNING_RATE,
        batch_size=FIXED_BATCH_SIZE,
        train_epochs=FIXED_TRAIN_EPOCHS,
        min_history_to_train=FIXED_MIN_HISTORY_TO_TRAIN,
        retrain_interval=FIXED_RETRAIN_INTERVAL,
    )
    return replace(config, training=training, ppo=ppo, dt=dt)


def stability_score(logs: pd.DataFrame) -> float:
    reward_tail = logs["episode_reward"].tail(min(10, len(logs))).astype(float)
    critic_tail = logs["critic_loss"].tail(min(10, len(logs))).astype(float)
    reward_std = float(reward_tail.std(ddof=0)) if len(reward_tail) > 1 else 0.0
    critic_std = float(critic_tail.std(ddof=0)) if len(critic_tail) > 1 else 0.0
    return reward_std + 0.01 * critic_std


def stability_label(reward_gap: float, score: float) -> str:
    gap_abs = abs(reward_gap)
    if gap_abs <= 15.0 and score <= 10.0:
        return "stable"
    if gap_abs <= 30.0 and score <= 18.0:
        return "moderate"
    return "unstable"


def extract_metrics(
    run_dir: Path,
    history_window: int,
    prediction_horizon: int,
    hidden_size: int,
    seed: int,
) -> dict[str, float | int | str]:
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    best_reward = float(summary["best_reward"])
    final_reward = float(summary["final_reward"])
    reward_gap = float(final_reward - best_reward)
    score = stability_score(logs)
    metrics: dict[str, float | int | str] = {
        "history_window": history_window,
        "prediction_horizon": prediction_horizon,
        "hidden_size": hidden_size,
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
        "stability_score": score,
        "training_stability": stability_label(reward_gap, score),
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
            "stability_score",
        ],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def run_search(
    history_window_grid: list[int],
    prediction_horizon_grid: list[int],
    hidden_size_grid: list[int],
    seed: int,
    output_root: Path,
) -> Path:
    search_tag = datetime.now().strftime("dt_core_search_%Y%m%d_%H%M%S")
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
            "train_epochs": FIXED_TRAIN_EPOCHS,
            "min_history_to_train": FIXED_MIN_HISTORY_TO_TRAIN,
            "retrain_interval": FIXED_RETRAIN_INTERVAL,
        },
        "grid": {
            "history_window": history_window_grid,
            "prediction_horizon": prediction_horizon_grid,
            "hidden_size": hidden_size_grid,
        },
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []

    for history_window in history_window_grid:
        for prediction_horizon in prediction_horizon_grid:
            for hidden_size in hidden_size_grid:
                run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                run_dir = root_dir / (
                    f"hw_{history_window:02d}_ph_{prediction_horizon:02d}_"
                    f"hs_{hidden_size:03d}_{run_tag}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)

                config = build_dt_search_config(
                    history_window=history_window,
                    prediction_horizon=prediction_horizon,
                    hidden_size=hidden_size,
                    seed=seed,
                )
                (run_dir / "experiment_config.json").write_text(
                    json.dumps(
                        {
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

                print(
                    "[train] "
                    f"history_window={history_window}, prediction_horizon={prediction_horizon}, "
                    f"hidden_size={hidden_size}, seed={seed} -> {run_dir}"
                )
                set_global_seeds(seed)
                train_agent_with_early_stopping(
                    config=config,
                    checkpoint_dir=str(run_dir),
                    early_stopping=early_stopping,
                )
                metrics = extract_metrics(
                    run_dir=run_dir,
                    history_window=history_window,
                    prediction_horizon=prediction_horizon,
                    hidden_size=hidden_size,
                    seed=seed,
                )
                all_metrics.append(metrics)
                manifest["runs"].append(
                    {
                        "history_window": history_window,
                        "prediction_horizon": prediction_horizon,
                        "hidden_size": hidden_size,
                        "seed": seed,
                        "run_dir": str(run_dir),
                        "analysis_summary": str(run_dir / "analysis_summary.json"),
                    }
                )

    summary_df = pd.DataFrame(all_metrics)
    ranked_df = rank_runs(summary_df)

    summary_csv = root_dir / "dt_search_summary.csv"
    summary_json = root_dir / "dt_search_summary.json"
    summary_md = root_dir / "dt_search_summary.md"
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
        history_window_grid=args.history_window_grid,
        prediction_horizon_grid=args.prediction_horizon_grid,
        hidden_size_grid=args.hidden_size_grid,
        seed=args.seed,
        output_root=Path(args.output_root),
    )
    print(f"DT core search completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
