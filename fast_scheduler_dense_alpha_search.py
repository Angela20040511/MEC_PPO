from __future__ import annotations

import argparse
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

FIXED_NOMA_QUANTILE = 0.65
FIXED_MAX_CLUSTER_SIZE = 2

DEFAULT_ALPHA_P_GRID = (0.2, 0.4, 0.6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense-topology alpha_p search under fixed PPO, DT, q, and Umax."
    )
    parser.add_argument(
        "--alpha-p-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHA_P_GRID),
        help="Candidate alpha_p values.",
    )
    parser.add_argument(
        "--noma-quantile",
        type=float,
        default=FIXED_NOMA_QUANTILE,
        help="Fixed q selected from the previous stage.",
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


def build_dense_alpha_config(
    alpha_p: float,
    noma_quantile: float,
    max_cluster_size: int,
    seed: int,
) -> MECConfig:
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
        alpha_p=alpha_p,
        noma_quantile=noma_quantile,
        max_cluster_size=max_cluster_size,
    )
    return replace(config, training=training, ppo=ppo, dt=dt, system=system)


def extract_metrics(
    run_dir: Path,
    alpha_p: float,
    noma_quantile: float,
    max_cluster_size: int,
) -> dict[str, float | int | str]:
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    best_reward = float(summary["best_reward"])
    final_reward = float(summary["final_reward"])
    reward_gap = final_reward - best_reward
    metrics: dict[str, float | int | str] = {
        "alpha_p": alpha_p,
        "noma_quantile": noma_quantile,
        "max_cluster_size": max_cluster_size,
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
    alpha_p_grid: list[float],
    noma_quantile: float,
    max_cluster_size: int,
    seed: int,
    output_root: Path,
) -> Path:
    search_tag = datetime.now().strftime("fast_scheduler_dense_alpha_search_%Y%m%d_%H%M%S")
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
            "max_cluster_size": max_cluster_size,
        },
        "alpha_p_grid": alpha_p_grid,
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []

    for alpha_p in alpha_p_grid:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root_dir / f"alpha_p_{str(f'{alpha_p:.2f}').replace('.', 'p')}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = build_dense_alpha_config(
            alpha_p=alpha_p,
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
            f"Umax={max_cluster_size}, alpha_p={alpha_p:.2f}, seed={seed} -> {run_dir}"
        )
        set_global_seeds(seed)
        train_agent_with_early_stopping(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
        )
        metrics = extract_metrics(
            run_dir=run_dir,
            alpha_p=alpha_p,
            noma_quantile=noma_quantile,
            max_cluster_size=max_cluster_size,
        )
        all_metrics.append(metrics)
        manifest["runs"].append(
            {
                "alpha_p": alpha_p,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    ranked_df = rank_runs(pd.DataFrame(all_metrics))
    summary_csv = root_dir / "fast_scheduler_dense_alpha_summary.csv"
    summary_json = root_dir / "fast_scheduler_dense_alpha_summary.json"
    summary_md = root_dir / "fast_scheduler_dense_alpha_summary.md"
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
        alpha_p_grid=args.alpha_p_grid,
        noma_quantile=args.noma_quantile,
        max_cluster_size=args.max_cluster_size,
        seed=args.seed,
        output_root=Path(args.output_root),
    )
    print(f"Dense alpha_p search completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
