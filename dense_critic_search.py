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
FIXED_ALPHA_P = 0.4
FIXED_RHO0 = 0.75
FIXED_GAMMA_Q = 0.10
FIXED_GAMMA_E = 0.15

FIXED_START_CHECK_EPOCH = 1
FIXED_MAX_EPOCHS = 6
FIXED_PATIENCE = 3
FIXED_MIN_DELTA = 1.0

DEFAULT_CRITIC_LR_GRID = (2e-4, 5e-4, 1e-3)
DEFAULT_VALUE_COEFF_GRID = (0.25, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense-topology critic-side search over critic_learning_rate and value_coeff."
    )
    parser.add_argument(
        "--critic-lr-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_CRITIC_LR_GRID),
        help="Candidate critic learning rates.",
    )
    parser.add_argument(
        "--value-coeff-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_VALUE_COEFF_GRID),
        help="Candidate PPO value coefficients.",
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


def build_dense_critic_config(
    critic_learning_rate: float,
    value_coeff: float,
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
        critic_learning_rate=critic_learning_rate,
        entropy_coeff=FIXED_ENTROPY_COEFF,
        value_coeff=value_coeff,
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
        noma_quantile=FIXED_NOMA_QUANTILE,
        max_cluster_size=FIXED_MAX_CLUSTER_SIZE,
        alpha_p=FIXED_ALPHA_P,
        rho0=FIXED_RHO0,
        gamma_q=FIXED_GAMMA_Q,
        gamma_e=FIXED_GAMMA_E,
    )
    return replace(config, training=training, ppo=ppo, dt=dt, system=system)


def extract_metrics(
    run_dir: Path,
    critic_learning_rate: float,
    value_coeff: float,
) -> dict[str, float | int | str]:
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    best_reward = float(summary["best_reward"])
    final_reward = float(summary["final_reward"])
    reward_gap = final_reward - best_reward
    best_epoch = int(best_model_info["best_epoch"])
    metrics: dict[str, float | int | str] = {
        "critic_learning_rate": critic_learning_rate,
        "value_coeff": value_coeff,
        "run_dir": str(run_dir),
        "epochs_completed": int(summary["epochs_completed"]),
        "best_reward": best_reward,
        "final_reward": final_reward,
        "best_epoch": best_epoch,
        "reward_gap": reward_gap,
        "reward_gap_abs": abs(reward_gap),
        "final_critic_loss": float(logs["critic_loss"].iloc[-1]),
        "avg_theta_last10": float(logs["avg_theta"].tail(min(10, len(logs))).mean()),
        "best_epoch_is_one": int(best_epoch == 1),
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
            "best_epoch_is_one",
            "best_epoch",
            "final_reward",
            "reward_gap_abs",
            "final_critic_loss",
            "best_reward",
        ],
        ascending=[True, False, False, True, True, False],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def run_search(
    critic_lr_grid: list[float],
    value_coeff_grid: list[float],
    seed: int,
    output_root: Path,
) -> Path:
    root_dir = output_root / datetime.now().strftime("dense_critic_search_%Y%m%d_%H%M%S")
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
            "time_steps": FIXED_TIME_STEPS,
            "q": FIXED_NOMA_QUANTILE,
            "Umax": FIXED_MAX_CLUSTER_SIZE,
            "alpha_p": FIXED_ALPHA_P,
            "rho0": FIXED_RHO0,
            "gamma_q": FIXED_GAMMA_Q,
            "gamma_e": FIXED_GAMMA_E,
            "start_check_epoch": FIXED_START_CHECK_EPOCH,
            "max_epochs": FIXED_MAX_EPOCHS,
            "patience": FIXED_PATIENCE,
            "min_delta": FIXED_MIN_DELTA,
        },
        "grid": {
            "critic_learning_rate": critic_lr_grid,
            "value_coeff": value_coeff_grid,
        },
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []
    for critic_learning_rate in critic_lr_grid:
        for value_coeff in value_coeff_grid:
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_dir = root_dir / (
                f"critic_lr_{str(f'{critic_learning_rate:.0e}').replace('-', 'm')}_"
                f"value_coeff_{str(value_coeff).replace('.', 'p')}_{run_tag}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            config = build_dense_critic_config(
                critic_learning_rate=critic_learning_rate,
                value_coeff=value_coeff,
                seed=seed,
            )
            (run_dir / "experiment_config.json").write_text(
                json.dumps(
                    {
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
                f"critic_lr={critic_learning_rate:g}, value_coeff={value_coeff:g}, seed={seed} -> {run_dir}"
            )
            set_global_seeds(seed)
            train_agent_with_early_stopping(
                config=config,
                checkpoint_dir=str(run_dir),
                early_stopping=early_stopping,
            )
            metrics = extract_metrics(
                run_dir=run_dir,
                critic_learning_rate=critic_learning_rate,
                value_coeff=value_coeff,
            )
            all_metrics.append(metrics)
            manifest["runs"].append(
                {
                    "critic_learning_rate": critic_learning_rate,
                    "value_coeff": value_coeff,
                    "run_dir": str(run_dir),
                    "analysis_summary": str(run_dir / "analysis_summary.json"),
                }
            )

    ranked_df = rank_runs(pd.DataFrame(all_metrics))
    summary_csv = root_dir / "dense_critic_summary.csv"
    summary_json = root_dir / "dense_critic_summary.json"
    summary_md = root_dir / "dense_critic_summary.md"
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
        critic_lr_grid=[float(value) for value in args.critic_lr_grid],
        value_coeff_grid=[float(value) for value in args.value_coeff_grid],
        seed=args.seed,
        output_root=Path(args.output_root),
    )
    print(f"Dense critic search completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
