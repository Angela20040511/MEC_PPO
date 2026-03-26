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

FIXED_START_CHECK_EPOCH = 1
FIXED_MAX_EPOCHS = 6
FIXED_PATIENCE = 3
FIXED_MIN_DELTA = 1.0

FIXED_GAMMA_E = 0.15
FIXED_GAMMA_Q = 0.20
FIXED_RHO0 = 0.85

DEFAULT_RHO0_GRID = (0.75, 0.85, 0.95)
DEFAULT_GAMMA_Q_GRID = (0.10, 0.20, 0.30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage dense rho0/gamma_q search focused on stability."
    )
    parser.add_argument(
        "--rho0-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_RHO0_GRID),
        help="Candidate rho0 values for stage 1.",
    )
    parser.add_argument(
        "--gamma-q-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_GAMMA_Q_GRID),
        help="Candidate gamma_q values for stage 2.",
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


def build_dense_config(
    rho0: float,
    gamma_q: float,
    gamma_e: float,
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
        noma_quantile=FIXED_NOMA_QUANTILE,
        max_cluster_size=FIXED_MAX_CLUSTER_SIZE,
        alpha_p=FIXED_ALPHA_P,
        rho0=rho0,
        gamma_q=gamma_q,
        gamma_e=gamma_e,
    )
    return replace(config, training=training, ppo=ppo, dt=dt, system=system)


def extract_metrics(run_dir: Path, extra_fields: dict[str, float | int | str]) -> dict[str, float | int | str]:
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    best_reward = float(summary["best_reward"])
    final_reward = float(summary["final_reward"])
    reward_gap = final_reward - best_reward
    best_epoch = int(best_model_info["best_epoch"])
    metrics: dict[str, float | int | str] = {
        **extra_fields,
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
            "final_reward",
            "reward_gap_abs",
            "final_critic_loss",
            "best_reward",
            "best_epoch_is_one",
            "best_epoch",
        ],
        ascending=[False, True, True, False, True, False],
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def write_stage_outputs(stage_dir: Path, stage_name: str, ranked_df: pd.DataFrame) -> dict[str, str]:
    summary_csv = stage_dir / f"{stage_name}_summary.csv"
    summary_json = stage_dir / f"{stage_name}_summary.json"
    summary_md = stage_dir / f"{stage_name}_summary.md"
    ranked_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        ranked_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(dataframe_to_markdown(ranked_df), encoding="utf-8")
    return {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
    }


def run_rho0_stage(
    root_dir: Path,
    rho0_grid: list[float],
    seed: int,
) -> tuple[pd.DataFrame, float]:
    stage_dir = root_dir / "rho0_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStoppingConfig(
        patience=FIXED_PATIENCE,
        min_delta=FIXED_MIN_DELTA,
        monitor_start_epoch=FIXED_START_CHECK_EPOCH,
    )
    manifest: dict[str, object] = {
        "stage": "rho0",
        "stage_dir": str(stage_dir),
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
            "q": FIXED_NOMA_QUANTILE,
            "Umax": FIXED_MAX_CLUSTER_SIZE,
            "alpha_p": FIXED_ALPHA_P,
            "start_check_epoch": FIXED_START_CHECK_EPOCH,
            "max_epochs": FIXED_MAX_EPOCHS,
            "patience": FIXED_PATIENCE,
            "min_delta": FIXED_MIN_DELTA,
            "gamma_q": FIXED_GAMMA_Q,
            "gamma_e": FIXED_GAMMA_E,
        },
        "rho0_grid": rho0_grid,
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []
    for rho0 in rho0_grid:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = stage_dir / f"rho0_{str(f'{rho0:.2f}').replace('.', 'p')}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = build_dense_config(
            rho0=rho0,
            gamma_q=FIXED_GAMMA_Q,
            gamma_e=FIXED_GAMMA_E,
            seed=seed,
        )
        (run_dir / "experiment_config.json").write_text(
            json.dumps(
                {
                    "stage": "rho0",
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
            f"stage=rho0, topology={FIXED_TOPOLOGY}, rho0={rho0:.2f}, gamma_q={FIXED_GAMMA_Q:.2f}, "
            f"seed={seed} -> {run_dir}"
        )
        set_global_seeds(seed)
        train_agent_with_early_stopping(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
        )
        metrics = extract_metrics(
            run_dir=run_dir,
            extra_fields={
                "rho0": rho0,
                "gamma_q": FIXED_GAMMA_Q,
                "gamma_e": FIXED_GAMMA_E,
            },
        )
        all_metrics.append(metrics)
        manifest["runs"].append(
            {
                "rho0": rho0,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    ranked_df = rank_runs(pd.DataFrame(all_metrics))
    manifest["summary"] = write_stage_outputs(stage_dir, "rho0_stage", ranked_df)
    (stage_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    best_rho0 = float(ranked_df.iloc[0]["rho0"])
    return ranked_df, best_rho0


def run_gamma_q_stage(
    root_dir: Path,
    gamma_q_grid: list[float],
    rho0: float,
    seed: int,
) -> pd.DataFrame:
    stage_dir = root_dir / "gamma_q_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStoppingConfig(
        patience=FIXED_PATIENCE,
        min_delta=FIXED_MIN_DELTA,
        monitor_start_epoch=FIXED_START_CHECK_EPOCH,
    )
    manifest: dict[str, object] = {
        "stage": "gamma_q",
        "stage_dir": str(stage_dir),
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
            "q": FIXED_NOMA_QUANTILE,
            "Umax": FIXED_MAX_CLUSTER_SIZE,
            "alpha_p": FIXED_ALPHA_P,
            "start_check_epoch": FIXED_START_CHECK_EPOCH,
            "max_epochs": FIXED_MAX_EPOCHS,
            "patience": FIXED_PATIENCE,
            "min_delta": FIXED_MIN_DELTA,
            "rho0": rho0,
            "gamma_e": FIXED_GAMMA_E,
        },
        "gamma_q_grid": gamma_q_grid,
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []
    for gamma_q in gamma_q_grid:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = stage_dir / f"gamma_q_{str(f'{gamma_q:.2f}').replace('.', 'p')}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = build_dense_config(
            rho0=rho0,
            gamma_q=gamma_q,
            gamma_e=FIXED_GAMMA_E,
            seed=seed,
        )
        (run_dir / "experiment_config.json").write_text(
            json.dumps(
                {
                    "stage": "gamma_q",
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
            f"stage=gamma_q, topology={FIXED_TOPOLOGY}, rho0={rho0:.2f}, gamma_q={gamma_q:.2f}, "
            f"seed={seed} -> {run_dir}"
        )
        set_global_seeds(seed)
        train_agent_with_early_stopping(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
        )
        metrics = extract_metrics(
            run_dir=run_dir,
            extra_fields={
                "rho0": rho0,
                "gamma_q": gamma_q,
                "gamma_e": FIXED_GAMMA_E,
            },
        )
        all_metrics.append(metrics)
        manifest["runs"].append(
            {
                "gamma_q": gamma_q,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    ranked_df = rank_runs(pd.DataFrame(all_metrics))
    manifest["summary"] = write_stage_outputs(stage_dir, "gamma_q_stage", ranked_df)
    (stage_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ranked_df


def main() -> None:
    args = parse_args()
    root_dir = Path(args.output_root) / datetime.now().strftime(
        "fast_scheduler_dense_rho_gamma_search_%Y%m%d_%H%M%S"
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    rho0_df, best_rho0 = run_rho0_stage(
        root_dir=root_dir,
        rho0_grid=[float(value) for value in args.rho0_grid],
        seed=args.seed,
    )
    gamma_q_df = run_gamma_q_stage(
        root_dir=root_dir,
        gamma_q_grid=[float(value) for value in args.gamma_q_grid],
        rho0=best_rho0,
        seed=args.seed,
    )

    final_summary = {
        "root_dir": str(root_dir),
        "recommended_rho0": best_rho0,
        "recommended_gamma_q": float(gamma_q_df.iloc[0]["gamma_q"]),
        "rho0_stage_summary": str(root_dir / "rho0_stage" / "rho0_stage_summary.csv"),
        "gamma_q_stage_summary": str(root_dir / "gamma_q_stage" / "gamma_q_stage_summary.csv"),
    }
    (root_dir / "final_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Dense rho0/gamma_q search completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
