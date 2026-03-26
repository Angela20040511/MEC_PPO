from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import MECConfig, build_config
from fast_scheduler_dense_core_search import train_agent_with_early_stopping
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds

FIXED_SEED = 2025
FIXED_TOPOLOGY = "dense"

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

FIXED_ENTROPY_COEFF = 1e-3
FIXED_ACTOR_LR = 5e-4
FIXED_UPDATE_EPOCHS = 10

DEFAULT_ACTOR_LR_GRID = (2e-4, 3e-4, 4e-4)
DEFAULT_UPDATE_EPOCHS_GRID = (4, 6, 8)
DEFAULT_ENTROPY_COEFF_GRID = (5e-4, 1e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Staged dense PPO tuning after scheduler parameters are fixed."
    )
    parser.add_argument(
        "--actor-lr-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_ACTOR_LR_GRID),
        help="Candidate actor learning rates for stage 1.",
    )
    parser.add_argument(
        "--update-epochs-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_UPDATE_EPOCHS_GRID),
        help="Candidate PPO update_epochs values for stage 2.",
    )
    parser.add_argument(
        "--entropy-coeff-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_ENTROPY_COEFF_GRID),
        help="Candidate entropy coefficients for stage 3.",
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


def build_dense_ppo_config(
    actor_learning_rate: float,
    update_epochs: int,
    entropy_coeff: float,
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
        actor_learning_rate=actor_learning_rate,
        entropy_coeff=entropy_coeff,
        update_epochs=update_epochs,
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
    extra_fields: dict[str, float | int | str],
) -> dict[str, float | int | str]:
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
            "best_epoch_is_one",
            "best_epoch",
            "final_critic_loss",
            "best_reward",
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


def run_stage(
    *,
    root_dir: Path,
    stage_name: str,
    param_name: str,
    param_values: list[float | int],
    actor_learning_rate: float,
    update_epochs: int,
    entropy_coeff: float,
    seed: int,
) -> pd.DataFrame:
    stage_dir = root_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStoppingConfig(
        patience=FIXED_PATIENCE,
        min_delta=FIXED_MIN_DELTA,
        monitor_start_epoch=FIXED_START_CHECK_EPOCH,
    )
    manifest: dict[str, Any] = {
        "stage": stage_name,
        "stage_dir": str(stage_dir),
        "fixed_params": {
            "topology": FIXED_TOPOLOGY,
            "seed": seed,
            "time_steps": FIXED_TIME_STEPS,
            "dt_history_window": FIXED_DT_HISTORY_WINDOW,
            "dt_prediction_horizon": FIXED_DT_PREDICTION_HORIZON,
            "dt_hidden_size": FIXED_DT_HIDDEN_SIZE,
            "dt_retrain_interval": FIXED_DT_RETRAIN_INTERVAL,
            "dt_train_epochs": FIXED_DT_TRAIN_EPOCHS,
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
            "actor_learning_rate": actor_learning_rate,
            "update_epochs": update_epochs,
            "entropy_coeff": entropy_coeff,
        },
        "param_name": param_name,
        "param_values": param_values,
        "runs": [],
    }

    all_metrics: list[dict[str, float | int | str]] = []
    for value in param_values:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        value_str = f"{float(value):.0e}" if param_name == "actor_learning_rate" else str(value)
        value_str = value_str.replace(".", "p").replace("-", "m")
        run_dir = stage_dir / f"{param_name}_{value_str}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        stage_actor_lr = float(value) if param_name == "actor_learning_rate" else actor_learning_rate
        stage_update_epochs = int(value) if param_name == "update_epochs" else update_epochs
        stage_entropy_coeff = float(value) if param_name == "entropy_coeff" else entropy_coeff
        config = build_dense_ppo_config(
            actor_learning_rate=stage_actor_lr,
            update_epochs=stage_update_epochs,
            entropy_coeff=stage_entropy_coeff,
            seed=seed,
        )

        (run_dir / "experiment_config.json").write_text(
            json.dumps(
                {
                    "stage": stage_name,
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
            f"stage={stage_name}, actor_lr={stage_actor_lr:g}, update_epochs={stage_update_epochs}, "
            f"entropy_coeff={stage_entropy_coeff:g}, seed={seed} -> {run_dir}"
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
                "actor_learning_rate": stage_actor_lr,
                "update_epochs": stage_update_epochs,
                "entropy_coeff": stage_entropy_coeff,
            },
        )
        all_metrics.append(metrics)
        manifest["runs"].append(
            {
                param_name: value,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    ranked_df = rank_runs(pd.DataFrame(all_metrics))
    manifest["summary"] = write_stage_outputs(stage_dir, stage_name, ranked_df)
    (stage_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return ranked_df


def main() -> None:
    args = parse_args()
    root_dir = Path(args.output_root) / datetime.now().strftime("dense_ppo_tuning_%Y%m%d_%H%M%S")
    root_dir.mkdir(parents=True, exist_ok=True)

    stage1_df = run_stage(
        root_dir=root_dir,
        stage_name="actor_lr_stage",
        param_name="actor_learning_rate",
        param_values=[float(value) for value in args.actor_lr_grid],
        actor_learning_rate=FIXED_ACTOR_LR,
        update_epochs=FIXED_UPDATE_EPOCHS,
        entropy_coeff=FIXED_ENTROPY_COEFF,
        seed=args.seed,
    )
    best_actor_lr = float(stage1_df.iloc[0]["actor_learning_rate"])

    stage2_df = run_stage(
        root_dir=root_dir,
        stage_name="update_epochs_stage",
        param_name="update_epochs",
        param_values=[int(value) for value in args.update_epochs_grid],
        actor_learning_rate=best_actor_lr,
        update_epochs=FIXED_UPDATE_EPOCHS,
        entropy_coeff=FIXED_ENTROPY_COEFF,
        seed=args.seed,
    )
    best_update_epochs = int(stage2_df.iloc[0]["update_epochs"])
    best_entropy_coeff = FIXED_ENTROPY_COEFF
    entropy_stage_ran = bool(int(stage2_df.iloc[0]["best_epoch"]) == 1)
    stage3_summary_path: str | None = None

    if entropy_stage_ran:
        stage3_df = run_stage(
            root_dir=root_dir,
            stage_name="entropy_coeff_stage",
            param_name="entropy_coeff",
            param_values=[float(value) for value in args.entropy_coeff_grid],
            actor_learning_rate=best_actor_lr,
            update_epochs=best_update_epochs,
            entropy_coeff=FIXED_ENTROPY_COEFF,
            seed=args.seed,
        )
        best_entropy_coeff = float(stage3_df.iloc[0]["entropy_coeff"])
        stage3_summary_path = str(root_dir / "entropy_coeff_stage" / "entropy_coeff_stage_summary.csv")
    else:
        stage3_df = None

    final_summary = {
        "root_dir": str(root_dir),
        "recommended_actor_learning_rate": best_actor_lr,
        "recommended_update_epochs": best_update_epochs,
        "recommended_entropy_coeff": best_entropy_coeff,
        "entropy_stage_ran": entropy_stage_ran,
        "actor_lr_stage_summary": str(root_dir / "actor_lr_stage" / "actor_lr_stage_summary.csv"),
        "update_epochs_stage_summary": str(root_dir / "update_epochs_stage" / "update_epochs_stage_summary.csv"),
        "entropy_coeff_stage_summary": stage3_summary_path,
    }
    (root_dir / "final_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Dense PPO tuning completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
