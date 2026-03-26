import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from config import MECConfig, build_config
from dense_actor_input_experiment import build_state_layout
from dense_critic_diagnosis import (
    PROBE_STATE_COUNT,
    collect_probe_states,
    dataframe_to_markdown,
    extract_metrics,
    plot_single_run_outputs,
    train_with_diagnosis,
)
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds
from rl.ppo_agent import PPOAgent

FIXED_SEED = 2025
FIXED_TOPOLOGY = "dense"
FIXED_REWARD_MODE = "per_sensor"
FIXED_VALUE_TARGET_MODE = "popart_return_norm"
FIXED_VALUE_LOSS_MODE = "huber"
FIXED_CRITIC_ARCH_MODE = "stronger_critic_backbone"
FIXED_CRITIC_INPUT_MODE = "normalized_augmented_critic_input"
FIXED_ACTOR_INPUT_MODE = "normalized_augmented_actor_input"
FIXED_ACTOR_RAW_INPUT_MODE = "baseline_actor_raw"

FIXED_ACTOR_LEARNING_RATE = 5e-4
FIXED_CRITIC_LEARNING_RATE = 5e-4
FIXED_ENTROPY_COEFF = 1e-3
FIXED_VALUE_COEFF = 0.5
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
DEFAULT_POLICY_RATIO_MODE = "current_joint_sum_ratio"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense mainline policy-update diagnosis with fixed actor/critic settings."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FIXED_SEED,
        help="Fixed seed for this controlled diagnosis run.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="checkpoints",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--policy-ratio-mode",
        type=str,
        default=DEFAULT_POLICY_RATIO_MODE,
        choices=[
            "current_joint_sum_ratio",
            "mean_logprob_ratio",
            "block_mean_ratio",
            "blockwise_surrogate_mean",
            "blockwise_weighted_surrogate_mean",
            "blockwise_scaled_advantage_surrogate_mean",
            "blockwise_value_scaled_advantage_surrogate_mean",
            "blockwise_td_advantage_surrogate_mean",
            "blockwise_delta_cost_td_advantage_surrogate_mean",
            "blockwise_action_conditioned_td_advantage_surrogate_mean",
            "blockwise_action_conditioned_path_value_td_advantage_surrogate_mean",
            "blockwise_path_specific_td_supervised_advantage_surrogate_mean",
            "theta_route_split_advantage_surrogate",
            "blockwise_pseudolocal_advantage_surrogate_mean",
        ],
        help="Policy ratio aggregation mode to use.",
    )
    return parser.parse_args()


def build_policy_update_diagnosis_config(seed: int, policy_ratio_mode: str) -> MECConfig:
    config = build_config(topology_mode=FIXED_TOPOLOGY)
    training = replace(
        config.training,
        num_epochs=FIXED_MAX_EPOCHS,
        time_steps=FIXED_TIME_STEPS,
        seed=seed,
    )
    ppo = replace(
        config.ppo,
        critic_arch_mode=FIXED_CRITIC_ARCH_MODE,
        critic_input_mode=FIXED_CRITIC_INPUT_MODE,
        actor_input_mode=FIXED_ACTOR_INPUT_MODE,
        actor_raw_input_mode=FIXED_ACTOR_RAW_INPUT_MODE,
        actor_learning_rate=FIXED_ACTOR_LEARNING_RATE,
        critic_learning_rate=FIXED_CRITIC_LEARNING_RATE,
        entropy_coeff=FIXED_ENTROPY_COEFF,
        value_coeff=FIXED_VALUE_COEFF,
        value_target_mode=FIXED_VALUE_TARGET_MODE,
        value_loss_mode=FIXED_VALUE_LOSS_MODE,
        update_epochs=FIXED_UPDATE_EPOCHS,
        policy_update_diagnosis_enabled=True,
        policy_ratio_mode=policy_ratio_mode,
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
        reward_mode=FIXED_REWARD_MODE,
        noma_quantile=FIXED_NOMA_QUANTILE,
        max_cluster_size=FIXED_MAX_CLUSTER_SIZE,
        alpha_p=FIXED_ALPHA_P,
        rho0=FIXED_RHO0,
        gamma_q=FIXED_GAMMA_Q,
        gamma_e=FIXED_GAMMA_E,
    )
    return replace(config, training=training, ppo=ppo, dt=dt, system=system)


def compute_bucket_slope(run_dir: Path) -> float:
    payload = pd.read_csv(run_dir / "target_bucket_prediction_mean.csv")
    if len(payload) < 2:
        return 0.0
    slope, _ = np.polyfit(
        payload["bucket_target_mean"].to_numpy(),
        payload["bucket_prediction_mean"].to_numpy(),
        deg=1,
    )
    return float(slope)


def run_experiment(seed: int, output_root: Path, policy_ratio_mode: str) -> Path:
    root_dir = output_root / datetime.now().strftime("dense_policy_update_diagnosis_%Y%m%d_%H%M%S")
    root_dir.mkdir(parents=True, exist_ok=True)

    config = build_policy_update_diagnosis_config(seed=seed, policy_ratio_mode=policy_ratio_mode)
    state_layout = build_state_layout(config)
    probe_states = collect_probe_states(config=config, seed=seed, probe_count=PROBE_STATE_COUNT)
    np.save(root_dir / "probe_states.npy", probe_states)

    early_stopping = EarlyStoppingConfig(
        patience=FIXED_PATIENCE,
        min_delta=FIXED_MIN_DELTA,
        monitor_start_epoch=FIXED_START_CHECK_EPOCH,
    )

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root_dir / f"policy_update_diagnosis_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    agent_preview = PPOAgent(
        config=config.ppo,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        critic_state_layout=state_layout,
    )
    (run_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "training": asdict(config.training),
                "ppo": asdict(config.ppo),
                "dt": asdict(config.dt),
                "system": asdict(config.system),
                "critic_state_layout": state_layout,
                "actor_raw_layout": agent_preview.describe_actor_raw_layout(),
                "early_stopping": asdict(early_stopping),
                "probe_state_count": PROBE_STATE_COUNT,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[diagnose] dense policy update diagnosis -> {run_dir}")
    set_global_seeds(seed)
    train_with_diagnosis(
        config=config,
        checkpoint_dir=str(run_dir),
        early_stopping=early_stopping,
        probe_states=probe_states,
        agent_kwargs={"critic_state_layout": state_layout},
    )
    plot_single_run_outputs(run_dir)
    train_summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    final_epoch = int(train_summary.get("final_epoch", -1))
    final_probe_policy_payload = (
        run_dir / f"probe_policy_update_payload_epoch_{final_epoch:02d}.csv"
        if final_epoch >= 0
        else run_dir / "probe_policy_update_payload_epoch_00.csv"
    )

    metrics = extract_metrics(run_dir=run_dir)
    metrics["final_critic_backbone_to_head_grad_ratio"] = float(
        metrics["final_critic_backbone_grad_norm"]
    ) / (float(metrics["final_critic_head_grad_norm"]) + 1e-8)
    metrics["final_target_bucket_prediction_slope"] = compute_bucket_slope(run_dir)
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_df = pd.DataFrame([metrics])
    summary_csv = root_dir / "policy_update_diagnosis_summary.csv"
    summary_json = root_dir / "policy_update_diagnosis_summary.json"
    summary_md = root_dir / "policy_update_diagnosis_summary.md"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(dataframe_to_markdown(summary_df), encoding="utf-8")

    plots_dir = run_dir / "plots"
    manifest = {
        "root_dir": str(root_dir),
        "run_dir": str(run_dir),
        "fixed_params": {
            "seed": seed,
            "topology": FIXED_TOPOLOGY,
            "reward_mode": FIXED_REWARD_MODE,
            "value_target_mode": FIXED_VALUE_TARGET_MODE,
            "value_loss_mode": FIXED_VALUE_LOSS_MODE,
            "critic_arch_mode": FIXED_CRITIC_ARCH_MODE,
            "critic_input_mode": FIXED_CRITIC_INPUT_MODE,
            "actor_input_mode": FIXED_ACTOR_INPUT_MODE,
            "actor_raw_input_mode": FIXED_ACTOR_RAW_INPUT_MODE,
            "policy_ratio_mode": policy_ratio_mode,
            "actor_learning_rate": FIXED_ACTOR_LEARNING_RATE,
            "critic_learning_rate": FIXED_CRITIC_LEARNING_RATE,
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
            "probe_state_count": PROBE_STATE_COUNT,
        },
        "summary": {
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
        "diagnostic_files": {
            "train_summary": str(run_dir / "train_summary.json"),
            "best_model_info": str(run_dir / "best_model_info.json"),
            "train_logs": str(run_dir / "train_logs.csv"),
            "best_model": str(run_dir / "best_model.pt"),
            "experiment_config": str(run_dir / "experiment_config.json"),
            "analysis_summary": str(run_dir / "analysis_summary.json"),
            "advantage_alignment_by_epoch": str(run_dir / "advantage_alignment_by_epoch.csv"),
            "ratio_clip_by_epoch": str(run_dir / "ratio_clip_by_epoch.csv"),
            "advantage_bucket_update_stats": str(run_dir / "advantage_bucket_update_stats.csv"),
            "final_probe_policy_payload": str(final_probe_policy_payload),
        },
        "plots": {
            "reward_curve": str(plots_dir / "reward_curve.png"),
            "policy_loss_curve": str(plots_dir / "policy_loss_curve.png"),
            "approx_kl_curve": str(plots_dir / "approx_kl_curve.png"),
            "clip_fraction_curve": str(plots_dir / "clip_fraction_curve.png"),
            "ratio_stats_curve": str(plots_dir / "ratio_stats_curve.png"),
            "advantage_stats_curve": str(plots_dir / "advantage_stats_curve.png"),
            "advantage_alignment_curve": str(plots_dir / "advantage_alignment_curve.png"),
            "advantage_bucket_update_compare": str(
                plots_dir / "advantage_bucket_update_compare.png"
            ),
            "selected_action_prob_gain_by_adv_bucket": str(
                plots_dir / "selected_action_prob_gain_by_adv_bucket.png"
            ),
            "probe_policy_kl_curve": str(plots_dir / "probe_policy_kl_curve.png"),
            "selected_action_change_rate_curve": str(
                plots_dir / "selected_action_change_rate_curve.png"
            ),
            "critic_loss_curve": str(plots_dir / "critic_loss_curve.png"),
            "value_explained_variance_curve": str(plots_dir / "value_explained_variance_curve.png"),
            "prediction_target_corr_curve": str(plots_dir / "prediction_target_corr_curve.png"),
            "prediction_std_over_target_std_curve": str(
                plots_dir / "prediction_std_over_target_std_curve.png"
            ),
            "critic_grad_norm_curve": str(plots_dir / "critic_grad_norm_curve.png"),
            "target_bucket_prediction_mean": str(plots_dir / "target_bucket_prediction_mean.png"),
        },
    }
    (root_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root_dir


def main() -> None:
    args = parse_args()
    root_dir = run_experiment(
        seed=args.seed,
        output_root=Path(args.output_root),
        policy_ratio_mode=args.policy_ratio_mode,
    )
    print(f"Dense policy update diagnosis completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
