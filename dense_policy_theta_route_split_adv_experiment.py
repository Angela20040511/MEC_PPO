import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from dense_actor_input_experiment import build_state_layout
from dense_critic_diagnosis import (
    PROBE_STATE_COUNT,
    collect_probe_states,
    dataframe_to_markdown,
    extract_metrics,
    plot_single_run_outputs,
    train_with_diagnosis,
)
from dense_policy_ratio_mode_experiment import (
    build_action_block_layout,
    build_policy_ratio_mode_config,
    compute_bucket_slope,
    load_mode_bucket_stats,
    load_mode_logs,
    plot_block_clip_fraction_compare,
    plot_block_logprob_scale_compare,
    plot_block_surrogate_scale_compare,
    plot_bucket_metric_compare,
    plot_logprob_scale_compare,
    plot_metric_compare,
    plot_two_panel_compare,
)
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds
from rl.ppo_agent import PPOAgent

FIXED_SEED = 2025
FIXED_START_CHECK_EPOCH = 1
FIXED_PATIENCE = 3
FIXED_MIN_DELTA = 1.0
DEFAULT_POLICY_RATIO_MODES = (
    "blockwise_action_conditioned_path_value_td_advantage_surrogate_mean",
    "theta_route_split_advantage_surrogate",
)
THETA_ROUTE_SPLIT_MODE_DEFINITIONS = {
    "blockwise_action_conditioned_path_value_td_advantage_surrogate_mean": (
        "Each block still uses one scalar A_b to drive [theta_b, route_logit_BS1, route_logit_BS2] jointly."
    ),
    "theta_route_split_advantage_surrogate": (
        "A_theta_b = [r_offload_b + gamma * (1 - done) * V_offload_b(s_next) - V_offload_b(s)] "
        "- [r_local_b + gamma * (1 - done) * V_local_b(s_next) - V_local_b(s)]; "
        "A_route_b = [r_bs1_b + gamma * (1 - done) * V_bs1_b(s_next) - V_bs1_b(s)] "
        "- [r_bs2_b + gamma * (1 - done) * V_bs2_b(s_next) - V_bs2_b(s)]; "
        "theta ratio uses only the theta dimension, route ratio uses only the two route logits, "
        "then block surrogate = 0.5 * (surrogate_theta + surrogate_route)."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled dense comparison for path-value block advantage vs theta-route split advantage."
    )
    parser.add_argument(
        "--policy-ratio-modes",
        nargs="+",
        default=list(DEFAULT_POLICY_RATIO_MODES),
        help="Policy surrogate modes to compare.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FIXED_SEED,
        help="Fixed seed for this controlled experiment.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="checkpoints",
        help="Root directory for experiment outputs.",
    )
    return parser.parse_args()


def plot_theta_route_alignment_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("advantage_action_alignment", "Overall Advantage Alignment"),
            ("theta_advantage_alignment", "Theta Advantage Alignment"),
            ("route_advantage_alignment", "Route Advantage Alignment"),
        ],
        output_path=output_path,
    )


def plot_theta_route_prob_gain_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("theta_selected_action_prob_gain", "Theta Selected Action Prob Gain"),
            ("route_selected_action_prob_gain", "Route Selected Action Prob Gain"),
            ("negative_advantage_action_prob_gain", "Overall Negative-Adv Prob Gain"),
            ("theta_negative_adv_prob_gain", "Theta Negative-Adv Prob Gain"),
            ("route_negative_adv_prob_gain", "Route Negative-Adv Prob Gain"),
            ("offload_vs_local_value_gap_mean", "Offload vs Local Value Gap Mean"),
        ],
        output_path=output_path,
    )


def plot_theta_route_clip_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("clip_fraction", "Overall Clip Fraction"),
            ("theta_clip_fraction", "Theta Clip Fraction"),
            ("route_clip_fraction", "Route Clip Fraction"),
            ("positive_adv_clip_fraction", "Positive-Adv Clip Fraction"),
            ("negative_adv_clip_fraction", "Negative-Adv Clip Fraction"),
            ("bs1_vs_bs2_value_gap_mean", "BS1 vs BS2 Value Gap Mean"),
        ],
        output_path=output_path,
    )


def plot_comparisons(root_dir: Path, summary_df: pd.DataFrame) -> None:
    plots_dir = root_dir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    mode_logs = load_mode_logs(summary_df)
    mode_bucket_stats = load_mode_bucket_stats(summary_df)

    plot_metric_compare(
        mode_logs,
        column="episode_reward",
        title="Reward Curve by Policy Surrogate Mode",
        ylabel="episode_reward",
        output_path=plots_dir / "reward_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="policy_loss",
        title="Policy Loss by Policy Surrogate Mode",
        ylabel="policy_loss",
        output_path=plots_dir / "policy_loss_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="clip_fraction",
        title="Clip Fraction by Policy Surrogate Mode",
        ylabel="clip_fraction",
        output_path=plots_dir / "clip_fraction_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="approx_kl",
        title="Approx KL by Policy Surrogate Mode",
        ylabel="approx_kl",
        output_path=plots_dir / "approx_kl_curve_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("ratio_mean", "Ratio Mean"),
            ("ratio_std", "Ratio Std"),
            ("ratio_min", "Ratio Min"),
            ("ratio_max", "Ratio Max"),
        ],
        output_path=plots_dir / "ratio_stats_curve_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("advantage_action_alignment", "Advantage / Log-Prob Alignment"),
            ("delta_log_prob_selected_action_mean", "Selected-Action Log-Prob Delta Mean"),
            ("delta_log_prob_selected_action_std", "Selected-Action Log-Prob Delta Std"),
        ],
        output_path=plots_dir / "advantage_alignment_curve_compare.png",
    )
    plot_bucket_metric_compare(
        mode_bucket_stats=mode_bucket_stats,
        value_column="bucket_mean_selected_action_prob_gain",
        title="Selected Action Prob Gain by Advantage Bucket",
        ylabel="bucket_mean_selected_action_prob_gain",
        output_path=plots_dir / "selected_action_prob_gain_by_adv_bucket_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("probe_policy_pairwise_kl_mean", "Probe Policy Pairwise KL Mean"),
            ("probe_policy_pairwise_l1_mean", "Probe Policy Pairwise L1 Mean"),
        ],
        output_path=plots_dir / "probe_policy_kl_curve_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("selected_action_change_rate", "Selected Action Change Rate"),
            ("top1_action_change_rate", "Top-1 Action Change Rate"),
            ("top1_prob_delta_mean", "Top-1 Probability Delta Mean"),
        ],
        output_path=plots_dir / "selected_action_change_rate_curve_compare.png",
    )
    plot_theta_route_alignment_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "theta_route_alignment_curve_compare.png",
    )
    plot_theta_route_prob_gain_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "theta_route_prob_gain_curve_compare.png",
    )
    plot_theta_route_clip_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "theta_route_clip_curve_compare.png",
    )
    plot_logprob_scale_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "logprob_scale_curve_compare.png",
    )
    plot_block_logprob_scale_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "block_logprob_scale_curve_compare.png",
    )
    plot_block_clip_fraction_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "block_clip_fraction_curve_compare.png",
    )
    plot_block_surrogate_scale_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "block_surrogate_scale_curve_compare.png",
    )
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("value_explained_variance", "Value Explained Variance"),
            ("prediction_target_corr", "Prediction Target Correlation"),
            ("prediction_std_over_target_std", "Prediction Std / Target Std"),
        ],
        output_path=plots_dir / "critic_health_compare.png",
    )


def run_experiment(policy_ratio_modes: list[str], seed: int, output_root: Path) -> Path:
    root_dir = output_root / datetime.now().strftime(
        "dense_policy_theta_route_split_adv_experiment_%Y%m%d_%H%M%S"
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStoppingConfig(
        patience=FIXED_PATIENCE,
        min_delta=FIXED_MIN_DELTA,
        monitor_start_epoch=FIXED_START_CHECK_EPOCH,
    )
    probe_config = build_policy_ratio_mode_config(
        policy_ratio_mode=policy_ratio_modes[0],
        seed=seed,
    )
    probe_states = collect_probe_states(
        config=probe_config,
        seed=seed,
        probe_count=PROBE_STATE_COUNT,
    )
    np.save(root_dir / "probe_states.npy", probe_states)

    manifest: dict[str, object] = {
        "root_dir": str(root_dir),
        "fixed_params": {
            "seed": seed,
            "probe_state_count": PROBE_STATE_COUNT,
            "theta_route_split_mode_definitions": THETA_ROUTE_SPLIT_MODE_DEFINITIONS,
            "start_check_epoch": FIXED_START_CHECK_EPOCH,
            "patience": FIXED_PATIENCE,
            "min_delta": FIXED_MIN_DELTA,
        },
        "policy_ratio_modes": policy_ratio_modes,
        "runs": [],
    }
    all_metrics: list[dict[str, float | int | str]] = []

    for policy_ratio_mode in policy_ratio_modes:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root_dir / f"policy_ratio_{policy_ratio_mode}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = build_policy_ratio_mode_config(
            policy_ratio_mode=policy_ratio_mode,
            seed=seed,
        )
        state_layout = build_state_layout(config)
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
                    "action_block_layout": build_action_block_layout(config),
                    "theta_route_split_mode_definition": THETA_ROUTE_SPLIT_MODE_DEFINITIONS.get(
                        policy_ratio_mode,
                        "",
                    ),
                    "theta_route_split_mode_definitions": THETA_ROUTE_SPLIT_MODE_DEFINITIONS,
                    "actor_raw_layout": agent_preview.describe_actor_raw_layout(),
                    "early_stopping": asdict(early_stopping),
                    "probe_state_count": PROBE_STATE_COUNT,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[train] policy_ratio_mode={policy_ratio_mode}, seed={seed} -> {run_dir}")
        set_global_seeds(seed)
        train_with_diagnosis(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
            probe_states=probe_states,
            agent_kwargs={"critic_state_layout": state_layout},
        )
        plot_single_run_outputs(run_dir)

        metrics = extract_metrics(run_dir=run_dir)
        metrics["final_critic_backbone_to_head_grad_ratio"] = float(
            metrics["final_critic_backbone_grad_norm"]
        ) / (float(metrics["final_critic_head_grad_norm"]) + 1e-8)
        metrics["final_target_bucket_prediction_slope"] = compute_bucket_slope(run_dir)
        (run_dir / "analysis_summary.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_metrics.append(metrics)
        manifest["runs"].append(
            {
                "policy_ratio_mode": policy_ratio_mode,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
                "theta_route_split_stats": str(run_dir / "theta_route_split_stats_by_epoch.csv"),
            }
        )

    summary_df = pd.DataFrame(all_metrics)
    summary_csv = root_dir / "policy_theta_route_split_adv_summary.csv"
    summary_json = root_dir / "policy_theta_route_split_adv_summary.json"
    summary_md = root_dir / "policy_theta_route_split_adv_summary.md"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(dataframe_to_markdown(summary_df), encoding="utf-8")

    plot_comparisons(root_dir=root_dir, summary_df=summary_df)

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
    root_dir = run_experiment(
        policy_ratio_modes=args.policy_ratio_modes,
        seed=args.seed,
        output_root=Path(args.output_root),
    )
    print(
        "Dense policy theta-route split advantage experiment completed. "
        f"Results saved to: {root_dir}"
    )


if __name__ == "__main__":
    main()
