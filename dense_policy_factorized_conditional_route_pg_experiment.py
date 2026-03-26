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
from dense_policy_hierarchical_hard_mask_experiment import plot_route_mask_compare
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
FIXED_ROUTE_MASK_THRESHOLD = 0.5
DEFAULT_POLICY_RATIO_MODES = (
    "hierarchical_actor_factorized_ratio_pg",
    "hierarchical_actor_factorized_conditional_route_pg",
)
FACTORIZED_CONDITIONAL_ROUTE_MODE_DEFINITIONS = {
    "hierarchical_actor_factorized_ratio_pg": (
        "Control group: separate theta / route backbones, hard route mask, branchwise normalized "
        "advantages, alternating theta-first then route-second updates, and fully factorized theta / "
        "route PPO objectives on the raw Gaussian action coordinates."
    ),
    "hierarchical_actor_factorized_conditional_route_pg": (
        "Test group: keep the same structure, advantages, alternating updates, and theta branch "
        "objective, but replace the route branch log-prob with a semantic conditional route-margin "
        "log-prob so route PPO updates track BS1-vs-BS2 conditional preference rather than the raw "
        "two-logit Gaussian coordinates."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled dense comparison for factorized-ratio PG vs factorized conditional-route PG."
        )
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
            ("offload_rate_mean", "Offload Rate Mean"),
            ("bs1_vs_bs2_entropy", "BS1 vs BS2 Entropy"),
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
            ("theta_route_feature_correlation", "Theta / Route Feature Correlation"),
        ],
        output_path=output_path,
    )


def plot_theta_route_head_grad_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("theta_head_grad_norm", "Theta Head Grad Norm"),
            ("route_head_grad_norm", "Route Head Grad Norm"),
            ("theta_route_head_correlation", "Theta / Route Head Correlation"),
        ],
        output_path=output_path,
    )


def plot_theta_route_backbone_grad_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("theta_backbone_grad_norm", "Theta Backbone Grad Norm"),
            ("route_backbone_grad_norm", "Route Backbone Grad Norm"),
            ("theta_route_feature_correlation", "Theta / Route Feature Correlation"),
        ],
        output_path=output_path,
    )


def plot_factorized_ratio_stats_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("theta_ratio_mean", "Theta Ratio Mean"),
            ("theta_ratio_std", "Theta Ratio Std"),
            ("theta_ratio_max", "Theta Ratio Max"),
            ("route_ratio_mean", "Route Ratio Mean"),
            ("route_ratio_std", "Route Ratio Std"),
            ("route_ratio_max", "Route Ratio Max"),
            ("theta_approx_kl", "Theta Approx KL"),
            ("route_approx_kl", "Route Approx KL"),
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
        title="Reward Curve by Factorized Conditional-Route PG Mode",
        ylabel="episode_reward",
        output_path=plots_dir / "reward_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="policy_loss",
        title="Policy Loss by Factorized Conditional-Route PG Mode",
        ylabel="policy_loss",
        output_path=plots_dir / "policy_loss_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="clip_fraction",
        title="Clip Fraction by Factorized Conditional-Route PG Mode",
        ylabel="clip_fraction",
        output_path=plots_dir / "clip_fraction_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="approx_kl",
        title="Approx KL by Factorized Conditional-Route PG Mode",
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
    plot_theta_route_head_grad_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "theta_route_head_grad_curve_compare.png",
    )
    plot_theta_route_backbone_grad_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "theta_route_backbone_grad_curve_compare.png",
    )
    plot_factorized_ratio_stats_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "factorized_ratio_stats_curve_compare.png",
    )
    plot_route_mask_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "route_mask_curve_compare.png",
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
            ("target_bucket_prediction_slope", "Target-Bucket Prediction Slope"),
        ],
        output_path=plots_dir / "critic_health_compare.png",
    )


def run_experiment(policy_ratio_modes: list[str], seed: int, output_root: Path) -> Path:
    root_dir = output_root / datetime.now().strftime(
        "dense_policy_factorized_conditional_route_pg_experiment_%Y%m%d_%H%M%S"
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
            "factorized_conditional_route_mode_definitions": FACTORIZED_CONDITIONAL_ROUTE_MODE_DEFINITIONS,
            "route_mask_threshold": FIXED_ROUTE_MASK_THRESHOLD,
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
                    "actor_structure": agent_preview.describe_actor_structure(),
                    "factorized_conditional_route_mode_definition": FACTORIZED_CONDITIONAL_ROUTE_MODE_DEFINITIONS.get(
                        policy_ratio_mode,
                        "",
                    ),
                    "factorized_conditional_route_mode_definitions": FACTORIZED_CONDITIONAL_ROUTE_MODE_DEFINITIONS,
                    "route_gate_definition": "route_gate_b = detach(theta_b)",
                    "route_mask_definition": (
                        f"route_mask_b = 1[detach(theta_b) > {FIXED_ROUTE_MASK_THRESHOLD}]"
                    ),
                    "theta_advantage_normalization_definition": (
                        "A_theta_norm = (A_theta - mean(A_theta_all_blocks)) / "
                        "(std(A_theta_all_blocks) + eps)"
                    ),
                    "route_advantage_normalization_definition": (
                        "A_route_norm = (A_route - mean(A_route_active_blocks)) / "
                        "(std(A_route_active_blocks) + eps); if active route count < 2 or std is "
                        "too small, fall back to unnormalized active A_route."
                    ),
                    "theta_old_new_logprob_definition": (
                        "theta_old_logprob_b = old log_prob(raw_theta_b); "
                        "theta_new_logprob_b = new log_prob(raw_theta_b)"
                    ),
                    "route_old_new_logprob_definition": (
                        "route_old_logprob_b = old log_prob(route_logit_BS1_b, route_logit_BS2_b); "
                        "route_new_logprob_b = new log_prob(route_logit_BS1_b, route_logit_BS2_b), "
                        "computed and summarized on active route blocks only"
                    ),
                    "conditional_route_old_new_logprob_definition": (
                        "For hierarchical_actor_factorized_conditional_route_pg only: "
                        "route_margin_b = route_logit_BS1_b - route_logit_BS2_b; "
                        "route_old_logprob_b = old log_prob(route_margin_b); "
                        "route_new_logprob_b = new log_prob(route_margin_b)."
                    ),
                    "theta_ratio_definition": "theta_ratio_b = exp(theta_new_logprob_b - theta_old_logprob_b)",
                    "route_ratio_definition": "route_ratio_b = exp(route_new_logprob_b - route_old_logprob_b)",
                    "theta_surrogate_definition": (
                        "surrogate_theta_b = min(theta_ratio_b * A_theta_norm_b, "
                        "clip(theta_ratio_b) * A_theta_norm_b)"
                    ),
                    "route_surrogate_definition": (
                        "surrogate_route_b = min(route_ratio_b * A_route_norm_b, "
                        "clip(route_ratio_b) * A_route_norm_b), active route blocks only"
                    ),
                    "theta_loss_definition": "theta_loss = -mean_b surrogate_theta_b over all blocks",
                    "route_loss_definition": (
                        "route_loss = -mean_b surrogate_route_b over active route blocks only"
                    ),
                    "actor_loss_definition": (
                        "actor_loss = 0.5 * theta_loss + 0.5 * route_loss for logging only; "
                        "backprop uses theta_loss in the theta-only step and route_loss in the "
                        "route-only step, without a joint actor surrogate."
                    ),
                    "theta_only_update_definition": (
                        "theta-only step optimizes theta PPO loss with theta branch parameters only."
                    ),
                    "route_only_update_definition": (
                        "route-only step optimizes route PPO loss with route branch parameters only "
                        "on the same rollout minibatch."
                    ),
                    "update_order_definition": "theta-first, then route-second within each PPO minibatch",
                    "rollout_reuse_definition": (
                        "Both branch steps reuse the same rollout batch and cached old branch log-probs."
                    ),
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
                "actor_structure_mode": metrics.get("actor_structure_mode", ""),
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
                "alternating_branch_pg_stats": str(
                    run_dir / "alternating_branch_pg_stats_by_epoch.csv"
                ),
                "factorized_ratio_pg_stats": str(
                    run_dir / "factorized_ratio_pg_stats_by_epoch.csv"
                ),
                "theta_route_split_stats": str(run_dir / "theta_route_split_stats_by_epoch.csv"),
            }
        )

    summary_df = pd.DataFrame(all_metrics)
    summary_csv = root_dir / "policy_factorized_conditional_route_pg_summary.csv"
    summary_json = root_dir / "policy_factorized_conditional_route_pg_summary.json"
    summary_md = root_dir / "policy_factorized_conditional_route_pg_summary.md"
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
        "Dense policy factorized-conditional-route PG experiment completed. "
        f"Results saved to: {root_dir}"
    )


if __name__ == "__main__":
    main()
