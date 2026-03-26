import argparse
import json
from dataclasses import asdict, replace
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
FIXED_THETA_KL_TARGET = 0.03
FIXED_ROUTE_KL_TARGET = 0.015
FIXED_COUPLED_STOP_MIN_THETA_UPDATES_PER_EPOCH = 1
FIXED_ROUTE_CONFIDENT_MASK_MARGIN = 0.02
FIXED_ROUTE_UPDATE_CAP_PER_EPOCH = 2
FIXED_ROUTE_ALIGNMENT_GATE_THRESHOLD = 0.015
FIXED_ROUTE_STEP_ALIGNMENT_GATE_THRESHOLD = 0.0
FIXED_COUPLED_STOP_SEVERITY_FACTOR = 1.5
FIXED_COUPLED_STOP_WARMUP_EPOCHS = 1
DEFAULT_POLICY_RATIO_MODES = (
    "hierarchical_actor_factorized_ratio_pg",
    "hierarchical_actor_factorized_trust_region_pg",
)
FACTORIZED_TRUST_REGION_MODE_DEFINITIONS = {
    "hierarchical_actor_factorized_ratio_pg": (
        "Control group: fully factorized theta / route PPO objective with separate backbones, hard "
        "route mask, branchwise normalized advantages, and alternating theta-first then "
        "route-second updates, but without branch-specific KL early stop."
    ),
    "hierarchical_actor_factorized_trust_region_pg": (
        "Test group: keep the same fully factorized theta / route PPO objective and alternating "
        "theta-first then route-second updates, while adding independent theta / route KL targets "
        "with branch-specific early stop inside each PPO epoch."
    ),
    "hierarchical_actor_factorized_trust_region_pg_route_first": (
        "Test group (route-first): keep the same fully factorized theta / route PPO objective and "
        "branch-specific KL early stop, but switch alternating order to route-first then "
        "theta-second within each PPO minibatch."
    ),
    "hierarchical_actor_factorized_trust_region_pg_coupled_stop": (
        "Test group (coupled-stop): keep theta-first alternating order and branch-specific KL "
        "early stop, and additionally freeze theta updates for the rest of the current PPO epoch "
        "once route KL early-stop is triggered."
    ),
    "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop": (
        "Test group (route-first + coupled-stop): keep fully factorized branch objectives and "
        "branch-specific KL early stop, use route-first then theta-second order, and freeze theta "
        "for the rest of the PPO epoch after route KL early-stop triggers."
    ),
    "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor": (
        "Test group (route-first + coupled-stop + theta-floor): same as route-first + coupled-stop, "
        "but coupled-stop can freeze theta only after at least a minimum number of theta updates "
        "have been completed within the current PPO epoch."
    ),
    "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask": (
        "Test group (route-first + coupled-stop + route-confident-mask): same as route-first + "
        "coupled-stop, but route branch updates only use high-confidence routing blocks with "
        "theta > 0.5 + margin during training."
    ),
    "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap": (
        "Test group (route-first + coupled-stop + route-cap): same as route-first + coupled-stop, "
        "but route branch updates are capped to a fixed maximum number per PPO epoch."
    ),
    "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise": (
        "Test group (route-first + coupled-stop + epochwise-branch-order): same as route-first + "
        "coupled-stop, but alternating branch updates are scheduled by PPO epoch phases "
        "(route-only first half epochs, theta-only second half epochs)."
    ),
    "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate": (
        "Test group (route-first + coupled-stop + severity-gate): same as route-first + "
        "coupled-stop, but theta freezing is triggered only when route KL exceeds "
        "route_kl_target * coupled_stop_severity_factor."
    ),
    "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate": (
        "Test group (route-first + coupled-stop + alignment-gate): same as route-first + "
        "coupled-stop, but each route step is accepted only if minibatch route-step alignment "
        "meets a configured threshold; otherwise that route step is rolled back."
    ),
    "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate": (
        "Test group (route-first + coupled-stop + severity+alignment freeze gate): same as "
        "route-first + coupled-stop + severity-gate, but theta freezing additionally requires "
        "the current minibatch route-step alignment to be below route_alignment_gate_threshold."
    ),
    "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1": (
        "Test group (route-first + coupled-stop + severity-gate + warmup1): same as "
        "route-first + coupled-stop + severity-gate, but coupled-stop theta freezing is disabled "
        "for the first outer training epoch and activated from epoch 2 onward."
    ),
    "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv": (
        "Test group (route-first + coupled-stop + route-counterfactual-adv): same as route-first "
        "+ coupled-stop, but route branch uses selected-route minus mean-remote-path TD advantage "
        "as the route training signal."
    ),
    "hierarchical_actor_route_step_alignment_gate": (
        "Test group (route-step-alignment-gate): same as route-first + coupled-stop + severity-gate, "
        "while additionally applying step-level route alignment quality control. After each route-only "
        "step, compute minibatch route-step alignment on active route blocks and rollback that route "
        "update when alignment is below threshold."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled dense comparison for factorized-ratio PG vs factorized trust-region PG."
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


def plot_branch_kl_early_stop_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("theta_approx_kl", "Theta Approx KL"),
            ("route_approx_kl", "Route Approx KL"),
            ("theta_kl_target", "Theta KL Target"),
            ("route_kl_target", "Route KL Target"),
            ("theta_early_stop_count", "Theta Early Stop Count"),
            ("route_early_stop_count", "Route Early Stop Count"),
            ("theta_update_count", "Theta Update Count"),
            ("route_update_count", "Route Update Count"),
            ("coupled_stop_trigger_count", "Coupled Stop Trigger Count"),
            (
                "coupled_stop_blocked_by_theta_floor_count",
                "Coupled Stop Blocked By Theta Floor Count",
            ),
            (
                "coupled_stop_blocked_by_severity_gate_count",
                "Coupled Stop Blocked By Severity Gate Count",
            ),
            (
                "coupled_stop_min_theta_updates_per_epoch",
                "Coupled Stop Min Theta Updates / Epoch",
            ),
            ("coupled_stop_severity_factor", "Coupled Stop Severity Factor"),
            (
                "coupled_stop_severity_freeze_kl_threshold",
                "Coupled Stop Severity Freeze KL Threshold",
            ),
            ("route_update_cap_trigger_count", "Route Update Cap Trigger Count"),
            ("route_update_cap_per_epoch", "Route Update Cap / Epoch"),
            ("route_alignment_gate_threshold", "Route Alignment Gate Threshold"),
            ("route_alignment_gate_reject_rate", "Route Alignment Gate Reject Rate"),
            ("route_alignment_gate_score_mean", "Route Alignment Gate Score Mean"),
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
        title="Reward Curve by Factorized Trust-Region PG Mode",
        ylabel="episode_reward",
        output_path=plots_dir / "reward_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="policy_loss",
        title="Policy Loss by Factorized Trust-Region PG Mode",
        ylabel="policy_loss",
        output_path=plots_dir / "policy_loss_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="clip_fraction",
        title="Clip Fraction by Factorized Trust-Region PG Mode",
        ylabel="clip_fraction",
        output_path=plots_dir / "clip_fraction_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="approx_kl",
        title="Approx KL by Factorized Trust-Region PG Mode",
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
    plot_branch_kl_early_stop_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "branch_kl_early_stop_curve_compare.png",
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
        "dense_policy_factorized_trust_region_pg_experiment_%Y%m%d_%H%M%S"
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
            "factorized_trust_region_mode_definitions": FACTORIZED_TRUST_REGION_MODE_DEFINITIONS,
            "route_mask_threshold": FIXED_ROUTE_MASK_THRESHOLD,
            "theta_kl_target": FIXED_THETA_KL_TARGET,
            "route_kl_target": FIXED_ROUTE_KL_TARGET,
            "coupled_stop_min_theta_updates_per_epoch": (
                FIXED_COUPLED_STOP_MIN_THETA_UPDATES_PER_EPOCH
            ),
            "coupled_stop_severity_factor": FIXED_COUPLED_STOP_SEVERITY_FACTOR,
            "coupled_stop_warmup_epochs": FIXED_COUPLED_STOP_WARMUP_EPOCHS,
            "route_confident_mask_margin": FIXED_ROUTE_CONFIDENT_MASK_MARGIN,
            "route_update_cap_per_epoch": FIXED_ROUTE_UPDATE_CAP_PER_EPOCH,
            "route_alignment_gate_threshold": FIXED_ROUTE_ALIGNMENT_GATE_THRESHOLD,
            "route_step_alignment_gate_threshold": FIXED_ROUTE_STEP_ALIGNMENT_GATE_THRESHOLD,
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
        config = replace(
            config,
            ppo=replace(
                config.ppo,
                theta_kl_target=FIXED_THETA_KL_TARGET,
                route_kl_target=FIXED_ROUTE_KL_TARGET,
                coupled_stop_severity_factor=(
                    FIXED_COUPLED_STOP_SEVERITY_FACTOR
                    if policy_ratio_mode
                    in {
                        "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
                        "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate",
                        "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1",
                        "hierarchical_actor_route_step_alignment_gate",
                    }
                    else 1.0
                ),
                coupled_stop_warmup_epochs=(
                    FIXED_COUPLED_STOP_WARMUP_EPOCHS
                    if policy_ratio_mode
                    == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1"
                    else 0
                ),
                coupled_stop_min_theta_updates_per_epoch=(
                    FIXED_COUPLED_STOP_MIN_THETA_UPDATES_PER_EPOCH
                    if policy_ratio_mode
                    == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor"
                    else 0
                ),
                route_confident_mask_margin=(
                    FIXED_ROUTE_CONFIDENT_MASK_MARGIN
                    if policy_ratio_mode
                    == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask"
                    else 0.0
                ),
                route_update_cap_per_epoch=(
                    FIXED_ROUTE_UPDATE_CAP_PER_EPOCH
                    if policy_ratio_mode
                    == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap"
                    else 0
                ),
                route_alignment_gate_threshold=(
                    FIXED_ROUTE_STEP_ALIGNMENT_GATE_THRESHOLD
                    if policy_ratio_mode == "hierarchical_actor_route_step_alignment_gate"
                    else (
                        FIXED_ROUTE_ALIGNMENT_GATE_THRESHOLD
                        if policy_ratio_mode
                        in {
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate",
                        }
                        else 0.0
                    )
                ),
            ),
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
                    "factorized_trust_region_mode_definition": FACTORIZED_TRUST_REGION_MODE_DEFINITIONS.get(
                        policy_ratio_mode,
                        "",
                    ),
                    "factorized_trust_region_mode_definitions": FACTORIZED_TRUST_REGION_MODE_DEFINITIONS,
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
                    "theta_approx_kl_definition": (
                        "theta_approx_kl = mean_b(old_theta_logprob_b - new_theta_logprob_b)"
                    ),
                    "route_approx_kl_definition": (
                        "route_approx_kl = mean_active_b(old_route_logprob_b - new_route_logprob_b)"
                    ),
                    "theta_kl_target": FIXED_THETA_KL_TARGET,
                    "route_kl_target": FIXED_ROUTE_KL_TARGET,
                    "theta_early_stop_definition": (
                        "Per-minibatch post-step KL early stop. If post-step theta_approx_kl "
                        "exceeds theta_kl_target, remaining theta updates in the current PPO "
                        "epoch are skipped."
                    ),
                    "route_early_stop_definition": (
                        "Per-minibatch post-step KL early stop on active route blocks only. If "
                        "post-step route_approx_kl exceeds route_kl_target, remaining route "
                        "updates in the current PPO epoch are skipped."
                    ),
                    "trust_region_basis": (
                        "Independent per-minibatch KL against rollout-old branch log-probs; not "
                        "cumulative KL and not KL-penalty."
                    ),
                    "update_order_definition": (
                        "route-only first half PPO update epochs, theta-only second half PPO update epochs"
                        if policy_ratio_mode
                        == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise"
                        else (
                            "route-first, then theta-second within each PPO minibatch"
                            if policy_ratio_mode
                            in {
                                "hierarchical_actor_factorized_trust_region_pg_route_first",
                                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop",
                                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor",
                                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask",
                                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap",
                                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
                                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate",
                                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1",
                                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
                                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv",
                                "hierarchical_actor_route_step_alignment_gate",
                            }
                            else "theta-first, then route-second within each PPO minibatch"
                        )
                    ),
                    "coupled_stop_definition": (
                        "If route branch KL early-stop is triggered, theta branch is also frozen for "
                        "the rest of the current PPO epoch."
                        if policy_ratio_mode
                        in {
                            "hierarchical_actor_factorized_trust_region_pg_coupled_stop",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv",
                            "hierarchical_actor_route_step_alignment_gate",
                        }
                        else "disabled"
                    ),
                    "theta_floor_definition": (
                        "When coupled-stop is enabled, theta freeze is allowed only after the current "
                        "PPO epoch has reached the configured minimum theta update count."
                        if policy_ratio_mode
                        == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor"
                        else "disabled"
                    ),
                    "coupled_stop_min_theta_updates_per_epoch": (
                        FIXED_COUPLED_STOP_MIN_THETA_UPDATES_PER_EPOCH
                        if policy_ratio_mode
                        == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor"
                        else 0
                    ),
                    "coupled_stop_severity_gate_definition": (
                        "When coupled-stop is enabled, theta freeze requires route_approx_kl to "
                        "exceed route_kl_target * coupled_stop_severity_factor."
                        if policy_ratio_mode
                        in {
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1",
                            "hierarchical_actor_route_step_alignment_gate",
                        }
                        else "disabled"
                    ),
                    "coupled_stop_severity_factor": (
                        FIXED_COUPLED_STOP_SEVERITY_FACTOR
                        if policy_ratio_mode
                        in {
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate",
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1",
                            "hierarchical_actor_route_step_alignment_gate",
                        }
                        else 1.0
                    ),
                    "route_confident_mask_definition": (
                        "Route branch update masks require theta > 0.5 + route_confident_mask_margin "
                        "during training."
                        if policy_ratio_mode
                        == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask"
                        else "disabled"
                    ),
                    "route_confident_mask_margin": (
                        FIXED_ROUTE_CONFIDENT_MASK_MARGIN
                        if policy_ratio_mode
                        == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask"
                        else 0.0
                    ),
                    "route_update_cap_definition": (
                        "Route branch updates are capped per PPO epoch; once cap is reached, route "
                        "updates for the rest of this PPO epoch are skipped."
                        if policy_ratio_mode
                        == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap"
                        else "disabled"
                    ),
                    "route_update_cap_per_epoch": (
                        FIXED_ROUTE_UPDATE_CAP_PER_EPOCH
                        if policy_ratio_mode
                        == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap"
                        else 0
                    ),
                    "route_alignment_gate_definition": (
                        "After each route step, compute minibatch route-step alignment and rollback "
                        "the route update when alignment is below route_alignment_gate_threshold."
                        if policy_ratio_mode
                        in {
                            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
                            "hierarchical_actor_route_step_alignment_gate",
                        }
                        else "disabled"
                    ),
                    "coupled_stop_alignment_freeze_gate_definition": (
                        "When coupled-stop and severity-gate conditions are met, theta freeze is "
                        "further gated by route-step alignment; freeze only when alignment is below "
                        "route_alignment_gate_threshold."
                        if policy_ratio_mode
                        == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate"
                        else "disabled"
                    ),
                    "route_alignment_gate_threshold": (
                        FIXED_ROUTE_STEP_ALIGNMENT_GATE_THRESHOLD
                        if policy_ratio_mode == "hierarchical_actor_route_step_alignment_gate"
                        else (
                            FIXED_ROUTE_ALIGNMENT_GATE_THRESHOLD
                            if policy_ratio_mode
                            in {
                                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
                                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate",
                            }
                            else 0.0
                        )
                    ),
                    "coupled_stop_warmup_definition": (
                        "Coupled-stop theta freeze is disabled during the first outer training epoch "
                        "and enabled afterward."
                        if policy_ratio_mode
                        == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1"
                        else "disabled"
                    ),
                    "coupled_stop_warmup_epochs": (
                        FIXED_COUPLED_STOP_WARMUP_EPOCHS
                        if policy_ratio_mode
                        == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1"
                        else 0
                    ),
                    "route_counterfactual_adv_definition": (
                        "Route branch advantage uses selected-remote-path TD advantage minus "
                        "mean remote-path TD advantage within each block."
                        if policy_ratio_mode
                        == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv"
                        else "disabled"
                    ),
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
                "factorized_trust_region_pg_stats": str(
                    run_dir / "factorized_trust_region_pg_stats_by_epoch.csv"
                ),
                "theta_route_split_stats": str(run_dir / "theta_route_split_stats_by_epoch.csv"),
            }
        )

    summary_df = pd.DataFrame(all_metrics)
    summary_csv = root_dir / "policy_factorized_trust_region_pg_summary.csv"
    summary_json = root_dir / "policy_factorized_trust_region_pg_summary.json"
    summary_md = root_dir / "policy_factorized_trust_region_pg_summary.md"
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
        "Dense policy factorized-trust-region PG experiment completed. "
        f"Results saved to: {root_dir}"
    )


if __name__ == "__main__":
    main()
