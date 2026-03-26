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
    "blockwise_action_conditioned_td_advantage_surrogate_mean",
    "blockwise_action_conditioned_path_value_td_advantage_surrogate_mean",
    "blockwise_path_specific_td_supervised_advantage_surrogate_mean",
)
BLOCK_PATH_COST_DEFINITION = (
    "cost_local_b = workload_b + virtual_queue_b + local_queue_b; "
    "cost_bs1_b = workload_b + virtual_queue_b + access_queue_sensor(b) + bs_queue_BS1(task_b); "
    "cost_bs2_b = workload_b + virtual_queue_b + access_queue_sensor(b) + bs_queue_BS2(task_b)."
)
BLOCK_PATH_VALUE_HEAD_DEFINITION = (
    "critic_block_path_value_head outputs 24 x 3 path values on top of the existing critic_backbone; "
    "each block emits [V_local_b(s), V_bs1_b(s), V_bs2_b(s)] while the global critic head stays unchanged."
)
PATH_TD_SUPERVISION_MODE_DEFINITIONS = {
    "blockwise_action_conditioned_td_advantage_surrogate_mean": (
        "p_local = 1 - sigmoid(theta_b); "
        "p_bs1, p_bs2 = sigmoid(theta_b) * softmax(route_logits_b); "
        "cost_act_b_now = p_local * cost_local_b(now) + p_bs1 * cost_bs1_b(now) + p_bs2 * cost_bs2_b(now); "
        "cost_act_b_next = p_local * cost_local_b(next) + p_bs1 * cost_bs1_b(next) + p_bs2 * cost_bs2_b(next); "
        "delta_cost_b = cost_act_b_next - cost_act_b_now; "
        "r_b = -delta_cost_b = cost_act_b_now - cost_act_b_next; "
        "A_b = r_b + gamma * (1 - done) * V_b(s_next) - V_b(s)."
    ),
    "blockwise_action_conditioned_path_value_td_advantage_surrogate_mean": (
        "p_local = 1 - sigmoid(theta_b); "
        "p_bs1, p_bs2 = sigmoid(theta_b) * softmax(route_logits_b); "
        "cost_act_b_now = p_local * cost_local_b(now) + p_bs1 * cost_bs1_b(now) + p_bs2 * cost_bs2_b(now); "
        "cost_act_b_next = p_local * cost_local_b(next) + p_bs1 * cost_bs1_b(next) + p_bs2 * cost_bs2_b(next); "
        "delta_cost_b = cost_act_b_next - cost_act_b_now; "
        "r_b = -delta_cost_b = cost_act_b_now - cost_act_b_next; "
        "V_act_b(s) = p_local * V_local_b(s) + p_bs1 * V_bs1_b(s) + p_bs2 * V_bs2_b(s); "
        "V_act_b(s_next) = p_local * V_local_b(s_next) + p_bs1 * V_bs1_b(s_next) + p_bs2 * V_bs2_b(s_next); "
        "A_b = r_b + gamma * (1 - done) * V_act_b(s_next) - V_act_b(s)."
    ),
    "blockwise_path_specific_td_supervised_advantage_surrogate_mean": (
        "p_local = 1 - sigmoid(theta_b); "
        "p_bs1, p_bs2 = sigmoid(theta_b) * softmax(route_logits_b); "
        "cost_act_b_now = p_local * cost_local_b(now) + p_bs1 * cost_bs1_b(now) + p_bs2 * cost_bs2_b(now); "
        "cost_act_b_next = p_local * cost_local_b(next) + p_bs1 * cost_bs1_b(next) + p_bs2 * cost_bs2_b(next); "
        "delta_cost_b = cost_act_b_next - cost_act_b_now; "
        "r_b = -delta_cost_b = cost_act_b_now - cost_act_b_next; "
        "r_local_b = cost_local_b(now) - cost_local_b(next); "
        "r_bs1_b = cost_bs1_b(now) - cost_bs1_b(next); "
        "r_bs2_b = cost_bs2_b(now) - cost_bs2_b(next); "
        "T_local_b = r_local_b + gamma * (1 - done) * V_local_b(s_next); "
        "T_bs1_b = r_bs1_b + gamma * (1 - done) * V_bs1_b(s_next); "
        "T_bs2_b = r_bs2_b + gamma * (1 - done) * V_bs2_b(s_next); "
        "critic_block_path_value_head is directly supervised against [T_local_b, T_bs1_b, T_bs2_b]; "
        "actor still uses V_act_b(s) = p_local * V_local_b(s) + p_bs1 * V_bs1_b(s) + p_bs2 * V_bs2_b(s), "
        "A_b = r_b + gamma * (1 - done) * V_act_b(s_next) - V_act_b(s)."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled dense comparison for action-conditioned TD, path-value bootstrap, and path-specific TD supervision."
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


def plot_block_td_value_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("block_value_pred_mean", "Block Bootstrap Value Mean"),
            ("block_td_target_mean", "Block TD Target Mean"),
            ("block_td_target_std", "Block TD Target Std"),
            ("block_td_advantage_entropy", "Block TD Advantage Entropy"),
            ("top_k_block_td_advantage_share", "Top-K Block TD Advantage Share"),
            ("active_block_count_mean", "Active Block Count Mean"),
        ],
        output_path=output_path,
    )


def plot_block_path_value_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("path_value_local_mean", "Path Value Local Mean"),
            ("path_value_bs1_mean", "Path Value BS1 Mean"),
            ("path_value_bs2_mean", "Path Value BS2 Mean"),
            ("block_action_conditioned_value_now_mean", "Action-Conditioned Value Now Mean"),
            ("block_action_conditioned_value_next_mean", "Action-Conditioned Value Next Mean"),
            ("block_td_advantage_mean", "Block TD Advantage Mean"),
        ],
        output_path=output_path,
    )


def plot_path_td_target_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("path_td_target_local_mean", "Path TD Target Local Mean"),
            ("path_td_target_bs1_mean", "Path TD Target BS1 Mean"),
            ("path_td_target_bs2_mean", "Path TD Target BS2 Mean"),
            ("path_td_target_local_std", "Path TD Target Local Std"),
            ("path_td_target_bs1_std", "Path TD Target BS1 Std"),
            ("path_td_target_bs2_std", "Path TD Target BS2 Std"),
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
    plot_block_td_value_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "block_td_value_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="block_td_advantage_entropy",
        title="Block TD Advantage Entropy by Policy Surrogate Mode",
        ylabel="block_td_advantage_entropy",
        output_path=plots_dir / "block_td_advantage_entropy_curve_compare.png",
    )
    plot_metric_compare(
        mode_logs,
        column="top_k_block_td_advantage_share",
        title="Top-K Block TD Advantage Share by Policy Surrogate Mode",
        ylabel="top_k_block_td_advantage_share",
        output_path=plots_dir / "top_k_block_td_advantage_share_curve_compare.png",
    )
    plot_block_path_value_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "block_path_value_curve_compare.png",
    )
    plot_path_td_target_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "path_td_target_curve_compare.png",
    )
    plot_bucket_metric_compare(
        mode_bucket_stats=mode_bucket_stats,
        value_column="bucket_mean_delta_log_prob_selected_action",
        title="Advantage Bucket Mean Delta Log-Prob",
        ylabel="bucket_mean_delta_log_prob_selected_action",
        output_path=plots_dir / "advantage_bucket_update_compare.png",
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
            ("value_explained_variance", "Value Explained Variance"),
            ("prediction_target_corr", "Prediction Target Correlation"),
            ("prediction_std_over_target_std", "Prediction Std / Target Std"),
        ],
        output_path=plots_dir / "critic_health_compare.png",
    )


def run_experiment(policy_ratio_modes: list[str], seed: int, output_root: Path) -> Path:
    root_dir = output_root / datetime.now().strftime(
        "dense_policy_block_path_td_supervision_experiment_%Y%m%d_%H%M%S"
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
            "block_path_cost_definition": BLOCK_PATH_COST_DEFINITION,
            "block_path_value_head_definition": BLOCK_PATH_VALUE_HEAD_DEFINITION,
            "path_td_supervision_mode_definitions": PATH_TD_SUPERVISION_MODE_DEFINITIONS,
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
                    "block_path_cost_definition": BLOCK_PATH_COST_DEFINITION,
                    "block_path_value_head_definition": BLOCK_PATH_VALUE_HEAD_DEFINITION,
                    "path_td_supervision_mode_definition": PATH_TD_SUPERVISION_MODE_DEFINITIONS.get(
                        policy_ratio_mode,
                        "",
                    ),
                    "path_td_supervision_mode_definitions": PATH_TD_SUPERVISION_MODE_DEFINITIONS,
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
                "block_path_value_stats": str(run_dir / "block_path_value_stats_by_epoch.csv"),
                "path_td_target_stats": str(run_dir / "path_td_target_stats_by_epoch.csv"),
            }
        )

    summary_df = pd.DataFrame(all_metrics)
    summary_csv = root_dir / "policy_block_path_td_supervision_summary.csv"
    summary_json = root_dir / "policy_block_path_td_supervision_summary.json"
    summary_md = root_dir / "policy_block_path_td_supervision_summary.md"
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
        "reward_curve_compare": str(root_dir / "comparison_plots" / "reward_curve_compare.png"),
        "policy_loss_curve_compare": str(
            root_dir / "comparison_plots" / "policy_loss_curve_compare.png"
        ),
        "approx_kl_curve_compare": str(
            root_dir / "comparison_plots" / "approx_kl_curve_compare.png"
        ),
        "clip_fraction_curve_compare": str(
            root_dir / "comparison_plots" / "clip_fraction_curve_compare.png"
        ),
        "ratio_stats_curve_compare": str(
            root_dir / "comparison_plots" / "ratio_stats_curve_compare.png"
        ),
        "advantage_alignment_curve_compare": str(
            root_dir / "comparison_plots" / "advantage_alignment_curve_compare.png"
        ),
        "advantage_bucket_update_compare": str(
            root_dir / "comparison_plots" / "advantage_bucket_update_compare.png"
        ),
        "selected_action_prob_gain_by_adv_bucket_compare": str(
            root_dir
            / "comparison_plots"
            / "selected_action_prob_gain_by_adv_bucket_compare.png"
        ),
        "probe_policy_kl_curve_compare": str(
            root_dir / "comparison_plots" / "probe_policy_kl_curve_compare.png"
        ),
        "selected_action_change_rate_curve_compare": str(
            root_dir / "comparison_plots" / "selected_action_change_rate_curve_compare.png"
        ),
        "block_clip_fraction_curve_compare": str(
            root_dir / "comparison_plots" / "block_clip_fraction_curve_compare.png"
        ),
        "block_surrogate_scale_curve_compare": str(
            root_dir / "comparison_plots" / "block_surrogate_scale_curve_compare.png"
        ),
        "block_td_advantage_entropy_curve_compare": str(
            root_dir / "comparison_plots" / "block_td_advantage_entropy_curve_compare.png"
        ),
        "top_k_block_td_advantage_share_curve_compare": str(
            root_dir / "comparison_plots" / "top_k_block_td_advantage_share_curve_compare.png"
        ),
        "block_path_value_curve_compare": str(
            root_dir / "comparison_plots" / "block_path_value_curve_compare.png"
        ),
        "path_td_target_curve_compare": str(
            root_dir / "comparison_plots" / "path_td_target_curve_compare.png"
        ),
        "critic_health_compare": str(root_dir / "comparison_plots" / "critic_health_compare.png"),
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
        "Dense policy block path-TD-supervision experiment completed. "
        f"Results saved to: {root_dir}"
    )


if __name__ == "__main__":
    main()
