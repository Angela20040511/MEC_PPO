import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

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
from dense_policy_joint_reward_aligned_credit_experiment import (
    JOINT_REWARD_ALIGNED_MODE_DEFINITIONS,
)
from dense_policy_ratio_mode_experiment import (
    FIXED_CRITIC_LEARNING_RATE,
    build_action_block_layout,
    build_policy_ratio_mode_config,
    compute_bucket_slope,
    load_mode_logs,
    plot_two_panel_compare,
)
from dense_policy_true_conditional_route_experiment import (
    FIXED_MIN_DELTA,
    FIXED_PATIENCE,
    FIXED_SEED,
    FIXED_START_CHECK_EPOCH,
)
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds
from rl.ppo_agent import PPOAgent


POLICY_RATIO_MODE = "hierarchical_actor_joint_reward_aligned_credit"
DEFAULT_GROUPS = (
    ("baseline_current_batch_only", False, 1.0),
    ("blended_current_heldout", True, 1.0),
    ("blended_current_heldout_quarter_step", True, 0.25),
)
BLEND_CURRENT_WEIGHT = 0.5
BLEND_HELDOUT_WEIGHT = 0.5
BLEND_HELDOUT_BATCH_COUNT = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Short formal training validation for current-batch-only versus blended "
            "current+heldout critic value loss on the joint reward-aligned actor mainline."
        )
    )
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=[group_name for group_name, _enabled, _scale in DEFAULT_GROUPS],
        help="Subset of blended-loss groups to run.",
    )
    return parser.parse_args()


def _plot_reward_compare(mode_logs: dict[str, pd.DataFrame], output_path: Path) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("episode_reward", "Episode Reward"),
            ("best_reward_so_far", "Best Reward So Far"),
        ],
        output_path=output_path,
    )


def _plot_advantage_alignment_compare(
    mode_logs: dict[str, pd.DataFrame], output_path: Path
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("overall_advantage_action_alignment", "Overall Advantage Alignment"),
            ("theta_advantage_alignment", "Theta Advantage Alignment"),
            ("route_advantage_alignment", "Route Advantage Alignment"),
        ],
        output_path=output_path,
    )


def _plot_critic_health_compare(mode_logs: dict[str, pd.DataFrame], output_path: Path) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("value_explained_variance", "Value Explained Variance"),
            ("prediction_target_corr", "Prediction-Target Corr"),
            ("prediction_std_over_target_std", "Prediction/Target Std Ratio"),
            ("target_bucket_prediction_slope", "Target Bucket Slope"),
        ],
        output_path=output_path,
    )


def _plot_drift_proxy_compare(mode_logs: dict[str, pd.DataFrame], output_path: Path) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("critic_loss_current_batch", "Current-Batch Value Loss"),
            ("critic_loss_heldout_batch", "Held-Out Value Loss"),
        ],
        output_path=output_path,
    )


def _plot_comparisons(root_dir: Path, summary_df: pd.DataFrame) -> None:
    plots_dir = root_dir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    mode_logs = load_mode_logs(summary_df)
    _plot_reward_compare(mode_logs, plots_dir / "reward_curve_compare.png")
    _plot_advantage_alignment_compare(
        mode_logs,
        plots_dir / "advantage_alignment_curve_compare.png",
    )
    _plot_critic_health_compare(mode_logs, plots_dir / "critic_health_curve_compare.png")
    _plot_drift_proxy_compare(mode_logs, plots_dir / "critic_drift_compare.png")


def _group_label(group_name: str) -> str:
    return group_name


def _configure_agent_for_group(
    agent: PPOAgent,
    blended_enabled: bool,
) -> None:
    agent.critic_blended_value_loss_enabled = bool(blended_enabled)
    agent.critic_blended_current_weight = float(BLEND_CURRENT_WEIGHT)
    agent.critic_blended_heldout_weight = float(BLEND_HELDOUT_WEIGHT)
    agent.critic_blended_heldout_batch_count = int(BLEND_HELDOUT_BATCH_COUNT)


def run_experiment(seed: int, output_root: Path, group_names: list[str]) -> Path:
    root_dir = output_root / datetime.now().strftime(
        "critic_blended_loss_training_validation_%Y%m%d_%H%M%S"
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStoppingConfig(
        patience=FIXED_PATIENCE,
        min_delta=FIXED_MIN_DELTA,
        monitor_start_epoch=FIXED_START_CHECK_EPOCH,
    )
    probe_config = build_policy_ratio_mode_config(
        policy_ratio_mode=POLICY_RATIO_MODE,
        seed=seed,
    )
    probe_states = collect_probe_states(
        config=probe_config,
        seed=seed,
        probe_count=PROBE_STATE_COUNT,
    )

    manifest: dict[str, object] = {
        "root_dir": str(root_dir),
        "seed": seed,
        "policy_ratio_mode": POLICY_RATIO_MODE,
        "base_critic_learning_rate": FIXED_CRITIC_LEARNING_RATE,
        "blend_current_weight": BLEND_CURRENT_WEIGHT,
        "blend_heldout_weight": BLEND_HELDOUT_WEIGHT,
        "blend_heldout_batch_count": BLEND_HELDOUT_BATCH_COUNT,
        "probe_state_count": PROBE_STATE_COUNT,
        "training_num_epochs": int(probe_config.training.num_epochs),
        "training_time_steps": int(probe_config.training.time_steps),
        "update_epochs": int(probe_config.ppo.update_epochs),
        "groups": [],
    }
    all_metrics: list[dict[str, object]] = []

    available_groups = {
        group_name: (blended_enabled, lr_scale)
        for group_name, blended_enabled, lr_scale in DEFAULT_GROUPS
    }
    selected_groups: list[tuple[str, bool, float]] = []
    for group_name in group_names:
        if group_name not in available_groups:
            raise ValueError(f"Unknown group name: {group_name}")
        blended_enabled, lr_scale = available_groups[group_name]
        selected_groups.append((group_name, blended_enabled, lr_scale))

    manifest["selected_groups"] = [group_name for group_name, _enabled, _scale in selected_groups]

    for group_name, blended_enabled, lr_scale in selected_groups:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root_dir / f"{group_name}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = build_policy_ratio_mode_config(
            policy_ratio_mode=POLICY_RATIO_MODE,
            seed=seed,
        )
        scaled_ppo = replace(
            config.ppo,
            critic_learning_rate=float(config.ppo.critic_learning_rate) * float(lr_scale),
        )
        config = replace(config, ppo=scaled_ppo)

        state_layout = build_state_layout(config)
        agent_preview = PPOAgent(
            config=config.ppo,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            critic_state_layout=state_layout,
        )
        _configure_agent_for_group(agent_preview, blended_enabled=blended_enabled)

        (run_dir / "experiment_config.json").write_text(
            json.dumps(
                {
                    "group_name": group_name,
                    "critic_lr_scale": float(lr_scale),
                    "training": asdict(config.training),
                    "ppo": asdict(config.ppo),
                    "dt": asdict(config.dt),
                    "system": asdict(config.system),
                    "critic_state_layout": state_layout,
                    "action_block_layout": build_action_block_layout(config),
                    "actor_structure": agent_preview.describe_actor_structure(),
                    "actor_raw_layout": agent_preview.describe_actor_raw_layout(),
                    "mode_definition": JOINT_REWARD_ALIGNED_MODE_DEFINITIONS[
                        POLICY_RATIO_MODE
                    ],
                    "critic_value_loss_sampling": {
                        "blended_enabled": bool(blended_enabled),
                        "current_weight": float(BLEND_CURRENT_WEIGHT),
                        "heldout_weight": float(BLEND_HELDOUT_WEIGHT),
                        "heldout_batch_count_per_epoch": int(BLEND_HELDOUT_BATCH_COUNT),
                        "heldout_refresh_rule": (
                            "At each training epoch start, reserve fixed held-out minibatches "
                            "from the current rollout buffer and reuse them throughout that epoch."
                        ),
                    },
                    "only_change": (
                        "critic main value loss sampling changed to current-only or blended "
                        "current+heldout; optional critic lr scaling for the quarter-step group; "
                        "everything else unchanged"
                    ),
                    "early_stopping": asdict(early_stopping),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[train] {group_name} policy_ratio_mode={POLICY_RATIO_MODE}, seed={seed}, "
            f"critic_lr={config.ppo.critic_learning_rate}, blended={blended_enabled} -> {run_dir}"
        )
        set_global_seeds(seed)
        train_with_diagnosis(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
            probe_states=probe_states,
            agent_kwargs={"critic_state_layout": state_layout},
            agent_setup_hook=lambda agent, enabled=blended_enabled: _configure_agent_for_group(
                agent,
                blended_enabled=enabled,
            ),
        )
        plot_single_run_outputs(run_dir)

        metrics = extract_metrics(run_dir=run_dir)
        logs = pd.read_csv(run_dir / "train_logs.csv")
        final_row = logs.iloc[-1]

        def final_float(column: str, default: float = 0.0) -> float:
            return float(final_row[column]) if column in final_row.index else float(default)

        metrics["group_name"] = group_name
        metrics["comparison_label"] = _group_label(group_name)
        metrics["policy_ratio_mode"] = _group_label(group_name)
        metrics["critic_lr_scale"] = float(lr_scale)
        metrics["critic_learning_rate"] = float(config.ppo.critic_learning_rate)
        metrics["base_critic_learning_rate"] = float(FIXED_CRITIC_LEARNING_RATE)
        metrics["critic_blended_value_loss_enabled"] = float(blended_enabled)
        metrics["critic_blended_current_weight"] = float(BLEND_CURRENT_WEIGHT)
        metrics["critic_blended_heldout_weight"] = float(BLEND_HELDOUT_WEIGHT)
        metrics["critic_blended_heldout_batch_count"] = float(BLEND_HELDOUT_BATCH_COUNT)
        metrics["final_critic_loss_current_batch"] = final_float("critic_loss_current_batch")
        metrics["final_critic_loss_heldout_batch"] = final_float("critic_loss_heldout_batch")
        metrics["final_critic_loss_current_minus_heldout"] = (
            metrics["final_critic_loss_current_batch"]
            - metrics["final_critic_loss_heldout_batch"]
        )
        metrics["final_target_bucket_prediction_slope"] = compute_bucket_slope(run_dir)
        (run_dir / "analysis_summary.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_metrics.append(metrics)
        manifest["groups"].append(
            {
                "group_name": group_name,
                "blended_enabled": bool(blended_enabled),
                "critic_lr_scale": float(lr_scale),
                "critic_learning_rate": float(config.ppo.critic_learning_rate),
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    summary_df = pd.DataFrame(all_metrics)
    summary_csv = root_dir / "critic_blended_loss_training_validation_summary.csv"
    summary_json = root_dir / "critic_blended_loss_training_validation_summary.json"
    summary_md = root_dir / "critic_blended_loss_training_validation_summary.md"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(dataframe_to_markdown(summary_df), encoding="utf-8")

    _plot_comparisons(root_dir=root_dir, summary_df=summary_df)

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
        seed=args.seed,
        output_root=Path(args.output_root),
        group_names=list(args.groups),
    )
    print(
        "Critic blended-loss training validation completed. "
        f"Results saved to: {root_dir}"
    )


if __name__ == "__main__":
    main()
