import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import torch

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
    (
        "baseline_critic_adam",
        {
            "optimizer_name": "adam",
            "critic_learning_rate": FIXED_CRITIC_LEARNING_RATE,
            "momentum": 0.0,
        },
    ),
    (
        "critic_sgd_momentum",
        {
            "optimizer_name": "sgd_momentum",
            "critic_learning_rate": 1e-3,
            "momentum": 0.9,
        },
    ),
    (
        "critic_sgd_plain",
        {
            "optimizer_name": "sgd_plain",
            "critic_learning_rate": 1e-3,
            "momentum": 0.0,
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Short formal training validation for replacing the critic Adam optimizer "
            "with simpler SGD-based rules on the joint reward-aligned actor mainline."
        )
    )
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=[group_name for group_name, _spec in DEFAULT_GROUPS],
        help="Subset of critic optimizer groups to run.",
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
            ("critic_grad_norm", "Critic Grad Norm"),
            ("critic_backbone_grad_norm", "Critic Backbone Grad Norm"),
            ("critic_head_grad_norm", "Critic Head Grad Norm"),
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


def _configure_agent_critic_optimizer(agent: PPOAgent, group_spec: dict[str, float | str]) -> None:
    optimizer_name = str(group_spec["optimizer_name"])
    critic_learning_rate = float(group_spec["critic_learning_rate"])
    momentum = float(group_spec["momentum"])
    if optimizer_name == "adam":
        agent.critic_optimizer = torch.optim.Adam(
            agent.critic_params,
            lr=critic_learning_rate,
        )
        return
    if optimizer_name in {"sgd_momentum", "sgd_plain"}:
        agent.critic_optimizer = torch.optim.SGD(
            agent.critic_params,
            lr=critic_learning_rate,
            momentum=momentum,
        )
        return
    raise ValueError(f"Unsupported critic optimizer group: {optimizer_name}")


def _group_label(group_name: str) -> str:
    return group_name


def run_experiment(seed: int, output_root: Path, group_names: list[str]) -> Path:
    root_dir = output_root / datetime.now().strftime(
        "critic_simpler_optimizer_training_validation_%Y%m%d_%H%M%S"
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

    available_groups = {group_name: spec for group_name, spec in DEFAULT_GROUPS}
    selected_groups: list[tuple[str, dict[str, float | str]]] = []
    for group_name in group_names:
        if group_name not in available_groups:
            raise ValueError(f"Unknown group name: {group_name}")
        selected_groups.append((group_name, available_groups[group_name]))

    manifest: dict[str, object] = {
        "root_dir": str(root_dir),
        "seed": seed,
        "policy_ratio_mode": POLICY_RATIO_MODE,
        "base_critic_learning_rate": FIXED_CRITIC_LEARNING_RATE,
        "probe_state_count": PROBE_STATE_COUNT,
        "training_num_epochs": int(probe_config.training.num_epochs),
        "training_time_steps": int(probe_config.training.time_steps),
        "update_epochs": int(probe_config.ppo.update_epochs),
        "selected_groups": [group_name for group_name, _spec in selected_groups],
        "groups": [],
    }
    all_metrics: list[dict[str, object]] = []

    for group_name, group_spec in selected_groups:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root_dir / f"{group_name}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = build_policy_ratio_mode_config(
            policy_ratio_mode=POLICY_RATIO_MODE,
            seed=seed,
        )
        state_layout = build_state_layout(config)
        agent_preview = PPOAgent(
            config=config.ppo,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            critic_state_layout=state_layout,
        )
        _configure_agent_critic_optimizer(agent_preview, group_spec)

        (run_dir / "experiment_config.json").write_text(
            json.dumps(
                {
                    "group_name": group_name,
                    "critic_optimizer": {
                        "name": str(group_spec["optimizer_name"]),
                        "learning_rate": float(group_spec["critic_learning_rate"]),
                        "momentum": float(group_spec["momentum"]),
                    },
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
                    "only_change": (
                        "critic optimizer rule replaced for the main critic while actor "
                        "optimizer, reward, target, architecture, and PPO hyperparameters "
                        "remain unchanged"
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
            f"critic_optimizer={group_spec['optimizer_name']}, "
            f"critic_lr={group_spec['critic_learning_rate']} -> {run_dir}"
        )
        set_global_seeds(seed)
        train_with_diagnosis(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
            probe_states=probe_states,
            agent_kwargs={"critic_state_layout": state_layout},
            agent_setup_hook=lambda agent, spec=group_spec: _configure_agent_critic_optimizer(
                agent,
                spec,
            ),
        )

        plot_single_run_outputs(run_dir)

        metrics = extract_metrics(run_dir=run_dir)
        metrics["group_name"] = group_name
        metrics["comparison_label"] = _group_label(group_name)
        metrics["policy_ratio_mode"] = POLICY_RATIO_MODE
        metrics["critic_optimizer_name"] = str(group_spec["optimizer_name"])
        metrics["critic_learning_rate"] = float(group_spec["critic_learning_rate"])
        metrics["critic_momentum"] = float(group_spec["momentum"])
        metrics["base_critic_learning_rate"] = float(FIXED_CRITIC_LEARNING_RATE)
        metrics["final_target_bucket_prediction_slope"] = compute_bucket_slope(run_dir)
        (run_dir / "analysis_summary.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_metrics.append(metrics)
        manifest["groups"].append(
            {
                "group_name": group_name,
                "critic_optimizer_name": str(group_spec["optimizer_name"]),
                "critic_learning_rate": float(group_spec["critic_learning_rate"]),
                "critic_momentum": float(group_spec["momentum"]),
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
            }
        )

    summary_df = pd.DataFrame(all_metrics)
    summary_csv = root_dir / "critic_simpler_optimizer_training_validation_summary.csv"
    summary_json = root_dir / "critic_simpler_optimizer_training_validation_summary.json"
    summary_md = root_dir / "critic_simpler_optimizer_training_validation_summary.md"
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
        "Critic simpler optimizer training validation completed. "
        f"Results saved to: {root_dir}"
    )


if __name__ == "__main__":
    main()
