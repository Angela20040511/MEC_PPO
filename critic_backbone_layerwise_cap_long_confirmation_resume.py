import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from critic_backbone_layerwise_cap_long_confirmation_validation import (
    DEFAULT_SELECTED_GROUPS,
    GROUP_SPECS,
    TRACKED_HOTSPOT_LAYERS,
    TRAINING_OVERRIDES,
    VALIDATION_STEM,
)
from critic_backbone_preconditioner_clip_telemetry_validation import (
    ACTIVE_GRAD_THRESHOLD,
    POLICY_RATIO_MODE,
    _apply_config_overrides,
    _build_summary_row,
    _configure_agent_for_group,
    _load_group_backbone_epoch_telemetry,
    _load_group_logs,
    _plot_advantage_alignment_compare,
    _plot_backbone_preconditioner_telemetry_compare,
    _plot_critic_drift_compare,
    _plot_critic_health_compare,
    _plot_hotspot_layer_telemetry_compare,
    _plot_reward_compare,
)
from dense_actor_input_experiment import build_state_layout
from dense_critic_diagnosis import (
    PROBE_STATE_COUNT,
    collect_probe_states,
    dataframe_to_markdown,
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
)
from dense_policy_true_conditional_route_experiment import (
    FIXED_MIN_DELTA,
    FIXED_PATIENCE,
    FIXED_SEED,
    FIXED_START_CHECK_EPOCH,
)
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds
from rl.ppo_agent import PPOAgent


REQUIRED_RUN_FILES = (
    "experiment_config.json",
    "train_summary.json",
    "train_logs.csv",
    "best_model.pt",
    "preconditioner_telemetry_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume helper for critic_backbone_layerwise_cap_long_confirmation_validation: "
            "reuse completed runs under an existing root, only rerun missing groups, "
            "and regenerate root-level summaries/compare plots."
        )
    )
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Specific missing groups to run. If omitted, auto-run missing groups only.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only rebuild root summary and compare plots from completed runs.",
    )
    return parser.parse_args()


def _match_group_name(dir_name: str) -> str | None:
    for group_name in sorted(GROUP_SPECS.keys(), key=len, reverse=True):
        if dir_name == group_name or dir_name.startswith(f"{group_name}_"):
            return group_name
    return None


def _is_complete_run_dir(run_dir: Path) -> bool:
    return all((run_dir / filename).exists() for filename in REQUIRED_RUN_FILES)


def _scan_root_runs(root_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {group_name: [] for group_name in GROUP_SPECS}
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        group_name = _match_group_name(child.name)
        if group_name is None:
            continue
        grouped[group_name].append(child)
    return grouped


def _latest_completed_runs(root_dir: Path) -> dict[str, Path]:
    grouped = _scan_root_runs(root_dir)
    completed: dict[str, Path] = {}
    for group_name, run_dirs in grouped.items():
        complete_dirs = [run_dir for run_dir in run_dirs if _is_complete_run_dir(run_dir)]
        if not complete_dirs:
            continue
        complete_dirs.sort(key=lambda path: path.stat().st_mtime)
        completed[group_name] = complete_dirs[-1]
    return completed


def _build_probe_config(seed: int) -> Any:
    probe_config = build_policy_ratio_mode_config(
        policy_ratio_mode=POLICY_RATIO_MODE,
        seed=seed,
    )
    return _apply_config_overrides(
        probe_config,
        training_overrides=TRAINING_OVERRIDES,
        ppo_overrides=None,
    )


def _run_single_group(
    root_dir: Path,
    group_name: str,
    seed: int,
    probe_states: Any,
    early_stopping: EarlyStoppingConfig,
) -> Path:
    if group_name not in GROUP_SPECS:
        raise ValueError(f"Unknown group name: {group_name}")

    group_spec = GROUP_SPECS[group_name]
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root_dir / f"{group_name}_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = _build_probe_config(seed)
    state_layout = build_state_layout(config)
    agent_preview = PPOAgent(
        config=config.ppo,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        critic_state_layout=state_layout,
    )
    _configure_agent_for_group(agent_preview, group_spec, TRACKED_HOTSPOT_LAYERS)

    (run_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "group_name": group_name,
                "training": asdict(config.training),
                "ppo": asdict(config.ppo),
                "dt": asdict(config.dt),
                "system": asdict(config.system),
                "critic_state_layout": state_layout,
                "action_block_layout": build_action_block_layout(config),
                "actor_structure": agent_preview.describe_actor_structure(),
                "actor_raw_layout": agent_preview.describe_actor_raw_layout(),
                "mode_definition": JOINT_REWARD_ALIGNED_MODE_DEFINITIONS[POLICY_RATIO_MODE],
                "critic_backbone_preconditioner_active_threshold": float(
                    ACTIVE_GRAD_THRESHOLD
                ),
                "critic_backbone_preconditioner_cap": (
                    float(group_spec["backbone_preconditioner_cap"])
                    if group_spec.get("backbone_preconditioner_cap") is not None
                    else None
                ),
                "clip_focus_parameter_names": list(group_spec["clip_focus_parameter_names"])
                if group_spec.get("clip_focus_parameter_names") is not None
                else [],
                "layerwise_backbone_preconditioner_caps": group_spec.get(
                    "layerwise_backbone_preconditioner_caps",
                    {},
                ),
                "tracked_hotspot_layers": list(TRACKED_HOTSPOT_LAYERS),
                "critic_head_optimizer": {
                    "name": "adam",
                    "learning_rate": float(FIXED_CRITIC_LEARNING_RATE),
                    "untouched": True,
                },
                "only_change": group_spec["description"],
                "early_stopping": asdict(early_stopping),
                "resume_root_dir": str(root_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[resume-train] {group_name} seed={seed} "
        f"layerwise_caps={group_spec.get('layerwise_backbone_preconditioner_caps')} "
        f"-> {run_dir}"
    )
    set_global_seeds(seed)
    agent, _logs, _summary, _payload = train_with_diagnosis(
        config=config,
        checkpoint_dir=str(run_dir),
        early_stopping=early_stopping,
        probe_states=probe_states,
        agent_kwargs={"critic_state_layout": state_layout},
        agent_setup_hook=lambda agent_obj, spec=group_spec: _configure_agent_for_group(
            agent_obj,
            spec,
            TRACKED_HOTSPOT_LAYERS,
        ),
    )
    plot_single_run_outputs(run_dir)
    if hasattr(agent.critic_optimizer, "finalize_telemetry"):
        agent.critic_optimizer.finalize_telemetry(run_dir)
    return run_dir


def _load_telemetry_summary(run_dir: Path) -> dict[str, Any]:
    telemetry_path = run_dir / "preconditioner_telemetry_summary.json"
    if not telemetry_path.exists():
        return {}
    return json.loads(telemetry_path.read_text(encoding="utf-8"))


def _write_root_summary(root_dir: Path, completed_runs: dict[str, Path], seed: int) -> None:
    summary_rows: list[dict[str, Any]] = []
    selected_groups = [group for group in DEFAULT_SELECTED_GROUPS if group in completed_runs]
    for group_name in selected_groups:
        run_dir = completed_runs[group_name]
        group_spec = GROUP_SPECS[group_name]
        telemetry_summary = _load_telemetry_summary(run_dir)
        summary_rows.append(
            _build_summary_row(
                run_dir=run_dir,
                group_name=group_name,
                group_spec=group_spec,
                telemetry_summary=telemetry_summary,
                tracked_hotspot_layers=TRACKED_HOTSPOT_LAYERS,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = root_dir / f"{VALIDATION_STEM}_summary.csv"
    summary_json = root_dir / f"{VALIDATION_STEM}_summary.json"
    summary_md = root_dir / f"{VALIDATION_STEM}_summary.md"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown_columns = [
        "group_name",
        "best_epoch",
        "best_reward",
        "final_reward",
        "reward_gap",
        "overall_advantage_action_alignment",
        "theta_advantage_alignment",
        "route_advantage_alignment",
        "value_explained_variance",
        "prediction_target_corr",
        "prediction_std_over_target_std",
        "target_bucket_prediction_slope",
        "joint_action_decision_agreement_ratio_under_reward_aligned",
        "final_critic_backbone_delta_norm",
        "backbone_active_preconditioner_p95",
        "backbone_active_preconditioner_p99",
        "backbone_clipped_active_fraction",
        "backbone_clipped_active_count",
        "critic_backbone_preconditioner_cap",
        "clip_focus_parameter_names",
    ]
    summary_md.write_text(
        dataframe_to_markdown(summary_df[markdown_columns]),
        encoding="utf-8",
    )

    group_logs = _load_group_logs(summary_df)
    backbone_telemetry_logs = _load_group_backbone_epoch_telemetry(summary_df)
    _plot_reward_compare(group_logs, root_dir / "reward_curve_compare.png")
    _plot_advantage_alignment_compare(
        group_logs,
        root_dir / "advantage_alignment_curve_compare.png",
    )
    _plot_critic_health_compare(group_logs, root_dir / "critic_health_curve_compare.png")
    _plot_critic_drift_compare(group_logs, root_dir / "critic_drift_compare.png")
    _plot_backbone_preconditioner_telemetry_compare(
        backbone_telemetry_logs,
        root_dir / "backbone_preconditioner_telemetry_compare.png",
    )
    _plot_hotspot_layer_telemetry_compare(
        summary_df,
        root_dir / "hotspot_layer_telemetry_compare.png",
    )

    manifest = {
        "root_dir": str(root_dir),
        "seed": int(seed),
        "policy_ratio_mode": POLICY_RATIO_MODE,
        "training_overrides": dict(TRAINING_OVERRIDES),
        "tracked_hotspot_layers": list(TRACKED_HOTSPOT_LAYERS),
        "completed_groups": [
            {
                "group_name": group_name,
                "run_dir": str(completed_runs[group_name]),
            }
            for group_name in selected_groups
        ],
        "summary": {
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
            "reward_curve_compare": str(root_dir / "reward_curve_compare.png"),
            "advantage_alignment_curve_compare": str(
                root_dir / "advantage_alignment_curve_compare.png"
            ),
            "critic_health_curve_compare": str(root_dir / "critic_health_curve_compare.png"),
            "critic_drift_compare": str(root_dir / "critic_drift_compare.png"),
            "backbone_preconditioner_telemetry_compare": str(
                root_dir / "backbone_preconditioner_telemetry_compare.png"
            ),
            "hotspot_layer_telemetry_compare": str(
                root_dir / "hotspot_layer_telemetry_compare.png"
            ),
        },
    }
    (root_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    completed_runs = _latest_completed_runs(root_dir)
    requested_groups = list(args.groups) if args.groups else list(DEFAULT_SELECTED_GROUPS)

    if not args.summary_only:
        early_stopping = EarlyStoppingConfig(
            patience=FIXED_PATIENCE,
            min_delta=FIXED_MIN_DELTA,
            monitor_start_epoch=FIXED_START_CHECK_EPOCH,
        )
        probe_config = _build_probe_config(args.seed)
        probe_states = collect_probe_states(
            config=probe_config,
            seed=args.seed,
            probe_count=PROBE_STATE_COUNT,
        )
        for group_name in requested_groups:
            if group_name in completed_runs:
                print(f"[resume-skip] {group_name} already completed at {completed_runs[group_name]}")
                continue
            run_dir = _run_single_group(
                root_dir=root_dir,
                group_name=group_name,
                seed=args.seed,
                probe_states=probe_states,
                early_stopping=early_stopping,
            )
            print(f"[resume-done] {group_name} -> {run_dir}")
            completed_runs = _latest_completed_runs(root_dir)

    completed_runs = _latest_completed_runs(root_dir)
    _write_root_summary(root_dir=root_dir, completed_runs=completed_runs, seed=args.seed)
    print(f"[resume-summary] root summary regenerated under: {root_dir}")


if __name__ == "__main__":
    main()
