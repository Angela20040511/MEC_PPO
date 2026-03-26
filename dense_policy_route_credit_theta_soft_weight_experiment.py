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
    load_mode_logs,
    plot_metric_compare,
    plot_two_panel_compare,
)
from dense_policy_true_conditional_route_experiment import (
    FIXED_MIN_DELTA,
    FIXED_PATIENCE,
    FIXED_ROUTE_MASK_THRESHOLD,
    FIXED_SEED,
    FIXED_START_CHECK_EPOCH,
    plot_comparisons as plot_true_conditional_comparisons,
)
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds
from rl.ppo_agent import PPOAgent

DEFAULT_POLICY_RATIO_MODES = (
    "hierarchical_actor_true_conditional_route_policy",
    "hierarchical_actor_route_credit_theta_soft_weight",
)

ROUTE_CREDIT_THETA_SOFT_WEIGHT_MODE_DEFINITIONS = {
    "hierarchical_actor_true_conditional_route_policy": (
        "Control group: true two-level conditional policy with theta Bernoulli local-vs-offload "
        "bookkeeping and conditional route categorical BS1-vs-BS2 bookkeeping on offload-active "
        "samples, without extra route-credit reweighting."
    ),
    "hierarchical_actor_route_credit_theta_soft_weight": (
        "Test group: keep the same true conditional two-level policy and alternating updates, but "
        "softly reweight route credit by normalized positive theta credit on offload-active "
        "samples instead of hard-gating route credit."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Controlled dense comparison for true conditional route policy vs soft theta-weighted route credit."
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


def plot_route_credit_weight_compare(
    mode_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    plot_two_panel_compare(
        mode_logs,
        panel_specs=[
            ("route_credit_weight_mean", "Route Credit Weight Mean"),
            ("route_credit_weight_std", "Route Credit Weight Std"),
            ("route_credit_weight_max", "Route Credit Weight Max"),
        ],
        output_path=output_path,
    )


def plot_comparisons(root_dir: Path, summary_df: pd.DataFrame) -> None:
    plot_true_conditional_comparisons(root_dir=root_dir, summary_df=summary_df)
    plots_dir = root_dir / "comparison_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    mode_logs = load_mode_logs(summary_df)
    plot_route_credit_weight_compare(
        mode_logs=mode_logs,
        output_path=plots_dir / "route_credit_weight_curve_compare.png",
    )


def run_experiment(policy_ratio_modes: list[str], seed: int, output_root: Path) -> Path:
    root_dir = output_root / datetime.now().strftime(
        "dense_policy_route_credit_theta_soft_weight_experiment_%Y%m%d_%H%M%S"
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
            "route_credit_theta_soft_weight_mode_definitions": (
                ROUTE_CREDIT_THETA_SOFT_WEIGHT_MODE_DEFINITIONS
            ),
            "route_mask_threshold": FIXED_ROUTE_MASK_THRESHOLD,
            "start_check_epoch": FIXED_START_CHECK_EPOCH,
            "patience": FIXED_PATIENCE,
            "min_delta": FIXED_MIN_DELTA,
            "soft_weight_eps": 0.1,
            "soft_weight_max": 3.0,
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
                    "route_credit_theta_soft_weight_mode_definition": (
                        ROUTE_CREDIT_THETA_SOFT_WEIGHT_MODE_DEFINITIONS.get(
                            policy_ratio_mode,
                            "",
                        )
                    ),
                    "route_credit_theta_soft_weight_mode_definitions": (
                        ROUTE_CREDIT_THETA_SOFT_WEIGHT_MODE_DEFINITIONS
                    ),
                    "route_mask_definition": (
                        f"route_mask_b = 1[detach(theta_b) > {FIXED_ROUTE_MASK_THRESHOLD}]"
                    ),
                    "true_conditional_route_policy_definition": (
                        "theta branch uses Bernoulli local-vs-offload bookkeeping; route branch "
                        "uses conditional Categorical BS1-vs-BS2 bookkeeping on offload-active "
                        "samples only."
                    ),
                    "route_credit_weight_definition": (
                        "A_theta_norm is the branchwise-normalized theta advantage. "
                        "For offload-active samples: "
                        "route_credit_weight_raw_b = relu(A_theta_norm_b) + 0.1; "
                        "route_credit_weight_b = route_credit_weight_raw_b / "
                        "mean_active(route_credit_weight_raw); "
                        "then clamp route_credit_weight_b to a max of 3.0."
                    ),
                    "route_effective_advantage_definition": (
                        "route_effective_advantage_b = route_credit_weight_b * A_route_norm_b "
                        "on offload-active samples."
                    ),
                    "route_fallback_definition": (
                        "No hard fallback to the old route loss is used in this soft-weight experiment."
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
                "factorized_ratio_pg_stats": str(
                    run_dir / "factorized_ratio_pg_stats_by_epoch.csv"
                ),
                "conditional_policy_stats": str(run_dir / "conditional_policy_stats_by_epoch.csv"),
                "route_credit_theta_soft_weight_stats": str(
                    run_dir / "route_credit_theta_soft_weight_stats_by_epoch.csv"
                ),
                "theta_route_split_stats": str(run_dir / "theta_route_split_stats_by_epoch.csv"),
            }
        )

    summary_df = pd.DataFrame(all_metrics)
    summary_csv = root_dir / "policy_route_credit_theta_soft_weight_summary.csv"
    summary_json = root_dir / "policy_route_credit_theta_soft_weight_summary.json"
    summary_md = root_dir / "policy_route_credit_theta_soft_weight_summary.md"
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
        "Dense policy route-credit theta soft-weight experiment completed. "
        f"Results saved to: {root_dir}"
    )


if __name__ == "__main__":
    main()
