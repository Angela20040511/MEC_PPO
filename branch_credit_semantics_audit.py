import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dense_actor_input_experiment import build_state_layout
from dense_policy_ratio_mode_experiment import build_policy_ratio_mode_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator


BASELINE_MODE = "hierarchical_actor_true_conditional_route_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static semantics audit for current A_theta and A_route against direct "
            "path-cost-based branch decision labels."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help=(
            "Optional baseline checkpoint path. If omitted, the latest true-conditional "
            "baseline checkpoint from the rewired candidate-score experiment is used."
        ),
    )
    return parser.parse_args()


def _find_latest_baseline_checkpoint() -> Path:
    root = Path("checkpoints")
    experiment_roots = sorted(
        path
        for path in root.glob("dense_policy_route_candidate_score_credit_experiment_*")
        if path.is_dir()
    )
    if not experiment_roots:
        raise FileNotFoundError(
            "No dense_policy_route_candidate_score_credit_experiment_* directory found."
        )
    latest_root = experiment_roots[-1]
    baseline_runs = sorted(
        latest_root.glob(f"policy_ratio_{BASELINE_MODE}_*/best_model.pt")
    )
    if not baseline_runs:
        raise FileNotFoundError(
            f"No baseline best_model.pt found under {latest_root}"
        )
    return baseline_runs[-1]


def _build_agent(seed: int) -> tuple[PPOAgent, Any]:
    config = build_policy_ratio_mode_config(policy_ratio_mode=BASELINE_MODE, seed=seed)
    state_layout = build_state_layout(config)
    agent = PPOAgent(
        config=config.ppo,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        critic_state_layout=state_layout,
    )
    return agent, config


def _collect_rollout(agent: PPOAgent, config: Any, seed: int) -> dict[str, torch.Tensor]:
    simulator = Simulator(config)
    state = simulator.reset(seed=seed)
    done = False

    for _ in range(config.training.time_steps):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, _ = simulator.step(action)
        agent.store_transition(state, action, log_prob, reward, done, value, next_state)
        state = next_state
        if done:
            break

    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)
    return agent.buffer.as_tensors(agent.device)


def _binary_confusion(pred_positive: pd.Series, true_positive: pd.Series) -> dict[str, int]:
    pred = pred_positive.astype(bool)
    true = true_positive.astype(bool)
    return {
        "tp": int((pred & true).sum()),
        "fp": int((pred & ~true).sum()),
        "tn": int((~pred & ~true).sum()),
        "fn": int((~pred & true).sum()),
    }


def _series_stats(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
    }


def _bucket_summary(df: pd.DataFrame, bucket_col: str, value_col: str, sign_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if df.empty or bucket_col not in df:
        return rows
    for bucket, group in df.groupby(bucket_col, dropna=False):
        if group.empty:
            continue
        rows.append(
            {
                "bucket": str(bucket),
                "count": int(len(group)),
                f"{value_col}_mean": float(group[value_col].mean()),
                f"{value_col}_median": float(group[value_col].median()),
                f"{value_col}_positive_ratio": float((group[value_col] > 0.0).mean()),
                f"{sign_col}_mean": float(group[sign_col].mean()),
                f"{sign_col}_median": float(group[sign_col].median()),
                f"{sign_col}_positive_ratio": float((group[sign_col] > 0.0).mean()),
            }
        )
    return rows


def _decision_rate(series: pd.Series, positive_mask: pd.Series) -> dict[str, float]:
    mask = positive_mask.astype(bool)
    if mask.any():
        positive_rate = float(series[mask].mean())
    else:
        positive_rate = 0.0
    negative_mask = ~mask
    if negative_mask.any():
        negative_rate = float(series[negative_mask].mean())
    else:
        negative_rate = 0.0
    return {
        "positive_group_rate": positive_rate,
        "negative_group_rate": negative_rate,
    }


def _correlation_summary(df: pd.DataFrame, left: str, right: str) -> dict[str, float]:
    if df.empty:
        return {
            "pearson": 0.0,
            "spearman": 0.0,
            "sign_agreement_ratio": 0.0,
        }
    return {
        "pearson": float(df[left].corr(df[right], method="pearson")),
        "spearman": float(df[left].corr(df[right], method="spearman")),
        "sign_agreement_ratio": float(((df[left] > 0.0) == (df[right] > 0.0)).mean()),
    }


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else _find_latest_baseline_checkpoint()
    )
    agent, config = _build_agent(args.seed)
    agent.load(str(checkpoint_path), load_optimizer=False)
    data = _collect_rollout(agent, config, args.seed)

    states = data["states"]
    next_states = data["next_states"]
    actions = data["actions"]
    dones = data["dones"]
    task_count = int(agent.critic_state_layout["task_count"])
    sensor_count = int(agent.critic_state_layout["sensor_count"])

    with torch.no_grad():
        critic_inputs, _, _ = agent._prepare_critic_inputs(states, update_stats=False)
        next_critic_inputs, _, _ = agent._prepare_critic_inputs(next_states, update_stats=False)
        split_advantages = agent._theta_route_split_advantages(
            states,
            next_states,
            dones,
            actions,
            critic_inputs=critic_inputs,
            next_critic_inputs=next_critic_inputs,
        )
        _, block_path_cost_local, block_path_cost_bs = agent._block_path_costs(states)
        theta_terms = agent._true_conditional_theta_policy_terms(actions, actions)
        route_terms = agent._true_conditional_route_policy_terms(actions, actions)

    a_theta = split_advantages["theta_advantages"].detach().cpu().numpy()
    a_route = split_advantages["route_advantages"].detach().cpu().numpy()
    route_masks = split_advantages["route_masks"].detach().cpu().numpy()
    actual_offload = theta_terms["offload_active_mask"].detach().cpu().numpy().astype(np.int64)
    actual_route = route_terms["selected_indices"].detach().cpu().numpy().astype(np.int64)

    local_score = (-block_path_cost_local).detach().cpu().numpy()
    bs_scores = (-block_path_cost_bs).detach().cpu().numpy()
    bs1_score = bs_scores[..., 0]
    bs2_score = bs_scores[..., 1]
    best_offload_score = np.maximum(bs1_score, bs2_score)
    theta_true_gap = best_offload_score - local_score
    route_true_gap = bs1_score - bs2_score

    rows: list[dict[str, Any]] = []
    sample_count, block_count = a_theta.shape
    for sample_id in range(sample_count):
        for block_id in range(block_count):
            sensor_id = int(block_id // task_count)
            task_id = int(block_id % task_count)
            offload_active = bool(route_masks[sample_id, block_id] > 0.5)
            actual_route_decision = int(actual_route[sample_id, block_id])
            route_sign_match = (
                bool((a_route[sample_id, block_id] > 0.0) == (route_true_gap[sample_id, block_id] > 0.0))
                if offload_active
                else np.nan
            )
            route_decision_match = (
                bool((actual_route_decision == 0) == (route_true_gap[sample_id, block_id] > 0.0))
                if offload_active
                else np.nan
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "block_id": block_id,
                    "sensor_id": sensor_id,
                    "task_id": task_id,
                    "A_theta": float(a_theta[sample_id, block_id]),
                    "A_route": float(a_route[sample_id, block_id]),
                    "local_score": float(local_score[sample_id, block_id]),
                    "bs1_score": float(bs1_score[sample_id, block_id]),
                    "bs2_score": float(bs2_score[sample_id, block_id]),
                    "best_offload_score": float(best_offload_score[sample_id, block_id]),
                    "theta_true_gap": float(theta_true_gap[sample_id, block_id]),
                    "route_true_gap": float(route_true_gap[sample_id, block_id]),
                    "offload_active_mask": int(offload_active),
                    "theta_sign_match": int(
                        (a_theta[sample_id, block_id] > 0.0)
                        == (theta_true_gap[sample_id, block_id] > 0.0)
                    ),
                    "route_sign_match": route_sign_match,
                    "actual_theta_decision": int(actual_offload[sample_id, block_id]),
                    "actual_route_decision": actual_route_decision,
                    "theta_decision_match": int(
                        (actual_offload[sample_id, block_id] == 1)
                        == (theta_true_gap[sample_id, block_id] > 0.0)
                    ),
                    "route_decision_match": route_decision_match,
                }
            )

    raw_df = pd.DataFrame(rows)
    theta_df = raw_df.copy()
    route_df = raw_df[raw_df["offload_active_mask"] == 1].copy()

    if not theta_df.empty:
        theta_df["theta_true_gap_bucket"] = pd.qcut(
            theta_df["theta_true_gap"],
            q=5,
            duplicates="drop",
        )
        theta_df["A_theta_bucket"] = pd.qcut(
            theta_df["A_theta"],
            q=5,
            duplicates="drop",
        )
    if not route_df.empty:
        route_df["route_true_gap_bucket"] = pd.qcut(
            route_df["route_true_gap"],
            q=5,
            duplicates="drop",
        )
        route_df["A_route_bucket"] = pd.qcut(
            route_df["A_route"],
            q=5,
            duplicates="drop",
        )

    theta_corr = _correlation_summary(theta_df, "A_theta", "theta_true_gap")
    route_corr = _correlation_summary(route_df, "A_route", "route_true_gap")

    theta_confusion = _binary_confusion(theta_df["A_theta"] > 0.0, theta_df["theta_true_gap"] > 0.0)
    route_confusion = _binary_confusion(route_df["A_route"] > 0.0, route_df["route_true_gap"] > 0.0)

    theta_true_gap_bucket_summary = _bucket_summary(
        theta_df,
        "theta_true_gap_bucket",
        "A_theta",
        "theta_true_gap",
    )
    theta_a_bucket_summary = _bucket_summary(
        theta_df,
        "A_theta_bucket",
        "theta_true_gap",
        "A_theta",
    )
    route_true_gap_bucket_summary = _bucket_summary(
        route_df,
        "route_true_gap_bucket",
        "A_route",
        "route_true_gap",
    )
    route_a_bucket_summary = _bucket_summary(
        route_df,
        "A_route_bucket",
        "route_true_gap",
        "A_route",
    )

    theta_decision_agreement_ratio = float(theta_df["theta_decision_match"].mean()) if not theta_df.empty else 0.0
    route_decision_agreement_ratio = float(route_df["route_decision_match"].mean()) if not route_df.empty else 0.0

    theta_true_gap_decision_rates = _decision_rate(
        theta_df["actual_theta_decision"],
        theta_df["theta_true_gap"] > 0.0,
    )
    theta_a_decision_rates = _decision_rate(
        theta_df["actual_theta_decision"],
        theta_df["A_theta"] > 0.0,
    )
    route_true_gap_decision_rates = _decision_rate(
        (route_df["actual_route_decision"] == 0).astype(float),
        route_df["route_true_gap"] > 0.0,
    )
    route_a_decision_rates = _decision_rate(
        (route_df["actual_route_decision"] == 0).astype(float),
        route_df["A_route"] > 0.0,
    )

    output_root = Path(args.output_root)
    root_dir = output_root / datetime.now().strftime("branch_credit_semantics_audit_%Y%m%d_%H%M%S")
    root_dir.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(root_dir / "branch_credit_semantics_audit_raw.csv", index=False)
    theta_df.to_csv(root_dir / "theta_credit_semantics.csv", index=False)
    route_df.to_csv(root_dir / "route_credit_semantics.csv", index=False)

    summary = {
        "baseline_mode": BASELINE_MODE,
        "checkpoint_path": str(checkpoint_path),
        "rollout_seed": args.seed,
        "sample_count": int(sample_count),
        "block_count": int(block_count),
        "sensor_count": int(sensor_count),
        "task_count": int(task_count),
        "block_mapping_definition": "sensor_id = block_id // task_count, task_id = block_id % task_count",
        "score_definition": {
            "local_score": "-local_path_cost",
            "bs1_score": "-bs1_path_cost",
            "bs2_score": "-bs2_path_cost",
            "best_offload_score": "max(bs1_score, bs2_score)",
            "theta_true_gap": "best_offload_score - local_score",
            "route_true_gap": "bs1_score - bs2_score",
        },
        "theta_audit": {
            "correlation": theta_corr,
            "confusion_matrix": theta_confusion,
            "a_theta_stats": _series_stats(theta_df["A_theta"]),
            "theta_true_gap_stats": _series_stats(theta_df["theta_true_gap"]),
            "theta_true_gap_bucket_summary": theta_true_gap_bucket_summary,
            "a_theta_bucket_summary": theta_a_bucket_summary,
            "decision": {
                "offload_decision_agreement_ratio": theta_decision_agreement_ratio,
                "actual_offload_rate_given_theta_true_gap_sign": theta_true_gap_decision_rates,
                "actual_offload_rate_given_a_theta_sign": theta_a_decision_rates,
            },
        },
        "route_audit": {
            "active_route_sample_count": int(len(route_df)),
            "correlation": route_corr,
            "confusion_matrix": route_confusion,
            "a_route_stats": _series_stats(route_df["A_route"]),
            "route_true_gap_stats": _series_stats(route_df["route_true_gap"]),
            "route_true_gap_bucket_summary": route_true_gap_bucket_summary,
            "a_route_bucket_summary": route_a_bucket_summary,
            "decision": {
                "route_decision_agreement_ratio": route_decision_agreement_ratio,
                "actual_bs1_rate_given_route_true_gap_sign": route_true_gap_decision_rates,
                "actual_bs1_rate_given_a_route_sign": route_a_decision_rates,
            },
        },
    }

    (root_dir / "branch_credit_semantics_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    recommendation_lines = [
        "# Branch Credit Semantics Audit",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- baseline mode: `{BASELINE_MODE}`",
        f"- rollout samples: `{sample_count}`",
        "",
        "## Theta",
        "",
        (
            f"- Pearson: `{theta_corr['pearson']:.4f}`, Spearman: `{theta_corr['spearman']:.4f}`, "
            f"sign agreement: `{theta_corr['sign_agreement_ratio']:.4f}`"
        ),
        f"- confusion: `{theta_confusion}`",
        (
            f"- offload decision agreement: `{theta_decision_agreement_ratio:.4f}`"
        ),
        "",
        "## Route",
        "",
        (
            f"- Pearson: `{route_corr['pearson']:.4f}`, Spearman: `{route_corr['spearman']:.4f}`, "
            f"sign agreement: `{route_corr['sign_agreement_ratio']:.4f}`"
        ),
        f"- confusion: `{route_confusion}`",
        (
            f"- route decision agreement: `{route_decision_agreement_ratio:.4f}`"
        ),
        "",
    ]
    (root_dir / "branch_credit_semantics_audit_summary.md").write_text(
        "\n".join(recommendation_lines),
        encoding="utf-8",
    )

    print(f"[branch_credit_semantics_audit] wrote results to {root_dir}")


if __name__ == "__main__":
    main()
