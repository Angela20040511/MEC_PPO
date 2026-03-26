import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from branch_credit_semantics_audit import _build_agent, _collect_rollout, _series_stats
from fine_grained_epoch_early_stop_search import set_global_seeds


BASELINE_MODE = "hierarchical_actor_true_conditional_route_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static audit for whether path-cost-based joint 3-action semantics align "
            "with the reward / return / advantage objective used by current training."
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
            "baseline checkpoint found under checkpoints/ is used."
        ),
    )
    return parser.parse_args()


def _find_latest_baseline_checkpoint() -> Path:
    matches = sorted(
        Path("checkpoints").glob(f"**/policy_ratio_{BASELINE_MODE}_*/best_model.pt")
    )
    if not matches:
        raise FileNotFoundError(
            f"No baseline best_model.pt found for mode {BASELINE_MODE}"
        )
    return matches[-1]


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    if left.empty or right.empty:
        return 0.0
    value = left.corr(right, method=method)
    return 0.0 if pd.isna(value) else float(value)


def _binary_confusion(pred_positive: pd.Series, true_positive: pd.Series) -> dict[str, int]:
    pred = pred_positive.astype(bool)
    true = true_positive.astype(bool)
    return {
        "tp": int((pred & true).sum()),
        "fp": int((pred & ~true).sum()),
        "tn": int((~pred & ~true).sum()),
        "fn": int((~pred & true).sum()),
    }


def _bucket_labels(series: pd.Series, bucket_count: int = 5) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="object")
    if series.nunique(dropna=False) <= 1:
        return pd.Series(["all"] * len(series), index=series.index, dtype="object")
    return pd.qcut(series, q=min(bucket_count, series.nunique()), duplicates="drop").astype(str)


def _bucket_summary(
    df: pd.DataFrame,
    bucket_col: str,
    value_col: str,
    target_col: str,
) -> list[dict[str, Any]]:
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
                f"{target_col}_mean": float(group[target_col].mean()),
                f"{target_col}_median": float(group[target_col].median()),
                f"{target_col}_positive_ratio": float((group[target_col] > 0.0).mean()),
            }
        )
    return rows


def _conditional_rate(
    mask: pd.Series,
    predicate: pd.Series,
) -> float:
    valid = mask.astype(bool)
    if not valid.any():
        return 0.0
    return float(predicate[valid].mean())


def _action_name_from_index(index: int) -> str:
    if index == 0:
        return "local"
    if index == 1:
        return "bs1"
    if index == 2:
        return "bs2"
    return f"unknown_{index}"


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
    raw_advantages = data["advantages"].detach().cpu()
    raw_returns = data["returns"].detach().cpu()
    task_count = int(agent.critic_state_layout["task_count"])

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
        block_path_rewards, _, _ = agent._block_path_td_rewards(states, next_states)
        theta_terms = agent._true_conditional_theta_policy_terms(actions, actions)
        route_terms = agent._true_conditional_route_policy_terms(actions, actions)

    a_theta = split_advantages["theta_advantages"].detach().cpu().numpy()
    a_route = split_advantages["route_advantages"].detach().cpu().numpy()
    theta_true_gap = (
        torch.max(-block_path_cost_bs[..., 0], -block_path_cost_bs[..., 1]) + block_path_cost_local
    ).detach().cpu().numpy()
    route_true_gap = (
        (-block_path_cost_bs[..., 0]) - (-block_path_cost_bs[..., 1])
    ).detach().cpu().numpy()

    offload_active_mask = theta_terms["offload_active_mask"].detach().cpu().numpy().astype(bool)
    selected_route_index = route_terms["selected_indices"].detach().cpu().numpy().astype(np.int64)
    local_prob = theta_terms["local_probs"].detach().cpu().numpy()
    offload_prob = theta_terms["offload_probs"].detach().cpu().numpy()
    route_old_probs = route_terms["probs"].detach().cpu().numpy()

    score_local = (-block_path_cost_local).detach().cpu().numpy()
    score_bs1 = (-block_path_cost_bs[..., 0]).detach().cpu().numpy()
    score_bs2 = (-block_path_cost_bs[..., 1]).detach().cpu().numpy()
    block_path_rewards_np = block_path_rewards.detach().cpu().numpy()
    selected_joint_reward_proxy = np.zeros_like(score_local)

    joint_score_tensor = np.stack([score_local, score_bs1, score_bs2], axis=-1)
    best_joint_action = np.argmax(joint_score_tensor, axis=-1)
    best_joint_score = joint_score_tensor.max(axis=-1)
    sorted_joint_scores = np.sort(joint_score_tensor, axis=-1)
    second_best_joint_score = sorted_joint_scores[..., -2]

    actual_joint_action = np.zeros_like(best_joint_action)
    actual_joint_action[offload_active_mask & (selected_route_index == 0)] = 1
    actual_joint_action[offload_active_mask & (selected_route_index == 1)] = 2
    selected_joint_score = np.take_along_axis(
        joint_score_tensor,
        actual_joint_action[..., None],
        axis=-1,
    ).squeeze(-1)
    selected_joint_reward_proxy = np.take_along_axis(
        block_path_rewards_np,
        actual_joint_action[..., None],
        axis=-1,
    ).squeeze(-1)

    induced_joint_action = np.where(a_theta <= 0.0, 0, np.where(a_route >= 0.0, 1, 2))

    rows: list[dict[str, Any]] = []
    sample_count, block_count = a_theta.shape
    for sample_id in range(sample_count):
        for block_id in range(block_count):
            sensor_id = int(block_id // task_count)
            task_id = int(block_id % task_count)
            row = {
                "sample_id": sample_id,
                "block_id": block_id,
                "sensor_id": sensor_id,
                "task_id": task_id,
                "score_local": float(score_local[sample_id, block_id]),
                "score_bs1": float(score_bs1[sample_id, block_id]),
                "score_bs2": float(score_bs2[sample_id, block_id]),
                "best_joint_action_under_score": _action_name_from_index(
                    int(best_joint_action[sample_id, block_id])
                ),
                "best_joint_action_under_score_index": int(best_joint_action[sample_id, block_id]),
                "best_joint_score": float(best_joint_score[sample_id, block_id]),
                "actual_theta_decision": int(offload_active_mask[sample_id, block_id]),
                "actual_route_decision": (
                    int(selected_route_index[sample_id, block_id])
                    if offload_active_mask[sample_id, block_id]
                    else -1
                ),
                "actual_joint_action": _action_name_from_index(
                    int(actual_joint_action[sample_id, block_id])
                ),
                "actual_joint_action_index": int(actual_joint_action[sample_id, block_id]),
                "selected_joint_score": float(selected_joint_score[sample_id, block_id]),
                "selected_joint_reward_proxy": float(
                    selected_joint_reward_proxy[sample_id, block_id]
                ),
                "A_theta": float(a_theta[sample_id, block_id]),
                "A_route": float(a_route[sample_id, block_id]),
                "advantage": float(raw_advantages[sample_id].item()),
                "return": float(raw_returns[sample_id].item()),
                "theta_true_gap": float(theta_true_gap[sample_id, block_id]),
                "route_true_gap": float(route_true_gap[sample_id, block_id]),
                "joint_action_match_under_score": int(
                    actual_joint_action[sample_id, block_id] == best_joint_action[sample_id, block_id]
                ),
                "joint_score_margin": float(
                    best_joint_score[sample_id, block_id] - selected_joint_score[sample_id, block_id]
                ),
                "best_minus_selected_score": float(
                    best_joint_score[sample_id, block_id] - selected_joint_score[sample_id, block_id]
                ),
                "best_minus_second_best_score": float(
                    best_joint_score[sample_id, block_id] - second_best_joint_score[sample_id, block_id]
                ),
                "best_offload_score": float(
                    max(score_bs1[sample_id, block_id], score_bs2[sample_id, block_id])
                ),
                "joint_selected_is_best": int(
                    selected_joint_score[sample_id, block_id] >= best_joint_score[sample_id, block_id] - 1e-12
                ),
                "induced_joint_action_from_A": _action_name_from_index(
                    int(induced_joint_action[sample_id, block_id])
                ),
                "induced_joint_action_from_A_index": int(induced_joint_action[sample_id, block_id]),
                "induced_joint_match_under_score": int(
                    induced_joint_action[sample_id, block_id] == best_joint_action[sample_id, block_id]
                ),
                "pi_old_local": float(local_prob[sample_id, block_id]),
                "pi_old_offload": float(offload_prob[sample_id, block_id]),
                "pi_old_bs1_conditional": float(route_old_probs[sample_id, block_id, 0]),
                "pi_old_bs2_conditional": float(route_old_probs[sample_id, block_id, 1]),
            }
            rows.append(row)

    raw_df = pd.DataFrame(rows)
    raw_df["selected_score_bucket"] = _bucket_labels(raw_df["selected_joint_score"])
    raw_df["advantage_bucket"] = _bucket_labels(raw_df["advantage"])
    raw_df["return_bucket"] = _bucket_labels(raw_df["return"])
    raw_df["theta_gap_bucket"] = _bucket_labels(raw_df["theta_true_gap"])
    raw_df["route_gap_bucket"] = _bucket_labels(raw_df["route_true_gap"])

    route_df = raw_df[raw_df["actual_theta_decision"] == 1].copy()

    summary = {
        "metadata": {
            "baseline_mode": BASELINE_MODE,
            "checkpoint_path": str(checkpoint_path),
            "seed": args.seed,
            "sample_count": int(sample_count),
            "block_count": int(block_count),
            "row_count": int(len(raw_df)),
            "route_active_row_count": int(len(route_df)),
        },
        "joint_score_vs_objective": {
            "selected_joint_score_vs_advantage": {
                "pearson": _safe_corr(raw_df["selected_joint_score"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_joint_score"], raw_df["advantage"], "spearman"),
            },
            "selected_joint_score_vs_return": {
                "pearson": _safe_corr(raw_df["selected_joint_score"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_joint_score"], raw_df["return"], "spearman"),
            },
            "best_joint_score_vs_advantage": {
                "pearson": _safe_corr(raw_df["best_joint_score"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["best_joint_score"], raw_df["advantage"], "spearman"),
            },
            "best_joint_score_vs_return": {
                "pearson": _safe_corr(raw_df["best_joint_score"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["best_joint_score"], raw_df["return"], "spearman"),
            },
            "selected_joint_score_bucket_vs_advantage": _bucket_summary(
                raw_df, "selected_score_bucket", "selected_joint_score", "advantage"
            ),
            "selected_joint_score_bucket_vs_return": _bucket_summary(
                raw_df, "selected_score_bucket", "selected_joint_score", "return"
            ),
            "advantage_bucket_vs_selected_joint_score": _bucket_summary(
                raw_df, "advantage_bucket", "advantage", "selected_joint_score"
            ),
            "return_bucket_vs_selected_joint_score": _bucket_summary(
                raw_df, "return_bucket", "return", "selected_joint_score"
            ),
        },
        "joint_action_semantics": {
            "joint_action_decision_agreement_ratio": float(
                raw_df["joint_action_match_under_score"].mean()
            ),
            "actual_local_rate_when_best_action_local": _conditional_rate(
                raw_df["best_joint_action_under_score_index"] == 0,
                raw_df["actual_joint_action_index"] == 0,
            ),
            "actual_bs1_rate_when_best_action_bs1": _conditional_rate(
                raw_df["best_joint_action_under_score_index"] == 1,
                raw_df["actual_joint_action_index"] == 1,
            ),
            "actual_bs2_rate_when_best_action_bs2": _conditional_rate(
                raw_df["best_joint_action_under_score_index"] == 2,
                raw_df["actual_joint_action_index"] == 2,
            ),
            "best_action_counts": {
                "local": int((raw_df["best_joint_action_under_score_index"] == 0).sum()),
                "bs1": int((raw_df["best_joint_action_under_score_index"] == 1).sum()),
                "bs2": int((raw_df["best_joint_action_under_score_index"] == 2).sum()),
            },
        },
        "branch_credit_semantics": {
            "A_theta_vs_theta_true_gap": {
                "pearson": _safe_corr(raw_df["A_theta"], raw_df["theta_true_gap"], "pearson"),
                "spearman": _safe_corr(raw_df["A_theta"], raw_df["theta_true_gap"], "spearman"),
                "sign_agreement_ratio": float(
                    ((raw_df["A_theta"] > 0.0) == (raw_df["theta_true_gap"] > 0.0)).mean()
                ),
                "confusion": _binary_confusion(
                    raw_df["A_theta"] > 0.0,
                    raw_df["theta_true_gap"] > 0.0,
                ),
            },
            "A_route_vs_route_true_gap": {
                "pearson": _safe_corr(route_df["A_route"], route_df["route_true_gap"], "pearson"),
                "spearman": _safe_corr(route_df["A_route"], route_df["route_true_gap"], "spearman"),
                "sign_agreement_ratio": float(
                    ((route_df["A_route"] > 0.0) == (route_df["route_true_gap"] > 0.0)).mean()
                )
                if not route_df.empty
                else 0.0,
                "confusion": _binary_confusion(
                    route_df["A_route"] > 0.0,
                    route_df["route_true_gap"] > 0.0,
                )
                if not route_df.empty
                else {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            },
            "joint_action_induced_by_A_match_ratio": float(
                raw_df["induced_joint_match_under_score"].mean()
            ),
        },
        "path_cost_vs_reward_conflict": {
            "path_cost_best_action_has_negative_advantage_ratio": float(
                (raw_df.loc[raw_df["joint_action_match_under_score"] == 1, "advantage"] < 0.0).mean()
            ),
            "path_cost_best_action_has_negative_return_ratio": float(
                (raw_df.loc[raw_df["joint_action_match_under_score"] == 1, "return"] < 0.0).mean()
            ),
            "path_cost_suboptimal_action_has_positive_advantage_ratio": float(
                (raw_df.loc[raw_df["joint_action_match_under_score"] == 0, "advantage"] > 0.0).mean()
            ),
            "path_cost_suboptimal_action_has_positive_return_ratio": float(
                (raw_df.loc[raw_df["joint_action_match_under_score"] == 0, "return"] > 0.0).mean()
            ),
            "actual_offload_rate_when_path_cost_prefers_local": _conditional_rate(
                raw_df["best_joint_action_under_score_index"] == 0,
                raw_df["actual_theta_decision"] == 1,
            ),
            "actual_bs1_rate_when_path_cost_prefers_bs2": _conditional_rate(
                raw_df["best_joint_action_under_score_index"] == 2,
                raw_df["actual_joint_action_index"] == 1,
            ),
            "actual_bs2_rate_when_path_cost_prefers_bs1": _conditional_rate(
                raw_df["best_joint_action_under_score_index"] == 1,
                raw_df["actual_joint_action_index"] == 2,
            ),
            "best_minus_selected_score_stats": _series_stats(raw_df["best_minus_selected_score"]),
            "best_minus_second_best_score_stats": _series_stats(raw_df["best_minus_second_best_score"]),
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"joint_credit_reward_semantics_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = output_dir / "joint_credit_reward_semantics_audit_raw.csv"
    joint_action_semantics_path = output_dir / "joint_action_semantics.csv"
    joint_score_vs_advantage_path = output_dir / "joint_score_vs_advantage.csv"
    joint_score_vs_return_path = output_dir / "joint_score_vs_return.csv"
    summary_json_path = output_dir / "joint_credit_reward_semantics_audit_summary.json"
    summary_md_path = output_dir / "joint_credit_reward_semantics_audit_summary.md"

    raw_df.to_csv(raw_csv_path, index=False)
    raw_df[
        [
            "sample_id",
            "block_id",
            "best_joint_action_under_score",
            "actual_joint_action",
            "joint_action_match_under_score",
            "selected_joint_score",
            "best_joint_score",
            "best_minus_selected_score",
            "advantage",
            "return",
        ]
    ].to_csv(joint_action_semantics_path, index=False)
    raw_df[
        [
            "sample_id",
            "block_id",
            "selected_joint_score",
            "best_joint_score",
            "advantage",
            "selected_joint_reward_proxy",
        ]
    ].to_csv(joint_score_vs_advantage_path, index=False)
    raw_df[
        [
            "sample_id",
            "block_id",
            "selected_joint_score",
            "best_joint_score",
            "return",
            "selected_joint_reward_proxy",
        ]
    ].to_csv(joint_score_vs_return_path, index=False)

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    md_lines = [
        "# Joint Credit Reward Semantics Audit",
        "",
        f"- baseline_mode: `{BASELINE_MODE}`",
        f"- checkpoint_path: `{checkpoint_path}`",
        f"- sample_count: {sample_count}",
        f"- block_count: {block_count}",
        f"- row_count: {len(raw_df)}",
        "",
        "## Joint Score vs Objective",
        f"- selected_joint_score vs advantage: pearson={summary['joint_score_vs_objective']['selected_joint_score_vs_advantage']['pearson']:.4f}, spearman={summary['joint_score_vs_objective']['selected_joint_score_vs_advantage']['spearman']:.4f}",
        f"- selected_joint_score vs return: pearson={summary['joint_score_vs_objective']['selected_joint_score_vs_return']['pearson']:.4f}, spearman={summary['joint_score_vs_objective']['selected_joint_score_vs_return']['spearman']:.4f}",
        f"- best_joint_score vs advantage: pearson={summary['joint_score_vs_objective']['best_joint_score_vs_advantage']['pearson']:.4f}, spearman={summary['joint_score_vs_objective']['best_joint_score_vs_advantage']['spearman']:.4f}",
        f"- best_joint_score vs return: pearson={summary['joint_score_vs_objective']['best_joint_score_vs_return']['pearson']:.4f}, spearman={summary['joint_score_vs_objective']['best_joint_score_vs_return']['spearman']:.4f}",
        "",
        "## Joint Action Agreement",
        f"- joint_action_decision_agreement_ratio: {summary['joint_action_semantics']['joint_action_decision_agreement_ratio']:.4f}",
        f"- actual_local_rate_when_best_action_local: {summary['joint_action_semantics']['actual_local_rate_when_best_action_local']:.4f}",
        f"- actual_bs1_rate_when_best_action_bs1: {summary['joint_action_semantics']['actual_bs1_rate_when_best_action_bs1']:.4f}",
        f"- actual_bs2_rate_when_best_action_bs2: {summary['joint_action_semantics']['actual_bs2_rate_when_best_action_bs2']:.4f}",
        "",
        "## Branch Credit Semantics",
        f"- A_theta vs theta_true_gap: pearson={summary['branch_credit_semantics']['A_theta_vs_theta_true_gap']['pearson']:.4f}, spearman={summary['branch_credit_semantics']['A_theta_vs_theta_true_gap']['spearman']:.4f}, sign_agreement={summary['branch_credit_semantics']['A_theta_vs_theta_true_gap']['sign_agreement_ratio']:.4f}",
        f"- A_route vs route_true_gap: pearson={summary['branch_credit_semantics']['A_route_vs_route_true_gap']['pearson']:.4f}, spearman={summary['branch_credit_semantics']['A_route_vs_route_true_gap']['spearman']:.4f}, sign_agreement={summary['branch_credit_semantics']['A_route_vs_route_true_gap']['sign_agreement_ratio']:.4f}",
        f"- joint_action_induced_by_A_match_ratio: {summary['branch_credit_semantics']['joint_action_induced_by_A_match_ratio']:.4f}",
        "",
        "## Path Cost vs Reward Conflict",
        f"- path_cost_best_action_has_negative_advantage_ratio: {summary['path_cost_vs_reward_conflict']['path_cost_best_action_has_negative_advantage_ratio']:.4f}",
        f"- path_cost_best_action_has_negative_return_ratio: {summary['path_cost_vs_reward_conflict']['path_cost_best_action_has_negative_return_ratio']:.4f}",
        f"- path_cost_suboptimal_action_has_positive_advantage_ratio: {summary['path_cost_vs_reward_conflict']['path_cost_suboptimal_action_has_positive_advantage_ratio']:.4f}",
        f"- path_cost_suboptimal_action_has_positive_return_ratio: {summary['path_cost_vs_reward_conflict']['path_cost_suboptimal_action_has_positive_return_ratio']:.4f}",
        f"- actual_offload_rate_when_path_cost_prefers_local: {summary['path_cost_vs_reward_conflict']['actual_offload_rate_when_path_cost_prefers_local']:.4f}",
        f"- actual_bs1_rate_when_path_cost_prefers_bs2: {summary['path_cost_vs_reward_conflict']['actual_bs1_rate_when_path_cost_prefers_bs2']:.4f}",
        f"- actual_bs2_rate_when_path_cost_prefers_bs1: {summary['path_cost_vs_reward_conflict']['actual_bs2_rate_when_path_cost_prefers_bs1']:.4f}",
        "",
    ]
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote audit summary to {summary_json_path}")


if __name__ == "__main__":
    main()
