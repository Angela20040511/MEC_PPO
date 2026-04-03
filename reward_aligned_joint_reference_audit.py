import argparse
import copy
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from branch_credit_semantics_audit import _build_agent, _series_stats
from fine_grained_epoch_early_stop_search import set_global_seeds
from simulator.simulator import Simulator


BASELINE_MODE = "hierarchical_actor_true_conditional_route_policy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static audit comparing path-cost joint reference against a reward-aligned "
            "joint reference built from one-step simulator reward counterfactuals."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    parser.add_argument(
        "--audit-steps",
        type=int,
        default=24,
        help="Number of rollout steps to audit with full counterfactual scoring.",
    )
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


def _action_name(index: int) -> str:
    if index == 0:
        return "local"
    if index == 1:
        return "bs1"
    if index == 2:
        return "bs2"
    return f"unknown_{index}"


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


def _conditional_rate(mask: pd.Series, predicate: pd.Series) -> float:
    valid = mask.astype(bool)
    if not valid.any():
        return 0.0
    return float(predicate[valid].mean())


def _set_block_joint_candidate(action: np.ndarray, block_slice: slice, candidate_index: int) -> np.ndarray:
    variant = np.array(action, copy=True)
    block = variant[block_slice].copy()
    if block.shape[0] != 3:
        raise ValueError(f"Expected 3-dim action block, got {block.shape[0]}")
    if candidate_index == 0:
        block[0] = -20.0
        block[1:] = 0.0
    elif candidate_index == 1:
        block[0] = 20.0
        block[1] = 20.0
        block[2] = -20.0
    elif candidate_index == 2:
        block[0] = 20.0
        block[1] = -20.0
        block[2] = 20.0
    else:
        raise ValueError(f"Unsupported candidate index: {candidate_index}")
    variant[block_slice] = block
    return variant


def _collect_rollout_with_counterfactuals(
    agent: Any,
    config: Any,
    seed: int,
    audit_steps: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    simulator = Simulator(config)
    state = simulator.reset(seed=seed)
    done = False
    block_slices = agent._action_block_slices()
    sample_payloads: list[dict[str, Any]] = []

    max_steps = min(int(audit_steps), int(config.training.time_steps))
    for sample_id in range(max_steps):
        action, log_prob, value, policy_cache = agent.select_action_with_info(state)
        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        sample_counterfactual_scores = np.zeros((len(block_slices), 3), dtype=np.float32)
        sample_counterfactual_delay = np.zeros((len(block_slices), 3), dtype=np.float32)
        sample_counterfactual_energy = np.zeros((len(block_slices), 3), dtype=np.float32)
        sample_counterfactual_backlog = np.zeros((len(block_slices), 3), dtype=np.float32)

        for block_id, block_slice in enumerate(block_slices):
            for candidate_index in range(3):
                variant_action = _set_block_joint_candidate(action_np, block_slice, candidate_index)
                simulator_cf = copy.deepcopy(simulator)
                _, reward_cf, _, info_cf = simulator_cf.step(variant_action)
                sample_counterfactual_scores[block_id, candidate_index] = float(reward_cf)
                sample_counterfactual_delay[block_id, candidate_index] = float(info_cf["reward_delay_term"])
                sample_counterfactual_energy[block_id, candidate_index] = float(info_cf["reward_energy_term"])
                sample_counterfactual_backlog[block_id, candidate_index] = float(info_cf["reward_backlog_term"])

        next_state, reward, done, info = simulator.step(action_np)
        agent.store_transition(
            state,
            action_np,
            log_prob,
            reward,
            done,
            value,
            next_state,
            policy_cache=policy_cache,
        )
        sample_payloads.append(
            {
                "sample_id": sample_id,
                "counterfactual_scores": sample_counterfactual_scores,
                "counterfactual_delay_terms": sample_counterfactual_delay,
                "counterfactual_energy_terms": sample_counterfactual_energy,
                "counterfactual_backlog_terms": sample_counterfactual_backlog,
                "actual_reward": float(reward),
                "actual_reward_delay_term": float(info["reward_delay_term"]),
                "actual_reward_energy_term": float(info["reward_energy_term"]),
                "actual_reward_backlog_term": float(info["reward_backlog_term"]),
            }
        )
        state = next_state
        if done:
            break

    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)
    return agent.buffer.as_tensors(agent.device), sample_payloads


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
    data, sample_payloads = _collect_rollout_with_counterfactuals(
        agent,
        config,
        args.seed,
        args.audit_steps,
    )

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
        theta_terms = agent._true_conditional_theta_policy_terms(actions, actions)
        route_terms = agent._true_conditional_route_policy_terms(actions, actions)

    a_theta = split_advantages["theta_advantages"].detach().cpu().numpy()
    a_route = split_advantages["route_advantages"].detach().cpu().numpy()
    offload_active_mask = theta_terms["offload_active_mask"].detach().cpu().numpy().astype(bool)
    selected_route_index = route_terms["selected_indices"].detach().cpu().numpy().astype(np.int64)

    score_local_path = (-block_path_cost_local).detach().cpu().numpy()
    score_bs1_path = (-block_path_cost_bs[..., 0]).detach().cpu().numpy()
    score_bs2_path = (-block_path_cost_bs[..., 1]).detach().cpu().numpy()
    path_joint_scores = np.stack([score_local_path, score_bs1_path, score_bs2_path], axis=-1)

    reward_joint_scores = np.stack(
        [payload["counterfactual_scores"] for payload in sample_payloads],
        axis=0,
    )
    reward_joint_delay_terms = np.stack(
        [payload["counterfactual_delay_terms"] for payload in sample_payloads],
        axis=0,
    )
    reward_joint_energy_terms = np.stack(
        [payload["counterfactual_energy_terms"] for payload in sample_payloads],
        axis=0,
    )
    reward_joint_backlog_terms = np.stack(
        [payload["counterfactual_backlog_terms"] for payload in sample_payloads],
        axis=0,
    )

    actual_joint_action = np.zeros_like(a_theta, dtype=np.int64)
    actual_joint_action[offload_active_mask & (selected_route_index == 0)] = 1
    actual_joint_action[offload_active_mask & (selected_route_index == 1)] = 2

    selected_joint_score_path = np.take_along_axis(
        path_joint_scores,
        actual_joint_action[..., None],
        axis=-1,
    ).squeeze(-1)
    selected_joint_score_reward = np.take_along_axis(
        reward_joint_scores,
        actual_joint_action[..., None],
        axis=-1,
    ).squeeze(-1)

    best_joint_action_under_path = np.argmax(path_joint_scores, axis=-1)
    best_joint_action_under_reward = np.argmax(reward_joint_scores, axis=-1)
    best_joint_score_path = path_joint_scores.max(axis=-1)
    best_joint_score_reward = reward_joint_scores.max(axis=-1)

    theta_true_gap_path = np.maximum(score_bs1_path, score_bs2_path) - score_local_path
    route_true_gap_path = score_bs1_path - score_bs2_path
    theta_true_gap_reward = np.maximum(
        reward_joint_scores[..., 1], reward_joint_scores[..., 2]
    ) - reward_joint_scores[..., 0]
    route_true_gap_reward = reward_joint_scores[..., 1] - reward_joint_scores[..., 2]

    rows: list[dict[str, Any]] = []
    sample_count, block_count = a_theta.shape
    for sample_id in range(sample_count):
        payload = sample_payloads[sample_id]
        for block_id in range(block_count):
            sensor_id = int(block_id // task_count)
            task_id = int(block_id % task_count)
            rows.append(
                {
                    "sample_id": sample_id,
                    "block_id": block_id,
                    "sensor_id": sensor_id,
                    "task_id": task_id,
                    "score_local_path": float(score_local_path[sample_id, block_id]),
                    "score_bs1_path": float(score_bs1_path[sample_id, block_id]),
                    "score_bs2_path": float(score_bs2_path[sample_id, block_id]),
                    "score_local_reward_aligned": float(reward_joint_scores[sample_id, block_id, 0]),
                    "score_bs1_reward_aligned": float(reward_joint_scores[sample_id, block_id, 1]),
                    "score_bs2_reward_aligned": float(reward_joint_scores[sample_id, block_id, 2]),
                    "best_joint_action_under_path": _action_name(int(best_joint_action_under_path[sample_id, block_id])),
                    "best_joint_action_under_path_index": int(best_joint_action_under_path[sample_id, block_id]),
                    "best_joint_action_under_reward_aligned": _action_name(int(best_joint_action_under_reward[sample_id, block_id])),
                    "best_joint_action_under_reward_aligned_index": int(best_joint_action_under_reward[sample_id, block_id]),
                    "selected_joint_action": _action_name(int(actual_joint_action[sample_id, block_id])),
                    "selected_joint_action_index": int(actual_joint_action[sample_id, block_id]),
                    "selected_joint_score_path": float(selected_joint_score_path[sample_id, block_id]),
                    "selected_joint_score_reward_aligned": float(selected_joint_score_reward[sample_id, block_id]),
                    "best_joint_score_path": float(best_joint_score_path[sample_id, block_id]),
                    "best_joint_score_reward_aligned": float(best_joint_score_reward[sample_id, block_id]),
                    "A_theta": float(a_theta[sample_id, block_id]),
                    "A_route": float(a_route[sample_id, block_id]),
                    "advantage": float(raw_advantages[sample_id].item()),
                    "return": float(raw_returns[sample_id].item()),
                    "theta_true_gap_path": float(theta_true_gap_path[sample_id, block_id]),
                    "route_true_gap_path": float(route_true_gap_path[sample_id, block_id]),
                    "theta_true_gap_reward_aligned": float(theta_true_gap_reward[sample_id, block_id]),
                    "route_true_gap_reward_aligned": float(route_true_gap_reward[sample_id, block_id]),
                    "joint_action_match_under_path": int(actual_joint_action[sample_id, block_id] == best_joint_action_under_path[sample_id, block_id]),
                    "joint_action_match_under_reward_aligned": int(actual_joint_action[sample_id, block_id] == best_joint_action_under_reward[sample_id, block_id]),
                    "reward_aligned_delay_term_local": float(reward_joint_delay_terms[sample_id, block_id, 0]),
                    "reward_aligned_delay_term_bs1": float(reward_joint_delay_terms[sample_id, block_id, 1]),
                    "reward_aligned_delay_term_bs2": float(reward_joint_delay_terms[sample_id, block_id, 2]),
                    "reward_aligned_energy_term_local": float(reward_joint_energy_terms[sample_id, block_id, 0]),
                    "reward_aligned_energy_term_bs1": float(reward_joint_energy_terms[sample_id, block_id, 1]),
                    "reward_aligned_energy_term_bs2": float(reward_joint_energy_terms[sample_id, block_id, 2]),
                    "reward_aligned_backlog_term_local": float(reward_joint_backlog_terms[sample_id, block_id, 0]),
                    "reward_aligned_backlog_term_bs1": float(reward_joint_backlog_terms[sample_id, block_id, 1]),
                    "reward_aligned_backlog_term_bs2": float(reward_joint_backlog_terms[sample_id, block_id, 2]),
                    "actual_reward": float(payload["actual_reward"]),
                    "actual_reward_delay_term": float(payload["actual_reward_delay_term"]),
                    "actual_reward_energy_term": float(payload["actual_reward_energy_term"]),
                    "actual_reward_backlog_term": float(payload["actual_reward_backlog_term"]),
                }
            )

    raw_df = pd.DataFrame(rows)
    raw_df["selected_joint_score_reward_aligned_bucket"] = _bucket_labels(raw_df["selected_joint_score_reward_aligned"])
    raw_df["selected_joint_score_path_bucket"] = _bucket_labels(raw_df["selected_joint_score_path"])
    raw_df["advantage_bucket"] = _bucket_labels(raw_df["advantage"])
    raw_df["return_bucket"] = _bucket_labels(raw_df["return"])

    route_df = raw_df[raw_df["selected_joint_action_index"] != 0].copy()
    conflict_mask = raw_df["best_joint_action_under_path_index"] != raw_df["best_joint_action_under_reward_aligned_index"]
    conflict_pairs = Counter(
        (row["best_joint_action_under_path"], row["best_joint_action_under_reward_aligned"])
        for _, row in raw_df.loc[conflict_mask].iterrows()
    )

    summary = {
        "metadata": {
            "baseline_mode": BASELINE_MODE,
            "checkpoint_path": str(checkpoint_path),
            "seed": args.seed,
            "audit_steps": int(args.audit_steps),
            "sample_count": int(sample_count),
            "block_count": int(block_count),
            "row_count": int(len(raw_df)),
        },
        "reward_aligned_reference_definition": {
            "score_definition": (
                "For each block and candidate joint action in {local, bs1, bs2}, clone the "
                "simulator at the current state, replace only that block's action with a "
                "deterministic candidate action, keep all other blocks at the actual rollout "
                "action, and use the resulting one-step PPO reward as the score."
            ),
            "reward_components_used": {
                "delay_term": "simulator info['reward_delay_term']",
                "energy_term": "simulator info['reward_energy_term']",
                "backlog_term": "simulator info['reward_backlog_term']",
                "final_score": "reward = -cost - queue_penalty from simulator.step(...)",
            },
        },
        "joint_score_vs_objective": {
            "path_selected_vs_advantage": {
                "pearson": _safe_corr(raw_df["selected_joint_score_path"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_joint_score_path"], raw_df["advantage"], "spearman"),
            },
            "path_selected_vs_return": {
                "pearson": _safe_corr(raw_df["selected_joint_score_path"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_joint_score_path"], raw_df["return"], "spearman"),
            },
            "path_best_vs_advantage": {
                "pearson": _safe_corr(raw_df["best_joint_score_path"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["best_joint_score_path"], raw_df["advantage"], "spearman"),
            },
            "path_best_vs_return": {
                "pearson": _safe_corr(raw_df["best_joint_score_path"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["best_joint_score_path"], raw_df["return"], "spearman"),
            },
            "reward_aligned_selected_vs_advantage": {
                "pearson": _safe_corr(raw_df["selected_joint_score_reward_aligned"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_joint_score_reward_aligned"], raw_df["advantage"], "spearman"),
            },
            "reward_aligned_selected_vs_return": {
                "pearson": _safe_corr(raw_df["selected_joint_score_reward_aligned"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_joint_score_reward_aligned"], raw_df["return"], "spearman"),
            },
            "reward_aligned_best_vs_advantage": {
                "pearson": _safe_corr(raw_df["best_joint_score_reward_aligned"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["best_joint_score_reward_aligned"], raw_df["advantage"], "spearman"),
            },
            "reward_aligned_best_vs_return": {
                "pearson": _safe_corr(raw_df["best_joint_score_reward_aligned"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["best_joint_score_reward_aligned"], raw_df["return"], "spearman"),
            },
        },
        "joint_action_semantics": {
            "joint_action_decision_agreement_ratio_under_path": float(raw_df["joint_action_match_under_path"].mean()),
            "joint_action_decision_agreement_ratio_under_reward_aligned": float(raw_df["joint_action_match_under_reward_aligned"].mean()),
            "actual_local_rate_when_reward_best_local": _conditional_rate(raw_df["best_joint_action_under_reward_aligned_index"] == 0, raw_df["selected_joint_action_index"] == 0),
            "actual_bs1_rate_when_reward_best_bs1": _conditional_rate(raw_df["best_joint_action_under_reward_aligned_index"] == 1, raw_df["selected_joint_action_index"] == 1),
            "actual_bs2_rate_when_reward_best_bs2": _conditional_rate(raw_df["best_joint_action_under_reward_aligned_index"] == 2, raw_df["selected_joint_action_index"] == 2),
            "actual_local_rate_when_path_best_local": _conditional_rate(raw_df["best_joint_action_under_path_index"] == 0, raw_df["selected_joint_action_index"] == 0),
            "actual_bs1_rate_when_path_best_bs1": _conditional_rate(raw_df["best_joint_action_under_path_index"] == 1, raw_df["selected_joint_action_index"] == 1),
            "actual_bs2_rate_when_path_best_bs2": _conditional_rate(raw_df["best_joint_action_under_path_index"] == 2, raw_df["selected_joint_action_index"] == 2),
        },
        "reference_conflict": {
            "path_best_vs_reward_aligned_best_match_ratio": float((raw_df["best_joint_action_under_path_index"] == raw_df["best_joint_action_under_reward_aligned_index"]).mean()),
            "conflict_pair_counts": {f"{path}->{reward}": int(count) for (path, reward), count in sorted(conflict_pairs.items())},
        },
        "branch_credit_vs_references": {
            "A_theta_vs_theta_true_gap_path": {
                "pearson": _safe_corr(raw_df["A_theta"], raw_df["theta_true_gap_path"], "pearson"),
                "spearman": _safe_corr(raw_df["A_theta"], raw_df["theta_true_gap_path"], "spearman"),
                "sign_agreement_ratio": float(((raw_df["A_theta"] > 0.0) == (raw_df["theta_true_gap_path"] > 0.0)).mean()),
            },
            "A_theta_vs_theta_true_gap_reward_aligned": {
                "pearson": _safe_corr(raw_df["A_theta"], raw_df["theta_true_gap_reward_aligned"], "pearson"),
                "spearman": _safe_corr(raw_df["A_theta"], raw_df["theta_true_gap_reward_aligned"], "spearman"),
                "sign_agreement_ratio": float(((raw_df["A_theta"] > 0.0) == (raw_df["theta_true_gap_reward_aligned"] > 0.0)).mean()),
            },
            "A_route_vs_route_true_gap_path": {
                "pearson": _safe_corr(route_df["A_route"], route_df["route_true_gap_path"], "pearson"),
                "spearman": _safe_corr(route_df["A_route"], route_df["route_true_gap_path"], "spearman"),
                "sign_agreement_ratio": float(((route_df["A_route"] > 0.0) == (route_df["route_true_gap_path"] > 0.0)).mean()) if not route_df.empty else 0.0,
            },
            "A_route_vs_route_true_gap_reward_aligned": {
                "pearson": _safe_corr(route_df["A_route"], route_df["route_true_gap_reward_aligned"], "pearson"),
                "spearman": _safe_corr(route_df["A_route"], route_df["route_true_gap_reward_aligned"], "spearman"),
                "sign_agreement_ratio": float(((route_df["A_route"] > 0.0) == (route_df["route_true_gap_reward_aligned"] > 0.0)).mean()) if not route_df.empty else 0.0,
            },
        },
        "bucket_summaries": {
            "reward_aligned_selected_score_bucket_vs_advantage": _bucket_summary(raw_df, "selected_joint_score_reward_aligned_bucket", "selected_joint_score_reward_aligned", "advantage"),
            "reward_aligned_selected_score_bucket_vs_return": _bucket_summary(raw_df, "selected_joint_score_reward_aligned_bucket", "selected_joint_score_reward_aligned", "return"),
            "path_selected_score_bucket_vs_advantage": _bucket_summary(raw_df, "selected_joint_score_path_bucket", "selected_joint_score_path", "advantage"),
            "path_selected_score_bucket_vs_return": _bucket_summary(raw_df, "selected_joint_score_path_bucket", "selected_joint_score_path", "return"),
        },
        "score_distribution": {
            "path_selected_score_stats": _series_stats(raw_df["selected_joint_score_path"]),
            "reward_aligned_selected_score_stats": _series_stats(raw_df["selected_joint_score_reward_aligned"]),
            "reward_aligned_delay_term_stats": _series_stats(raw_df["actual_reward_delay_term"]),
            "reward_aligned_energy_term_stats": _series_stats(raw_df["actual_reward_energy_term"]),
            "reward_aligned_backlog_term_stats": _series_stats(raw_df["actual_reward_backlog_term"]),
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"reward_aligned_joint_reference_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = output_dir / "reward_aligned_joint_reference_audit_raw.csv"
    summary_json_path = output_dir / "reward_aligned_joint_reference_audit_summary.json"
    summary_md_path = output_dir / "reward_aligned_joint_reference_audit_summary.md"
    joint_action_path = output_dir / "joint_action_semantics_path_vs_reward_aligned.csv"
    score_adv_path = output_dir / "joint_score_vs_advantage_path_vs_reward_aligned.csv"
    score_return_path = output_dir / "joint_score_vs_return_path_vs_reward_aligned.csv"

    raw_df.to_csv(raw_csv_path, index=False)
    raw_df[[
        "sample_id",
        "block_id",
        "selected_joint_action",
        "best_joint_action_under_path",
        "best_joint_action_under_reward_aligned",
        "joint_action_match_under_path",
        "joint_action_match_under_reward_aligned",
    ]].to_csv(joint_action_path, index=False)
    raw_df[[
        "sample_id",
        "block_id",
        "selected_joint_score_path",
        "selected_joint_score_reward_aligned",
        "best_joint_score_path",
        "best_joint_score_reward_aligned",
        "advantage",
    ]].to_csv(score_adv_path, index=False)
    raw_df[[
        "sample_id",
        "block_id",
        "selected_joint_score_path",
        "selected_joint_score_reward_aligned",
        "best_joint_score_path",
        "best_joint_score_reward_aligned",
        "return",
    ]].to_csv(score_return_path, index=False)

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    md_lines = [
        "# Reward Aligned Joint Reference Audit",
        "",
        f"- baseline_mode: `{BASELINE_MODE}`",
        f"- checkpoint_path: `{checkpoint_path}`",
        f"- sample_count: {sample_count}",
        f"- block_count: {block_count}",
        f"- row_count: {len(raw_df)}",
        "",
        "## Joint Score vs Objective",
        f"- path selected vs advantage: pearson={summary['joint_score_vs_objective']['path_selected_vs_advantage']['pearson']:.4f}, spearman={summary['joint_score_vs_objective']['path_selected_vs_advantage']['spearman']:.4f}",
        f"- reward-aligned selected vs advantage: pearson={summary['joint_score_vs_objective']['reward_aligned_selected_vs_advantage']['pearson']:.4f}, spearman={summary['joint_score_vs_objective']['reward_aligned_selected_vs_advantage']['spearman']:.4f}",
        f"- path selected vs return: pearson={summary['joint_score_vs_objective']['path_selected_vs_return']['pearson']:.4f}, spearman={summary['joint_score_vs_objective']['path_selected_vs_return']['spearman']:.4f}",
        f"- reward-aligned selected vs return: pearson={summary['joint_score_vs_objective']['reward_aligned_selected_vs_return']['pearson']:.4f}, spearman={summary['joint_score_vs_objective']['reward_aligned_selected_vs_return']['spearman']:.4f}",
        "",
        "## Joint Action Agreement",
        f"- under path: {summary['joint_action_semantics']['joint_action_decision_agreement_ratio_under_path']:.4f}",
        f"- under reward-aligned: {summary['joint_action_semantics']['joint_action_decision_agreement_ratio_under_reward_aligned']:.4f}",
        "",
        "## Reference Conflict",
        f"- path_best_vs_reward_aligned_best_match_ratio: {summary['reference_conflict']['path_best_vs_reward_aligned_best_match_ratio']:.4f}",
        "",
        "## Branch Credit vs Reward-Aligned Gaps",
        f"- A_theta vs theta_true_gap_path: pearson={summary['branch_credit_vs_references']['A_theta_vs_theta_true_gap_path']['pearson']:.4f}, spearman={summary['branch_credit_vs_references']['A_theta_vs_theta_true_gap_path']['spearman']:.4f}, sign={summary['branch_credit_vs_references']['A_theta_vs_theta_true_gap_path']['sign_agreement_ratio']:.4f}",
        f"- A_theta vs theta_true_gap_reward_aligned: pearson={summary['branch_credit_vs_references']['A_theta_vs_theta_true_gap_reward_aligned']['pearson']:.4f}, spearman={summary['branch_credit_vs_references']['A_theta_vs_theta_true_gap_reward_aligned']['spearman']:.4f}, sign={summary['branch_credit_vs_references']['A_theta_vs_theta_true_gap_reward_aligned']['sign_agreement_ratio']:.4f}",
        f"- A_route vs route_true_gap_path: pearson={summary['branch_credit_vs_references']['A_route_vs_route_true_gap_path']['pearson']:.4f}, spearman={summary['branch_credit_vs_references']['A_route_vs_route_true_gap_path']['spearman']:.4f}, sign={summary['branch_credit_vs_references']['A_route_vs_route_true_gap_path']['sign_agreement_ratio']:.4f}",
        f"- A_route vs route_true_gap_reward_aligned: pearson={summary['branch_credit_vs_references']['A_route_vs_route_true_gap_reward_aligned']['pearson']:.4f}, spearman={summary['branch_credit_vs_references']['A_route_vs_route_true_gap_reward_aligned']['spearman']:.4f}, sign={summary['branch_credit_vs_references']['A_route_vs_route_true_gap_reward_aligned']['sign_agreement_ratio']:.4f}",
        "",
    ]
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote audit summary to {summary_json_path}")


if __name__ == "__main__":
    main()
