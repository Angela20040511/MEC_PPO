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

from branch_credit_semantics_audit import _series_stats
from dense_actor_input_experiment import build_state_layout
from dense_policy_ratio_mode_experiment import build_policy_ratio_mode_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator


BASELINE_MODE = "hierarchical_actor_joint_reward_aligned_credit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static decomposition audit for joint TD-aligned credit, separating "
            "counterfactual reward, bootstrap value, and combined TD scores."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    parser.add_argument(
        "--audit-steps",
        type=int,
        default=24,
        help="Number of rollout steps to audit with full counterfactual decomposition.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help=(
            "Optional baseline checkpoint path. If omitted, the latest "
            "joint reward-aligned baseline checkpoint is used."
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


def _set_block_joint_candidate(
    action: np.ndarray,
    block_slice: slice,
    candidate_index: int,
) -> np.ndarray:
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


def _collect_rollout_with_td_decomposition(
    agent: PPOAgent,
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
        reward_scores = np.zeros((len(block_slices), 3), dtype=np.float32)
        value_scores = np.zeros((len(block_slices), 3), dtype=np.float32)
        td_scores = np.zeros((len(block_slices), 3), dtype=np.float32)

        for block_id, block_slice in enumerate(block_slices):
            for candidate_index in range(3):
                variant_action = _set_block_joint_candidate(
                    action_np,
                    block_slice,
                    candidate_index,
                )
                simulator_cf = copy.deepcopy(simulator)
                next_state_cf, reward_cf, done_cf, _ = simulator_cf.step(variant_action)
                next_value_cf = 0.0 if done_cf else float(agent.evaluate_value(next_state_cf))
                reward_scores[block_id, candidate_index] = float(reward_cf)
                value_scores[block_id, candidate_index] = float(next_value_cf)
                td_scores[block_id, candidate_index] = float(
                    reward_cf + agent.config.gamma * next_value_cf
                )

        next_state, reward, done, _ = simulator.step(action_np)
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
                "reward_scores": reward_scores,
                "value_scores": value_scores,
                "td_scores": td_scores,
            }
        )
        state = next_state
        if done:
            break

    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)
    return agent.buffer.as_tensors(agent.device), sample_payloads


def _match_ratio(left: pd.Series, right: pd.Series) -> float:
    if left.empty or right.empty:
        return 0.0
    return float((left == right).mean())


def _conditional_rate(mask: pd.Series, predicate: pd.Series) -> float:
    valid = mask.astype(bool)
    if not valid.any():
        return 0.0
    return float(predicate[valid].mean())


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
    data, sample_payloads = _collect_rollout_with_td_decomposition(
        agent,
        config,
        args.seed,
        args.audit_steps,
    )

    states = data["states"]
    actions = data["actions"]
    raw_advantages = data["advantages"].detach().cpu()
    raw_returns = data["returns"].detach().cpu()
    task_count = int(agent.critic_state_layout["task_count"])

    with torch.no_grad():
        theta_terms = agent._true_conditional_theta_policy_terms(actions, actions)
        route_terms = agent._true_conditional_route_policy_terms(actions, actions)

    offload_active_mask = theta_terms["offload_active_mask"].detach().cpu().numpy().astype(bool)
    selected_route_index = route_terms["selected_indices"].detach().cpu().numpy().astype(np.int64)
    actual_joint_action = np.zeros_like(selected_route_index, dtype=np.int64)
    actual_joint_action[offload_active_mask & (selected_route_index == 0)] = 1
    actual_joint_action[offload_active_mask & (selected_route_index == 1)] = 2

    reward_scores = np.stack([payload["reward_scores"] for payload in sample_payloads], axis=0)
    value_scores = np.stack([payload["value_scores"] for payload in sample_payloads], axis=0)
    td_scores = np.stack([payload["td_scores"] for payload in sample_payloads], axis=0)
    gamma_v_scores = float(agent.config.gamma) * value_scores

    selected_reward_score = np.take_along_axis(
        reward_scores,
        actual_joint_action[..., None],
        axis=-1,
    ).squeeze(-1)
    selected_value_score = np.take_along_axis(
        value_scores,
        actual_joint_action[..., None],
        axis=-1,
    ).squeeze(-1)
    selected_td_score = np.take_along_axis(
        td_scores,
        actual_joint_action[..., None],
        axis=-1,
    ).squeeze(-1)

    best_action_under_reward = np.argmax(reward_scores, axis=-1)
    best_action_under_value = np.argmax(value_scores, axis=-1)
    best_action_under_td = np.argmax(td_scores, axis=-1)

    best_reward_score = reward_scores.max(axis=-1)
    best_value_score = value_scores.max(axis=-1)
    best_td_score = td_scores.max(axis=-1)

    reward_sorted = np.sort(reward_scores, axis=-1)
    value_sorted = np.sort(value_scores, axis=-1)
    td_sorted = np.sort(td_scores, axis=-1)
    reward_margin = reward_sorted[..., -1] - reward_sorted[..., -2]
    value_margin = value_sorted[..., -1] - value_sorted[..., -2]
    td_margin = td_sorted[..., -1] - td_sorted[..., -2]

    std_reward = reward_scores.std(axis=-1)
    std_gamma_v = gamma_v_scores.std(axis=-1)
    std_td = td_scores.std(axis=-1)

    mean_abs_r_term = np.abs(reward_scores).mean(axis=-1)
    mean_abs_gamma_v_term = np.abs(gamma_v_scores).mean(axis=-1)
    mean_abs_td_term = np.abs(td_scores).mean(axis=-1)
    gamma_v_over_r_abs_ratio = mean_abs_gamma_v_term / (mean_abs_r_term + 1e-6)

    rows: list[dict[str, Any]] = []
    sample_count, block_count = actual_joint_action.shape
    for sample_id in range(sample_count):
        for block_id in range(block_count):
            sensor_id = int(block_id // task_count)
            task_id = int(block_id % task_count)
            reward_action_name = _action_name(int(best_action_under_reward[sample_id, block_id]))
            value_action_name = _action_name(int(best_action_under_value[sample_id, block_id]))
            td_action_name = _action_name(int(best_action_under_td[sample_id, block_id]))
            rows.append(
                {
                    "sample_id": sample_id,
                    "block_id": block_id,
                    "sensor_id": sensor_id,
                    "task_id": task_id,
                    "r_local": float(reward_scores[sample_id, block_id, 0]),
                    "r_bs1": float(reward_scores[sample_id, block_id, 1]),
                    "r_bs2": float(reward_scores[sample_id, block_id, 2]),
                    "v_local": float(value_scores[sample_id, block_id, 0]),
                    "v_bs1": float(value_scores[sample_id, block_id, 1]),
                    "v_bs2": float(value_scores[sample_id, block_id, 2]),
                    "td_local": float(td_scores[sample_id, block_id, 0]),
                    "td_bs1": float(td_scores[sample_id, block_id, 1]),
                    "td_bs2": float(td_scores[sample_id, block_id, 2]),
                    "selected_joint_action": _action_name(int(actual_joint_action[sample_id, block_id])),
                    "selected_joint_action_index": int(actual_joint_action[sample_id, block_id]),
                    "selected_reward_score": float(selected_reward_score[sample_id, block_id]),
                    "selected_value_score": float(selected_value_score[sample_id, block_id]),
                    "selected_td_score": float(selected_td_score[sample_id, block_id]),
                    "best_action_under_reward": reward_action_name,
                    "best_action_under_reward_index": int(best_action_under_reward[sample_id, block_id]),
                    "best_action_under_value": value_action_name,
                    "best_action_under_value_index": int(best_action_under_value[sample_id, block_id]),
                    "best_action_under_td": td_action_name,
                    "best_action_under_td_index": int(best_action_under_td[sample_id, block_id]),
                    "best_reward_score": float(best_reward_score[sample_id, block_id]),
                    "best_value_score": float(best_value_score[sample_id, block_id]),
                    "best_td_score": float(best_td_score[sample_id, block_id]),
                    "advantage": float(raw_advantages[sample_id].item()),
                    "return": float(raw_returns[sample_id].item()),
                    "reward_vs_td_best_match": int(
                        best_action_under_reward[sample_id, block_id]
                        == best_action_under_td[sample_id, block_id]
                    ),
                    "reward_vs_value_best_match": int(
                        best_action_under_reward[sample_id, block_id]
                        == best_action_under_value[sample_id, block_id]
                    ),
                    "value_vs_td_best_match": int(
                        best_action_under_value[sample_id, block_id]
                        == best_action_under_td[sample_id, block_id]
                    ),
                    "reward_td_flip_type": (
                        f"reward:{reward_action_name}->td:{td_action_name}"
                        if reward_action_name != td_action_name
                        else "match"
                    ),
                    "reward_margin": float(reward_margin[sample_id, block_id]),
                    "value_margin": float(value_margin[sample_id, block_id]),
                    "td_margin": float(td_margin[sample_id, block_id]),
                    "std_reward_term": float(std_reward[sample_id, block_id]),
                    "std_gamma_v_term": float(std_gamma_v[sample_id, block_id]),
                    "std_td_term": float(std_td[sample_id, block_id]),
                    "mean_abs_r_term": float(mean_abs_r_term[sample_id, block_id]),
                    "mean_abs_gamma_v_term": float(mean_abs_gamma_v_term[sample_id, block_id]),
                    "mean_abs_td_term": float(mean_abs_td_term[sample_id, block_id]),
                    "gamma_v_over_r_abs_ratio_sample": float(
                        gamma_v_over_r_abs_ratio[sample_id, block_id]
                    ),
                }
            )

    raw_df = pd.DataFrame(rows)
    flip_df = raw_df[raw_df["reward_td_flip_type"] != "match"].copy()
    flip_counts = Counter(flip_df["reward_td_flip_type"])

    value_dominates_flip_mask = (
        (raw_df["reward_vs_td_best_match"] == 0)
        & (raw_df["std_gamma_v_term"] > raw_df["std_reward_term"])
    )
    reward_large_value_small_mask = (
        (raw_df["std_reward_term"] > raw_df["std_reward_term"].median())
        & (raw_df["std_gamma_v_term"] < raw_df["std_gamma_v_term"].median())
    )
    td_margin_smaller_mask = raw_df["td_margin"] < raw_df["reward_margin"]

    summary = {
        "metadata": {
            "baseline_mode": BASELINE_MODE,
            "checkpoint_path": str(checkpoint_path),
            "seed": args.seed,
            "audit_steps": int(args.audit_steps),
            "sample_count": int(sample_count),
            "block_count": int(block_count),
            "row_count": int(len(raw_df)),
            "gamma": float(agent.config.gamma),
            "value_source": (
                "agent.evaluate_value(next_state_candidate) via PPOAgent._value_from_state_tensor "
                "using the main scalar critic value head under popart_return_norm."
            ),
        },
        "score_definition": {
            "reward_term": "counterfactual simulator.step(...) reward for each candidate joint action",
            "value_term": "V(next_state_candidate) from agent.evaluate_value(next_state_candidate)",
            "td_term": "reward_term + gamma * value_term",
        },
        "correlations": {
            "selected_reward_vs_advantage": {
                "pearson": _safe_corr(raw_df["selected_reward_score"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_reward_score"], raw_df["advantage"], "spearman"),
            },
            "selected_reward_vs_return": {
                "pearson": _safe_corr(raw_df["selected_reward_score"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_reward_score"], raw_df["return"], "spearman"),
            },
            "selected_value_vs_advantage": {
                "pearson": _safe_corr(raw_df["selected_value_score"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_value_score"], raw_df["advantage"], "spearman"),
            },
            "selected_value_vs_return": {
                "pearson": _safe_corr(raw_df["selected_value_score"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_value_score"], raw_df["return"], "spearman"),
            },
            "selected_td_vs_advantage": {
                "pearson": _safe_corr(raw_df["selected_td_score"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_td_score"], raw_df["advantage"], "spearman"),
            },
            "selected_td_vs_return": {
                "pearson": _safe_corr(raw_df["selected_td_score"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_td_score"], raw_df["return"], "spearman"),
            },
            "best_reward_vs_advantage": {
                "pearson": _safe_corr(raw_df["best_reward_score"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["best_reward_score"], raw_df["advantage"], "spearman"),
            },
            "best_reward_vs_return": {
                "pearson": _safe_corr(raw_df["best_reward_score"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["best_reward_score"], raw_df["return"], "spearman"),
            },
            "best_value_vs_advantage": {
                "pearson": _safe_corr(raw_df["best_value_score"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["best_value_score"], raw_df["advantage"], "spearman"),
            },
            "best_value_vs_return": {
                "pearson": _safe_corr(raw_df["best_value_score"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["best_value_score"], raw_df["return"], "spearman"),
            },
            "best_td_vs_advantage": {
                "pearson": _safe_corr(raw_df["best_td_score"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["best_td_score"], raw_df["advantage"], "spearman"),
            },
            "best_td_vs_return": {
                "pearson": _safe_corr(raw_df["best_td_score"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["best_td_score"], raw_df["return"], "spearman"),
            },
        },
        "ranking_flip_analysis": {
            "reward_best_vs_td_best_match_ratio": _match_ratio(
                raw_df["best_action_under_reward_index"],
                raw_df["best_action_under_td_index"],
            ),
            "reward_best_vs_value_best_match_ratio": _match_ratio(
                raw_df["best_action_under_reward_index"],
                raw_df["best_action_under_value_index"],
            ),
            "value_best_vs_td_best_match_ratio": _match_ratio(
                raw_df["best_action_under_value_index"],
                raw_df["best_action_under_td_index"],
            ),
            "reward_td_flip_type_counts": {
                flip_type: int(count)
                for flip_type, count in sorted(flip_counts.items())
            },
        },
        "scale_analysis": {
            "std_reward_term_stats": _series_stats(raw_df["std_reward_term"]),
            "std_gamma_v_term_stats": _series_stats(raw_df["std_gamma_v_term"]),
            "std_td_term_stats": _series_stats(raw_df["std_td_term"]),
            "mean_abs_r_term": float(raw_df["mean_abs_r_term"].mean()),
            "mean_abs_gamma_v_term": float(raw_df["mean_abs_gamma_v_term"].mean()),
            "mean_abs_td_term": float(raw_df["mean_abs_td_term"].mean()),
            "gamma_v_over_r_abs_ratio": float(raw_df["gamma_v_over_r_abs_ratio_sample"].mean()),
            "reward_large_value_small_ratio": float(reward_large_value_small_mask.mean()),
            "value_dominates_ranking_flip_ratio": float(value_dominates_flip_mask.mean()),
            "td_margin_smaller_than_reward_margin_ratio": float(td_margin_smaller_mask.mean()),
        },
        "static_explanation_signals": {
            "reward_td_flip_ratio": float((raw_df["reward_vs_td_best_match"] == 0).mean()),
            "selected_td_vs_return_weaker_than_reward": bool(
                abs(_safe_corr(raw_df["selected_td_score"], raw_df["return"], "pearson"))
                < abs(_safe_corr(raw_df["selected_reward_score"], raw_df["return"], "pearson"))
            ),
            "selected_td_vs_advantage_weaker_than_reward": bool(
                abs(_safe_corr(raw_df["selected_td_score"], raw_df["advantage"], "pearson"))
                < abs(_safe_corr(raw_df["selected_reward_score"], raw_df["advantage"], "pearson"))
            ),
            "td_margin_collapse_signal": float(
                flip_df["td_margin"].mean() if not flip_df.empty else 0.0
            ),
            "reward_margin_on_flip_samples": float(
                flip_df["reward_margin"].mean() if not flip_df.empty else 0.0
            ),
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"joint_td_decomposition_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = output_dir / "joint_td_decomposition_audit_raw.csv"
    summary_json_path = output_dir / "joint_td_decomposition_audit_summary.json"
    summary_md_path = output_dir / "joint_td_decomposition_audit_summary.md"
    flip_csv_path = output_dir / "joint_td_flip_analysis.csv"
    scale_csv_path = output_dir / "joint_td_scale_analysis.csv"
    corr_csv_path = output_dir / "joint_td_corr_analysis.csv"

    raw_df.to_csv(raw_csv_path, index=False)
    pd.DataFrame(
        [
            {"flip_type": flip_type, "count": count}
            for flip_type, count in sorted(flip_counts.items())
        ]
    ).to_csv(flip_csv_path, index=False)
    raw_df[
        [
            "sample_id",
            "block_id",
            "std_reward_term",
            "std_gamma_v_term",
            "std_td_term",
            "mean_abs_r_term",
            "mean_abs_gamma_v_term",
            "mean_abs_td_term",
            "gamma_v_over_r_abs_ratio_sample",
            "reward_margin",
            "value_margin",
            "td_margin",
        ]
    ].to_csv(scale_csv_path, index=False)
    pd.DataFrame(
        [
            {"pair": key, **value}
            for key, value in summary["correlations"].items()
        ]
    ).to_csv(corr_csv_path, index=False)

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    md_lines = [
        "# Joint TD Decomposition Audit",
        "",
        f"- baseline_mode: `{BASELINE_MODE}`",
        f"- checkpoint_path: `{checkpoint_path}`",
        f"- sample_count: {sample_count}",
        f"- block_count: {block_count}",
        f"- row_count: {len(raw_df)}",
        f"- gamma: {agent.config.gamma:.4f}",
        "",
        "## Correlations",
        f"- selected reward vs return: pearson={summary['correlations']['selected_reward_vs_return']['pearson']:.4f}, spearman={summary['correlations']['selected_reward_vs_return']['spearman']:.4f}",
        f"- selected value vs return: pearson={summary['correlations']['selected_value_vs_return']['pearson']:.4f}, spearman={summary['correlations']['selected_value_vs_return']['spearman']:.4f}",
        f"- selected td vs return: pearson={summary['correlations']['selected_td_vs_return']['pearson']:.4f}, spearman={summary['correlations']['selected_td_vs_return']['spearman']:.4f}",
        f"- selected reward vs advantage: pearson={summary['correlations']['selected_reward_vs_advantage']['pearson']:.4f}, spearman={summary['correlations']['selected_reward_vs_advantage']['spearman']:.4f}",
        f"- selected value vs advantage: pearson={summary['correlations']['selected_value_vs_advantage']['pearson']:.4f}, spearman={summary['correlations']['selected_value_vs_advantage']['spearman']:.4f}",
        f"- selected td vs advantage: pearson={summary['correlations']['selected_td_vs_advantage']['pearson']:.4f}, spearman={summary['correlations']['selected_td_vs_advantage']['spearman']:.4f}",
        "",
        "## Ranking Flips",
        f"- reward best vs td best match ratio: {summary['ranking_flip_analysis']['reward_best_vs_td_best_match_ratio']:.4f}",
        f"- reward best vs value best match ratio: {summary['ranking_flip_analysis']['reward_best_vs_value_best_match_ratio']:.4f}",
        f"- value best vs td best match ratio: {summary['ranking_flip_analysis']['value_best_vs_td_best_match_ratio']:.4f}",
        "",
        "## Scale",
        f"- mean_abs_r_term: {summary['scale_analysis']['mean_abs_r_term']:.4f}",
        f"- mean_abs_gamma_v_term: {summary['scale_analysis']['mean_abs_gamma_v_term']:.4f}",
        f"- mean_abs_td_term: {summary['scale_analysis']['mean_abs_td_term']:.4f}",
        f"- gamma_v_over_r_abs_ratio: {summary['scale_analysis']['gamma_v_over_r_abs_ratio']:.4f}",
        f"- value_dominates_ranking_flip_ratio: {summary['scale_analysis']['value_dominates_ranking_flip_ratio']:.4f}",
        f"- td_margin_smaller_than_reward_margin_ratio: {summary['scale_analysis']['td_margin_smaller_than_reward_margin_ratio']:.4f}",
        "",
        "## Static Explanation Signals",
        f"- reward_td_flip_ratio: {summary['static_explanation_signals']['reward_td_flip_ratio']:.4f}",
        f"- selected_td_vs_return_weaker_than_reward: {summary['static_explanation_signals']['selected_td_vs_return_weaker_than_reward']}",
        f"- selected_td_vs_advantage_weaker_than_reward: {summary['static_explanation_signals']['selected_td_vs_advantage_weaker_than_reward']}",
        f"- reward_margin_on_flip_samples: {summary['static_explanation_signals']['reward_margin_on_flip_samples']:.4f}",
        f"- td_margin_collapse_signal: {summary['static_explanation_signals']['td_margin_collapse_signal']:.4f}",
        "",
        "## Top Flip Types",
    ]
    for flip_type, count in sorted(flip_counts.items(), key=lambda item: (-item[1], item[0]))[:10]:
        md_lines.append(f"- {flip_type}: {count}")

    with summary_md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines) + "\n")

    print(f"Saved summary JSON to {summary_json_path}")
    print(f"Saved summary MD to {summary_md_path}")
    print(f"Saved raw CSV to {raw_csv_path}")


if __name__ == "__main__":
    main()
