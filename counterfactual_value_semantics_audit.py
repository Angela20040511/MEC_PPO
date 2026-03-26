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
            "Static audit for why counterfactual V(next_state_candidate) disagrees "
            "with reward/return semantics in joint TD-aligned credit."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    parser.add_argument(
        "--audit-steps",
        type=int,
        default=24,
        help="Number of rollout steps to audit with counterfactual next-state evaluation.",
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


def _feature_group_stats(states: np.ndarray, slices: dict[str, slice]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name in ["workload", "access_queue", "virtual_queue", "local_queue", "bs_queue"]:
        block = states[:, slices[name]]
        per_state_mean = block.mean(axis=1)
        stats[name] = _series_stats(pd.Series(per_state_mean))
    return stats


def _collect_rollout_with_counterfactual_values(
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
        action, log_prob, value = agent.select_action(state)
        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        reward_scores = np.zeros((len(block_slices), 3), dtype=np.float32)
        value_scores = np.zeros((len(block_slices), 3), dtype=np.float32)
        next_states_cf = np.zeros(
            (len(block_slices), 3, int(config.state_dim)),
            dtype=np.float32,
        )

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
                next_states_cf[block_id, candidate_index] = np.asarray(
                    next_state_cf,
                    dtype=np.float32,
                )

        next_state, reward, done, _ = simulator.step(action_np)
        agent.store_transition(state, action_np, log_prob, reward, done, value, next_state)
        sample_payloads.append(
            {
                "sample_id": sample_id,
                "reward_scores": reward_scores,
                "value_scores": value_scores,
                "next_states_cf": next_states_cf,
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
    data, sample_payloads = _collect_rollout_with_counterfactual_values(
        agent,
        config,
        args.seed,
        args.audit_steps,
    )

    states = data["states"]
    next_states = data["next_states"]
    actions = data["actions"]
    raw_advantages = data["advantages"].detach().cpu()
    raw_returns = data["returns"].detach().cpu()
    task_count = int(agent.critic_state_layout["task_count"])

    with torch.no_grad():
        theta_terms = agent._true_conditional_theta_policy_terms(actions, actions)
        route_terms = agent._true_conditional_route_policy_terms(actions, actions)
        actual_next_values = (
            agent._value_from_state_tensor(next_states).squeeze(-1).detach().cpu().numpy()
        )

    offload_active_mask = theta_terms["offload_active_mask"].detach().cpu().numpy().astype(bool)
    selected_route_index = route_terms["selected_indices"].detach().cpu().numpy().astype(np.int64)
    actual_joint_action = np.zeros_like(selected_route_index, dtype=np.int64)
    actual_joint_action[offload_active_mask & (selected_route_index == 0)] = 1
    actual_joint_action[offload_active_mask & (selected_route_index == 1)] = 2

    reward_scores = np.stack([payload["reward_scores"] for payload in sample_payloads], axis=0)
    value_scores = np.stack([payload["value_scores"] for payload in sample_payloads], axis=0)
    counterfactual_next_states = np.stack(
        [payload["next_states_cf"] for payload in sample_payloads],
        axis=0,
    )

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
    best_action_under_reward = np.argmax(reward_scores, axis=-1)
    best_action_under_value = np.argmax(value_scores, axis=-1)
    best_reward_score = reward_scores.max(axis=-1)
    best_value_score = value_scores.max(axis=-1)

    candidate_flat_states = counterfactual_next_states.reshape(-1, counterfactual_next_states.shape[-1])
    actual_next_states_np = next_states.detach().cpu().numpy()
    actual_state_repeated = np.repeat(
        actual_next_states_np[:, None, None, :],
        repeats=counterfactual_next_states.shape[1],
        axis=1,
    )
    actual_state_repeated = np.repeat(actual_state_repeated, repeats=3, axis=2)

    actual_counterpart_flat = actual_state_repeated.reshape(-1, counterfactual_next_states.shape[-1])
    raw_state_distance = np.linalg.norm(candidate_flat_states - actual_counterpart_flat, axis=1)

    with torch.no_grad():
        candidate_state_tensor = torch.tensor(
            candidate_flat_states,
            dtype=torch.float32,
            device=agent.device,
        )
        actual_counterpart_tensor = torch.tensor(
            actual_counterpart_flat,
            dtype=torch.float32,
            device=agent.device,
        )
        candidate_critic_inputs, _, _ = agent._prepare_critic_inputs(
            candidate_state_tensor,
            update_stats=False,
        )
        actual_critic_inputs, _, _ = agent._prepare_critic_inputs(
            actual_counterpart_tensor,
            update_stats=False,
        )
        critic_input_distance = (
            torch.norm(candidate_critic_inputs - actual_critic_inputs, dim=1)
            .detach()
            .cpu()
            .numpy()
        )

    raw_state_distance = raw_state_distance.reshape(
        counterfactual_next_states.shape[0],
        counterfactual_next_states.shape[1],
        3,
    )
    critic_input_distance = critic_input_distance.reshape(
        counterfactual_next_states.shape[0],
        counterfactual_next_states.shape[1],
        3,
    )

    rows: list[dict[str, Any]] = []
    sample_count, block_count = actual_joint_action.shape
    for sample_id in range(sample_count):
        for block_id in range(block_count):
            sensor_id = int(block_id // task_count)
            task_id = int(block_id % task_count)
            reward_action_name = _action_name(int(best_action_under_reward[sample_id, block_id]))
            value_action_name = _action_name(int(best_action_under_value[sample_id, block_id]))
            for candidate_index in range(3):
                candidate_name = _action_name(candidate_index)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "block_id": block_id,
                        "sensor_id": sensor_id,
                        "task_id": task_id,
                        "candidate_action": candidate_name,
                        "candidate_action_index": candidate_index,
                        "r_candidate": float(reward_scores[sample_id, block_id, candidate_index]),
                        "v_candidate": float(value_scores[sample_id, block_id, candidate_index]),
                        "return": float(raw_returns[sample_id].item()),
                        "advantage": float(raw_advantages[sample_id].item()),
                        "best_action_under_reward": reward_action_name,
                        "best_action_under_reward_index": int(best_action_under_reward[sample_id, block_id]),
                        "best_action_under_value": value_action_name,
                        "best_action_under_value_index": int(best_action_under_value[sample_id, block_id]),
                        "reward_vs_value_best_match": int(
                            best_action_under_reward[sample_id, block_id]
                            == best_action_under_value[sample_id, block_id]
                        ),
                        "reward_value_flip_type": (
                            f"reward:{reward_action_name}->value:{value_action_name}"
                            if reward_action_name != value_action_name
                            else "match"
                        ),
                        "selected_joint_action": _action_name(int(actual_joint_action[sample_id, block_id])),
                        "selected_joint_action_index": int(actual_joint_action[sample_id, block_id]),
                        "selected_reward_score": float(selected_reward_score[sample_id, block_id]),
                        "selected_value_score": float(selected_value_score[sample_id, block_id]),
                        "best_reward_score": float(best_reward_score[sample_id, block_id]),
                        "best_value_score": float(best_value_score[sample_id, block_id]),
                        "actual_next_state_value": float(actual_next_values[sample_id]),
                        "is_counterfactual_state": 1,
                        "raw_state_distance_to_actual_next_state": float(
                            raw_state_distance[sample_id, block_id, candidate_index]
                        ),
                        "critic_input_distance_to_actual_next_state": float(
                            critic_input_distance[sample_id, block_id, candidate_index]
                        ),
                    }
                )

    raw_df = pd.DataFrame(rows)
    flip_df = raw_df[raw_df["reward_value_flip_type"] != "match"].copy()
    flip_counts = Counter(flip_df["reward_value_flip_type"])

    per_sample_candidate_next_states = candidate_flat_states
    slices = agent._state_slices()
    feature_shift_rows: list[dict[str, Any]] = []
    actual_feature_stats = _feature_group_stats(actual_next_states_np, slices)
    counterfactual_feature_stats = _feature_group_stats(per_sample_candidate_next_states, slices)
    for feature_name in ["workload", "access_queue", "virtual_queue", "local_queue", "bs_queue"]:
        feature_shift_rows.append(
            {
                "feature": feature_name,
                "actual_mean": actual_feature_stats[feature_name]["mean"],
                "actual_std": actual_feature_stats[feature_name]["std"],
                "counterfactual_mean": counterfactual_feature_stats[feature_name]["mean"],
                "counterfactual_std": counterfactual_feature_stats[feature_name]["std"],
                "mean_shift": (
                    counterfactual_feature_stats[feature_name]["mean"]
                    - actual_feature_stats[feature_name]["mean"]
                ),
                "std_shift": (
                    counterfactual_feature_stats[feature_name]["std"]
                    - actual_feature_stats[feature_name]["std"]
                ),
            }
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
        "value_path_audit": {
            "evaluate_value_path": (
                "PPOAgent.evaluate_value(next_state_candidate) -> "
                "PPOAgent._value_from_state_tensor(state_tensor) -> "
                "PPOAgent._prepare_critic_inputs(update_stats=False) -> "
                "ActorCritic.value_from_critic_input(critic_input, mean, std) -> "
                "normalized critic head output * popart_std + popart_mean"
            ),
            "training_value_semantics": (
                "Same main scalar critic value used for training returns under "
                "value_target_mode=popart_return_norm; evaluate_value returns the "
                "raw, de-normalized value estimate, not normalized target-space output."
            ),
        },
        "candidate_value_vs_objective": {
            "v_local_vs_advantage": {
                "pearson": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "local", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "local", "advantage"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "local", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "local", "advantage"],
                    "spearman",
                ),
            },
            "v_local_vs_return": {
                "pearson": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "local", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "local", "return"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "local", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "local", "return"],
                    "spearman",
                ),
            },
            "v_bs1_vs_advantage": {
                "pearson": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "bs1", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "bs1", "advantage"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "bs1", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "bs1", "advantage"],
                    "spearman",
                ),
            },
            "v_bs1_vs_return": {
                "pearson": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "bs1", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "bs1", "return"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "bs1", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "bs1", "return"],
                    "spearman",
                ),
            },
            "v_bs2_vs_advantage": {
                "pearson": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "bs2", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "bs2", "advantage"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "bs2", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "bs2", "advantage"],
                    "spearman",
                ),
            },
            "v_bs2_vs_return": {
                "pearson": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "bs2", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "bs2", "return"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "bs2", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "bs2", "return"],
                    "spearman",
                ),
            },
            "selected_value_vs_advantage": {
                "pearson": _safe_corr(raw_df["selected_value_score"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_value_score"], raw_df["advantage"], "spearman"),
            },
            "selected_value_vs_return": {
                "pearson": _safe_corr(raw_df["selected_value_score"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_value_score"], raw_df["return"], "spearman"),
            },
            "best_value_vs_advantage": {
                "pearson": _safe_corr(raw_df["best_value_score"], raw_df["advantage"], "pearson"),
                "spearman": _safe_corr(raw_df["best_value_score"], raw_df["advantage"], "spearman"),
            },
            "best_value_vs_return": {
                "pearson": _safe_corr(raw_df["best_value_score"], raw_df["return"], "pearson"),
                "spearman": _safe_corr(raw_df["best_value_score"], raw_df["return"], "spearman"),
            },
        },
        "candidate_value_vs_reward": {
            "v_local_vs_r_local": {
                "pearson": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "local", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "local", "r_candidate"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "local", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "local", "r_candidate"],
                    "spearman",
                ),
            },
            "v_bs1_vs_r_bs1": {
                "pearson": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "bs1", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "bs1", "r_candidate"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "bs1", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "bs1", "r_candidate"],
                    "spearman",
                ),
            },
            "v_bs2_vs_r_bs2": {
                "pearson": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "bs2", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "bs2", "r_candidate"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    raw_df.loc[raw_df["candidate_action"] == "bs2", "v_candidate"],
                    raw_df.loc[raw_df["candidate_action"] == "bs2", "r_candidate"],
                    "spearman",
                ),
            },
            "selected_value_vs_selected_reward": {
                "pearson": _safe_corr(raw_df["selected_value_score"], raw_df["selected_reward_score"], "pearson"),
                "spearman": _safe_corr(raw_df["selected_value_score"], raw_df["selected_reward_score"], "spearman"),
            },
            "best_value_vs_best_reward": {
                "pearson": _safe_corr(raw_df["best_value_score"], raw_df["best_reward_score"], "pearson"),
                "spearman": _safe_corr(raw_df["best_value_score"], raw_df["best_reward_score"], "spearman"),
            },
        },
        "value_ranking_flips": {
            "reward_best_vs_value_best_match_ratio": float(
                raw_df["reward_vs_value_best_match"].mean()
            ),
            "reward_value_flip_type_counts": {
                flip_type: int(count)
                for flip_type, count in sorted(flip_counts.items())
            },
        },
        "distribution_shift": {
            "actual_rollout_next_state_value_stats": _series_stats(pd.Series(actual_next_values)),
            "counterfactual_next_state_value_stats": _series_stats(pd.Series(value_scores.reshape(-1))),
            "raw_state_distance_stats": _series_stats(pd.Series(raw_state_distance.reshape(-1))),
            "critic_input_distance_stats": _series_stats(pd.Series(critic_input_distance.reshape(-1))),
            "feature_group_stats": feature_shift_rows,
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"counterfactual_value_semantics_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = output_dir / "counterfactual_value_semantics_audit_raw.csv"
    summary_json_path = output_dir / "counterfactual_value_semantics_audit_summary.json"
    summary_md_path = output_dir / "counterfactual_value_semantics_audit_summary.md"
    flip_csv_path = output_dir / "counterfactual_value_flip_analysis.csv"
    corr_csv_path = output_dir / "counterfactual_value_corr_analysis.csv"
    shift_csv_path = output_dir / "counterfactual_value_distribution_shift.csv"

    raw_df.to_csv(raw_csv_path, index=False)
    pd.DataFrame(
        [
            {"flip_type": flip_type, "count": count}
            for flip_type, count in sorted(flip_counts.items())
        ]
    ).to_csv(flip_csv_path, index=False)
    pd.DataFrame(
        [
            {"pair": key, **value}
            for section_name in ["candidate_value_vs_objective", "candidate_value_vs_reward"]
            for key, value in summary[section_name].items()
        ]
    ).to_csv(corr_csv_path, index=False)
    pd.DataFrame(feature_shift_rows).to_csv(shift_csv_path, index=False)

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    md_lines = [
        "# Counterfactual Value Semantics Audit",
        "",
        f"- baseline_mode: `{BASELINE_MODE}`",
        f"- checkpoint_path: `{checkpoint_path}`",
        f"- sample_count: {sample_count}",
        f"- block_count: {block_count}",
        f"- row_count: {len(raw_df)}",
        "",
        "## Value vs Objective",
        f"- selected value vs return: pearson={summary['candidate_value_vs_objective']['selected_value_vs_return']['pearson']:.4f}, spearman={summary['candidate_value_vs_objective']['selected_value_vs_return']['spearman']:.4f}",
        f"- selected value vs advantage: pearson={summary['candidate_value_vs_objective']['selected_value_vs_advantage']['pearson']:.4f}, spearman={summary['candidate_value_vs_objective']['selected_value_vs_advantage']['spearman']:.4f}",
        f"- best value vs return: pearson={summary['candidate_value_vs_objective']['best_value_vs_return']['pearson']:.4f}, spearman={summary['candidate_value_vs_objective']['best_value_vs_return']['spearman']:.4f}",
        "",
        "## Value vs Reward",
        f"- selected value vs selected reward: pearson={summary['candidate_value_vs_reward']['selected_value_vs_selected_reward']['pearson']:.4f}, spearman={summary['candidate_value_vs_reward']['selected_value_vs_selected_reward']['spearman']:.4f}",
        f"- best value vs best reward: pearson={summary['candidate_value_vs_reward']['best_value_vs_best_reward']['pearson']:.4f}, spearman={summary['candidate_value_vs_reward']['best_value_vs_best_reward']['spearman']:.4f}",
        "",
        "## Ranking Flips",
        f"- reward best vs value best match ratio: {summary['value_ranking_flips']['reward_best_vs_value_best_match_ratio']:.4f}",
        "",
        "## Distribution Shift",
        f"- actual next-state value mean/std: {summary['distribution_shift']['actual_rollout_next_state_value_stats']['mean']:.4f} / {summary['distribution_shift']['actual_rollout_next_state_value_stats']['std']:.4f}",
        f"- counterfactual next-state value mean/std: {summary['distribution_shift']['counterfactual_next_state_value_stats']['mean']:.4f} / {summary['distribution_shift']['counterfactual_next_state_value_stats']['std']:.4f}",
        f"- raw state distance mean/std: {summary['distribution_shift']['raw_state_distance_stats']['mean']:.4f} / {summary['distribution_shift']['raw_state_distance_stats']['std']:.4f}",
        f"- critic input distance mean/std: {summary['distribution_shift']['critic_input_distance_stats']['mean']:.4f} / {summary['distribution_shift']['critic_input_distance_stats']['std']:.4f}",
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
