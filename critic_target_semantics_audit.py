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
            "Static audit for main critic target semantics on real rollout states "
            "and counterfactual next states."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    parser.add_argument("--audit-steps", type=int, default=24)
    parser.add_argument("--checkpoint-path", type=str, default="")
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


def _simulate_running_return_stats(
    agent: PPOAgent,
    returns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_mean = float(returns.mean().item())
    batch_var = float(returns.var(unbiased=False).item())
    batch_count = float(returns.numel())
    if agent.running_return_count == 0.0:
        target_mean = batch_mean
        target_var = max(batch_var, 1e-8)
    else:
        delta = batch_mean - agent.running_return_mean
        total_count = agent.running_return_count + batch_count
        mean = agent.running_return_mean + delta * batch_count / total_count
        m_a = agent.running_return_var * agent.running_return_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * agent.running_return_count * batch_count / total_count
        target_mean = mean
        target_var = max(m2 / total_count, 1e-8)
    target_mean_tensor = torch.tensor(target_mean, dtype=returns.dtype, device=returns.device)
    target_std_tensor = torch.tensor(
        float(np.sqrt(target_var)),
        dtype=returns.dtype,
        device=returns.device,
    )
    return target_mean_tensor, target_std_tensor


def _feature_group_stats(states: np.ndarray, slices: dict[str, slice]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name in ["workload", "access_queue", "virtual_queue", "local_queue", "bs_queue"]:
        block = states[:, slices[name]]
        per_state_mean = block.mean(axis=1)
        stats[name] = _series_stats(pd.Series(per_state_mean))
    return stats


def _collect_rollout_with_counterfactuals(
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
        next_states_cf = np.zeros((len(block_slices), 3, int(config.state_dim)), dtype=np.float32)

        for block_id, block_slice in enumerate(block_slices):
            for candidate_index in range(3):
                variant_action = _set_block_joint_candidate(action_np, block_slice, candidate_index)
                simulator_cf = copy.deepcopy(simulator)
                next_state_cf, reward_cf, done_cf, _ = simulator_cf.step(variant_action)
                next_value_cf = 0.0 if done_cf else float(agent.evaluate_value(next_state_cf))
                reward_scores[block_id, candidate_index] = float(reward_cf)
                value_scores[block_id, candidate_index] = float(next_value_cf)
                next_states_cf[block_id, candidate_index] = np.asarray(next_state_cf, dtype=np.float32)

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
                "next_states_cf": next_states_cf,
                "actual_reward": float(reward),
                "done": bool(done),
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
    returns = data["returns"]
    advantages = data["advantages"]
    dones = data["dones"]
    task_count = int(agent.critic_state_layout["task_count"])

    with torch.no_grad():
        critic_inputs, _, _ = agent._prepare_critic_inputs(states, update_stats=False)
        real_value_raw = agent._value_from_state_tensor(states).squeeze(-1)
        actual_next_value_raw = agent._value_from_state_tensor(next_states).squeeze(-1)
        real_value_normalized = agent.network.normalized_value_from_critic_input(
            critic_inputs
        ).squeeze(-1)
        theta_terms = agent._true_conditional_theta_policy_terms(actions, actions)
        route_terms = agent._true_conditional_route_policy_terms(actions, actions)

    target_mean, target_std = _simulate_running_return_stats(agent, returns)
    value_targets = (returns - target_mean) / (target_std + 1e-8)
    value_pred_normalized_equivalent = (real_value_raw - target_mean) / (target_std + 1e-8)
    one_step_rewards = torch.tensor(
        [payload["actual_reward"] for payload in sample_payloads],
        dtype=torch.float32,
        device=states.device,
    )
    one_step_td_targets = (
        one_step_rewards + float(agent.config.gamma) * actual_next_value_raw * (1.0 - dones)
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

    candidate_flat_states = counterfactual_next_states.reshape(
        -1,
        counterfactual_next_states.shape[-1],
    )
    actual_next_states_np = next_states.detach().cpu().numpy()
    actual_state_repeated = np.repeat(
        actual_next_states_np[:, None, None, :],
        repeats=counterfactual_next_states.shape[1],
        axis=1,
    )
    actual_state_repeated = np.repeat(actual_state_repeated, repeats=3, axis=2)
    actual_counterpart_flat = actual_state_repeated.reshape(
        -1,
        counterfactual_next_states.shape[-1],
    )
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
        actual_counterpart_critic_inputs, _, _ = agent._prepare_critic_inputs(
            actual_counterpart_tensor,
            update_stats=False,
        )
        critic_input_distance = (
            torch.norm(candidate_critic_inputs - actual_counterpart_critic_inputs, dim=1)
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

    real_rows: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []

    states_np = states.detach().cpu().numpy()
    slices = agent._state_slices()
    real_state_feature_stats = _feature_group_stats(states_np, slices)
    actual_next_state_feature_stats = _feature_group_stats(actual_next_states_np, slices)
    counterfactual_feature_stats = _feature_group_stats(candidate_flat_states, slices)

    real_value_raw_np = real_value_raw.detach().cpu().numpy()
    real_value_normalized_np = real_value_normalized.detach().cpu().numpy()
    value_targets_np = value_targets.detach().cpu().numpy()
    value_pred_normalized_equivalent_np = value_pred_normalized_equivalent.detach().cpu().numpy()
    returns_np = returns.detach().cpu().numpy()
    advantages_np = advantages.detach().cpu().numpy()
    one_step_rewards_np = one_step_rewards.detach().cpu().numpy()
    one_step_td_targets_np = one_step_td_targets.detach().cpu().numpy()
    actual_next_value_np = actual_next_value_raw.detach().cpu().numpy()

    for sample_id in range(states.shape[0]):
        for block_id in range(actual_joint_action.shape[1]):
            sensor_id = int(block_id // task_count)
            task_id = int(block_id % task_count)
            reward_action_name = _action_name(int(best_action_under_reward[sample_id, block_id]))
            value_action_name = _action_name(int(best_action_under_value[sample_id, block_id]))
            real_rows.append(
                {
                    "sample_id": sample_id,
                    "timestep_id": sample_id,
                    "state_type": "real_rollout",
                    "block_id": block_id,
                    "sensor_id": sensor_id,
                    "task_id": task_id,
                    "candidate_action": _action_name(int(actual_joint_action[sample_id, block_id])),
                    "value_pred": float(real_value_raw_np[sample_id]),
                    "value_pred_normalized": float(real_value_normalized_np[sample_id]),
                    "value_pred_normalized_equivalent": float(value_pred_normalized_equivalent_np[sample_id]),
                    "return": float(returns_np[sample_id]),
                    "value_target": float(value_targets_np[sample_id]),
                    "advantage": float(advantages_np[sample_id]),
                    "one_step_reward": float(one_step_rewards_np[sample_id]),
                    "one_step_td_target": float(one_step_td_targets_np[sample_id]),
                    "reward_best_vs_value_best_match": int(
                        best_action_under_reward[sample_id, block_id]
                        == best_action_under_value[sample_id, block_id]
                    ),
                    "reward_value_flip_type": (
                        f"reward:{reward_action_name}->value:{value_action_name}"
                        if reward_action_name != value_action_name
                        else "match"
                    ),
                    "workload_mean": float(states_np[sample_id, slices["workload"]].mean()),
                    "virtual_queue_mean": float(states_np[sample_id, slices["virtual_queue"]].mean()),
                    "local_queue_mean": float(states_np[sample_id, slices["local_queue"]].mean()),
                    "access_queue_mean": float(states_np[sample_id, slices["access_queue"]].mean()),
                    "bs_queue_mean": float(states_np[sample_id, slices["bs_queue"]].mean()),
                }
            )
            for candidate_index in range(3):
                counterfactual_rows.append(
                    {
                        "sample_id": sample_id,
                        "timestep_id": sample_id,
                        "state_type": "counterfactual",
                        "block_id": block_id,
                        "sensor_id": sensor_id,
                        "task_id": task_id,
                        "candidate_action": _action_name(candidate_index),
                        "value_pred": float(value_scores[sample_id, block_id, candidate_index]),
                        "value_pred_normalized": np.nan,
                        "value_pred_normalized_equivalent": np.nan,
                        "return": float(returns_np[sample_id]),
                        "value_target": float(value_targets_np[sample_id]),
                        "advantage": float(advantages_np[sample_id]),
                        "one_step_reward": float(reward_scores[sample_id, block_id, candidate_index]),
                        "one_step_td_target": np.nan,
                        "reward_best_vs_value_best_match": int(
                            best_action_under_reward[sample_id, block_id]
                            == best_action_under_value[sample_id, block_id]
                        ),
                        "reward_value_flip_type": (
                            f"reward:{reward_action_name}->value:{value_action_name}"
                            if reward_action_name != value_action_name
                            else "match"
                        ),
                        "workload_mean": float(
                            counterfactual_next_states[sample_id, block_id, candidate_index, slices["workload"]].mean()
                        ),
                        "virtual_queue_mean": float(
                            counterfactual_next_states[sample_id, block_id, candidate_index, slices["virtual_queue"]].mean()
                        ),
                        "local_queue_mean": float(
                            counterfactual_next_states[sample_id, block_id, candidate_index, slices["local_queue"]].mean()
                        ),
                        "access_queue_mean": float(
                            counterfactual_next_states[sample_id, block_id, candidate_index, slices["access_queue"]].mean()
                        ),
                        "bs_queue_mean": float(
                            counterfactual_next_states[sample_id, block_id, candidate_index, slices["bs_queue"]].mean()
                        ),
                        "raw_state_distance_proxy": float(raw_state_distance[sample_id, block_id, candidate_index]),
                        "critic_input_distance_proxy": float(
                            critic_input_distance[sample_id, block_id, candidate_index]
                        ),
                    }
                )

    raw_df = pd.DataFrame(real_rows + counterfactual_rows)
    real_df = pd.DataFrame(real_rows)
    counterfactual_df = pd.DataFrame(counterfactual_rows)
    flip_df = counterfactual_df[counterfactual_df["reward_value_flip_type"] != "match"].copy()
    flip_counts = Counter(flip_df["reward_value_flip_type"])

    summary = {
        "metadata": {
            "baseline_mode": BASELINE_MODE,
            "checkpoint_path": str(checkpoint_path),
            "seed": args.seed,
            "audit_steps": int(args.audit_steps),
            "sample_count": int(states.shape[0]),
            "block_count": int(actual_joint_action.shape[1]),
            "row_count": int(len(raw_df)),
            "value_target_mode": str(agent.config.value_target_mode),
            "gamma": float(agent.config.gamma),
            "gae_lambda": float(agent.config.gae_lambda),
        },
        "value_path": {
            "evaluate_value_path": (
                "PPOAgent.evaluate_value(state) -> "
                "PPOAgent._value_from_state_tensor(state_tensor) -> "
                "PPOAgent._prepare_critic_inputs(update_stats=False) -> "
                "ActorCritic.value_from_critic_input(critic_input, mean, std) -> "
                "normalized critic prediction * popart_std + popart_mean"
            ),
            "training_value_head_semantics": (
                "Main scalar critic head. Under popart_return_norm, training loss fits "
                "normalized targets, while evaluate_value returns the corresponding raw, "
                "de-normalized value estimate in return space."
            ),
        },
        "training_target_pipeline": {
            "buffer_stored_quantities": {
                "state": "raw normalized state vector",
                "next_state": "next raw normalized state vector",
                "reward": "environment one-step reward",
                "done": "episode termination flag",
                "value": "raw scalar value from evaluate_value(state) at rollout time",
                "advantages": "GAE(lambda) from delta_t",
                "returns": "advantages + value",
            },
            "finish_trajectory_formula": {
                "delta_t": "reward_t + gamma * value_{t+1} * (1-done_t) - value_t",
                "gae_t": "delta_t + gamma * gae_lambda * (1-done_t) * gae_{t+1}",
                "return_t": "gae_t + value_t",
            },
            "value_target_construction": {
                "raw_target": "returns",
                "popart_stats_update": "running mean/std updated on rollout returns",
                "normalized_target": "(returns - target_mean) / (target_std + 1e-8)",
                "value_loss_target": "normalized_target",
            },
            "critic_prediction_used_in_loss": {
                "training_prediction": "network.normalized_value_from_critic_input(critic_input)",
                "raw_equivalent_prediction": "(evaluate_value(state) - target_mean) / (target_std + 1e-8)",
                "popart_rescale": "critic head is rescaled before loss so raw semantics stay continuous",
            },
            "inference_prediction": {
                "evaluate_value_output": "raw de-normalized scalar value in return space",
                "same_semantics_as_training": True,
            },
            "target_mean": float(target_mean.item()),
            "target_std": float(target_std.item()),
        },
        "real_rollout_value_semantics": {
            "value_vs_return": {
                "pearson": _safe_corr(real_df["value_pred"], real_df["return"], "pearson"),
                "spearman": _safe_corr(real_df["value_pred"], real_df["return"], "spearman"),
            },
            "value_vs_value_target": {
                "pearson": _safe_corr(
                    real_df["value_pred_normalized_equivalent"],
                    real_df["value_target"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    real_df["value_pred_normalized_equivalent"],
                    real_df["value_target"],
                    "spearman",
                ),
            },
            "normalized_value_vs_value_target": {
                "pearson": _safe_corr(
                    real_df["value_pred_normalized"],
                    real_df["value_target"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    real_df["value_pred_normalized"],
                    real_df["value_target"],
                    "spearman",
                ),
            },
            "value_vs_advantage": {
                "pearson": _safe_corr(real_df["value_pred"], real_df["advantage"], "pearson"),
                "spearman": _safe_corr(real_df["value_pred"], real_df["advantage"], "spearman"),
            },
            "value_vs_one_step_reward": {
                "pearson": _safe_corr(
                    real_df["value_pred"],
                    real_df["one_step_reward"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    real_df["value_pred"],
                    real_df["one_step_reward"],
                    "spearman",
                ),
            },
            "value_vs_one_step_td_target": {
                "pearson": _safe_corr(
                    real_df["value_pred"],
                    real_df["one_step_td_target"],
                    "pearson",
                ),
                "spearman": _safe_corr(
                    real_df["value_pred"],
                    real_df["one_step_td_target"],
                    "spearman",
                ),
            },
        },
        "counterfactual_value_semantics": {
            "selected_counterfactual_value_vs_return": {
                "pearson": _safe_corr(
                    pd.Series(selected_value_score.reshape(-1)),
                    pd.Series(np.repeat(returns_np, actual_joint_action.shape[1])),
                    "pearson",
                ),
                "spearman": _safe_corr(
                    pd.Series(selected_value_score.reshape(-1)),
                    pd.Series(np.repeat(returns_np, actual_joint_action.shape[1])),
                    "spearman",
                ),
            },
            "selected_counterfactual_value_vs_advantage": {
                "pearson": _safe_corr(
                    pd.Series(selected_value_score.reshape(-1)),
                    pd.Series(np.repeat(advantages_np, actual_joint_action.shape[1])),
                    "pearson",
                ),
                "spearman": _safe_corr(
                    pd.Series(selected_value_score.reshape(-1)),
                    pd.Series(np.repeat(advantages_np, actual_joint_action.shape[1])),
                    "spearman",
                ),
            },
            "selected_counterfactual_value_vs_reward_aligned_score": {
                "pearson": _safe_corr(
                    pd.Series(selected_value_score.reshape(-1)),
                    pd.Series(selected_reward_score.reshape(-1)),
                    "pearson",
                ),
                "spearman": _safe_corr(
                    pd.Series(selected_value_score.reshape(-1)),
                    pd.Series(selected_reward_score.reshape(-1)),
                    "spearman",
                ),
            },
            "best_counterfactual_value_vs_return": {
                "pearson": _safe_corr(
                    pd.Series(best_value_score.reshape(-1)),
                    pd.Series(np.repeat(returns_np, actual_joint_action.shape[1])),
                    "pearson",
                ),
                "spearman": _safe_corr(
                    pd.Series(best_value_score.reshape(-1)),
                    pd.Series(np.repeat(returns_np, actual_joint_action.shape[1])),
                    "spearman",
                ),
            },
            "best_counterfactual_value_vs_advantage": {
                "pearson": _safe_corr(
                    pd.Series(best_value_score.reshape(-1)),
                    pd.Series(np.repeat(advantages_np, actual_joint_action.shape[1])),
                    "pearson",
                ),
                "spearman": _safe_corr(
                    pd.Series(best_value_score.reshape(-1)),
                    pd.Series(np.repeat(advantages_np, actual_joint_action.shape[1])),
                    "spearman",
                ),
            },
            "best_counterfactual_value_vs_reward_aligned_score": {
                "pearson": _safe_corr(
                    pd.Series(best_value_score.reshape(-1)),
                    pd.Series(best_reward_score.reshape(-1)),
                    "pearson",
                ),
                "spearman": _safe_corr(
                    pd.Series(best_value_score.reshape(-1)),
                    pd.Series(best_reward_score.reshape(-1)),
                    "spearman",
                ),
            },
        },
        "distribution_comparison": {
            "real_rollout_state_value_stats": _series_stats(pd.Series(real_value_raw_np)),
            "real_rollout_next_state_value_stats": _series_stats(pd.Series(actual_next_value_np)),
            "counterfactual_next_state_value_stats": _series_stats(pd.Series(value_scores.reshape(-1))),
            "real_state_feature_stats": real_state_feature_stats,
            "real_next_state_feature_stats": actual_next_state_feature_stats,
            "counterfactual_next_state_feature_stats": counterfactual_feature_stats,
            "raw_state_distance_stats": _series_stats(pd.Series(raw_state_distance.reshape(-1))),
            "critic_input_distance_stats": _series_stats(pd.Series(critic_input_distance.reshape(-1))),
        },
        "reward_value_ranking": {
            "reward_best_vs_value_best_match_ratio": float(
                counterfactual_df["reward_best_vs_value_best_match"].mean()
            ),
            "reward_value_flip_type_counts": {
                flip_type: int(count)
                for flip_type, count in sorted(flip_counts.items())
            },
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"critic_target_semantics_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = output_dir / "critic_target_semantics_audit_raw.csv"
    summary_json_path = output_dir / "critic_target_semantics_audit_summary.json"
    summary_md_path = output_dir / "critic_target_semantics_audit_summary.md"
    corr_csv_path = output_dir / "critic_target_corr_analysis.csv"
    pipeline_md_path = output_dir / "critic_target_pipeline_trace.md"
    shift_csv_path = output_dir / "critic_target_distribution_shift.csv"

    raw_df.to_csv(raw_csv_path, index=False)
    pd.DataFrame(
        [
            {"section": "real_rollout_value_semantics", "pair": key, **value}
            for key, value in summary["real_rollout_value_semantics"].items()
        ]
        + [
            {"section": "counterfactual_value_semantics", "pair": key, **value}
            for key, value in summary["counterfactual_value_semantics"].items()
        ]
    ).to_csv(corr_csv_path, index=False)
    shift_rows = []
    for feature_name in ["workload", "access_queue", "virtual_queue", "local_queue", "bs_queue"]:
        shift_rows.append(
            {
                "feature": feature_name,
                "real_state_mean": real_state_feature_stats[feature_name]["mean"],
                "real_next_state_mean": actual_next_state_feature_stats[feature_name]["mean"],
                "counterfactual_next_state_mean": counterfactual_feature_stats[feature_name]["mean"],
                "real_state_std": real_state_feature_stats[feature_name]["std"],
                "real_next_state_std": actual_next_state_feature_stats[feature_name]["std"],
                "counterfactual_next_state_std": counterfactual_feature_stats[feature_name]["std"],
            }
        )
    pd.DataFrame(shift_rows).to_csv(shift_csv_path, index=False)

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    pipeline_lines = [
        "# Critic Target Pipeline Trace",
        "",
        "1. Rollout collection",
        "   - `select_action(state)` returns raw `value = evaluate_value(state)`.",
        "   - Buffer stores `reward`, `done`, `value`, `state`, `next_state`.",
        "",
        "2. Trajectory finish",
        "   - `delta_t = reward_t + gamma * value_{t+1} * (1-done_t) - value_t`",
        "   - `gae_t = delta_t + gamma * gae_lambda * (1-done_t) * gae_{t+1}`",
        "   - `return_t = gae_t + value_t`",
        "",
        "3. Training target under `popart_return_norm`",
        f"   - target_mean = {float(target_mean.item()):.6f}",
        f"   - target_std = {float(target_std.item()):.6f}",
        "   - `value_targets = (returns - target_mean) / (target_std + 1e-8)`",
        "",
        "4. Critic prediction used in loss",
        "   - `normalized_values = network.normalized_value_from_critic_input(critic_input)`",
        "   - loss fits `normalized_values` against `value_targets`",
        "",
        "5. Inference-time value",
        "   - `evaluate_value(state)` returns raw de-normalized value",
        "   - equivalent normalized prediction is `(evaluate_value(state) - target_mean) / target_std`",
    ]
    with pipeline_md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(pipeline_lines) + "\n")

    md_lines = [
        "# Critic Target Semantics Audit",
        "",
        f"- baseline_mode: `{BASELINE_MODE}`",
        f"- checkpoint_path: `{checkpoint_path}`",
        f"- sample_count: {int(states.shape[0])}",
        f"- block_count: {int(actual_joint_action.shape[1])}",
        f"- row_count: {len(raw_df)}",
        "",
        "## Real Rollout Value Semantics",
        f"- value vs return: pearson={summary['real_rollout_value_semantics']['value_vs_return']['pearson']:.4f}, spearman={summary['real_rollout_value_semantics']['value_vs_return']['spearman']:.4f}",
        f"- value vs value_target: pearson={summary['real_rollout_value_semantics']['value_vs_value_target']['pearson']:.4f}, spearman={summary['real_rollout_value_semantics']['value_vs_value_target']['spearman']:.4f}",
        f"- value vs advantage: pearson={summary['real_rollout_value_semantics']['value_vs_advantage']['pearson']:.4f}, spearman={summary['real_rollout_value_semantics']['value_vs_advantage']['spearman']:.4f}",
        f"- value vs one-step reward: pearson={summary['real_rollout_value_semantics']['value_vs_one_step_reward']['pearson']:.4f}, spearman={summary['real_rollout_value_semantics']['value_vs_one_step_reward']['spearman']:.4f}",
        f"- value vs one-step TD target: pearson={summary['real_rollout_value_semantics']['value_vs_one_step_td_target']['pearson']:.4f}, spearman={summary['real_rollout_value_semantics']['value_vs_one_step_td_target']['spearman']:.4f}",
        "",
        "## Counterfactual Value Semantics",
        f"- selected counterfactual value vs return: pearson={summary['counterfactual_value_semantics']['selected_counterfactual_value_vs_return']['pearson']:.4f}, spearman={summary['counterfactual_value_semantics']['selected_counterfactual_value_vs_return']['spearman']:.4f}",
        f"- selected counterfactual value vs advantage: pearson={summary['counterfactual_value_semantics']['selected_counterfactual_value_vs_advantage']['pearson']:.4f}, spearman={summary['counterfactual_value_semantics']['selected_counterfactual_value_vs_advantage']['spearman']:.4f}",
        f"- selected counterfactual value vs reward-aligned score: pearson={summary['counterfactual_value_semantics']['selected_counterfactual_value_vs_reward_aligned_score']['pearson']:.4f}, spearman={summary['counterfactual_value_semantics']['selected_counterfactual_value_vs_reward_aligned_score']['spearman']:.4f}",
        "",
        "## Distribution",
        f"- real rollout state value mean/std: {summary['distribution_comparison']['real_rollout_state_value_stats']['mean']:.4f} / {summary['distribution_comparison']['real_rollout_state_value_stats']['std']:.4f}",
        f"- real rollout next-state value mean/std: {summary['distribution_comparison']['real_rollout_next_state_value_stats']['mean']:.4f} / {summary['distribution_comparison']['real_rollout_next_state_value_stats']['std']:.4f}",
        f"- counterfactual next-state value mean/std: {summary['distribution_comparison']['counterfactual_next_state_value_stats']['mean']:.4f} / {summary['distribution_comparison']['counterfactual_next_state_value_stats']['std']:.4f}",
        f"- raw state distance mean/std: {summary['distribution_comparison']['raw_state_distance_stats']['mean']:.4f} / {summary['distribution_comparison']['raw_state_distance_stats']['std']:.4f}",
        f"- critic input distance mean/std: {summary['distribution_comparison']['critic_input_distance_stats']['mean']:.4f} / {summary['distribution_comparison']['critic_input_distance_stats']['std']:.4f}",
        "",
        "## Ranking Flips",
        f"- reward best vs value best match ratio: {summary['reward_value_ranking']['reward_best_vs_value_best_match_ratio']:.4f}",
    ]
    with summary_md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines) + "\n")

    print(f"Saved summary JSON to {summary_json_path}")
    print(f"Saved summary MD to {summary_md_path}")
    print(f"Saved raw CSV to {raw_csv_path}")


if __name__ == "__main__":
    main()
