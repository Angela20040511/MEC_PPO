from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import MECConfig, build_config
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator

FIXED_SEED = 2025
FIXED_TOPOLOGY = "dense"
FIXED_REWARD_MODE = "per_sensor"
FIXED_VALUE_TARGET_MODE = "popart_return_norm"
FIXED_VALUE_LOSS_MODE = "huber"

FIXED_ACTOR_LEARNING_RATE = 5e-4
FIXED_CRITIC_LEARNING_RATE = 5e-4
FIXED_ENTROPY_COEFF = 1e-3
FIXED_VALUE_COEFF = 0.5
FIXED_UPDATE_EPOCHS = 10
FIXED_TIME_STEPS = 100

FIXED_DT_HISTORY_WINDOW = 8
FIXED_DT_PREDICTION_HORIZON = 2
FIXED_DT_HIDDEN_SIZE = 128
FIXED_DT_RETRAIN_INTERVAL = 5
FIXED_DT_TRAIN_EPOCHS = 5
FIXED_DT_NUM_LAYERS = 1
FIXED_DT_LEARNING_RATE = 1e-3
FIXED_DT_BATCH_SIZE = 32
FIXED_DT_MIN_HISTORY_TO_TRAIN = 20

FIXED_NOMA_QUANTILE = 0.65
FIXED_MAX_CLUSTER_SIZE = 2
FIXED_ALPHA_P = 0.4
FIXED_RHO0 = 0.75
FIXED_GAMMA_Q = 0.10
FIXED_GAMMA_E = 0.15

FIXED_START_CHECK_EPOCH = 1
FIXED_MAX_EPOCHS = 6
FIXED_PATIENCE = 3
FIXED_MIN_DELTA = 1.0

PROBE_STATE_COUNT = 64
PROBE_ROLLOUT_SEED_OFFSET = 991


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense critic diagnosis with fixed PopArt + Huber mainline."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=FIXED_SEED,
        help="Fixed seed for this controlled diagnosis run.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="checkpoints",
        help="Root directory for diagnosis outputs.",
    )
    return parser.parse_args()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in df.to_numpy():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_diagnosis_config(seed: int) -> MECConfig:
    config = build_config(topology_mode=FIXED_TOPOLOGY)
    training = replace(
        config.training,
        num_epochs=FIXED_MAX_EPOCHS,
        time_steps=FIXED_TIME_STEPS,
        seed=seed,
    )
    ppo = replace(
        config.ppo,
        actor_learning_rate=FIXED_ACTOR_LEARNING_RATE,
        critic_learning_rate=FIXED_CRITIC_LEARNING_RATE,
        entropy_coeff=FIXED_ENTROPY_COEFF,
        value_coeff=FIXED_VALUE_COEFF,
        value_target_mode=FIXED_VALUE_TARGET_MODE,
        value_loss_mode=FIXED_VALUE_LOSS_MODE,
        update_epochs=FIXED_UPDATE_EPOCHS,
    )
    dt = replace(
        config.dt,
        history_window=FIXED_DT_HISTORY_WINDOW,
        prediction_horizon=FIXED_DT_PREDICTION_HORIZON,
        hidden_size=FIXED_DT_HIDDEN_SIZE,
        retrain_interval=FIXED_DT_RETRAIN_INTERVAL,
        train_epochs=FIXED_DT_TRAIN_EPOCHS,
        num_layers=FIXED_DT_NUM_LAYERS,
        learning_rate=FIXED_DT_LEARNING_RATE,
        batch_size=FIXED_DT_BATCH_SIZE,
        min_history_to_train=FIXED_DT_MIN_HISTORY_TO_TRAIN,
    )
    system = replace(
        config.system,
        reward_mode=FIXED_REWARD_MODE,
        noma_quantile=FIXED_NOMA_QUANTILE,
        max_cluster_size=FIXED_MAX_CLUSTER_SIZE,
        alpha_p=FIXED_ALPHA_P,
        rho0=FIXED_RHO0,
        gamma_q=FIXED_GAMMA_Q,
        gamma_e=FIXED_GAMMA_E,
    )
    return replace(config, training=training, ppo=ppo, dt=dt, system=system)


def save_logs(run_dir: Path, logs: list[dict[str, float | int | str]]) -> None:
    if not logs:
        return
    with (run_dir / "train_logs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)


def collect_probe_states(config: MECConfig, seed: int, probe_count: int) -> np.ndarray:
    simulator = Simulator(config)
    rng = np.random.default_rng(seed + PROBE_ROLLOUT_SEED_OFFSET)
    state = simulator.reset(seed=seed + PROBE_ROLLOUT_SEED_OFFSET)

    states: list[np.ndarray] = []
    while len(states) < probe_count:
        states.append(np.array(state, dtype=np.float32, copy=True))
        action = rng.uniform(-1.0, 1.0, size=config.action_dim).astype(np.float32)
        state, _, done, _ = simulator.step(action)
        if done:
            state = simulator.reset(seed=seed + PROBE_ROLLOUT_SEED_OFFSET + len(states))
    return np.stack(states)


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
        raise ValueError(f"Unsupported joint candidate index: {candidate_index}")
    variant[block_slice] = block
    return variant


def compute_joint_counterfactual_scores(
    agent: PPOAgent,
    simulator: Simulator,
    action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    block_slices = agent._action_block_slices()
    reward_scores = np.zeros((len(block_slices), 3), dtype=np.float32)
    td_scores = np.zeros((len(block_slices), 3), dtype=np.float32)
    flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
    for block_id, block_slice in enumerate(block_slices):
        for candidate_index in range(3):
            variant_action = _set_block_joint_candidate(
                flat_action,
                block_slice,
                candidate_index,
            )
            simulator_cf = copy.deepcopy(simulator)
            next_state_cf, reward_cf, done_cf, _ = simulator_cf.step(variant_action)
            reward_scores[block_id, candidate_index] = float(reward_cf)
            next_value_cf = 0.0 if done_cf else float(agent.evaluate_value(next_state_cf))
            td_scores[block_id, candidate_index] = float(
                reward_cf + agent.config.gamma * next_value_cf
            )
    return reward_scores, td_scores


def diagnose_probe_states(agent: PPOAgent, probe_states: np.ndarray) -> dict[str, float]:
    state_tensor = torch.tensor(probe_states, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        predictions = agent._value_from_state_tensor(state_tensor).squeeze(-1).cpu().numpy()

    prediction_mean = float(np.mean(predictions))
    prediction_std = float(np.std(predictions))
    prediction_min = float(np.min(predictions))
    prediction_max = float(np.max(predictions))
    q25, q75 = np.percentile(predictions, [25, 75])
    if len(predictions) > 1:
        diff_matrix = np.abs(predictions[:, None] - predictions[None, :])
        upper = diff_matrix[np.triu_indices(len(predictions), k=1)]
        pairwise_prediction_distance_mean = float(upper.mean()) if upper.size > 0 else 0.0
    else:
        pairwise_prediction_distance_mean = 0.0

    return {
        "probe_prediction_mean": prediction_mean,
        "probe_prediction_std": prediction_std,
        "probe_prediction_min": prediction_min,
        "probe_prediction_max": prediction_max,
        "probe_prediction_range": prediction_max - prediction_min,
        "probe_prediction_iqr": float(q75 - q25),
        "pairwise_prediction_distance_mean": pairwise_prediction_distance_mean,
    }


def decode_raw_action_batch_to_dispatch_probs(
    config: MECConfig,
    raw_actions: np.ndarray,
) -> tuple[np.ndarray, list[tuple[str, str]], list[str]]:
    """Decode raw actor outputs into local/BS dispatch probabilities per sensor-task pair."""
    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]

    decision_pairs = [
        (sensor_id, task_name)
        for sensor_id in config.sensor_ids
        for task_name in config.task_names
    ]
    base_station_ids = list(config.base_station_ids)
    choice_labels = ["local", *base_station_ids]
    dispatch_probs = np.zeros(
        (actions.shape[0], len(decision_pairs), len(choice_labels)),
        dtype=np.float32,
    )

    cursor = 0
    for decision_index, (sensor_id, _task_name) in enumerate(decision_pairs):
        raw_theta = actions[:, cursor]
        cursor += 1
        raw_logits = actions[:, cursor : cursor + len(base_station_ids)]
        cursor += len(base_station_ids)

        clipped_theta = np.clip(raw_theta, -40.0, 40.0)
        theta = 1.0 / (1.0 + np.exp(-clipped_theta))
        dispatch_probs[:, decision_index, 0] = 1.0 - theta

        reachable_bs = config.reachable_base_stations(sensor_id)
        reachable_indices = [base_station_ids.index(bs_id) for bs_id in reachable_bs]
        reachable_logits = raw_logits[:, reachable_indices]
        shifted = reachable_logits - np.max(reachable_logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        reachable_probs = exp_values / np.clip(exp_values.sum(axis=1, keepdims=True), 1e-8, None)
        for reachable_offset, bs_index in enumerate(reachable_indices):
            dispatch_probs[:, decision_index, 1 + bs_index] = theta * reachable_probs[
                :,
                reachable_offset,
            ]

    return dispatch_probs, decision_pairs, choice_labels


def capture_probe_policy_snapshot(
    agent: PPOAgent,
    probe_states: np.ndarray,
) -> dict[str, np.ndarray]:
    """Capture the current actor mean/std on a fixed probe-state set."""
    state_tensor = torch.tensor(probe_states, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        actor_inputs, _, _ = agent._prepare_actor_inputs(state_tensor, update_stats=False)
        distribution = agent.network.policy_from_actor_input(actor_inputs)
    return {
        "mean_actions": distribution.mean.detach().cpu().numpy().astype(np.float32),
        "std_actions": distribution.stddev.detach().cpu().numpy().astype(np.float32),
    }


def compare_probe_policy_snapshots(
    config: MECConfig,
    old_snapshot: dict[str, np.ndarray],
    new_snapshot: dict[str, np.ndarray],
    sample_noise: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compare old/new probe-state policies in decoded dispatch space and raw-logit space."""
    old_mean_actions = np.asarray(old_snapshot["mean_actions"], dtype=np.float32)
    old_std_actions = np.asarray(old_snapshot["std_actions"], dtype=np.float32)
    new_mean_actions = np.asarray(new_snapshot["mean_actions"], dtype=np.float32)
    new_std_actions = np.asarray(new_snapshot["std_actions"], dtype=np.float32)
    shared_noise = np.asarray(sample_noise, dtype=np.float32)

    old_dispatch_probs, decision_pairs, choice_labels = decode_raw_action_batch_to_dispatch_probs(
        config=config,
        raw_actions=old_mean_actions,
    )
    new_dispatch_probs, _, _ = decode_raw_action_batch_to_dispatch_probs(
        config=config,
        raw_actions=new_mean_actions,
    )

    old_sample_actions = old_mean_actions + old_std_actions * shared_noise
    new_sample_actions = new_mean_actions + new_std_actions * shared_noise
    old_sample_dispatch_probs, _, _ = decode_raw_action_batch_to_dispatch_probs(
        config=config,
        raw_actions=old_sample_actions,
    )
    new_sample_dispatch_probs, _, _ = decode_raw_action_batch_to_dispatch_probs(
        config=config,
        raw_actions=new_sample_actions,
    )

    eps = 1e-8
    decision_kl = np.sum(
        old_dispatch_probs * (np.log(old_dispatch_probs + eps) - np.log(new_dispatch_probs + eps)),
        axis=-1,
    )
    decision_l1 = np.sum(np.abs(new_dispatch_probs - old_dispatch_probs), axis=-1)
    old_top1 = np.argmax(old_dispatch_probs, axis=-1)
    new_top1 = np.argmax(new_dispatch_probs, axis=-1)
    old_sample_top1 = np.argmax(old_sample_dispatch_probs, axis=-1)
    new_sample_top1 = np.argmax(new_sample_dispatch_probs, axis=-1)
    old_top1_prob = np.max(old_dispatch_probs, axis=-1)
    new_top1_prob = np.max(new_dispatch_probs, axis=-1)
    logit_delta = new_mean_actions - old_mean_actions

    metrics = {
        "probe_policy_pairwise_kl_mean": float(np.mean(decision_kl)),
        "probe_policy_pairwise_l1_mean": float(np.mean(decision_l1)),
        "policy_logit_delta_mean": float(np.mean(logit_delta)),
        "policy_logit_delta_std": float(np.std(logit_delta)),
        "selected_action_change_rate": float(np.mean(old_sample_top1 != new_sample_top1)),
        "top1_action_change_rate": float(np.mean(old_top1 != new_top1)),
        "top1_prob_delta_mean": float(np.mean(new_top1_prob - old_top1_prob)),
    }

    payload_rows: list[dict[str, float | int | str]] = []
    for probe_index in range(old_dispatch_probs.shape[0]):
        for decision_index, (sensor_id, task_name) in enumerate(decision_pairs):
            row: dict[str, float | int | str] = {
                "probe_index": int(probe_index),
                "sensor_id": sensor_id,
                "task_name": task_name,
                "old_top1_choice": choice_labels[int(old_top1[probe_index, decision_index])],
                "new_top1_choice": choice_labels[int(new_top1[probe_index, decision_index])],
                "top1_changed": int(old_top1[probe_index, decision_index] != new_top1[probe_index, decision_index]),
                "old_sampled_choice": choice_labels[
                    int(old_sample_top1[probe_index, decision_index])
                ],
                "new_sampled_choice": choice_labels[
                    int(new_sample_top1[probe_index, decision_index])
                ],
                "sampled_changed": int(
                    old_sample_top1[probe_index, decision_index]
                    != new_sample_top1[probe_index, decision_index]
                ),
                "old_top1_prob": float(old_top1_prob[probe_index, decision_index]),
                "new_top1_prob": float(new_top1_prob[probe_index, decision_index]),
                "top1_prob_delta": float(
                    new_top1_prob[probe_index, decision_index]
                    - old_top1_prob[probe_index, decision_index]
                ),
                "decision_kl": float(decision_kl[probe_index, decision_index]),
                "decision_l1": float(decision_l1[probe_index, decision_index]),
            }
            for choice_index, choice_label in enumerate(choice_labels):
                row[f"old_{choice_label}_prob"] = float(
                    old_dispatch_probs[probe_index, decision_index, choice_index]
                )
                row[f"new_{choice_label}_prob"] = float(
                    new_dispatch_probs[probe_index, decision_index, choice_index]
                )
            payload_rows.append(row)

    return metrics, pd.DataFrame(payload_rows)


def save_probe_policy_update_payload(
    run_dir: Path,
    epoch: int,
    payload_df: pd.DataFrame,
) -> Path:
    """Persist decoded probe-policy comparisons for one epoch."""
    payload_path = run_dir / f"probe_policy_update_payload_epoch_{epoch:02d}.csv"
    payload_df.to_csv(payload_path, index=False, encoding="utf-8-sig")
    return payload_path


def save_prediction_target_payload(
    run_dir: Path,
    epoch: int,
    value_targets: list[float],
    predictions: list[float],
    raw_predictions: list[float],
    constant_predictions: list[float],
) -> Path:
    payload_path = run_dir / f"prediction_target_payload_epoch_{epoch:02d}.csv"
    rows = [
        {
            "value_target": target,
            "critic_prediction": prediction,
            "critic_raw_prediction": raw_prediction,
            "constant_prediction": constant_prediction,
        }
        for target, prediction, raw_prediction, constant_prediction in zip(
            value_targets,
            predictions,
            raw_predictions,
            constant_predictions,
            strict=True,
        )
    ]
    with payload_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return payload_path


def run_training_epoch(
    agent: PPOAgent,
    simulator: Simulator,
    config: MECConfig,
    epoch: int,
    probe_states: np.ndarray,
    probe_action_noise: np.ndarray,
    run_dir: Path,
) -> tuple[
    dict[str, float | int | str],
    Path,
    list[dict[str, float | int | str]],
    Path,
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    state = simulator.reset(seed=config.training.seed + epoch)

    episode_reward = 0.0
    episode_delay = 0.0
    episode_energy = 0.0
    episode_raw_total_backlog = 0.0
    episode_normalized_total_backlog = 0.0
    episode_normalized_avg_queue_len = 0.0
    episode_reward_delay_term = 0.0
    episode_reward_energy_term = 0.0
    episode_reward_backlog_term = 0.0
    episode_avg_theta = 0.0

    step_count = 0
    done = False
    for _ in range(config.training.time_steps):
        action, log_prob, value = agent.select_action(state)
        joint_reward_aligned_scores = None
        joint_td_aligned_scores = None
        if config.ppo.policy_ratio_mode in {
            "hierarchical_actor_theta_route_candidate_score_credit",
            "hierarchical_actor_joint_reward_aligned_credit",
            "hierarchical_actor_joint_td_aligned_credit",
        }:
            (
                joint_reward_aligned_scores,
                joint_td_aligned_scores,
            ) = compute_joint_counterfactual_scores(
                agent,
                simulator,
                np.asarray(action, dtype=np.float32),
            )
        next_state, reward, done, info = simulator.step(action)

        agent.store_transition(
            state,
            action,
            log_prob,
            reward,
            done,
            value,
            next_state,
            joint_reward_aligned_scores=joint_reward_aligned_scores,
            joint_td_aligned_scores=joint_td_aligned_scores,
        )
        state = next_state

        episode_reward += float(reward)
        episode_delay += float(info["raw_total_delay"])
        episode_energy += float(info["raw_total_energy"])
        episode_raw_total_backlog += float(info["raw_total_backlog"])
        episode_normalized_total_backlog += float(info["normalized_total_backlog"])
        episode_normalized_avg_queue_len += float(info["avg_queue_len"])
        episode_reward_delay_term += float(info["reward_delay_term"])
        episode_reward_energy_term += float(info["reward_energy_term"])
        episode_reward_backlog_term += float(info["reward_backlog_term"])
        episode_avg_theta += float(info["avg_theta"])

        step_count += 1
        if done:
            break

    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)
    old_probe_policy = capture_probe_policy_snapshot(agent, probe_states)
    losses = agent.train()
    new_probe_policy = capture_probe_policy_snapshot(agent, probe_states)
    probe_metrics = diagnose_probe_states(agent, probe_states)
    probe_policy_metrics, probe_policy_payload = compare_probe_policy_snapshots(
        config=config,
        old_snapshot=old_probe_policy,
        new_snapshot=new_probe_policy,
        sample_noise=probe_action_noise,
    )
    payload_path = save_prediction_target_payload(
        run_dir=run_dir,
        epoch=epoch,
        value_targets=losses["diagnostic_value_targets"],
        predictions=losses["diagnostic_prediction_values"],
        raw_predictions=losses["diagnostic_raw_predictions"],
        constant_predictions=losses["diagnostic_constant_predictions"],
    )
    probe_policy_payload_path = save_probe_policy_update_payload(
        run_dir=run_dir,
        epoch=epoch,
        payload_df=probe_policy_payload,
    )
    advantage_bucket_rows: list[dict[str, float | int | str]] = []
    for row in losses["diagnostic_advantage_bucket_rows"]:
        bucket_row = dict(row)
        bucket_row["epoch"] = int(epoch)
        advantage_bucket_rows.append(bucket_row)
    block_logprob_rows: list[dict[str, float | int | str]] = []
    for row in losses.get("diagnostic_block_logprob_rows", []):
        block_row = dict(row)
        block_row["epoch"] = int(epoch)
        block_logprob_rows.append(block_row)
    block_surrogate_rows: list[dict[str, float | int | str]] = []
    for row in losses.get("diagnostic_block_surrogate_rows", []):
        block_row = dict(row)
        block_row["epoch"] = int(epoch)
        block_surrogate_rows.append(block_row)
    block_adv_scale_rows: list[dict[str, float | int | str]] = []
    for row in losses.get("diagnostic_block_adv_scale_rows", []):
        block_row = dict(row)
        block_row["epoch"] = int(epoch)
        block_adv_scale_rows.append(block_row)
    block_value_scale_rows: list[dict[str, float | int | str]] = []
    for row in losses.get("diagnostic_block_value_scale_rows", []):
        block_row = dict(row)
        block_row["epoch"] = int(epoch)
        block_value_scale_rows.append(block_row)
    block_td_value_rows: list[dict[str, float | int | str]] = []
    for row in losses.get("diagnostic_block_td_value_rows", []):
        block_row = dict(row)
        block_row["epoch"] = int(epoch)
        block_td_value_rows.append(block_row)
    block_delta_cost_rows: list[dict[str, float | int | str]] = []
    for row in losses.get("diagnostic_block_delta_cost_rows", []):
        block_row = dict(row)
        block_row["epoch"] = int(epoch)
        block_delta_cost_rows.append(block_row)
    block_advantage_rows: list[dict[str, float | int | str]] = []
    for row in losses.get("diagnostic_block_advantage_rows", []):
        block_row = dict(row)
        block_row["epoch"] = int(epoch)
        block_advantage_rows.append(block_row)
    block_weight_rows: list[dict[str, float | int | str]] = []
    for row in losses.get("diagnostic_block_weight_rows", []):
        block_row = dict(row)
        block_row["epoch"] = int(epoch)
        block_weight_rows.append(block_row)
    theta_route_split_rows: list[dict[str, float | int | str]] = []
    for row in losses.get("diagnostic_theta_route_split_rows", []):
        split_row = dict(row)
        split_row["epoch"] = int(epoch)
        theta_route_split_rows.append(split_row)

    log = {
        "epoch": int(epoch),
        "critic_arch_mode": config.ppo.critic_arch_mode,
        "actor_input_mode": config.ppo.actor_input_mode,
        "actor_raw_input_mode": config.ppo.actor_raw_input_mode,
        "actor_structure_mode": config.ppo.actor_structure_mode,
        "critic_input_mode": config.ppo.critic_input_mode,
        "policy_surrogate_mode": losses["policy_surrogate_mode"],
        "policy_ratio_mode": config.ppo.policy_ratio_mode,
        "actor_structure_mode": losses["actor_structure_mode"],
        "reward_mode": config.system.reward_mode,
        "value_target_mode": config.ppo.value_target_mode,
        "value_loss_mode": config.ppo.value_loss_mode,
        "num_sensors": int(len(config.sensors)),
        "action_dim": int(config.action_dim),
        "action_block_count": losses["action_block_count"],
        "action_block_slices": losses["action_block_slices"],
        "episode_reward": episode_reward,
        "episode_delay": episode_delay,
        "episode_energy": episode_energy,
        "avg_raw_total_backlog": episode_raw_total_backlog / max(step_count, 1),
        "avg_normalized_total_backlog": episode_normalized_total_backlog / max(step_count, 1),
        "avg_normalized_avg_queue_len": episode_normalized_avg_queue_len / max(step_count, 1),
        "episode_reward_delay_term": episode_reward_delay_term,
        "episode_reward_energy_term": episode_reward_energy_term,
        "episode_reward_backlog_term": episode_reward_backlog_term,
        "avg_theta": episode_avg_theta / max(step_count, 1),
        "num_steps": int(step_count),
        "actor_loss": losses["actor_loss"],
        "critic_loss": losses["critic_loss"],
        "entropy": losses["entropy"],
        "policy_entropy": losses["policy_entropy"],
        "raw_return_mean": losses["raw_return_mean"],
        "raw_return_std": losses["raw_return_std"],
        "value_target_mean": losses["value_target_mean"],
        "value_target_std": losses["value_target_std"],
        "advantage_mean": losses["advantage_mean"],
        "advantage_std": losses["advantage_std"],
        "normalized_advantage_mean": losses["normalized_advantage_mean"],
        "normalized_advantage_std": losses["normalized_advantage_std"],
        "positive_advantage_ratio": losses["positive_advantage_ratio"],
        "negative_advantage_ratio": losses["negative_advantage_ratio"],
        "delta_log_prob_selected_action_mean": losses["delta_log_prob_selected_action_mean"],
        "delta_log_prob_selected_action_std": losses["delta_log_prob_selected_action_std"],
        "selected_action_prob_gain_mean": losses["selected_action_prob_gain_mean"],
        "advantage_action_alignment": losses["advantage_action_alignment"],
        "high_advantage_action_prob_gain": losses["high_advantage_action_prob_gain"],
        "mid_advantage_action_prob_gain": losses["mid_advantage_action_prob_gain"],
        "low_advantage_action_prob_gain": losses["low_advantage_action_prob_gain"],
        "negative_advantage_action_prob_gain": losses["negative_advantage_action_prob_gain"],
        "advantage_bucket_high_positive_threshold": losses[
            "advantage_bucket_high_positive_threshold"
        ],
        "advantage_bucket_near_zero_threshold": losses["advantage_bucket_near_zero_threshold"],
        "ratio_mean": losses["ratio_mean"],
        "ratio_std": losses["ratio_std"],
        "ratio_min": losses["ratio_min"],
        "ratio_max": losses["ratio_max"],
        "clip_fraction": losses["clip_fraction"],
        "positive_adv_clip_fraction": losses["positive_adv_clip_fraction"],
        "negative_adv_clip_fraction": losses["negative_adv_clip_fraction"],
        "approx_kl": losses["approx_kl"],
        "entropy_loss": losses["entropy_loss"],
        "policy_loss": losses["policy_loss"],
        "ratio_by_adv_sign": losses["ratio_by_adv_sign"],
        "logprob_delta_sum_mean": losses["logprob_delta_sum_mean"],
        "logprob_delta_sum_std": losses["logprob_delta_sum_std"],
        "logprob_delta_mean_mean": losses["logprob_delta_mean_mean"],
        "logprob_delta_mean_std": losses["logprob_delta_mean_std"],
        "block_logprob_delta_mean": losses["block_logprob_delta_mean"],
        "block_logprob_delta_std": losses["block_logprob_delta_std"],
        "block_weight_entropy": losses["block_weight_entropy"],
        "block_weight_max_mean": losses["block_weight_max_mean"],
        "top_k_block_weight_share": losses["top_k_block_weight_share"],
        "block_adv_scale_mean": losses["block_adv_scale_mean"],
        "block_adv_scale_std": losses["block_adv_scale_std"],
        "block_adv_scale_max": losses["block_adv_scale_max"],
        "block_adv_scale_min": losses["block_adv_scale_min"],
        "block_adv_scale_entropy": losses["block_adv_scale_entropy"],
        "top_k_block_adv_scale_share": losses["top_k_block_adv_scale_share"],
        "block_value_score_mean": losses["block_value_score_mean"],
        "block_value_score_std": losses["block_value_score_std"],
        "block_value_score_max": losses["block_value_score_max"],
        "block_value_score_min": losses["block_value_score_min"],
        "block_value_scale_mean": losses["block_value_scale_mean"],
        "block_value_scale_std": losses["block_value_scale_std"],
        "block_value_scale_max": losses["block_value_scale_max"],
        "block_value_scale_min": losses["block_value_scale_min"],
        "block_value_scale_entropy": losses["block_value_scale_entropy"],
        "top_k_block_value_scale_share": losses["top_k_block_value_scale_share"],
        "block_value_pred_mean": losses["block_value_pred_mean"],
        "block_value_pred_std": losses["block_value_pred_std"],
        "block_value_pred_max": losses["block_value_pred_max"],
        "block_value_pred_min": losses["block_value_pred_min"],
        "block_path_value_local_mean": losses["block_path_value_local_mean"],
        "block_path_value_local_std": losses["block_path_value_local_std"],
        "block_path_value_bs1_mean": losses["block_path_value_bs1_mean"],
        "block_path_value_bs1_std": losses["block_path_value_bs1_std"],
        "block_path_value_bs2_mean": losses["block_path_value_bs2_mean"],
        "block_path_value_bs2_std": losses["block_path_value_bs2_std"],
        "path_value_local_mean": losses["path_value_local_mean"],
        "path_value_local_std": losses["path_value_local_std"],
        "path_value_bs1_mean": losses["path_value_bs1_mean"],
        "path_value_bs1_std": losses["path_value_bs1_std"],
        "path_value_bs2_mean": losses["path_value_bs2_mean"],
        "path_value_bs2_std": losses["path_value_bs2_std"],
        "path_td_target_local_mean": losses["path_td_target_local_mean"],
        "path_td_target_local_std": losses["path_td_target_local_std"],
        "path_td_target_bs1_mean": losses["path_td_target_bs1_mean"],
        "path_td_target_bs1_std": losses["path_td_target_bs1_std"],
        "path_td_target_bs2_mean": losses["path_td_target_bs2_mean"],
        "path_td_target_bs2_std": losses["path_td_target_bs2_std"],
        "theta_advantage_mean": losses["theta_advantage_mean"],
        "theta_advantage_std": losses["theta_advantage_std"],
        "theta_advantage_max": losses["theta_advantage_max"],
        "theta_advantage_min": losses["theta_advantage_min"],
        "route_advantage_mean": losses["route_advantage_mean"],
        "route_advantage_std": losses["route_advantage_std"],
        "route_advantage_max": losses["route_advantage_max"],
        "route_advantage_min": losses["route_advantage_min"],
        "theta_adv_norm_mean": losses["theta_adv_norm_mean"],
        "theta_adv_norm_std": losses["theta_adv_norm_std"],
        "route_adv_norm_mean": losses["route_adv_norm_mean"],
        "route_adv_norm_std": losses["route_adv_norm_std"],
        "route_adv_norm_active_count": losses["route_adv_norm_active_count"],
        "route_adv_norm_used_fallback": losses["route_adv_norm_used_fallback"],
        "theta_advantage_alignment": losses["theta_advantage_alignment"],
        "route_advantage_alignment": losses["route_advantage_alignment"],
        "theta_adv_to_route_delta_alignment": losses.get(
            "theta_adv_to_route_delta_alignment", 0.0
        ),
        "route_adv_to_theta_delta_alignment": losses.get(
            "route_adv_to_theta_delta_alignment", 0.0
        ),
        "theta_alignment_margin_over_cross": losses.get(
            "theta_alignment_margin_over_cross", 0.0
        ),
        "route_alignment_margin_over_cross": losses.get(
            "route_alignment_margin_over_cross", 0.0
        ),
        "cross_branch_alignment_mean": losses.get("cross_branch_alignment_mean", 0.0),
        "theta_selected_action_prob_gain": losses["theta_selected_action_prob_gain"],
        "route_selected_action_prob_gain": losses["route_selected_action_prob_gain"],
        "theta_negative_adv_prob_gain": losses["theta_negative_adv_prob_gain"],
        "route_negative_adv_prob_gain": losses["route_negative_adv_prob_gain"],
        "theta_clip_fraction": losses["theta_clip_fraction"],
        "route_clip_fraction": losses["route_clip_fraction"],
        "theta_ratio_mean": losses["theta_ratio_mean"],
        "theta_ratio_std": losses["theta_ratio_std"],
        "theta_ratio_max": losses["theta_ratio_max"],
        "route_ratio_mean": losses["route_ratio_mean"],
        "route_ratio_std": losses["route_ratio_std"],
        "route_ratio_max": losses["route_ratio_max"],
        "offload_active_fraction": losses.get("offload_active_fraction", 0.0),
        "route_logprob_active_fraction": losses.get("route_logprob_active_fraction", 0.0),
        "theta_positive_fraction": losses.get("theta_positive_fraction", 0.0),
        "route_credit_gate_fraction": losses.get("route_credit_gate_fraction", 0.0),
        "route_credit_effective_fraction": losses.get("route_credit_effective_fraction", 0.0),
        "route_credit_effective_count": losses.get("route_credit_effective_count", 0.0),
        "route_credit_weight_mean": losses.get("route_credit_weight_mean", 1.0),
        "route_credit_weight_std": losses.get("route_credit_weight_std", 0.0),
        "route_credit_weight_min": losses.get("route_credit_weight_min", 1.0),
        "route_credit_weight_max": losses.get("route_credit_weight_max", 1.0),
        "theta_candidate_score_mean": losses.get("theta_candidate_score_mean", 0.0),
        "theta_candidate_score_std": losses.get("theta_candidate_score_std", 0.0),
        "theta_selected_score_mean": losses.get("theta_selected_score_mean", 0.0),
        "theta_expected_score_mean": losses.get("theta_expected_score_mean", 0.0),
        "theta_residual_credit_mean": losses.get("theta_residual_credit_mean", 0.0),
        "theta_residual_credit_std": losses.get("theta_residual_credit_std", 0.0),
        "theta_residual_credit_min": losses.get("theta_residual_credit_min", 0.0),
        "theta_residual_credit_max": losses.get("theta_residual_credit_max", 0.0),
        "route_candidate_score_mean": losses.get("route_candidate_score_mean", 0.0),
        "route_candidate_score_std": losses.get("route_candidate_score_std", 0.0),
        "route_residual_credit_mean": losses.get("route_residual_credit_mean", 0.0),
        "route_residual_credit_std": losses.get("route_residual_credit_std", 0.0),
        "route_residual_credit_min": losses.get("route_residual_credit_min", 0.0),
        "route_residual_credit_max": losses.get("route_residual_credit_max", 0.0),
        "route_expected_score_mean": losses.get("route_expected_score_mean", 0.0),
        "route_selected_score_mean": losses.get("route_selected_score_mean", 0.0),
        "route_decision_agreement_ratio": losses.get("route_decision_agreement_ratio", 0.0),
        "actual_bs1_rate_when_route_true_gap_positive": losses.get(
            "actual_bs1_rate_when_route_true_gap_positive",
            0.0,
        ),
        "actual_bs1_rate_when_route_true_gap_negative": losses.get(
            "actual_bs1_rate_when_route_true_gap_negative",
            0.0,
        ),
        "route_score_vector_mean_abs": losses.get("route_score_vector_mean_abs", 0.0),
        "route_score_vector_std": losses.get("route_score_vector_std", 0.0),
        "route_credit_fallback_trigger_count": losses.get(
            "route_credit_fallback_trigger_count",
            0,
        ),
        "route_credit_fallback_rate": losses.get("route_credit_fallback_rate", 0.0),
        "theta_old_logprob_mean": losses.get("theta_old_logprob_mean", 0.0),
        "theta_new_logprob_mean": losses.get("theta_new_logprob_mean", 0.0),
        "route_old_logprob_mean": losses.get("route_old_logprob_mean", 0.0),
        "route_new_logprob_mean": losses.get("route_new_logprob_mean", 0.0),
        "joint_old_logprob_mean": losses.get("joint_old_logprob_mean", 0.0),
        "joint_new_logprob_mean": losses.get("joint_new_logprob_mean", 0.0),
        "joint_ratio_mean": losses.get("joint_ratio_mean", 0.0),
        "joint_approx_kl": losses.get("joint_approx_kl", 0.0),
        "joint_clip_fraction": losses.get("joint_clip_fraction", 0.0),
        "joint_selected_score_mean": losses.get("joint_selected_score_mean", 0.0),
        "joint_expected_score_mean": losses.get("joint_expected_score_mean", 0.0),
        "joint_residual_credit_mean": losses.get("joint_residual_credit_mean", 0.0),
        "joint_residual_credit_std": losses.get("joint_residual_credit_std", 0.0),
        "joint_residual_credit_min": losses.get("joint_residual_credit_min", 0.0),
        "joint_residual_credit_max": losses.get("joint_residual_credit_max", 0.0),
        "joint_action_decision_agreement_ratio_under_reward_aligned": losses.get(
            "joint_action_decision_agreement_ratio_under_reward_aligned",
            0.0,
        ),
        "actual_local_rate_when_reward_aligned_best_is_local": losses.get(
            "actual_local_rate_when_reward_aligned_best_is_local",
            0.0,
        ),
        "actual_bs1_rate_when_reward_aligned_best_is_bs1": losses.get(
            "actual_bs1_rate_when_reward_aligned_best_is_bs1",
            0.0,
        ),
        "actual_bs2_rate_when_reward_aligned_best_is_bs2": losses.get(
            "actual_bs2_rate_when_reward_aligned_best_is_bs2",
            0.0,
        ),
        "joint_action_decision_agreement_ratio_under_td_aligned": losses.get(
            "joint_action_decision_agreement_ratio_under_td_aligned",
            0.0,
        ),
        "actual_local_rate_when_td_aligned_best_is_local": losses.get(
            "actual_local_rate_when_td_aligned_best_is_local",
            0.0,
        ),
        "actual_bs1_rate_when_td_aligned_best_is_bs1": losses.get(
            "actual_bs1_rate_when_td_aligned_best_is_bs1",
            0.0,
        ),
        "actual_bs2_rate_when_td_aligned_best_is_bs2": losses.get(
            "actual_bs2_rate_when_td_aligned_best_is_bs2",
            0.0,
        ),
        "offload_decision_agreement_ratio": losses.get(
            "offload_decision_agreement_ratio",
            0.0,
        ),
        "actual_offload_rate_when_theta_true_gap_positive": losses.get(
            "actual_offload_rate_when_theta_true_gap_positive",
            0.0,
        ),
        "actual_offload_rate_when_theta_true_gap_negative": losses.get(
            "actual_offload_rate_when_theta_true_gap_negative",
            0.0,
        ),
        "theta_approx_kl": losses["theta_approx_kl"],
        "route_approx_kl": losses["route_approx_kl"],
        "conditional_policy_consistency_score": losses.get(
            "conditional_policy_consistency_score",
            0.0,
        ),
        "theta_loss_mean": losses["theta_loss_mean"],
        "route_loss_mean": losses["route_loss_mean"],
        "theta_route_loss_ratio": losses["theta_route_loss_ratio"],
        "route_gate_mean": losses["route_gate_mean"],
        "route_gate_std": losses["route_gate_std"],
        "route_gate_min": losses["route_gate_min"],
        "route_gate_max": losses["route_gate_max"],
        "route_gate_active_fraction": losses["route_gate_active_fraction"],
        "route_mask_mean": losses["route_mask_mean"],
        "route_mask_active_fraction": losses["route_mask_active_fraction"],
        "route_mask_threshold": losses.get("route_mask_threshold", 0.5),
        "route_confident_mask_margin": losses.get("route_confident_mask_margin", 0.0),
        "route_mask_count_mean": losses["route_mask_count_mean"],
        "route_mask_count_std": losses["route_mask_count_std"],
        "theta_head_grad_norm": losses["theta_head_grad_norm"],
        "route_head_grad_norm": losses["route_head_grad_norm"],
        "theta_backbone_grad_norm": losses["theta_backbone_grad_norm"],
        "route_backbone_grad_norm": losses["route_backbone_grad_norm"],
        "theta_kl_target": losses["theta_kl_target"],
        "route_kl_target": losses["route_kl_target"],
        "theta_early_stop_count": losses["theta_early_stop_count"],
        "route_early_stop_count": losses["route_early_stop_count"],
        "coupled_stop_trigger_count": losses.get("coupled_stop_trigger_count", 0),
        "coupled_stop_blocked_by_theta_floor_count": losses.get(
            "coupled_stop_blocked_by_theta_floor_count", 0
        ),
        "coupled_stop_blocked_by_severity_gate_count": losses.get(
            "coupled_stop_blocked_by_severity_gate_count", 0
        ),
        "coupled_stop_min_theta_updates_per_epoch": losses.get(
            "coupled_stop_min_theta_updates_per_epoch", 0
        ),
        "coupled_stop_severity_factor": losses.get("coupled_stop_severity_factor", 1.0),
        "coupled_stop_severity_freeze_kl_threshold": losses.get(
            "coupled_stop_severity_freeze_kl_threshold", 0.0
        ),
        "route_update_cap_trigger_count": losses.get("route_update_cap_trigger_count", 0),
        "route_update_cap_per_epoch": losses.get("route_update_cap_per_epoch", 0),
        "route_alignment_gate_threshold": losses.get("route_alignment_gate_threshold", 0.0),
        "route_alignment_gate_accept_count": losses.get("route_alignment_gate_accept_count", 0),
        "route_alignment_gate_reject_count": losses.get("route_alignment_gate_reject_count", 0),
        "route_alignment_gate_reject_rate": losses.get("route_alignment_gate_reject_rate", 0.0),
        "route_alignment_gate_score_mean": losses.get("route_alignment_gate_score_mean", 0.0),
        "route_step_alignment_mean": losses.get("route_step_alignment_mean", 0.0),
        "route_step_alignment_min": losses.get("route_step_alignment_min", 0.0),
        "route_step_alignment_max": losses.get("route_step_alignment_max", 0.0),
        "route_step_accept_count": losses.get("route_step_accept_count", 0),
        "route_step_reject_count": losses.get("route_step_reject_count", 0),
        "route_step_accept_rate": losses.get("route_step_accept_rate", 0.0),
        "route_step_reject_rate": losses.get("route_step_reject_rate", 0.0),
        "theta_update_count": losses["theta_update_count"],
        "route_update_count": losses["route_update_count"],
        "theta_only_step_kl": losses["theta_only_step_kl"],
        "route_only_step_kl": losses["route_only_step_kl"],
        "theta_only_step_prob_gain": losses["theta_only_step_prob_gain"],
        "route_only_step_prob_gain": losses["route_only_step_prob_gain"],
        "theta_after_route_shift": losses["theta_after_route_shift"],
        "route_after_theta_shift": losses["route_after_theta_shift"],
        "theta_route_feature_correlation": losses["theta_route_feature_correlation"],
        "theta_route_head_correlation": losses["theta_route_head_correlation"],
        "offload_rate_mean": losses["offload_rate_mean"],
        "bs1_vs_bs2_entropy": losses["bs1_vs_bs2_entropy"],
        "offload_vs_local_value_gap_mean": losses["offload_vs_local_value_gap_mean"],
        "offload_vs_local_value_gap_std": losses["offload_vs_local_value_gap_std"],
        "bs1_vs_bs2_value_gap_mean": losses["bs1_vs_bs2_value_gap_mean"],
        "bs1_vs_bs2_value_gap_std": losses["bs1_vs_bs2_value_gap_std"],
        "block_action_conditioned_value_now_mean": losses["block_action_conditioned_value_now_mean"],
        "block_action_conditioned_value_now_std": losses["block_action_conditioned_value_now_std"],
        "block_action_conditioned_value_next_mean": losses["block_action_conditioned_value_next_mean"],
        "block_action_conditioned_value_next_std": losses["block_action_conditioned_value_next_std"],
        "block_path_cost_local_mean": losses["block_path_cost_local_mean"],
        "block_path_cost_local_std": losses["block_path_cost_local_std"],
        "block_path_cost_bs1_mean": losses["block_path_cost_bs1_mean"],
        "block_path_cost_bs1_std": losses["block_path_cost_bs1_std"],
        "block_path_cost_bs2_mean": losses["block_path_cost_bs2_mean"],
        "block_path_cost_bs2_std": losses["block_path_cost_bs2_std"],
        "block_local_cost_now_mean": losses["block_local_cost_now_mean"],
        "block_local_cost_now_std": losses["block_local_cost_now_std"],
        "block_local_cost_next_mean": losses["block_local_cost_next_mean"],
        "block_local_cost_next_std": losses["block_local_cost_next_std"],
        "block_action_conditioned_cost_now_mean": losses["block_action_conditioned_cost_now_mean"],
        "block_action_conditioned_cost_now_std": losses["block_action_conditioned_cost_now_std"],
        "block_action_conditioned_cost_next_mean": losses["block_action_conditioned_cost_next_mean"],
        "block_action_conditioned_cost_next_std": losses["block_action_conditioned_cost_next_std"],
        "block_delta_cost_mean": losses["block_delta_cost_mean"],
        "block_delta_cost_std": losses["block_delta_cost_std"],
        "block_delta_cost_max": losses["block_delta_cost_max"],
        "block_delta_cost_min": losses["block_delta_cost_min"],
        "block_td_reward_mean": losses["block_td_reward_mean"],
        "block_td_reward_std": losses["block_td_reward_std"],
        "block_td_reward_max": losses["block_td_reward_max"],
        "block_td_reward_min": losses["block_td_reward_min"],
        "block_td_target_mean": losses["block_td_target_mean"],
        "block_td_target_std": losses["block_td_target_std"],
        "block_td_target_max": losses["block_td_target_max"],
        "block_td_target_min": losses["block_td_target_min"],
        "block_td_advantage_mean": losses["block_td_advantage_mean"],
        "block_td_advantage_std": losses["block_td_advantage_std"],
        "block_td_advantage_max": losses["block_td_advantage_max"],
        "block_td_advantage_min": losses["block_td_advantage_min"],
        "block_td_advantage_entropy": losses["block_td_advantage_entropy"],
        "top_k_block_td_advantage_share": losses["top_k_block_td_advantage_share"],
        "block_advantage_mean": losses["block_advantage_mean"],
        "block_advantage_std": losses["block_advantage_std"],
        "block_advantage_max": losses["block_advantage_max"],
        "block_advantage_min": losses["block_advantage_min"],
        "block_advantage_entropy": losses["block_advantage_entropy"],
        "top_k_block_advantage_share": losses["top_k_block_advantage_share"],
        "active_block_count_mean": losses["active_block_count_mean"],
        "block_ratio_proxy_mean": losses["block_ratio_proxy_mean"],
        "block_ratio_proxy_std": losses["block_ratio_proxy_std"],
        "block_clip_fraction_mean": losses["block_clip_fraction_mean"],
        "block_clip_fraction_std": losses["block_clip_fraction_std"],
        "block_positive_adv_clip_fraction_mean": losses[
            "block_positive_adv_clip_fraction_mean"
        ],
        "block_negative_adv_clip_fraction_mean": losses[
            "block_negative_adv_clip_fraction_mean"
        ],
        "block_ratio_mean": losses["block_ratio_mean"],
        "block_ratio_std": losses["block_ratio_std"],
        "block_ratio_max": losses["block_ratio_max"],
        "block_surrogate_mean": losses["block_surrogate_mean"],
        "block_surrogate_std": losses["block_surrogate_std"],
        "per_block_selected_action_prob_gain_mean": losses[
            "per_block_selected_action_prob_gain_mean"
        ],
        "per_block_selected_action_prob_gain_std": losses[
            "per_block_selected_action_prob_gain_std"
        ],
        "sum_to_mean_scale_ratio": losses["sum_to_mean_scale_ratio"],
        "sum_to_blockmean_scale_ratio": losses["sum_to_blockmean_scale_ratio"],
        "mean_to_blockmean_scale_ratio": losses["mean_to_blockmean_scale_ratio"],
        "running_return_mean": losses["running_return_mean"],
        "running_return_std": losses["running_return_std"],
        "popart_mean": losses["popart_mean"],
        "popart_std": losses["popart_std"],
        "actor_grad_norm": losses["actor_grad_norm"],
        "action_distribution_std": losses["action_distribution_std"],
        "action_prob_std": losses["action_prob_std"],
        "policy_std_mean": losses["policy_std_mean"],
        "policy_logit_std": losses["policy_logit_std"],
        "policy_confidence_mean": losses["policy_confidence_mean"],
        "selected_action_histogram": losses["selected_action_histogram"],
        "critic_raw_prediction_mean": losses["critic_raw_prediction_mean"],
        "critic_raw_prediction_std": losses["critic_raw_prediction_std"],
        "critic_normalized_prediction_mean": losses["critic_normalized_prediction_mean"],
        "critic_normalized_prediction_std": losses["critic_normalized_prediction_std"],
        "value_explained_variance": losses["value_explained_variance"],
        "prediction_target_corr": losses["prediction_target_corr"],
        "prediction_std_over_target_std": losses["prediction_std_over_target_std"],
        "constant_baseline_mse": losses["constant_baseline_mse"],
        "critic_prediction_mse": losses["critic_prediction_mse"],
        "critic_vs_constant_mse_gain": losses["critic_vs_constant_mse_gain"],
        "constant_baseline_huber": losses["constant_baseline_huber"],
        "critic_prediction_huber": losses["critic_prediction_huber"],
        "critic_vs_constant_huber_gain": losses["critic_vs_constant_huber_gain"],
        "critic_hidden_feature_mean": losses["critic_hidden_feature_mean"],
        "critic_hidden_feature_std": losses["critic_hidden_feature_std"],
        "critic_hidden_feature_dim_std_mean": losses["critic_hidden_feature_dim_std_mean"],
        "critic_hidden_feature_dim_std_min": losses["critic_hidden_feature_dim_std_min"],
        "critic_head_weight_norm": losses["critic_head_weight_norm"],
        "critic_head_bias_mean": losses["critic_head_bias_mean"],
        "critic_backbone_grad_norm": losses["critic_backbone_grad_norm"],
        "critic_head_grad_norm": losses["critic_head_grad_norm"],
        "actor_input_mean": losses["actor_input_mean"],
        "actor_input_std": losses["actor_input_std"],
        "actor_input_dim_std_mean": losses["actor_input_dim_std_mean"],
        "actor_input_dim_std_min": losses["actor_input_dim_std_min"],
        "actor_input_clip_fraction": losses["actor_input_clip_fraction"],
        "actor_input_derived_dim": losses["actor_input_derived_dim"],
        "actor_input_total_dim": losses["actor_input_total_dim"],
        "actor_derived_feature_mean": losses["actor_derived_feature_mean"],
        "actor_derived_feature_std": losses["actor_derived_feature_std"],
        "actor_raw_dim": losses["actor_raw_dim"],
        "actor_raw_pruned_dim_count": losses["actor_raw_pruned_dim_count"],
        "actor_raw_dim_std_mean": losses["actor_raw_dim_std_mean"],
        "actor_raw_dim_std_min": losses["actor_raw_dim_std_min"],
        "actor_raw_zero_var_dim_count": losses["actor_raw_zero_var_dim_count"],
        "actor_raw_static_zero_var_dim_count": losses["actor_raw_static_zero_var_dim_count"],
        "critic_input_mean": losses["critic_input_mean"],
        "critic_input_std": losses["critic_input_std"],
        "critic_input_dim_std_mean": losses["critic_input_dim_std_mean"],
        "critic_input_dim_std_min": losses["critic_input_dim_std_min"],
        "critic_input_clip_fraction": losses["critic_input_clip_fraction"],
        "critic_input_derived_dim": losses["critic_input_derived_dim"],
        "critic_input_total_dim": losses["critic_input_total_dim"],
        "critic_derived_feature_mean": losses["critic_derived_feature_mean"],
        "critic_derived_feature_std": losses["critic_derived_feature_std"],
        "prediction_target_payload_path": str(payload_path),
        "probe_policy_update_payload_path": str(probe_policy_payload_path),
    }
    log.update(probe_metrics)
    log.update(probe_policy_metrics)
    return (
        log,
        payload_path,
        advantage_bucket_rows,
        probe_policy_payload_path,
        block_logprob_rows,
        block_surrogate_rows,
        block_adv_scale_rows,
        block_value_scale_rows,
        block_td_value_rows,
        block_delta_cost_rows,
        block_advantage_rows,
        block_weight_rows,
        theta_route_split_rows,
    )


def train_with_diagnosis(
    config: MECConfig,
    checkpoint_dir: str,
    early_stopping: EarlyStoppingConfig,
    probe_states: np.ndarray,
    agent_kwargs: dict[str, Any] | None = None,
    agent_setup_hook: Any | None = None,
) -> tuple[PPOAgent, list[dict[str, float | int | str]], dict[str, object], Path]:
    simulator = Simulator(config)
    agent = PPOAgent(
        config=config.ppo,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        **(agent_kwargs or {}),
    )
    if agent_setup_hook is not None:
        agent_setup_hook(agent)
    logs: list[dict[str, float | int | str]] = []

    run_dir = Path(checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    best_reward = float("-inf")
    best_epoch = -1
    monitor_reference = float("-inf")
    epochs_without_significant_improvement = 0
    stopped_early = False
    stop_reason = "reached_max_epochs"
    final_payload_path: Path | None = None
    final_probe_policy_payload_path: Path | None = None
    all_advantage_bucket_rows: list[dict[str, float | int | str]] = []
    all_block_logprob_rows: list[dict[str, float | int | str]] = []
    all_block_surrogate_rows: list[dict[str, float | int | str]] = []
    all_block_adv_scale_rows: list[dict[str, float | int | str]] = []
    all_block_value_scale_rows: list[dict[str, float | int | str]] = []
    all_block_td_value_rows: list[dict[str, float | int | str]] = []
    all_block_delta_cost_rows: list[dict[str, float | int | str]] = []
    all_block_advantage_rows: list[dict[str, float | int | str]] = []
    all_block_weight_rows: list[dict[str, float | int | str]] = []
    all_theta_route_split_rows: list[dict[str, float | int | str]] = []
    probe_action_noise = np.random.default_rng(
        config.training.seed + config.ppo.policy_update_probe_noise_seed
    ).standard_normal(size=(len(probe_states), config.action_dim)).astype(np.float32)

    for epoch in range(config.training.num_epochs):
        (
            log,
            payload_path,
            advantage_bucket_rows,
            probe_policy_payload_path,
            block_logprob_rows,
            block_surrogate_rows,
            block_adv_scale_rows,
            block_value_scale_rows,
            block_td_value_rows,
            block_delta_cost_rows,
            block_advantage_rows,
            block_weight_rows,
            theta_route_split_rows,
        ) = run_training_epoch(
            agent=agent,
            simulator=simulator,
            config=config,
            epoch=epoch,
            probe_states=probe_states,
            probe_action_noise=probe_action_noise,
            run_dir=run_dir,
        )
        final_payload_path = payload_path
        final_probe_policy_payload_path = probe_policy_payload_path
        all_advantage_bucket_rows.extend(advantage_bucket_rows)
        all_block_logprob_rows.extend(block_logprob_rows)
        all_block_surrogate_rows.extend(block_surrogate_rows)
        all_block_adv_scale_rows.extend(block_adv_scale_rows)
        all_block_value_scale_rows.extend(block_value_scale_rows)
        all_block_td_value_rows.extend(block_td_value_rows)
        all_block_delta_cost_rows.extend(block_delta_cost_rows)
        all_block_advantage_rows.extend(block_advantage_rows)
        all_block_weight_rows.extend(block_weight_rows)
        all_theta_route_split_rows.extend(theta_route_split_rows)
        logs.append(log)
        episode_reward = float(log["episode_reward"])

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_epoch = epoch
            agent.save(str(run_dir / "best_model.pt"))
            (run_dir / "best_model_info.json").write_text(
                json.dumps(
                    {
                        "best_epoch": best_epoch,
                        "best_reward": best_reward,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        if epoch == early_stopping.monitor_start_epoch - 1:
            monitor_reference = best_reward
            epochs_without_significant_improvement = 0
        elif epoch >= early_stopping.monitor_start_epoch:
            if episode_reward >= monitor_reference + early_stopping.min_delta:
                monitor_reference = episode_reward
                epochs_without_significant_improvement = 0
            else:
                epochs_without_significant_improvement += 1

            if epochs_without_significant_improvement >= early_stopping.patience:
                stopped_early = True
                stop_reason = (
                    "no significant reward improvement "
                    f"for {early_stopping.patience} epochs after epoch "
                    f"{early_stopping.monitor_start_epoch}"
                )
                break

    if (run_dir / "best_model.pt").exists():
        agent.load(str(run_dir / "best_model.pt"), load_optimizer=False)

    save_logs(run_dir, logs)
    logs_df = pd.DataFrame(logs)
    if not logs_df.empty:
        advantage_alignment_columns = [
            "epoch",
            "advantage_mean",
            "advantage_std",
            "positive_advantage_ratio",
            "negative_advantage_ratio",
            "delta_log_prob_selected_action_mean",
            "delta_log_prob_selected_action_std",
            "selected_action_prob_gain_mean",
            "advantage_action_alignment",
            "theta_advantage_alignment",
            "route_advantage_alignment",
            "theta_adv_to_route_delta_alignment",
            "route_adv_to_theta_delta_alignment",
            "theta_alignment_margin_over_cross",
            "route_alignment_margin_over_cross",
            "cross_branch_alignment_mean",
            "theta_selected_action_prob_gain",
            "route_selected_action_prob_gain",
            "theta_negative_adv_prob_gain",
            "route_negative_adv_prob_gain",
            "theta_clip_fraction",
            "route_clip_fraction",
            "high_advantage_action_prob_gain",
            "mid_advantage_action_prob_gain",
            "low_advantage_action_prob_gain",
            "negative_advantage_action_prob_gain",
            "advantage_bucket_high_positive_threshold",
            "advantage_bucket_near_zero_threshold",
        ]
        ratio_clip_columns = [
            "epoch",
            "ratio_mean",
            "ratio_std",
            "ratio_min",
            "ratio_max",
            "clip_fraction",
            "positive_adv_clip_fraction",
            "negative_adv_clip_fraction",
            "approx_kl",
            "entropy_loss",
            "policy_loss",
            "ratio_by_adv_sign",
        ]
        logprob_scale_columns = [
            "epoch",
            "policy_surrogate_mode",
            "policy_ratio_mode",
            "actor_structure_mode",
            "action_dim",
            "action_block_count",
            "action_block_slices",
            "delta_log_prob_selected_action_mean",
            "delta_log_prob_selected_action_std",
            "logprob_delta_sum_mean",
            "logprob_delta_sum_std",
            "logprob_delta_mean_mean",
            "logprob_delta_mean_std",
            "block_logprob_delta_mean",
            "block_logprob_delta_std",
            "block_weight_entropy",
            "block_weight_max_mean",
            "top_k_block_weight_share",
            "block_adv_scale_mean",
            "block_adv_scale_std",
            "block_adv_scale_max",
            "block_adv_scale_min",
            "block_adv_scale_entropy",
            "top_k_block_adv_scale_share",
            "block_value_score_mean",
            "block_value_score_std",
            "block_value_score_max",
            "block_value_score_min",
            "block_value_scale_mean",
            "block_value_scale_std",
            "block_value_scale_max",
            "block_value_scale_min",
            "block_value_scale_entropy",
            "top_k_block_value_scale_share",
            "block_value_pred_mean",
            "block_value_pred_std",
            "block_value_pred_max",
            "block_value_pred_min",
            "block_td_target_mean",
            "block_td_target_std",
            "block_td_target_max",
            "block_td_target_min",
            "block_td_advantage_mean",
            "block_td_advantage_std",
            "block_td_advantage_max",
            "block_td_advantage_min",
            "block_td_advantage_entropy",
            "top_k_block_td_advantage_share",
            "block_advantage_mean",
            "block_advantage_std",
            "block_advantage_max",
            "block_advantage_min",
            "block_advantage_entropy",
            "top_k_block_advantage_share",
            "active_block_count_mean",
            "block_ratio_proxy_mean",
            "block_ratio_proxy_std",
            "block_clip_fraction_mean",
            "block_clip_fraction_std",
            "block_positive_adv_clip_fraction_mean",
            "block_negative_adv_clip_fraction_mean",
            "block_ratio_mean",
            "block_ratio_std",
            "block_ratio_max",
            "block_surrogate_mean",
            "block_surrogate_std",
            "per_block_selected_action_prob_gain_mean",
            "per_block_selected_action_prob_gain_std",
            "sum_to_mean_scale_ratio",
            "sum_to_blockmean_scale_ratio",
            "mean_to_blockmean_scale_ratio",
        ]
        logs_df[advantage_alignment_columns].to_csv(
            run_dir / "advantage_alignment_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
        logs_df[
            [
                "epoch",
                "theta_advantage_mean",
                "theta_advantage_std",
                "theta_advantage_max",
                "theta_advantage_min",
                "route_advantage_mean",
                "route_advantage_std",
                "route_advantage_max",
                "route_advantage_min",
                "theta_adv_norm_mean",
                "theta_adv_norm_std",
                "route_adv_norm_mean",
                "route_adv_norm_std",
                "route_adv_norm_active_count",
                "route_adv_norm_used_fallback",
                "theta_advantage_alignment",
                "route_advantage_alignment",
                "theta_adv_to_route_delta_alignment",
                "route_adv_to_theta_delta_alignment",
                "theta_alignment_margin_over_cross",
                "route_alignment_margin_over_cross",
                "cross_branch_alignment_mean",
                "theta_selected_action_prob_gain",
                "route_selected_action_prob_gain",
                "theta_negative_adv_prob_gain",
                "route_negative_adv_prob_gain",
                "theta_clip_fraction",
                "route_clip_fraction",
                "theta_ratio_mean",
                "theta_ratio_std",
                "theta_ratio_max",
                "route_ratio_mean",
                "route_ratio_std",
                "route_ratio_max",
                "theta_approx_kl",
                "route_approx_kl",
                "theta_loss_mean",
                "route_loss_mean",
                "theta_route_loss_ratio",
                "theta_head_grad_norm",
                "route_head_grad_norm",
                "theta_backbone_grad_norm",
                "route_backbone_grad_norm",
                "theta_kl_target",
                "route_kl_target",
                "theta_early_stop_count",
                "route_early_stop_count",
                "coupled_stop_trigger_count",
                "coupled_stop_blocked_by_theta_floor_count",
                "coupled_stop_blocked_by_severity_gate_count",
                "coupled_stop_min_theta_updates_per_epoch",
                "coupled_stop_severity_factor",
                "coupled_stop_severity_freeze_kl_threshold",
                "theta_update_count",
                "route_update_count",
                "theta_only_step_kl",
                "route_only_step_kl",
                "theta_only_step_prob_gain",
                "route_only_step_prob_gain",
                "theta_after_route_shift",
                "route_after_theta_shift",
                "theta_route_feature_correlation",
                "theta_route_head_correlation",
                "offload_rate_mean",
                "bs1_vs_bs2_entropy",
                "offload_vs_local_value_gap_mean",
                "offload_vs_local_value_gap_std",
                "bs1_vs_bs2_value_gap_mean",
                "bs1_vs_bs2_value_gap_std",
            ]
        ].to_csv(
            run_dir / "theta_route_alignment_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
        logs_df[ratio_clip_columns].to_csv(
            run_dir / "ratio_clip_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
        logs_df[logprob_scale_columns].to_csv(
            run_dir / "logprob_scale_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    block_logprob_df = pd.DataFrame(all_block_logprob_rows)
    if not block_logprob_df.empty:
        block_logprob_df.to_csv(
            run_dir / "block_logprob_scale_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    block_surrogate_df = pd.DataFrame(all_block_surrogate_rows)
    if not block_surrogate_df.empty:
        block_surrogate_df.to_csv(
            run_dir / "block_surrogate_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
        block_surrogate_df.to_csv(
            run_dir / "per_block_update_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    block_adv_scale_df = pd.DataFrame(all_block_adv_scale_rows)
    if not block_adv_scale_df.empty:
        block_adv_scale_df.to_csv(
            run_dir / "block_adv_scale_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    block_value_scale_df = pd.DataFrame(all_block_value_scale_rows)
    if not block_value_scale_df.empty:
        block_value_scale_df.to_csv(
            run_dir / "block_value_scale_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    block_td_value_df = pd.DataFrame(all_block_td_value_rows)
    if not block_td_value_df.empty:
        block_td_value_df.to_csv(
            run_dir / "block_td_value_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
        block_td_value_df.to_csv(
            run_dir / "block_path_value_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
        block_td_value_df.to_csv(
            run_dir / "path_td_target_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    block_delta_cost_df = pd.DataFrame(all_block_delta_cost_rows)
    if not block_delta_cost_df.empty:
        block_delta_cost_df.to_csv(
            run_dir / "block_delta_cost_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
        block_delta_cost_df.to_csv(
            run_dir / "block_action_conditioned_cost_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    block_advantage_df = pd.DataFrame(all_block_advantage_rows)
    if not block_advantage_df.empty:
        block_advantage_df.to_csv(
            run_dir / "block_advantage_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    block_weight_df = pd.DataFrame(all_block_weight_rows)
    if not block_weight_df.empty:
        block_weight_df.to_csv(
            run_dir / "block_weight_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    theta_route_split_df = pd.DataFrame(all_theta_route_split_rows)
    if not theta_route_split_df.empty:
        theta_route_split_df.to_csv(
            run_dir / "theta_route_split_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    route_gate_columns = [
        "epoch",
        "policy_surrogate_mode",
        "actor_structure_mode",
        "route_gate_mean",
        "route_gate_std",
        "route_gate_min",
        "route_gate_max",
        "route_gate_active_fraction",
        "offload_rate_mean",
        "theta_head_grad_norm",
        "route_head_grad_norm",
        "theta_backbone_grad_norm",
        "route_backbone_grad_norm",
    ]
    available_route_gate_columns = [
        column for column in route_gate_columns if logs and column in logs[0]
    ]
    if available_route_gate_columns:
        pd.DataFrame(logs)[available_route_gate_columns].to_csv(
            run_dir / "route_gate_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    route_mask_columns = [
        "epoch",
        "policy_surrogate_mode",
        "actor_structure_mode",
        "route_mask_mean",
        "route_mask_active_fraction",
        "route_mask_count_mean",
        "route_mask_count_std",
        "offload_rate_mean",
        "theta_head_grad_norm",
        "route_head_grad_norm",
        "theta_backbone_grad_norm",
        "route_backbone_grad_norm",
    ]
    available_route_mask_columns = [
        column for column in route_mask_columns if logs and column in logs[0]
    ]
    if available_route_mask_columns:
        pd.DataFrame(logs)[available_route_mask_columns].to_csv(
            run_dir / "route_mask_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    branchwise_balanced_pg_columns = [
        "epoch",
        "policy_surrogate_mode",
        "actor_structure_mode",
        "theta_adv_norm_mean",
        "theta_adv_norm_std",
        "route_adv_norm_mean",
        "route_adv_norm_std",
        "route_adv_norm_active_count",
        "route_adv_norm_used_fallback",
        "theta_loss_mean",
        "route_loss_mean",
        "theta_route_loss_ratio",
        "theta_clip_fraction",
        "route_clip_fraction",
        "route_mask_mean",
        "route_mask_active_fraction",
    ]
    available_branchwise_balanced_pg_columns = [
        column for column in branchwise_balanced_pg_columns if logs and column in logs[0]
    ]
    if available_branchwise_balanced_pg_columns:
        pd.DataFrame(logs)[available_branchwise_balanced_pg_columns].to_csv(
            run_dir / "branchwise_balanced_pg_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    alternating_branch_pg_columns = [
        "epoch",
        "policy_surrogate_mode",
        "actor_structure_mode",
        "theta_only_step_kl",
        "route_only_step_kl",
        "theta_only_step_prob_gain",
        "route_only_step_prob_gain",
        "theta_after_route_shift",
        "route_after_theta_shift",
        "theta_head_grad_norm",
        "route_head_grad_norm",
        "theta_backbone_grad_norm",
        "route_backbone_grad_norm",
        "theta_kl_target",
        "route_kl_target",
        "theta_early_stop_count",
        "route_early_stop_count",
        "coupled_stop_trigger_count",
        "coupled_stop_blocked_by_theta_floor_count",
        "coupled_stop_blocked_by_severity_gate_count",
        "coupled_stop_min_theta_updates_per_epoch",
        "coupled_stop_severity_factor",
        "coupled_stop_severity_freeze_kl_threshold",
        "route_update_cap_trigger_count",
        "route_update_cap_per_epoch",
        "route_alignment_gate_threshold",
        "route_alignment_gate_accept_count",
        "route_alignment_gate_reject_count",
        "route_alignment_gate_reject_rate",
        "route_alignment_gate_score_mean",
        "route_step_alignment_mean",
        "route_step_alignment_min",
        "route_step_alignment_max",
        "route_step_accept_count",
        "route_step_reject_count",
        "route_step_accept_rate",
        "route_step_reject_rate",
        "theta_update_count",
        "route_update_count",
        "route_mask_mean",
        "route_mask_active_fraction",
        "route_mask_threshold",
        "route_confident_mask_margin",
    ]
    available_alternating_branch_pg_columns = [
        column for column in alternating_branch_pg_columns if logs and column in logs[0]
    ]
    if available_alternating_branch_pg_columns:
        pd.DataFrame(logs)[available_alternating_branch_pg_columns].to_csv(
            run_dir / "alternating_branch_pg_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    factorized_ratio_pg_columns = [
        "epoch",
        "policy_surrogate_mode",
        "actor_structure_mode",
        "theta_ratio_mean",
        "theta_ratio_std",
        "theta_ratio_max",
        "route_ratio_mean",
        "route_ratio_std",
        "route_ratio_max",
        "offload_active_fraction",
        "route_logprob_active_fraction",
        "theta_positive_fraction",
        "route_credit_gate_fraction",
        "route_credit_effective_fraction",
        "route_credit_effective_count",
        "route_credit_weight_mean",
        "route_credit_weight_std",
        "route_credit_weight_min",
        "route_credit_weight_max",
        "theta_candidate_score_mean",
        "theta_candidate_score_std",
        "theta_selected_score_mean",
        "theta_expected_score_mean",
        "theta_residual_credit_mean",
        "theta_residual_credit_std",
        "theta_residual_credit_min",
        "theta_residual_credit_max",
        "route_candidate_score_mean",
        "route_candidate_score_std",
        "route_residual_credit_mean",
        "route_residual_credit_std",
        "route_residual_credit_min",
        "route_residual_credit_max",
        "route_expected_score_mean",
        "route_selected_score_mean",
        "route_decision_agreement_ratio",
        "actual_bs1_rate_when_route_true_gap_positive",
        "actual_bs1_rate_when_route_true_gap_negative",
        "route_score_vector_mean_abs",
        "route_score_vector_std",
        "route_credit_fallback_trigger_count",
        "route_credit_fallback_rate",
        "theta_old_logprob_mean",
        "theta_new_logprob_mean",
        "route_old_logprob_mean",
        "route_new_logprob_mean",
        "joint_old_logprob_mean",
        "joint_new_logprob_mean",
        "joint_ratio_mean",
        "joint_approx_kl",
        "joint_clip_fraction",
        "joint_selected_score_mean",
        "joint_expected_score_mean",
        "joint_residual_credit_mean",
        "joint_residual_credit_std",
        "joint_residual_credit_min",
        "joint_residual_credit_max",
        "joint_action_decision_agreement_ratio_under_reward_aligned",
        "actual_local_rate_when_reward_aligned_best_is_local",
        "actual_bs1_rate_when_reward_aligned_best_is_bs1",
        "actual_bs2_rate_when_reward_aligned_best_is_bs2",
        "offload_decision_agreement_ratio",
        "actual_offload_rate_when_theta_true_gap_positive",
        "actual_offload_rate_when_theta_true_gap_negative",
        "theta_approx_kl",
        "route_approx_kl",
        "theta_clip_fraction",
        "route_clip_fraction",
        "conditional_policy_consistency_score",
        "theta_only_step_kl",
        "route_only_step_kl",
        "theta_kl_target",
        "route_kl_target",
        "theta_early_stop_count",
        "route_early_stop_count",
        "coupled_stop_trigger_count",
        "coupled_stop_blocked_by_theta_floor_count",
        "coupled_stop_blocked_by_severity_gate_count",
        "coupled_stop_min_theta_updates_per_epoch",
        "coupled_stop_severity_factor",
        "coupled_stop_severity_freeze_kl_threshold",
        "route_update_cap_trigger_count",
        "route_update_cap_per_epoch",
        "route_alignment_gate_threshold",
        "route_alignment_gate_accept_count",
        "route_alignment_gate_reject_count",
        "route_alignment_gate_reject_rate",
        "route_alignment_gate_score_mean",
        "route_step_alignment_mean",
        "route_step_alignment_min",
        "route_step_alignment_max",
        "route_step_accept_count",
        "route_step_reject_count",
        "route_step_accept_rate",
        "route_step_reject_rate",
        "theta_update_count",
        "route_update_count",
        "route_mask_mean",
        "route_mask_active_fraction",
        "route_mask_threshold",
        "route_confident_mask_margin",
    ]
    available_factorized_ratio_pg_columns = [
        column for column in factorized_ratio_pg_columns if logs and column in logs[0]
    ]
    if available_factorized_ratio_pg_columns:
        pd.DataFrame(logs)[available_factorized_ratio_pg_columns].to_csv(
            run_dir / "factorized_ratio_pg_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(logs)[available_factorized_ratio_pg_columns].to_csv(
            run_dir / "factorized_trust_region_pg_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
        conditional_policy_columns = [
            "epoch",
            "offload_active_fraction",
            "route_logprob_active_fraction",
            "theta_positive_fraction",
            "route_credit_gate_fraction",
            "route_credit_effective_fraction",
            "route_credit_effective_count",
            "route_credit_fallback_trigger_count",
            "route_credit_weight_mean",
            "route_credit_weight_std",
            "route_credit_weight_min",
            "route_credit_weight_max",
            "theta_selected_score_mean",
            "theta_expected_score_mean",
            "theta_residual_credit_mean",
            "theta_residual_credit_std",
            "theta_candidate_score_mean",
            "theta_candidate_score_std",
            "route_residual_credit_mean",
            "route_residual_credit_std",
            "route_candidate_score_mean",
            "route_candidate_score_std",
            "route_expected_score_mean",
            "route_selected_score_mean",
            "theta_ratio_mean",
            "route_ratio_mean",
            "theta_approx_kl",
            "route_approx_kl",
            "theta_clip_fraction",
            "route_clip_fraction",
            "advantage_action_alignment",
            "offload_decision_agreement_ratio",
            "actual_offload_rate_when_theta_true_gap_positive",
            "actual_offload_rate_when_theta_true_gap_negative",
            "route_decision_agreement_ratio",
            "actual_bs1_rate_when_route_true_gap_positive",
            "actual_bs1_rate_when_route_true_gap_negative",
            "conditional_policy_consistency_score",
        ]
        available_conditional_policy_columns = [
            column for column in conditional_policy_columns if logs and column in logs[0]
        ]
        if available_conditional_policy_columns:
            pd.DataFrame(logs)[available_conditional_policy_columns].to_csv(
                run_dir / "conditional_policy_stats_by_epoch.csv",
                index=False,
                encoding="utf-8-sig",
            )
        route_credit_gate_columns = [
            "epoch",
            "offload_active_fraction",
            "theta_positive_fraction",
            "route_credit_weight_mean",
            "route_credit_weight_std",
            "route_credit_effective_fraction",
            "route_credit_effective_count",
            "theta_ratio_mean",
            "route_ratio_mean",
            "theta_approx_kl",
            "route_approx_kl",
            "theta_clip_fraction",
            "route_clip_fraction",
            "advantage_action_alignment",
        ]
        available_route_credit_gate_columns = [
            column for column in route_credit_gate_columns if logs and column in logs[0]
        ]
        if available_route_credit_gate_columns:
            pd.DataFrame(logs)[available_route_credit_gate_columns].to_csv(
                run_dir / "route_credit_theta_gate_stats_by_epoch.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(logs)[available_route_credit_gate_columns].to_csv(
                run_dir / "route_credit_theta_soft_weight_stats_by_epoch.csv",
                index=False,
                encoding="utf-8-sig",
            )
        route_residual_credit_columns = [
            "epoch",
            "offload_active_fraction",
            "route_logprob_active_fraction",
            "route_residual_credit_mean",
            "route_residual_credit_std",
            "route_expected_score_mean",
            "route_selected_score_mean",
            "route_ratio_mean",
            "route_approx_kl",
            "route_clip_fraction",
            "advantage_action_alignment",
        ]
        available_route_residual_credit_columns = [
            column for column in route_residual_credit_columns if logs and column in logs[0]
        ]
        if available_route_residual_credit_columns:
            pd.DataFrame(logs)[available_route_residual_credit_columns].to_csv(
                run_dir / "route_residual_credit_stats_by_epoch.csv",
                index=False,
                encoding="utf-8-sig",
            )
        route_candidate_score_credit_columns = [
            "epoch",
            "offload_active_fraction",
            "route_logprob_active_fraction",
            "route_selected_score_mean",
            "route_expected_score_mean",
            "route_residual_credit_mean",
            "route_residual_credit_std",
            "route_ratio_mean",
            "route_approx_kl",
            "route_clip_fraction",
            "advantage_action_alignment",
        ]
        available_route_candidate_score_credit_columns = [
            column
            for column in route_candidate_score_credit_columns
            if logs and column in logs[0]
        ]
        if available_route_candidate_score_credit_columns:
            pd.DataFrame(logs)[available_route_candidate_score_credit_columns].to_csv(
                run_dir / "route_candidate_score_credit_stats_by_epoch.csv",
                index=False,
                encoding="utf-8-sig",
            )
        theta_candidate_score_credit_columns = [
            "epoch",
            "theta_selected_score_mean",
            "theta_expected_score_mean",
            "theta_residual_credit_mean",
            "theta_residual_credit_std",
            "theta_ratio_mean",
            "theta_approx_kl",
            "theta_clip_fraction",
            "advantage_action_alignment",
            "offload_decision_agreement_ratio",
            "actual_offload_rate_when_theta_true_gap_positive",
            "actual_offload_rate_when_theta_true_gap_negative",
        ]
        available_theta_candidate_score_credit_columns = [
            column
            for column in theta_candidate_score_credit_columns
            if logs and column in logs[0]
        ]
        if available_theta_candidate_score_credit_columns:
            pd.DataFrame(logs)[available_theta_candidate_score_credit_columns].to_csv(
                run_dir / "theta_candidate_score_credit_stats_by_epoch.csv",
                index=False,
                encoding="utf-8-sig",
            )
        theta_route_candidate_score_credit_columns = [
            "epoch",
            "advantage_action_alignment",
            "theta_advantage_alignment",
            "route_advantage_alignment",
            "theta_approx_kl",
            "route_approx_kl",
            "theta_clip_fraction",
            "route_clip_fraction",
            "offload_decision_agreement_ratio",
            "route_decision_agreement_ratio",
            "theta_residual_credit_mean",
            "route_residual_credit_mean",
        ]
        available_theta_route_candidate_score_credit_columns = [
            column
            for column in theta_route_candidate_score_credit_columns
            if logs and column in logs[0]
        ]
        if available_theta_route_candidate_score_credit_columns:
            pd.DataFrame(logs)[available_theta_route_candidate_score_credit_columns].to_csv(
                run_dir / "theta_route_candidate_score_credit_stats_by_epoch.csv",
                index=False,
                encoding="utf-8-sig",
            )
        joint_reward_aligned_credit_columns = [
            "epoch",
            "advantage_action_alignment",
            "joint_approx_kl",
            "joint_clip_fraction",
            "joint_selected_score_mean",
            "joint_expected_score_mean",
            "joint_residual_credit_mean",
            "joint_residual_credit_std",
            "joint_action_decision_agreement_ratio_under_reward_aligned",
            "best_reward_so_far",
        ]
        available_joint_reward_aligned_credit_columns = [
            column
            for column in joint_reward_aligned_credit_columns
            if logs and column in logs[0]
        ]
        if available_joint_reward_aligned_credit_columns:
            pd.DataFrame(logs)[available_joint_reward_aligned_credit_columns].to_csv(
                run_dir / "joint_reward_aligned_credit_stats_by_epoch.csv",
                index=False,
                encoding="utf-8-sig",
            )
        joint_td_aligned_credit_columns = [
            "epoch",
            "advantage_action_alignment",
            "joint_approx_kl",
            "joint_clip_fraction",
            "joint_selected_score_mean",
            "joint_expected_score_mean",
            "joint_residual_credit_mean",
            "joint_residual_credit_std",
            "joint_action_decision_agreement_ratio_under_td_aligned",
            "best_reward_so_far",
        ]
        available_joint_td_aligned_credit_columns = [
            column
            for column in joint_td_aligned_credit_columns
            if logs and column in logs[0]
        ]
        if available_joint_td_aligned_credit_columns:
            pd.DataFrame(logs)[available_joint_td_aligned_credit_columns].to_csv(
                run_dir / "joint_td_aligned_credit_stats_by_epoch.csv",
                index=False,
                encoding="utf-8-sig",
            )
    separate_backbone_columns = [
        "epoch",
        "policy_surrogate_mode",
        "actor_structure_mode",
        "theta_backbone_grad_norm",
        "route_backbone_grad_norm",
        "theta_route_feature_correlation",
        "theta_head_grad_norm",
        "route_head_grad_norm",
        "theta_route_head_correlation",
        "route_mask_mean",
        "route_mask_active_fraction",
        "offload_rate_mean",
    ]
    available_separate_backbone_columns = [
        column for column in separate_backbone_columns if logs and column in logs[0]
    ]
    if available_separate_backbone_columns:
        pd.DataFrame(logs)[available_separate_backbone_columns].to_csv(
            run_dir / "separate_backbone_stats_by_epoch.csv",
            index=False,
            encoding="utf-8-sig",
        )
    bucket_df = pd.DataFrame(all_advantage_bucket_rows)
    if not bucket_df.empty:
        bucket_df.to_csv(
            run_dir / "advantage_bucket_update_stats.csv",
            index=False,
            encoding="utf-8-sig",
        )
    final_epoch = len(logs) - 1 if logs else -1
    final_reward = float(logs[-1]["episode_reward"]) if logs else None
    summary: dict[str, object] = {
        "critic_arch_mode": config.ppo.critic_arch_mode,
        "actor_input_mode": config.ppo.actor_input_mode,
        "actor_raw_input_mode": config.ppo.actor_raw_input_mode,
        "actor_structure_mode": config.ppo.actor_structure_mode,
        "critic_input_mode": config.ppo.critic_input_mode,
        "policy_surrogate_mode": config.ppo.policy_ratio_mode,
        "policy_ratio_mode": config.ppo.policy_ratio_mode,
        "theta_kl_target": config.ppo.theta_kl_target,
        "route_kl_target": config.ppo.route_kl_target,
        "reward_mode": config.system.reward_mode,
        "value_target_mode": config.ppo.value_target_mode,
        "value_loss_mode": config.ppo.value_loss_mode,
        "topology": FIXED_TOPOLOGY,
        "action_dim": int(config.action_dim),
        "action_block_count": int(logs[-1]["action_block_count"]) if logs else 0,
        "action_block_slices": str(logs[-1]["action_block_slices"]) if logs else "[]",
        "max_epochs": config.training.num_epochs,
        "epochs_completed": len(logs),
        "best_epoch": best_epoch,
        "best_reward": best_reward,
        "final_epoch": final_epoch,
        "final_reward": final_reward,
        "reward_gap": (final_reward - best_reward) if final_reward is not None else None,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "monitor_start_epoch": early_stopping.monitor_start_epoch,
        "early_stopping_patience": early_stopping.patience,
        "min_delta": early_stopping.min_delta,
        "restored_best_model": True,
        "probe_state_count": int(len(probe_states)),
        "final_prediction_target_payload": str(final_payload_path) if final_payload_path else None,
        "final_probe_policy_update_payload": (
            str(final_probe_policy_payload_path) if final_probe_policy_payload_path else None
        ),
    }
    (run_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return agent, logs, summary, final_payload_path if final_payload_path is not None else run_dir


def extract_metrics(run_dir: Path) -> dict[str, float | int | str]:
    summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    best_model_info = json.loads((run_dir / "best_model_info.json").read_text(encoding="utf-8"))
    logs = pd.read_csv(run_dir / "train_logs.csv")

    best_reward = float(summary["best_reward"])
    final_reward = float(summary["final_reward"])
    reward_gap = final_reward - best_reward
    best_epoch = int(best_model_info["best_epoch"])
    final_row = logs.iloc[-1]

    def final_float(column: str, default: float = 0.0) -> float:
        return float(final_row[column]) if column in final_row.index else float(default)

    def final_int(column: str, default: int = 0) -> int:
        return int(final_row[column]) if column in final_row.index else int(default)

    def final_str(column: str, default: str = "") -> str:
        return str(final_row[column]) if column in final_row.index else default

    metrics: dict[str, float | int | str] = {
        "critic_arch_mode": str(summary.get("critic_arch_mode", "baseline_critic")),
        "actor_input_mode": str(summary.get("actor_input_mode", "current_actor_input")),
        "actor_raw_input_mode": str(summary.get("actor_raw_input_mode", "baseline_actor_raw")),
        "actor_structure_mode": str(
            summary.get("actor_structure_mode", final_str("actor_structure_mode", "flat_joint_actor"))
        ),
        "critic_input_mode": str(summary.get("critic_input_mode", "current_stronger_input")),
        "policy_surrogate_mode": str(
            summary.get(
                "policy_surrogate_mode",
                summary.get("policy_ratio_mode", "current_joint_sum_ratio"),
            )
        ),
        "policy_ratio_mode": str(summary.get("policy_ratio_mode", "current_joint_sum_ratio")),
        "run_dir": str(run_dir),
        "action_dim": int(summary.get("action_dim", final_int("action_dim"))),
        "action_block_count": int(summary.get("action_block_count", final_int("action_block_count", 1))),
        "action_block_slices": str(
            summary.get("action_block_slices", final_str("action_block_slices", "[]"))
        ),
        "epochs_completed": int(summary["epochs_completed"]),
        "best_reward": best_reward,
        "final_reward": final_reward,
        "best_epoch": best_epoch,
        "reward_gap": reward_gap,
        "reward_gap_abs": abs(reward_gap),
        "final_actor_loss": final_float("actor_loss"),
        "final_critic_loss": final_float("critic_loss"),
        "final_policy_entropy": final_float("policy_entropy", final_float("entropy")),
        "final_advantage_mean": final_float("advantage_mean"),
        "final_advantage_std": final_float("advantage_std"),
        "final_normalized_advantage_mean": final_float("normalized_advantage_mean"),
        "final_normalized_advantage_std": final_float("normalized_advantage_std"),
        "final_positive_advantage_ratio": final_float("positive_advantage_ratio"),
        "final_negative_advantage_ratio": final_float("negative_advantage_ratio"),
        "final_delta_log_prob_selected_action_mean": final_float(
            "delta_log_prob_selected_action_mean"
        ),
        "final_delta_log_prob_selected_action_std": final_float(
            "delta_log_prob_selected_action_std"
        ),
        "final_selected_action_prob_gain_mean": final_float("selected_action_prob_gain_mean"),
        "final_advantage_action_alignment": final_float("advantage_action_alignment"),
        "final_high_advantage_action_prob_gain": final_float(
            "high_advantage_action_prob_gain"
        ),
        "final_mid_advantage_action_prob_gain": final_float("mid_advantage_action_prob_gain"),
        "final_low_advantage_action_prob_gain": final_float("low_advantage_action_prob_gain"),
        "final_negative_advantage_action_prob_gain": final_float(
            "negative_advantage_action_prob_gain"
        ),
        "final_advantage_bucket_high_positive_threshold": final_float(
            "advantage_bucket_high_positive_threshold"
        ),
        "final_advantage_bucket_near_zero_threshold": final_float(
            "advantage_bucket_near_zero_threshold"
        ),
        "final_ratio_mean": final_float("ratio_mean", 1.0),
        "final_ratio_std": final_float("ratio_std"),
        "final_ratio_min": final_float("ratio_min", 1.0),
        "final_ratio_max": final_float("ratio_max", 1.0),
        "final_clip_fraction": final_float("clip_fraction"),
        "final_positive_adv_clip_fraction": final_float("positive_adv_clip_fraction"),
        "final_negative_adv_clip_fraction": final_float("negative_adv_clip_fraction"),
        "final_approx_kl": final_float("approx_kl"),
        "final_entropy_loss": final_float("entropy_loss"),
        "final_policy_loss": final_float("policy_loss"),
        "final_ratio_by_adv_sign": final_str("ratio_by_adv_sign"),
        "final_logprob_delta_sum_mean": final_float("logprob_delta_sum_mean"),
        "final_logprob_delta_sum_std": final_float("logprob_delta_sum_std"),
        "final_logprob_delta_mean_mean": final_float("logprob_delta_mean_mean"),
        "final_logprob_delta_mean_std": final_float("logprob_delta_mean_std"),
        "final_block_logprob_delta_mean": final_float("block_logprob_delta_mean"),
        "final_block_logprob_delta_std": final_float("block_logprob_delta_std"),
        "final_block_weight_entropy": final_float("block_weight_entropy"),
        "final_block_weight_max_mean": final_float("block_weight_max_mean"),
        "final_top_k_block_weight_share": final_float("top_k_block_weight_share"),
        "final_block_adv_scale_mean": final_float("block_adv_scale_mean", 1.0),
        "final_block_adv_scale_std": final_float("block_adv_scale_std"),
        "final_block_adv_scale_max": final_float("block_adv_scale_max", 1.0),
        "final_block_adv_scale_min": final_float("block_adv_scale_min", 1.0),
        "final_block_adv_scale_entropy": final_float("block_adv_scale_entropy"),
        "final_top_k_block_adv_scale_share": final_float("top_k_block_adv_scale_share"),
        "final_block_value_score_mean": final_float("block_value_score_mean"),
        "final_block_value_score_std": final_float("block_value_score_std"),
        "final_block_value_score_max": final_float("block_value_score_max"),
        "final_block_value_score_min": final_float("block_value_score_min"),
        "final_block_value_scale_mean": final_float("block_value_scale_mean", 1.0),
        "final_block_value_scale_std": final_float("block_value_scale_std"),
        "final_block_value_scale_max": final_float("block_value_scale_max", 1.0),
        "final_block_value_scale_min": final_float("block_value_scale_min", 1.0),
        "final_block_value_scale_entropy": final_float("block_value_scale_entropy"),
        "final_top_k_block_value_scale_share": final_float(
            "top_k_block_value_scale_share"
        ),
        "final_block_value_pred_mean": final_float("block_value_pred_mean"),
        "final_block_value_pred_std": final_float("block_value_pred_std"),
        "final_block_value_pred_max": final_float("block_value_pred_max"),
        "final_block_value_pred_min": final_float("block_value_pred_min"),
        "final_block_path_value_local_mean": final_float("block_path_value_local_mean"),
        "final_block_path_value_local_std": final_float("block_path_value_local_std"),
        "final_block_path_value_bs1_mean": final_float("block_path_value_bs1_mean"),
        "final_block_path_value_bs1_std": final_float("block_path_value_bs1_std"),
        "final_block_path_value_bs2_mean": final_float("block_path_value_bs2_mean"),
        "final_block_path_value_bs2_std": final_float("block_path_value_bs2_std"),
        "final_path_value_local_mean": final_float("path_value_local_mean"),
        "final_path_value_local_std": final_float("path_value_local_std"),
        "final_path_value_bs1_mean": final_float("path_value_bs1_mean"),
        "final_path_value_bs1_std": final_float("path_value_bs1_std"),
        "final_path_value_bs2_mean": final_float("path_value_bs2_mean"),
        "final_path_value_bs2_std": final_float("path_value_bs2_std"),
        "final_path_td_target_local_mean": final_float("path_td_target_local_mean"),
        "final_path_td_target_local_std": final_float("path_td_target_local_std"),
        "final_path_td_target_bs1_mean": final_float("path_td_target_bs1_mean"),
        "final_path_td_target_bs1_std": final_float("path_td_target_bs1_std"),
        "final_path_td_target_bs2_mean": final_float("path_td_target_bs2_mean"),
        "final_path_td_target_bs2_std": final_float("path_td_target_bs2_std"),
        "final_theta_advantage_mean": final_float("theta_advantage_mean"),
        "final_theta_advantage_std": final_float("theta_advantage_std"),
        "final_theta_advantage_max": final_float("theta_advantage_max"),
        "final_theta_advantage_min": final_float("theta_advantage_min"),
        "final_route_advantage_mean": final_float("route_advantage_mean"),
        "final_route_advantage_std": final_float("route_advantage_std"),
        "final_route_advantage_max": final_float("route_advantage_max"),
        "final_route_advantage_min": final_float("route_advantage_min"),
        "final_theta_adv_norm_mean": final_float("theta_adv_norm_mean"),
        "final_theta_adv_norm_std": final_float("theta_adv_norm_std"),
        "final_route_adv_norm_mean": final_float("route_adv_norm_mean"),
        "final_route_adv_norm_std": final_float("route_adv_norm_std"),
        "final_route_adv_norm_active_count": final_float("route_adv_norm_active_count"),
        "final_route_adv_norm_used_fallback": final_float("route_adv_norm_used_fallback"),
        "final_theta_advantage_alignment": final_float("theta_advantage_alignment"),
        "final_route_advantage_alignment": final_float("route_advantage_alignment"),
        "final_theta_adv_to_route_delta_alignment": final_float(
            "theta_adv_to_route_delta_alignment"
        ),
        "final_route_adv_to_theta_delta_alignment": final_float(
            "route_adv_to_theta_delta_alignment"
        ),
        "final_theta_alignment_margin_over_cross": final_float(
            "theta_alignment_margin_over_cross"
        ),
        "final_route_alignment_margin_over_cross": final_float(
            "route_alignment_margin_over_cross"
        ),
        "final_cross_branch_alignment_mean": final_float(
            "cross_branch_alignment_mean"
        ),
        "final_theta_selected_action_prob_gain": final_float("theta_selected_action_prob_gain"),
        "final_route_selected_action_prob_gain": final_float("route_selected_action_prob_gain"),
        "final_theta_negative_adv_prob_gain": final_float("theta_negative_adv_prob_gain"),
        "final_route_negative_adv_prob_gain": final_float("route_negative_adv_prob_gain"),
        "final_theta_clip_fraction": final_float("theta_clip_fraction"),
        "final_route_clip_fraction": final_float("route_clip_fraction"),
        "final_theta_ratio_mean": final_float("theta_ratio_mean"),
        "final_theta_ratio_std": final_float("theta_ratio_std"),
        "final_theta_ratio_max": final_float("theta_ratio_max"),
        "final_route_ratio_mean": final_float("route_ratio_mean"),
        "final_route_ratio_std": final_float("route_ratio_std"),
        "final_route_ratio_max": final_float("route_ratio_max"),
        "final_offload_active_fraction": final_float("offload_active_fraction"),
        "final_route_logprob_active_fraction": final_float("route_logprob_active_fraction"),
        "final_theta_positive_fraction": final_float("theta_positive_fraction"),
        "final_route_credit_gate_fraction": final_float("route_credit_gate_fraction"),
        "final_route_credit_effective_fraction": final_float(
            "route_credit_effective_fraction"
        ),
        "final_route_credit_effective_count": final_float("route_credit_effective_count"),
        "final_route_credit_weight_mean": final_float("route_credit_weight_mean"),
        "final_route_credit_weight_std": final_float("route_credit_weight_std"),
        "final_route_credit_weight_min": final_float("route_credit_weight_min"),
        "final_route_credit_weight_max": final_float("route_credit_weight_max"),
        "final_theta_candidate_score_mean": final_float("theta_candidate_score_mean"),
        "final_theta_candidate_score_std": final_float("theta_candidate_score_std"),
        "final_theta_selected_score_mean": final_float("theta_selected_score_mean"),
        "final_theta_expected_score_mean": final_float("theta_expected_score_mean"),
        "final_theta_residual_credit_mean": final_float("theta_residual_credit_mean"),
        "final_theta_residual_credit_std": final_float("theta_residual_credit_std"),
        "final_theta_residual_credit_min": final_float("theta_residual_credit_min"),
        "final_theta_residual_credit_max": final_float("theta_residual_credit_max"),
        "final_route_candidate_score_mean": final_float("route_candidate_score_mean"),
        "final_route_candidate_score_std": final_float("route_candidate_score_std"),
        "final_route_residual_credit_mean": final_float("route_residual_credit_mean"),
        "final_route_residual_credit_std": final_float("route_residual_credit_std"),
        "final_route_residual_credit_min": final_float("route_residual_credit_min"),
        "final_route_residual_credit_max": final_float("route_residual_credit_max"),
        "final_route_expected_score_mean": final_float("route_expected_score_mean"),
        "final_route_selected_score_mean": final_float("route_selected_score_mean"),
        "final_route_decision_agreement_ratio": final_float(
            "route_decision_agreement_ratio"
        ),
        "final_actual_bs1_rate_when_route_true_gap_positive": final_float(
            "actual_bs1_rate_when_route_true_gap_positive"
        ),
        "final_actual_bs1_rate_when_route_true_gap_negative": final_float(
            "actual_bs1_rate_when_route_true_gap_negative"
        ),
        "final_route_score_vector_mean_abs": final_float("route_score_vector_mean_abs"),
        "final_route_score_vector_std": final_float("route_score_vector_std"),
        "final_route_credit_fallback_trigger_count": final_int(
            "route_credit_fallback_trigger_count"
        ),
        "final_route_credit_fallback_rate": final_float("route_credit_fallback_rate"),
        "final_theta_old_logprob_mean": final_float("theta_old_logprob_mean"),
        "final_theta_new_logprob_mean": final_float("theta_new_logprob_mean"),
        "final_route_old_logprob_mean": final_float("route_old_logprob_mean"),
        "final_route_new_logprob_mean": final_float("route_new_logprob_mean"),
        "final_joint_old_logprob_mean": final_float("joint_old_logprob_mean"),
        "final_joint_new_logprob_mean": final_float("joint_new_logprob_mean"),
        "final_joint_ratio_mean": final_float("joint_ratio_mean"),
        "final_joint_approx_kl": final_float("joint_approx_kl"),
        "final_joint_clip_fraction": final_float("joint_clip_fraction"),
        "final_joint_selected_score_mean": final_float("joint_selected_score_mean"),
        "final_joint_expected_score_mean": final_float("joint_expected_score_mean"),
        "final_joint_residual_credit_mean": final_float("joint_residual_credit_mean"),
        "final_joint_residual_credit_std": final_float("joint_residual_credit_std"),
        "final_joint_residual_credit_min": final_float("joint_residual_credit_min"),
        "final_joint_residual_credit_max": final_float("joint_residual_credit_max"),
        "final_joint_action_decision_agreement_ratio_under_reward_aligned": final_float(
            "joint_action_decision_agreement_ratio_under_reward_aligned"
        ),
        "final_actual_local_rate_when_reward_aligned_best_is_local": final_float(
            "actual_local_rate_when_reward_aligned_best_is_local"
        ),
        "final_actual_bs1_rate_when_reward_aligned_best_is_bs1": final_float(
            "actual_bs1_rate_when_reward_aligned_best_is_bs1"
        ),
        "final_actual_bs2_rate_when_reward_aligned_best_is_bs2": final_float(
            "actual_bs2_rate_when_reward_aligned_best_is_bs2"
        ),
        "final_joint_action_decision_agreement_ratio_under_td_aligned": final_float(
            "joint_action_decision_agreement_ratio_under_td_aligned"
        ),
        "final_actual_local_rate_when_td_aligned_best_is_local": final_float(
            "actual_local_rate_when_td_aligned_best_is_local"
        ),
        "final_actual_bs1_rate_when_td_aligned_best_is_bs1": final_float(
            "actual_bs1_rate_when_td_aligned_best_is_bs1"
        ),
        "final_actual_bs2_rate_when_td_aligned_best_is_bs2": final_float(
            "actual_bs2_rate_when_td_aligned_best_is_bs2"
        ),
        "final_offload_decision_agreement_ratio": final_float(
            "offload_decision_agreement_ratio"
        ),
        "final_actual_offload_rate_when_theta_true_gap_positive": final_float(
            "actual_offload_rate_when_theta_true_gap_positive"
        ),
        "final_actual_offload_rate_when_theta_true_gap_negative": final_float(
            "actual_offload_rate_when_theta_true_gap_negative"
        ),
        "final_theta_approx_kl": final_float("theta_approx_kl"),
        "final_route_approx_kl": final_float("route_approx_kl"),
        "final_conditional_policy_consistency_score": final_float(
            "conditional_policy_consistency_score"
        ),
        "final_theta_loss_mean": final_float("theta_loss_mean"),
        "final_route_loss_mean": final_float("route_loss_mean"),
        "final_theta_route_loss_ratio": final_float("theta_route_loss_ratio"),
        "final_route_gate_mean": final_float("route_gate_mean"),
        "final_route_gate_std": final_float("route_gate_std"),
        "final_route_gate_min": final_float("route_gate_min"),
        "final_route_gate_max": final_float("route_gate_max"),
        "final_route_gate_active_fraction": final_float("route_gate_active_fraction"),
        "final_route_mask_mean": final_float("route_mask_mean"),
        "final_route_mask_active_fraction": final_float("route_mask_active_fraction"),
        "final_route_mask_threshold": final_float("route_mask_threshold"),
        "final_route_confident_mask_margin": final_float("route_confident_mask_margin"),
        "final_route_mask_count_mean": final_float("route_mask_count_mean"),
        "final_route_mask_count_std": final_float("route_mask_count_std"),
        "final_theta_head_grad_norm": final_float("theta_head_grad_norm"),
        "final_route_head_grad_norm": final_float("route_head_grad_norm"),
        "final_theta_backbone_grad_norm": final_float("theta_backbone_grad_norm"),
        "final_route_backbone_grad_norm": final_float("route_backbone_grad_norm"),
        "final_theta_kl_target": final_float("theta_kl_target"),
        "final_route_kl_target": final_float("route_kl_target"),
        "final_theta_early_stop_count": final_int("theta_early_stop_count"),
        "final_route_early_stop_count": final_int("route_early_stop_count"),
        "final_coupled_stop_trigger_count": final_int("coupled_stop_trigger_count"),
        "final_coupled_stop_blocked_by_theta_floor_count": final_int(
            "coupled_stop_blocked_by_theta_floor_count"
        ),
        "final_coupled_stop_blocked_by_severity_gate_count": final_int(
            "coupled_stop_blocked_by_severity_gate_count"
        ),
        "final_coupled_stop_min_theta_updates_per_epoch": final_int(
            "coupled_stop_min_theta_updates_per_epoch"
        ),
        "final_coupled_stop_severity_factor": final_float(
            "coupled_stop_severity_factor"
        ),
        "final_coupled_stop_severity_freeze_kl_threshold": final_float(
            "coupled_stop_severity_freeze_kl_threshold"
        ),
        "final_route_update_cap_trigger_count": final_int("route_update_cap_trigger_count"),
        "final_route_update_cap_per_epoch": final_int("route_update_cap_per_epoch"),
        "final_route_alignment_gate_threshold": final_float(
            "route_alignment_gate_threshold"
        ),
        "final_route_alignment_gate_accept_count": final_int(
            "route_alignment_gate_accept_count"
        ),
        "final_route_alignment_gate_reject_count": final_int(
            "route_alignment_gate_reject_count"
        ),
        "final_route_alignment_gate_reject_rate": final_float(
            "route_alignment_gate_reject_rate"
        ),
        "final_route_alignment_gate_score_mean": final_float(
            "route_alignment_gate_score_mean"
        ),
        "final_route_step_alignment_mean": final_float("route_step_alignment_mean"),
        "final_route_step_alignment_min": final_float("route_step_alignment_min"),
        "final_route_step_alignment_max": final_float("route_step_alignment_max"),
        "final_route_step_accept_count": final_int("route_step_accept_count"),
        "final_route_step_reject_count": final_int("route_step_reject_count"),
        "final_route_step_accept_rate": final_float("route_step_accept_rate"),
        "final_route_step_reject_rate": final_float("route_step_reject_rate"),
        "final_theta_update_count": final_int("theta_update_count"),
        "final_route_update_count": final_int("route_update_count"),
        "final_theta_only_step_kl": final_float("theta_only_step_kl"),
        "final_route_only_step_kl": final_float("route_only_step_kl"),
        "final_theta_only_step_prob_gain": final_float("theta_only_step_prob_gain"),
        "final_route_only_step_prob_gain": final_float("route_only_step_prob_gain"),
        "final_theta_after_route_shift": final_float("theta_after_route_shift"),
        "final_route_after_theta_shift": final_float("route_after_theta_shift"),
        "final_theta_route_feature_correlation": final_float("theta_route_feature_correlation"),
        "final_theta_route_head_correlation": final_float("theta_route_head_correlation"),
        "final_offload_rate_mean": final_float("offload_rate_mean"),
        "final_bs1_vs_bs2_entropy": final_float("bs1_vs_bs2_entropy"),
        "final_offload_vs_local_value_gap_mean": final_float("offload_vs_local_value_gap_mean"),
        "final_offload_vs_local_value_gap_std": final_float("offload_vs_local_value_gap_std"),
        "final_bs1_vs_bs2_value_gap_mean": final_float("bs1_vs_bs2_value_gap_mean"),
        "final_bs1_vs_bs2_value_gap_std": final_float("bs1_vs_bs2_value_gap_std"),
        "final_block_action_conditioned_value_now_mean": final_float(
            "block_action_conditioned_value_now_mean"
        ),
        "final_block_action_conditioned_value_now_std": final_float(
            "block_action_conditioned_value_now_std"
        ),
        "final_block_action_conditioned_value_next_mean": final_float(
            "block_action_conditioned_value_next_mean"
        ),
        "final_block_action_conditioned_value_next_std": final_float(
            "block_action_conditioned_value_next_std"
        ),
        "final_block_path_cost_local_mean": final_float("block_path_cost_local_mean"),
        "final_block_path_cost_local_std": final_float("block_path_cost_local_std"),
        "final_block_path_cost_bs1_mean": final_float("block_path_cost_bs1_mean"),
        "final_block_path_cost_bs1_std": final_float("block_path_cost_bs1_std"),
        "final_block_path_cost_bs2_mean": final_float("block_path_cost_bs2_mean"),
        "final_block_path_cost_bs2_std": final_float("block_path_cost_bs2_std"),
        "final_block_local_cost_now_mean": final_float("block_local_cost_now_mean"),
        "final_block_local_cost_now_std": final_float("block_local_cost_now_std"),
        "final_block_local_cost_next_mean": final_float("block_local_cost_next_mean"),
        "final_block_local_cost_next_std": final_float("block_local_cost_next_std"),
        "final_block_action_conditioned_cost_now_mean": final_float(
            "block_action_conditioned_cost_now_mean"
        ),
        "final_block_action_conditioned_cost_now_std": final_float(
            "block_action_conditioned_cost_now_std"
        ),
        "final_block_action_conditioned_cost_next_mean": final_float(
            "block_action_conditioned_cost_next_mean"
        ),
        "final_block_action_conditioned_cost_next_std": final_float(
            "block_action_conditioned_cost_next_std"
        ),
        "final_block_delta_cost_mean": final_float("block_delta_cost_mean"),
        "final_block_delta_cost_std": final_float("block_delta_cost_std"),
        "final_block_delta_cost_max": final_float("block_delta_cost_max"),
        "final_block_delta_cost_min": final_float("block_delta_cost_min"),
        "final_block_td_reward_mean": final_float("block_td_reward_mean"),
        "final_block_td_reward_std": final_float("block_td_reward_std"),
        "final_block_td_reward_max": final_float("block_td_reward_max"),
        "final_block_td_reward_min": final_float("block_td_reward_min"),
        "final_block_td_target_mean": final_float("block_td_target_mean"),
        "final_block_td_target_std": final_float("block_td_target_std"),
        "final_block_td_target_max": final_float("block_td_target_max"),
        "final_block_td_target_min": final_float("block_td_target_min"),
        "final_block_td_advantage_mean": final_float("block_td_advantage_mean"),
        "final_block_td_advantage_std": final_float("block_td_advantage_std"),
        "final_block_td_advantage_max": final_float("block_td_advantage_max"),
        "final_block_td_advantage_min": final_float("block_td_advantage_min"),
        "final_block_td_advantage_entropy": final_float("block_td_advantage_entropy"),
        "final_top_k_block_td_advantage_share": final_float(
            "top_k_block_td_advantage_share"
        ),
        "final_block_advantage_mean": final_float("block_advantage_mean"),
        "final_block_advantage_std": final_float("block_advantage_std"),
        "final_block_advantage_max": final_float("block_advantage_max"),
        "final_block_advantage_min": final_float("block_advantage_min"),
        "final_block_advantage_entropy": final_float("block_advantage_entropy"),
        "final_top_k_block_advantage_share": final_float("top_k_block_advantage_share"),
        "final_active_block_count_mean": final_float("active_block_count_mean"),
        "final_block_ratio_proxy_mean": final_float("block_ratio_proxy_mean", 1.0),
        "final_block_ratio_proxy_std": final_float("block_ratio_proxy_std"),
        "final_block_clip_fraction_mean": final_float("block_clip_fraction_mean"),
        "final_block_clip_fraction_std": final_float("block_clip_fraction_std"),
        "final_block_positive_adv_clip_fraction_mean": final_float(
            "block_positive_adv_clip_fraction_mean"
        ),
        "final_block_negative_adv_clip_fraction_mean": final_float(
            "block_negative_adv_clip_fraction_mean"
        ),
        "final_block_ratio_mean": final_float("block_ratio_mean", 1.0),
        "final_block_ratio_std": final_float("block_ratio_std"),
        "final_block_ratio_max": final_float("block_ratio_max", 1.0),
        "final_block_surrogate_mean": final_float("block_surrogate_mean"),
        "final_block_surrogate_std": final_float("block_surrogate_std"),
        "final_per_block_selected_action_prob_gain_mean": final_float(
            "per_block_selected_action_prob_gain_mean"
        ),
        "final_per_block_selected_action_prob_gain_std": final_float(
            "per_block_selected_action_prob_gain_std"
        ),
        "final_sum_to_mean_scale_ratio": final_float("sum_to_mean_scale_ratio"),
        "final_sum_to_blockmean_scale_ratio": final_float("sum_to_blockmean_scale_ratio"),
        "final_mean_to_blockmean_scale_ratio": final_float("mean_to_blockmean_scale_ratio"),
        "final_actor_grad_norm": final_float("actor_grad_norm"),
        "final_action_distribution_std": final_float("action_distribution_std"),
        "final_action_prob_std": final_float("action_prob_std"),
        "final_policy_std_mean": final_float("policy_std_mean"),
        "final_policy_logit_std": final_float("policy_logit_std"),
        "final_policy_confidence_mean": final_float("policy_confidence_mean"),
        "final_selected_action_histogram": final_str("selected_action_histogram"),
        "final_probe_policy_pairwise_kl_mean": final_float("probe_policy_pairwise_kl_mean"),
        "final_probe_policy_pairwise_l1_mean": final_float("probe_policy_pairwise_l1_mean"),
        "final_policy_logit_delta_mean": final_float("policy_logit_delta_mean"),
        "final_policy_logit_delta_std": final_float("policy_logit_delta_std"),
        "final_selected_action_change_rate": final_float("selected_action_change_rate"),
        "final_top1_action_change_rate": final_float("top1_action_change_rate"),
        "final_top1_prob_delta_mean": final_float("top1_prob_delta_mean"),
        "final_popart_mean": final_float("popart_mean"),
        "final_popart_std": final_float("popart_std"),
        "final_value_target_mean": final_float("value_target_mean"),
        "final_value_target_std": final_float("value_target_std"),
        "final_critic_raw_prediction_mean": final_float("critic_raw_prediction_mean"),
        "final_critic_raw_prediction_std": final_float("critic_raw_prediction_std"),
        "final_critic_normalized_prediction_mean": final_float(
            "critic_normalized_prediction_mean"
        ),
        "final_critic_normalized_prediction_std": final_float(
            "critic_normalized_prediction_std"
        ),
        "final_value_explained_variance": final_float("value_explained_variance"),
        "final_prediction_target_corr": final_float("prediction_target_corr"),
        "final_prediction_std_over_target_std": final_float("prediction_std_over_target_std"),
        "final_probe_prediction_std": final_float("probe_prediction_std"),
        "final_probe_prediction_range": final_float("probe_prediction_range"),
        "final_pairwise_prediction_distance_mean": final_float(
            "pairwise_prediction_distance_mean"
        ),
        "final_critic_vs_constant_mse_gain": final_float("critic_vs_constant_mse_gain"),
        "final_critic_vs_constant_huber_gain": final_float("critic_vs_constant_huber_gain"),
        "final_critic_hidden_feature_std": final_float("critic_hidden_feature_std"),
        "final_critic_hidden_feature_dim_std_mean": final_float(
            "critic_hidden_feature_dim_std_mean"
        ),
        "final_critic_hidden_feature_dim_std_min": final_float(
            "critic_hidden_feature_dim_std_min"
        ),
        "final_critic_head_weight_norm": final_float("critic_head_weight_norm"),
        "final_critic_head_bias_mean": final_float("critic_head_bias_mean"),
        "final_critic_backbone_grad_norm": final_float("critic_backbone_grad_norm"),
        "final_critic_head_grad_norm": final_float("critic_head_grad_norm"),
        "final_actor_input_mean": final_float("actor_input_mean"),
        "final_actor_input_std": final_float("actor_input_std"),
        "final_actor_input_dim_std_mean": final_float("actor_input_dim_std_mean"),
        "final_actor_input_dim_std_min": final_float("actor_input_dim_std_min"),
        "final_actor_input_clip_fraction": final_float("actor_input_clip_fraction"),
        "final_actor_input_derived_dim": final_int("actor_input_derived_dim"),
        "final_actor_input_total_dim": final_int("actor_input_total_dim"),
        "final_actor_derived_feature_mean": final_float("actor_derived_feature_mean"),
        "final_actor_derived_feature_std": final_float("actor_derived_feature_std"),
        "final_actor_raw_dim": final_int("actor_raw_dim"),
        "final_actor_raw_pruned_dim_count": final_int("actor_raw_pruned_dim_count"),
        "final_actor_raw_dim_std_mean": final_float("actor_raw_dim_std_mean"),
        "final_actor_raw_dim_std_min": final_float("actor_raw_dim_std_min"),
        "final_actor_raw_zero_var_dim_count": final_int("actor_raw_zero_var_dim_count"),
        "final_actor_raw_static_zero_var_dim_count": final_int(
            "actor_raw_static_zero_var_dim_count"
        ),
        "final_critic_input_mean": final_float("critic_input_mean"),
        "final_critic_input_std": final_float("critic_input_std"),
        "final_critic_input_dim_std_mean": final_float("critic_input_dim_std_mean"),
        "final_critic_input_dim_std_min": final_float("critic_input_dim_std_min"),
        "final_critic_input_clip_fraction": final_float("critic_input_clip_fraction"),
        "final_critic_input_derived_dim": final_int("critic_input_derived_dim"),
        "final_critic_input_total_dim": final_int("critic_input_total_dim"),
        "final_critic_derived_feature_mean": final_float("critic_derived_feature_mean"),
        "final_critic_derived_feature_std": final_float("critic_derived_feature_std"),
        "best_epoch_is_one": int(best_epoch == 1),
    }
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def plot_single_series(
    df: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    values = df[column] if column in df.columns else np.zeros(len(df))
    plt.plot(df["epoch"], values, marker="o", label=column)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_multi_panel_series(
    df: pd.DataFrame,
    panel_specs: list[tuple[str, str, str]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(panel_specs), 1, figsize=(8, 4.5 * len(panel_specs)), sharex=True)
    if len(panel_specs) == 1:
        axes = [axes]
    for axis, (column, ylabel, title) in zip(axes, panel_specs, strict=True):
        values = df[column] if column in df.columns else np.zeros(len(df))
        axis.plot(df["epoch"], values, marker="o")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_single_run_outputs(run_dir: Path) -> None:
    logs = pd.read_csv(run_dir / "train_logs.csv")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_single_series(
        logs,
        column="critic_loss",
        title="Critic Loss",
        ylabel="critic_loss",
        output_path=plots_dir / "critic_loss_compare.png",
    )
    plot_single_series(
        logs,
        column="episode_reward",
        title="Reward Curve",
        ylabel="episode_reward",
        output_path=plots_dir / "reward_curve_compare.png",
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(logs["epoch"], logs["value_target_mean"], marker="o")
    axes[0].set_ylabel("value_target_mean")
    axes[0].set_title("Value Target Mean")
    axes[1].plot(logs["epoch"], logs["value_target_std"], marker="o")
    axes[1].set_ylabel("value_target_std")
    axes[1].set_title("Value Target Std")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "target_stats_compare.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(logs["epoch"], logs["critic_raw_prediction_mean"], marker="o")
    axes[0].set_ylabel("critic_raw_prediction_mean")
    axes[0].set_title("Critic Raw Prediction Mean")
    axes[1].plot(logs["epoch"], logs["critic_normalized_prediction_mean"], marker="o")
    axes[1].set_ylabel("critic_normalized_prediction_mean")
    axes[1].set_title("Critic Normalized Prediction Mean")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "prediction_mean_compare.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(logs["epoch"], logs["critic_raw_prediction_std"], marker="o")
    axes[0].set_ylabel("critic_raw_prediction_std")
    axes[0].set_title("Critic Raw Prediction Std")
    axes[1].plot(logs["epoch"], logs["critic_normalized_prediction_std"], marker="o")
    axes[1].set_ylabel("critic_normalized_prediction_std")
    axes[1].set_title("Critic Normalized Prediction Std")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "prediction_std_compare.png", dpi=150)
    plt.close(fig)

    plot_single_series(
        logs,
        column="value_explained_variance",
        title="Value Explained Variance",
        ylabel="value_explained_variance",
        output_path=plots_dir / "explained_variance_compare.png",
    )
    plot_single_series(
        logs,
        column="prediction_target_corr",
        title="Prediction Target Correlation",
        ylabel="prediction_target_corr",
        output_path=plots_dir / "prediction_target_corr_compare.png",
    )
    plot_single_series(
        logs,
        column="prediction_std_over_target_std",
        title="Prediction Std over Target Std",
        ylabel="prediction_std_over_target_std",
        output_path=plots_dir / "prediction_std_ratio_compare.png",
    )
    plot_single_series(
        logs,
        column="probe_prediction_std",
        title="Probe Prediction Std",
        ylabel="probe_prediction_std",
        output_path=plots_dir / "probe_prediction_std_curve.png",
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(logs["epoch"], logs["critic_vs_constant_mse_gain"], marker="o", label="mse gain")
    axes[0].set_ylabel("critic_vs_constant_mse_gain")
    axes[0].set_title("Critic vs Constant Baseline MSE Gain")
    axes[1].plot(
        logs["epoch"],
        logs["critic_vs_constant_huber_gain"],
        marker="o",
        label="huber gain",
        color="#ff7f0e",
    )
    axes[1].set_ylabel("critic_vs_constant_huber_gain")
    axes[1].set_title("Critic vs Constant Baseline Huber Gain")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "critic_vs_constant_gain_compare.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(logs["epoch"], logs["critic_hidden_feature_std"], marker="o", label="feature std")
    axes[0].plot(
        logs["epoch"],
        logs["critic_hidden_feature_dim_std_mean"],
        marker="s",
        label="dim std mean",
    )
    axes[0].legend()
    axes[0].set_ylabel("feature std")
    axes[0].set_title("Critic Hidden Feature Std")
    axes[1].plot(
        logs["epoch"],
        logs["critic_hidden_feature_dim_std_min"],
        marker="o",
        color="#2ca02c",
    )
    axes[1].set_ylabel("dim std min")
    axes[1].set_title("Critic Hidden Feature Dim Std Min")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "critic_hidden_feature_std_compare.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(logs["epoch"], logs["critic_backbone_grad_norm"], marker="o")
    axes[0].set_ylabel("critic_backbone_grad_norm")
    axes[0].set_title("Critic Backbone Grad Norm")
    axes[1].plot(logs["epoch"], logs["critic_head_grad_norm"], marker="o", color="#ff7f0e")
    axes[1].set_ylabel("critic_head_grad_norm")
    axes[1].set_title("Critic Head Grad Norm")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "critic_grad_norm_compare.png", dpi=150)
    plt.close(fig)

    plot_single_series(
        logs,
        column="actor_input_std",
        title="Actor Input Std",
        ylabel="actor_input_std",
        output_path=plots_dir / "actor_input_std_compare.png",
    )
    plot_multi_panel_series(
        logs,
        panel_specs=[
            (
                "actor_input_dim_std_mean",
                "actor_input_dim_std_mean",
                "Actor Input Dim Std Mean",
            ),
            (
                "actor_input_dim_std_min",
                "actor_input_dim_std_min",
                "Actor Input Dim Std Min",
            ),
        ],
        output_path=plots_dir / "actor_input_dim_std_compare.png",
    )
    plot_single_series(
        logs,
        column="actor_grad_norm",
        title="Actor Grad Norm",
        ylabel="actor_grad_norm",
        output_path=plots_dir / "actor_grad_norm_compare.png",
    )
    plot_single_series(
        logs,
        column="policy_entropy",
        title="Policy Entropy",
        ylabel="policy_entropy",
        output_path=plots_dir / "policy_entropy_compare.png",
    )
    plot_multi_panel_series(
        logs,
        panel_specs=[
            ("advantage_mean", "advantage_mean", "Advantage Mean"),
            ("advantage_std", "advantage_std", "Advantage Std"),
        ],
        output_path=plots_dir / "advantage_stats_compare.png",
    )
    plot_multi_panel_series(
        logs,
        panel_specs=[
            ("action_prob_std", "action_prob_std", "Action Distribution Std"),
            ("policy_std_mean", "policy_std_mean", "Policy Std Mean"),
            ("policy_confidence_mean", "policy_confidence_mean", "Policy Confidence Mean"),
        ],
        output_path=plots_dir / "action_distribution_compare.png",
    )
    plot_single_series(
        logs,
        column="actor_raw_zero_var_dim_count",
        title="Actor Raw Zero-Var Dim Count",
        ylabel="actor_raw_zero_var_dim_count",
        output_path=plots_dir / "actor_raw_zero_var_dim_compare.png",
    )
    plot_multi_panel_series(
        logs,
        panel_specs=[
            ("actor_raw_dim_std_mean", "actor_raw_dim_std_mean", "Actor Raw Dim Std Mean"),
            ("actor_raw_dim_std_min", "actor_raw_dim_std_min", "Actor Raw Dim Std Min"),
        ],
        output_path=plots_dir / "actor_raw_dim_std_compare.png",
    )

    final_payload_path = Path(str(logs["prediction_target_payload_path"].iloc[-1]))
    payload = pd.read_csv(final_payload_path)

    plt.figure(figsize=(7, 6))
    plt.scatter(payload["value_target"], payload["critic_prediction"], s=12, alpha=0.6)
    lower = min(payload["value_target"].min(), payload["critic_prediction"].min())
    upper = max(payload["value_target"].max(), payload["critic_prediction"].max())
    plt.plot([lower, upper], [lower, upper], linestyle="--", color="#d62728", linewidth=1.2)
    plt.xlabel("value_target")
    plt.ylabel("critic_prediction")
    plt.title("Prediction vs Target Scatter")
    plt.tight_layout()
    plt.savefig(plots_dir / "prediction_vs_target_scatter.png", dpi=150)
    plt.close()

    bucket_df = payload.copy()
    bucket_df["target_bucket"] = pd.qcut(
        bucket_df["value_target"],
        q=min(10, len(bucket_df)),
        duplicates="drop",
    )
    bucket_summary = (
        bucket_df.groupby("target_bucket", observed=True)
        .agg(
            bucket_target_mean=("value_target", "mean"),
            bucket_prediction_mean=("critic_prediction", "mean"),
            bucket_count=("critic_prediction", "size"),
        )
        .reset_index(drop=True)
    )
    bucket_summary.to_csv(run_dir / "target_bucket_prediction_mean.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    x = np.arange(len(bucket_summary))
    plt.plot(x, bucket_summary["bucket_target_mean"], marker="o", label="target mean")
    plt.plot(x, bucket_summary["bucket_prediction_mean"], marker="s", label="prediction mean")
    plt.xlabel("target quantile bucket")
    plt.ylabel("mean")
    plt.title("Target Bucket Prediction Mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "target_bucket_prediction_mean.png", dpi=150)
    plt.close()

    plot_single_series(
        logs,
        column="episode_reward",
        title="Reward Curve",
        ylabel="episode_reward",
        output_path=plots_dir / "reward_curve.png",
    )
    plot_single_series(
        logs,
        column="critic_loss",
        title="Critic Loss Curve",
        ylabel="critic_loss",
        output_path=plots_dir / "critic_loss_curve.png",
    )
    plot_single_series(
        logs,
        column="policy_loss",
        title="Policy Loss Curve",
        ylabel="policy_loss",
        output_path=plots_dir / "policy_loss_curve.png",
    )
    plot_single_series(
        logs,
        column="approx_kl",
        title="Approx KL Curve",
        ylabel="approx_kl",
        output_path=plots_dir / "approx_kl_curve.png",
    )
    plot_single_series(
        logs,
        column="value_explained_variance",
        title="Value Explained Variance",
        ylabel="value_explained_variance",
        output_path=plots_dir / "value_explained_variance_curve.png",
    )
    plot_single_series(
        logs,
        column="prediction_target_corr",
        title="Prediction Target Correlation",
        ylabel="prediction_target_corr",
        output_path=plots_dir / "prediction_target_corr_curve.png",
    )
    plot_single_series(
        logs,
        column="prediction_std_over_target_std",
        title="Prediction Std over Target Std",
        ylabel="prediction_std_over_target_std",
        output_path=plots_dir / "prediction_std_over_target_std_curve.png",
    )
    plot_multi_panel_series(
        logs,
        panel_specs=[
            ("ratio_mean", "ratio_mean", "Ratio Mean"),
            ("ratio_std", "ratio_std", "Ratio Std"),
            ("ratio_min", "ratio_min", "Ratio Min"),
            ("ratio_max", "ratio_max", "Ratio Max"),
        ],
        output_path=plots_dir / "ratio_stats_curve.png",
    )
    plot_multi_panel_series(
        logs,
        panel_specs=[
            ("advantage_mean", "advantage_mean", "Advantage Mean"),
            ("advantage_std", "advantage_std", "Advantage Std"),
            ("positive_advantage_ratio", "positive_advantage_ratio", "Positive Advantage Ratio"),
            ("negative_advantage_ratio", "negative_advantage_ratio", "Negative Advantage Ratio"),
        ],
        output_path=plots_dir / "advantage_stats_curve.png",
    )
    plot_multi_panel_series(
        logs,
        panel_specs=[
            (
                "probe_policy_pairwise_kl_mean",
                "probe_policy_pairwise_kl_mean",
                "Probe Policy Pairwise KL Mean",
            ),
            (
                "probe_policy_pairwise_l1_mean",
                "probe_policy_pairwise_l1_mean",
                "Probe Policy Pairwise L1 Mean",
            ),
        ],
        output_path=plots_dir / "probe_policy_kl_curve.png",
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    clip_specs = [
        ("clip_fraction", "clip_fraction", "#1f77b4"),
        ("positive_adv_clip_fraction", "positive_adv_clip_fraction", "#ff7f0e"),
        ("negative_adv_clip_fraction", "negative_adv_clip_fraction", "#2ca02c"),
    ]
    for column, label, color in clip_specs:
        values = logs[column] if column in logs.columns else np.zeros(len(logs))
        axes[0].plot(logs["epoch"], values, marker="o", label=label, color=color)
    axes[0].set_ylabel("clip_fraction")
    axes[0].set_title("Clip Fraction")
    axes[0].legend()
    entropy_loss_values = logs["entropy_loss"] if "entropy_loss" in logs.columns else np.zeros(len(logs))
    axes[1].plot(logs["epoch"], entropy_loss_values, marker="o", color="#9467bd")
    axes[1].set_ylabel("entropy_loss")
    axes[1].set_title("Entropy Loss")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "clip_fraction_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    delta_mean = (
        logs["delta_log_prob_selected_action_mean"]
        if "delta_log_prob_selected_action_mean" in logs.columns
        else np.zeros(len(logs))
    )
    delta_std = (
        logs["delta_log_prob_selected_action_std"]
        if "delta_log_prob_selected_action_std" in logs.columns
        else np.zeros(len(logs))
    )
    alignment = (
        logs["advantage_action_alignment"]
        if "advantage_action_alignment" in logs.columns
        else np.zeros(len(logs))
    )
    axes[0].plot(logs["epoch"], delta_mean, marker="o", label="delta_log_prob_mean")
    axes[0].plot(logs["epoch"], delta_std, marker="s", label="delta_log_prob_std")
    axes[0].set_ylabel("delta_log_prob")
    axes[0].set_title("Selected-Action Log-Prob Delta")
    axes[0].legend()
    axes[1].plot(logs["epoch"], alignment, marker="o", color="#d62728")
    axes[1].set_ylabel("corr")
    axes[1].set_title("Advantage / Log-Prob Alignment")
    gain_specs = [
        ("high_advantage_action_prob_gain", "high positive"),
        ("mid_advantage_action_prob_gain", "mid positive"),
        ("low_advantage_action_prob_gain", "near zero"),
        ("negative_advantage_action_prob_gain", "negative"),
    ]
    for column, label in gain_specs:
        values = logs[column] if column in logs.columns else np.zeros(len(logs))
        axes[2].plot(logs["epoch"], values, marker="o", label=label)
    axes[2].set_ylabel("selected_action_prob_gain")
    axes[2].set_title("Selected-Action Prob Gain by Advantage Bucket")
    axes[2].legend()
    axes[2].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "advantage_alignment_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    change_specs = [
        ("selected_action_change_rate", "selected_action_change_rate", "#1f77b4"),
        ("top1_action_change_rate", "top1_action_change_rate", "#ff7f0e"),
    ]
    for column, label, color in change_specs:
        values = logs[column] if column in logs.columns else np.zeros(len(logs))
        axes[0].plot(logs["epoch"], values, marker="o", label=label, color=color)
    axes[0].set_ylabel("change_rate")
    axes[0].set_title("Probe Selected-Action Change Rate")
    axes[0].legend()
    top1_prob_delta = (
        logs["top1_prob_delta_mean"] if "top1_prob_delta_mean" in logs.columns else np.zeros(len(logs))
    )
    axes[1].plot(logs["epoch"], top1_prob_delta, marker="o", color="#2ca02c")
    axes[1].set_ylabel("top1_prob_delta_mean")
    axes[1].set_title("Probe Top-1 Probability Delta")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "selected_action_change_rate_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    logprob_delta_sum_mean = (
        logs["logprob_delta_sum_mean"] if "logprob_delta_sum_mean" in logs.columns else np.zeros(len(logs))
    )
    logprob_delta_mean_mean = (
        logs["logprob_delta_mean_mean"] if "logprob_delta_mean_mean" in logs.columns else np.zeros(len(logs))
    )
    logprob_delta_sum_std = (
        logs["logprob_delta_sum_std"] if "logprob_delta_sum_std" in logs.columns else np.zeros(len(logs))
    )
    logprob_delta_mean_std = (
        logs["logprob_delta_mean_std"] if "logprob_delta_mean_std" in logs.columns else np.zeros(len(logs))
    )
    sum_to_mean_scale_ratio = (
        logs["sum_to_mean_scale_ratio"] if "sum_to_mean_scale_ratio" in logs.columns else np.zeros(len(logs))
    )
    axes[0].plot(logs["epoch"], logprob_delta_sum_mean, marker="o", label="sum mean")
    axes[0].plot(logs["epoch"], logprob_delta_mean_mean, marker="s", label="mean mean")
    axes[0].set_ylabel("mean")
    axes[0].set_title("Log-Prob Delta Mean")
    axes[0].legend()
    axes[1].plot(logs["epoch"], logprob_delta_sum_std, marker="o", label="sum std")
    axes[1].plot(logs["epoch"], logprob_delta_mean_std, marker="s", label="mean std")
    axes[1].set_ylabel("std")
    axes[1].set_title("Log-Prob Delta Std")
    axes[1].legend()
    axes[2].plot(logs["epoch"], sum_to_mean_scale_ratio, marker="o", color="#2ca02c")
    axes[2].set_ylabel("scale ratio")
    axes[2].set_title("Sum to Mean Scale Ratio")
    axes[2].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "logprob_scale_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    block_logprob_delta_mean = (
        logs["block_logprob_delta_mean"]
        if "block_logprob_delta_mean" in logs.columns
        else np.zeros(len(logs))
    )
    block_logprob_delta_std = (
        logs["block_logprob_delta_std"]
        if "block_logprob_delta_std" in logs.columns
        else np.zeros(len(logs))
    )
    block_ratio_proxy_mean = (
        logs["block_ratio_proxy_mean"]
        if "block_ratio_proxy_mean" in logs.columns
        else np.ones(len(logs))
    )
    block_ratio_proxy_std = (
        logs["block_ratio_proxy_std"]
        if "block_ratio_proxy_std" in logs.columns
        else np.zeros(len(logs))
    )
    sum_to_blockmean_scale_ratio = (
        logs["sum_to_blockmean_scale_ratio"]
        if "sum_to_blockmean_scale_ratio" in logs.columns
        else np.ones(len(logs))
    )
    mean_to_blockmean_scale_ratio = (
        logs["mean_to_blockmean_scale_ratio"]
        if "mean_to_blockmean_scale_ratio" in logs.columns
        else np.ones(len(logs))
    )
    axes[0].plot(logs["epoch"], block_logprob_delta_mean, marker="o", label="block mean")
    axes[0].plot(logs["epoch"], block_logprob_delta_std, marker="s", label="block std")
    axes[0].set_ylabel("block_delta")
    axes[0].set_title("Block-Mean Log-Prob Delta")
    axes[0].legend()
    axes[1].plot(logs["epoch"], block_ratio_proxy_mean, marker="o", label="ratio proxy mean")
    axes[1].plot(logs["epoch"], block_ratio_proxy_std, marker="s", label="ratio proxy std")
    axes[1].set_ylabel("block_ratio_proxy")
    axes[1].set_title("Block Ratio Proxy")
    axes[1].legend()
    axes[2].plot(
        logs["epoch"],
        sum_to_blockmean_scale_ratio,
        marker="o",
        label="sum_to_blockmean",
    )
    axes[2].plot(
        logs["epoch"],
        mean_to_blockmean_scale_ratio,
        marker="s",
        label="mean_to_blockmean",
    )
    axes[2].set_ylabel("scale ratio")
    axes[2].set_title("Block Scale Ratios")
    axes[2].legend()
    axes[2].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "block_logprob_scale_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(
        logs["epoch"],
        logs["block_clip_fraction_mean"]
        if "block_clip_fraction_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="o",
        label="block clip mean",
    )
    axes[0].plot(
        logs["epoch"],
        logs["block_clip_fraction_std"]
        if "block_clip_fraction_std" in logs.columns
        else np.zeros(len(logs)),
        marker="s",
        label="block clip std",
    )
    axes[0].plot(
        logs["epoch"],
        logs["block_positive_adv_clip_fraction_mean"]
        if "block_positive_adv_clip_fraction_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="^",
        label="block +adv clip mean",
    )
    axes[0].plot(
        logs["epoch"],
        logs["block_negative_adv_clip_fraction_mean"]
        if "block_negative_adv_clip_fraction_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="v",
        label="block -adv clip mean",
    )
    axes[0].set_ylabel("clip_fraction")
    axes[0].set_title("Block Clip Fraction")
    axes[0].legend()
    axes[1].plot(
        logs["epoch"],
        logs["block_ratio_mean"] if "block_ratio_mean" in logs.columns else np.ones(len(logs)),
        marker="o",
        label="block ratio mean",
    )
    axes[1].plot(
        logs["epoch"],
        logs["block_ratio_max"] if "block_ratio_max" in logs.columns else np.ones(len(logs)),
        marker="s",
        label="block ratio max",
    )
    axes[1].set_ylabel("block_ratio")
    axes[1].set_title("Block Ratio Scale")
    axes[1].legend()
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "block_clip_fraction_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(
        logs["epoch"],
        logs["block_surrogate_mean"]
        if "block_surrogate_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="o",
        label="block surrogate mean",
    )
    axes[0].plot(
        logs["epoch"],
        logs["block_surrogate_std"]
        if "block_surrogate_std" in logs.columns
        else np.zeros(len(logs)),
        marker="s",
        label="block surrogate std",
    )
    axes[0].set_ylabel("block_surrogate")
    axes[0].set_title("Block Surrogate Scale")
    axes[0].legend()
    axes[1].plot(
        logs["epoch"],
        logs["per_block_selected_action_prob_gain_mean"]
        if "per_block_selected_action_prob_gain_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="o",
        label="per-block gain mean",
    )
    axes[1].plot(
        logs["epoch"],
        logs["per_block_selected_action_prob_gain_std"]
        if "per_block_selected_action_prob_gain_std" in logs.columns
        else np.zeros(len(logs)),
        marker="s",
        label="per-block gain std",
    )
    axes[1].set_ylabel("per_block_gain")
    axes[1].set_title("Per-Block Selected Action Prob Gain")
    axes[1].legend()
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "block_surrogate_scale_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(
        logs["epoch"],
        logs["block_adv_scale_entropy"]
        if "block_adv_scale_entropy" in logs.columns
        else np.zeros(len(logs)),
        marker="o",
        label="block adv-scale entropy",
    )
    axes[0].plot(
        logs["epoch"],
        logs["top_k_block_adv_scale_share"]
        if "top_k_block_adv_scale_share" in logs.columns
        else np.zeros(len(logs)),
        marker="s",
        label="top-k adv-scale share",
    )
    axes[0].set_ylabel("adv_scale")
    axes[0].set_title("Block Advantage Scale Structure")
    axes[0].legend()
    axes[1].plot(
        logs["epoch"],
        logs["block_adv_scale_mean"]
        if "block_adv_scale_mean" in logs.columns
        else np.ones(len(logs)),
        marker="o",
        label="adv-scale mean",
    )
    axes[1].plot(
        logs["epoch"],
        logs["block_adv_scale_std"]
        if "block_adv_scale_std" in logs.columns
        else np.zeros(len(logs)),
        marker="s",
        label="adv-scale std",
    )
    axes[1].plot(
        logs["epoch"],
        logs["active_block_count_mean"]
        if "active_block_count_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="^",
        label="active block count",
    )
    axes[1].set_ylabel("summary")
    axes[1].set_title("Block Advantage Scale Summary")
    axes[1].legend()
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "block_adv_scale_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(
        logs["epoch"],
        logs["block_value_scale_entropy"]
        if "block_value_scale_entropy" in logs.columns
        else np.zeros(len(logs)),
        marker="o",
        label="block value-scale entropy",
    )
    axes[0].plot(
        logs["epoch"],
        logs["top_k_block_value_scale_share"]
        if "top_k_block_value_scale_share" in logs.columns
        else np.zeros(len(logs)),
        marker="s",
        label="top-k value-scale share",
    )
    axes[0].set_ylabel("value_scale")
    axes[0].set_title("Learned Block Value Scale Structure")
    axes[0].legend()
    axes[1].plot(
        logs["epoch"],
        logs["block_value_score_mean"]
        if "block_value_score_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="o",
        label="value score mean",
    )
    axes[1].plot(
        logs["epoch"],
        logs["block_value_scale_std"]
        if "block_value_scale_std" in logs.columns
        else np.zeros(len(logs)),
        marker="s",
        label="value-scale std",
    )
    axes[1].plot(
        logs["epoch"],
        logs["active_block_count_mean"]
        if "active_block_count_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="^",
        label="active block count",
    )
    axes[1].set_ylabel("summary")
    axes[1].set_title("Learned Block Value Scale Summary")
    axes[1].legend()
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "block_value_scale_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(
        logs["epoch"],
        logs["block_td_advantage_entropy"]
        if "block_td_advantage_entropy" in logs.columns
        else np.zeros(len(logs)),
        marker="o",
        label="block td-adv entropy",
    )
    axes[0].plot(
        logs["epoch"],
        logs["top_k_block_td_advantage_share"]
        if "top_k_block_td_advantage_share" in logs.columns
        else np.zeros(len(logs)),
        marker="s",
        label="top-k td-adv share",
    )
    axes[0].set_ylabel("td_advantage")
    axes[0].set_title("Block TD Advantage Structure")
    axes[0].legend()
    axes[1].plot(
        logs["epoch"],
        logs["block_value_pred_mean"]
        if "block_value_pred_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="o",
        label="value pred mean",
    )
    axes[1].plot(
        logs["epoch"],
        logs["block_td_target_mean"]
        if "block_td_target_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="s",
        label="td target mean",
    )
    axes[1].plot(
        logs["epoch"],
        logs["block_td_advantage_mean"]
        if "block_td_advantage_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="^",
        label="td advantage mean",
    )
    axes[1].set_ylabel("summary")
    axes[1].set_title("Block TD Value Summary")
    axes[1].legend()
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "block_td_value_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(
        logs["epoch"],
        logs["block_advantage_entropy"]
        if "block_advantage_entropy" in logs.columns
        else np.zeros(len(logs)),
        marker="o",
        label="block advantage entropy",
    )
    axes[0].plot(
        logs["epoch"],
        logs["top_k_block_advantage_share"]
        if "top_k_block_advantage_share" in logs.columns
        else np.zeros(len(logs)),
        marker="s",
        label="top-k block advantage share",
    )
    axes[0].set_ylabel("block_advantage")
    axes[0].set_title("Block Advantage Structure")
    axes[0].legend()
    axes[1].plot(
        logs["epoch"],
        logs["block_advantage_mean"]
        if "block_advantage_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="o",
        label="block advantage mean",
    )
    axes[1].plot(
        logs["epoch"],
        logs["block_advantage_std"]
        if "block_advantage_std" in logs.columns
        else np.zeros(len(logs)),
        marker="s",
        label="block advantage std",
    )
    axes[1].plot(
        logs["epoch"],
        logs["active_block_count_mean"]
        if "active_block_count_mean" in logs.columns
        else np.zeros(len(logs)),
        marker="^",
        label="active block count",
    )
    axes[1].set_ylabel("summary")
    axes[1].set_title("Block Advantage Summary")
    axes[1].legend()
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "block_advantage_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(logs["epoch"], logs["critic_backbone_grad_norm"], marker="o", label="backbone")
    axes[0].plot(logs["epoch"], logs["critic_head_grad_norm"], marker="s", label="head")
    axes[0].set_ylabel("grad_norm")
    axes[0].set_title("Critic Grad Norm")
    axes[0].legend()
    axes[1].plot(
        logs["epoch"],
        logs["critic_hidden_feature_dim_std_mean"],
        marker="o",
        color="#2ca02c",
    )
    axes[1].set_ylabel("feature_dim_std_mean")
    axes[1].set_title("Critic Hidden Feature Dim Std Mean")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(plots_dir / "critic_grad_norm_curve.png", dpi=150)
    plt.close(fig)

    bucket_stats_path = run_dir / "advantage_bucket_update_stats.csv"
    if bucket_stats_path.exists():
        bucket_stats = pd.read_csv(bucket_stats_path)
        if not bucket_stats.empty:
            plt.figure(figsize=(9, 6))
            for bucket_name, bucket_df in bucket_stats.groupby("bucket_name", observed=True):
                plt.plot(
                    bucket_df["epoch"],
                    bucket_df["bucket_mean_delta_log_prob_selected_action"],
                    marker="o",
                    label=str(bucket_name),
                )
            plt.xlabel("epoch")
            plt.ylabel("bucket_mean_delta_log_prob_selected_action")
            plt.title("Advantage Bucket Update Compare")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "advantage_bucket_update_compare.png", dpi=150)
            plt.close()

            plt.figure(figsize=(9, 6))
            for bucket_name, bucket_df in bucket_stats.groupby("bucket_name", observed=True):
                plt.plot(
                    bucket_df["epoch"],
                    bucket_df["bucket_mean_selected_action_prob_gain"],
                    marker="o",
                    label=str(bucket_name),
                )
            plt.xlabel("epoch")
            plt.ylabel("bucket_mean_selected_action_prob_gain")
            plt.title("Selected Action Prob Gain by Advantage Bucket")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "selected_action_prob_gain_by_adv_bucket.png", dpi=150)
            plt.close()


def run_diagnosis(seed: int, output_root: Path) -> Path:
    root_dir = output_root / datetime.now().strftime("dense_critic_diagnosis_%Y%m%d_%H%M%S")
    root_dir.mkdir(parents=True, exist_ok=True)

    config = build_diagnosis_config(seed=seed)
    probe_states = collect_probe_states(config=config, seed=seed, probe_count=PROBE_STATE_COUNT)
    np.save(root_dir / "probe_states.npy", probe_states)

    early_stopping = EarlyStoppingConfig(
        patience=FIXED_PATIENCE,
        min_delta=FIXED_MIN_DELTA,
        monitor_start_epoch=FIXED_START_CHECK_EPOCH,
    )
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root_dir / f"critic_diagnosis_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "training": asdict(config.training),
                "ppo": asdict(config.ppo),
                "dt": asdict(config.dt),
                "system": asdict(config.system),
                "early_stopping": asdict(early_stopping),
                "probe_state_count": PROBE_STATE_COUNT,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[diagnose] dense critic diagnosis -> {run_dir}")
    set_global_seeds(seed)
    _, _, _, _ = train_with_diagnosis(
        config=config,
        checkpoint_dir=str(run_dir),
        early_stopping=early_stopping,
        probe_states=probe_states,
    )
    plot_single_run_outputs(run_dir)

    metrics = extract_metrics(run_dir)
    summary_df = pd.DataFrame([metrics])
    summary_csv = root_dir / "critic_diagnosis_summary.csv"
    summary_json = root_dir / "critic_diagnosis_summary.json"
    summary_md = root_dir / "critic_diagnosis_summary.md"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(dataframe_to_markdown(summary_df), encoding="utf-8")

    manifest = {
        "root_dir": str(root_dir),
        "run_dir": str(run_dir),
        "fixed_params": {
            "seed": seed,
            "topology": FIXED_TOPOLOGY,
            "reward_mode": FIXED_REWARD_MODE,
            "value_target_mode": FIXED_VALUE_TARGET_MODE,
            "value_loss_mode": FIXED_VALUE_LOSS_MODE,
        },
        "summary": {
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    (root_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root_dir


def main() -> None:
    args = parse_args()
    root_dir = run_diagnosis(seed=args.seed, output_root=Path(args.output_root))
    print(f"Dense critic diagnosis completed. Results saved to: {root_dir}")


if __name__ == "__main__":
    main()
