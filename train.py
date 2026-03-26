"""训练与回放流程。"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from config import MECConfig, build_config, build_default_config
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator


def train_agent(
    config: MECConfig | None = None,
    checkpoint_dir: str = "checkpoints",
) -> tuple[PPOAgent, list[dict[str, float]]]:
    """训练 PPO 智能体并返回日志，同时保存 best 和 last 模型。"""
    config = config or build_default_config()
    simulator = Simulator(config)
    agent = PPOAgent(config=config.ppo, state_dim=config.state_dim, action_dim=config.action_dim)
    logs: list[dict[str, float]] = []

    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    best_reward = float("-inf")
    best_epoch = -1

    for epoch in range(config.training.num_epochs):
        state = simulator.reset(seed=config.training.seed + epoch)

        episode_reward = 0.0
        episode_cost = 0.0
        episode_queue_penalty = 0.0

        episode_delay = 0.0
        episode_local_delay_cost = 0.0
        episode_uplink_delay_cost = 0.0
        episode_backhaul_delay_cost = 0.0
        episode_bs_compute_delay_cost = 0.0

        episode_energy = 0.0
        episode_uplink_energy = 0.0
        episode_local_compute_energy = 0.0
        episode_bs_compute_energy = 0.0

        episode_normalized_avg_queue_len = 0.0
        episode_avg_theta = 0.0
        step_count = 0

        episode_reward_delay_term = 0.0
        episode_reward_energy_term = 0.0
        episode_reward_backlog_term = 0.0

        episode_raw_total_backlog = 0.0
        episode_normalized_total_backlog = 0.0

        done = False

        for _ in range(config.training.time_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, info = simulator.step(action)

            agent.store_transition(state, action, log_prob, reward, done, value)
            state = next_state

            episode_reward += reward
            episode_cost += float(info["cost"])
            episode_queue_penalty += float(info["queue_penalty"])

            episode_delay += float(info["total_delay"])
            episode_local_delay_cost += float(info["local_delay_cost"])
            episode_uplink_delay_cost += float(info["uplink_delay_cost"])
            episode_backhaul_delay_cost += float(info["backhaul_delay_cost"])
            episode_bs_compute_delay_cost += float(info["bs_compute_delay_cost"])

            episode_energy += float(info["total_energy"])
            episode_uplink_energy += float(info["uplink_energy"])
            episode_local_compute_energy += float(info["local_compute_energy"])
            episode_bs_compute_energy += float(info["bs_compute_energy"])

            episode_reward_delay_term += float(info["reward_delay_term"])
            episode_reward_energy_term += float(info["reward_energy_term"])
            episode_reward_backlog_term += float(info["reward_backlog_term"])

            episode_normalized_avg_queue_len += float(info["total_backlog"])
            episode_raw_total_backlog += float(info["raw_total_backlog"])
            episode_normalized_total_backlog += float(info["normalized_total_backlog"])
            episode_avg_theta += float(info["avg_theta"])

            step_count += 1

            if done:
                break

        last_value = 0.0 if done else agent.evaluate_value(state)
        agent.finish_trajectory(last_value)
        losses = agent.train()

        avg_normalized_avg_queue_len = episode_normalized_avg_queue_len / max(step_count, 1)
        avg_theta = episode_avg_theta / max(step_count, 1)

        log = {
            "epoch": float(epoch),
            "episode_reward": episode_reward,
            "episode_cost": episode_cost,
            "episode_queue_penalty": episode_queue_penalty,
            "episode_delay": episode_delay,
            "episode_local_delay_cost": episode_local_delay_cost,
            "episode_uplink_delay_cost": episode_uplink_delay_cost,
            "episode_backhaul_delay_cost": episode_backhaul_delay_cost,
            "episode_bs_compute_delay_cost": episode_bs_compute_delay_cost,
            "episode_energy": episode_energy,
            "episode_uplink_energy": episode_uplink_energy,
            "episode_local_compute_energy": episode_local_compute_energy,
            "episode_bs_compute_energy": episode_bs_compute_energy,
            "episode_reward_delay_term": episode_reward_delay_term,
            "episode_reward_energy_term": episode_reward_energy_term,
            "episode_reward_backlog_term": episode_reward_backlog_term,
            "avg_normalized_avg_queue_len": avg_normalized_avg_queue_len,
            "avg_raw_total_backlog": episode_raw_total_backlog / max(step_count, 1),
            "avg_normalized_total_backlog": episode_normalized_total_backlog / max(step_count, 1),
            "avg_theta": avg_theta,
            "num_steps": float(step_count),
            "actor_loss": losses["actor_loss"],
            "critic_loss": losses["critic_loss"],
            "entropy": losses["entropy"],
        }
        logs.append(log)

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_epoch = epoch
            agent.save(str(checkpoint_root / "best_model.pt"))
            with (checkpoint_root / "best_model_info.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_reward": best_reward,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    agent.save(str(checkpoint_root / "last_model.pt"))

    with (checkpoint_root / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_epochs": config.training.num_epochs,
                "best_epoch": best_epoch,
                "best_reward": best_reward,
                "final_epoch": len(logs) - 1 if logs else -1,
                "final_reward": logs[-1]["episode_reward"] if logs else None,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (checkpoint_root / "train_logs.json").open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    if logs:
        fieldnames = list(logs[0].keys())
        with (checkpoint_root / "train_logs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)

    return agent, logs


def rollout_agent(
    agent: PPOAgent,
    config: MECConfig | None = None,
    steps: int | None = None,
    deterministic: bool = True,
) -> list[dict[str, Any]]:
    """用训练好的策略执行一次回放。"""
    config = config or build_default_config()
    simulator = Simulator(config)
    state = simulator.reset(seed=config.training.seed)
    rollout_steps = steps or config.training.time_steps
    results: list[dict[str, Any]] = []

    for step in range(rollout_steps):
        action, _, _ = agent.select_action(state, deterministic=deterministic)
        state, reward, done, info = simulator.step(action)
        results.append(
            {
                "step": step,
                "reward": reward,
                "total_delay": info["total_delay"],
                "total_energy": info["total_energy"],
                "queue_penalty": info["queue_penalty"],
            }
        )
        if done:
            break
    return results


def build_runtime_config(
    num_epochs: int | None = None,
    time_steps: int | None = None,
    topology_mode: str = "default",
) -> MECConfig:
    """按运行参数覆盖默认训练配置。"""
    config = build_config(topology_mode=topology_mode)
    training = config.training
    if num_epochs is not None:
        training = replace(training, num_epochs=num_epochs)
    if time_steps is not None:
        training = replace(training, time_steps=time_steps)
    return replace(config, training=training)


if __name__ == "__main__":
    runtime_config = build_runtime_config()
    _, training_logs = train_agent(runtime_config)
    for log in training_logs:
        print(log)
