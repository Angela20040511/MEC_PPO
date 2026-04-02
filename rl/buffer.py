"""PPO 经验缓存。"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
import torch


@dataclass
class PPOBuffer:
    """保存一条轨迹的状态、动作和回报信息。"""

    states: list[np.ndarray] = field(default_factory=list)
    next_states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    log_prob_components: list[np.ndarray] = field(default_factory=list)
    action_means: list[np.ndarray] = field(default_factory=list)
    action_stds: list[np.ndarray] = field(default_factory=list)
    joint_reward_aligned_scores: list[np.ndarray] = field(default_factory=list)
    joint_td_aligned_scores: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        log_prob_components: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        next_state: np.ndarray | None = None,
        joint_reward_aligned_scores: np.ndarray | None = None,
        joint_td_aligned_scores: np.ndarray | None = None,
    ) -> None:
        """写入一条环境交互样本。"""
        self.states.append(state.astype(np.float32))
        self.next_states.append(
            (next_state if next_state is not None else state).astype(np.float32)
        )
        self.actions.append(action.astype(np.float32))
        self.log_probs.append(float(log_prob))
        self.log_prob_components.append(log_prob_components.astype(np.float32))
        self.action_means.append(action_mean.astype(np.float32))
        self.action_stds.append(action_std.astype(np.float32))
        if joint_reward_aligned_scores is not None:
            self.joint_reward_aligned_scores.append(
                joint_reward_aligned_scores.astype(np.float32)
            )
        if joint_td_aligned_scores is not None:
            self.joint_td_aligned_scores.append(
                joint_td_aligned_scores.astype(np.float32)
            )
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def finish_trajectory(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """在轨迹结束时用 GAE 计算优势函数和回报。"""
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)

        for index in reversed(range(len(self.rewards))):
            mask = 1.0 - float(self.dones[index])
            delta = self.rewards[index] + gamma * next_value * mask - self.values[index]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[index] = gae
            returns[index] = advantages[index] + self.values[index]
            next_value = self.values[index]

        self.advantages = advantages
        self.returns = returns

    def as_tensors(self, device: torch.device, strict: bool = True) -> dict[str, torch.Tensor]:
        """把缓存数据转成 PyTorch 张量。"""
        if self.advantages is None or self.returns is None:
            raise RuntimeError("finish_trajectory must be called before as_tensors")

        tensor_dict = {
            "states": torch.tensor(np.stack(self.states), dtype=torch.float32, device=device),
            "next_states": torch.tensor(
                np.stack(self.next_states),
                dtype=torch.float32,
                device=device,
            ),
            "actions": torch.tensor(np.stack(self.actions), dtype=torch.float32, device=device),
            "old_log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            "old_log_prob_components": torch.tensor(
                np.stack(self.log_prob_components),
                dtype=torch.float32,
                device=device,
            ),
            "old_action_means": torch.tensor(
                np.stack(self.action_means),
                dtype=torch.float32,
                device=device,
            ),
            "old_action_stds": torch.tensor(
                np.stack(self.action_stds),
                dtype=torch.float32,
                device=device,
            ),
            "advantages": torch.tensor(self.advantages, dtype=torch.float32, device=device),
            "returns": torch.tensor(self.returns, dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones, dtype=torch.float32, device=device),
        }
        reward_score_count = len(self.joint_reward_aligned_scores)
        td_score_count = len(self.joint_td_aligned_scores)
        state_count = len(self.states)
        if reward_score_count not in {0, state_count}:
            message = (
                "joint_reward_aligned_scores length mismatch: "
                f"scores={reward_score_count}, states={state_count}"
            )
            if strict:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        if reward_score_count == state_count and self.joint_reward_aligned_scores:
            tensor_dict["joint_reward_aligned_scores"] = torch.tensor(
                np.stack(self.joint_reward_aligned_scores),
                dtype=torch.float32,
                device=device,
            )
        if td_score_count not in {0, state_count}:
            message = (
                "joint_td_aligned_scores length mismatch: "
                f"scores={td_score_count}, states={state_count}"
            )
            if strict:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        if td_score_count == state_count and self.joint_td_aligned_scores:
            tensor_dict["joint_td_aligned_scores"] = torch.tensor(
                np.stack(self.joint_td_aligned_scores),
                dtype=torch.float32,
                device=device,
            )
        return tensor_dict

    def clear(self) -> None:
        """清空缓存。"""
        self.states.clear()
        self.next_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.log_prob_components.clear()
        self.action_means.clear()
        self.action_stds.clear()
        self.joint_reward_aligned_scores.clear()
        self.joint_td_aligned_scores.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        """返回当前样本数量。"""
        return len(self.states)
