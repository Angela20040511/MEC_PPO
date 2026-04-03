"""PPO agent implementation."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from config import PPOConfig
from rl.buffer import PPOBuffer
from rl.network import ActorCritic


class CriticOptimizerAblationCaptured(RuntimeError):
    """Raised to stop training after capturing a pre-step critic ablation snapshot."""


@dataclass
class PPOAgent:
    """PPO agent with separate actor and critic optimizers."""

    config: PPOConfig
    state_dim: int
    action_dim: int
    critic_state_layout: dict[str, int] | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    network: ActorCritic = field(init=False)
    actor_optimizer: torch.optim.Optimizer = field(init=False)
    theta_actor_optimizer: torch.optim.Optimizer | None = field(init=False, default=None)
    route_actor_optimizer: torch.optim.Optimizer | None = field(init=False, default=None)
    critic_optimizer: torch.optim.Optimizer = field(init=False)
    actor_params: list[nn.Parameter] = field(init=False, repr=False)
    theta_actor_params: list[nn.Parameter] = field(init=False, repr=False)
    route_actor_params: list[nn.Parameter] = field(init=False, repr=False)
    critic_params: list[nn.Parameter] = field(init=False, repr=False)
    buffer: PPOBuffer = field(default_factory=PPOBuffer)
    running_return_mean: float = field(init=False, default=0.0)
    running_return_var: float = field(init=False, default=1.0)
    running_return_count: float = field(init=False, default=0.0)
    actor_input_running_mean: np.ndarray | None = field(init=False, default=None, repr=False)
    actor_input_running_var: np.ndarray | None = field(init=False, default=None, repr=False)
    actor_input_count: float = field(init=False, default=0.0)
    critic_input_running_mean: np.ndarray | None = field(init=False, default=None, repr=False)
    critic_input_running_var: np.ndarray | None = field(init=False, default=None, repr=False)
    critic_input_count: float = field(init=False, default=0.0)
    actor_raw_dim: int = field(init=False, default=0)
    actor_input_dim: int = field(init=False, default=0)
    critic_input_dim: int = field(init=False, default=0)
    action_block_count: int = field(init=False, default=1)
    action_path_count: int = field(init=False, default=1)
    actor_derived_feature_dim: int = field(init=False, default=0)
    critic_derived_feature_dim: int = field(init=False, default=0)
    policy_update_call_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Create the network and optimizers."""
        self.actor_derived_feature_dim = self._actor_derived_feature_dim()
        self.critic_derived_feature_dim = self._critic_derived_feature_dim()
        self.actor_raw_dim = self._actor_raw_input_dim()
        self.actor_input_dim = self.actor_raw_dim + self.actor_derived_feature_dim
        self.critic_input_dim = self.state_dim + self.critic_derived_feature_dim
        self.action_block_count = len(self._action_block_slices())
        self.action_path_count = self._action_path_count()
        self.network = ActorCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_size=self.config.hidden_size,
            initial_log_std=self.config.actor_log_std,
            actor_structure_mode=self._actor_structure_mode(),
            critic_arch_mode=self.config.critic_arch_mode,
            actor_input_dim=self.actor_input_dim,
            critic_input_dim=self.critic_input_dim,
            actor_block_count=self.action_block_count,
            actor_path_count=self.action_path_count,
            critic_block_value_count=self.action_block_count,
            critic_block_path_count=self.action_path_count,
        ).to(self.device)
        self.actor_params = (
            self.network.actor_backbone_parameters()
            + self.network.actor_main_head_parameters()
            + self.network.actor_route_head_parameters()
            + [self.network.actor_log_std]
        )
        self.theta_actor_params = (
            self.network.actor_theta_backbone_parameters()
            + self.network.actor_theta_head_parameters()
            + [self.network.actor_log_std]
        )
        self.route_actor_params = (
            self.network.actor_route_backbone_parameters()
            + self.network.actor_route_head_parameters()
            + [self.network.actor_log_std]
        )
        self.critic_params = (
            list(self.network.critic_backbone.parameters())
            + list(self.network.critic_head.parameters())
            + list(self.network.critic_block_value_head.parameters())
            + list(self.network.critic_block_path_value_head.parameters())
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor_params,
            lr=self.config.actor_learning_rate,
        )
        if self._uses_dual_optimizer_alternating_branch_pg():
            self.theta_actor_optimizer = torch.optim.Adam(
                self.theta_actor_params,
                lr=self.config.actor_learning_rate,
            )
            self.route_actor_optimizer = torch.optim.Adam(
                self.route_actor_params,
                lr=self.config.actor_learning_rate,
            )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_params,
            lr=self.config.critic_learning_rate,
        )

    def _uses_normalized_augmented_critic_input(self) -> bool:
        """Return whether critic-only input processing is enabled."""
        return self.config.critic_input_mode == "normalized_augmented_critic_input"

    def _uses_normalized_augmented_actor_input(self) -> bool:
        """Return whether actor-only input processing is enabled."""
        return self.config.actor_input_mode == "normalized_augmented_actor_input"

    def _actor_raw_input_mode(self) -> str:
        """Return the current actor raw-state composition mode."""
        return self.config.actor_raw_input_mode

    def _actor_structure_mode(self) -> str:
        """Return the configured actor output structure mode."""
        return getattr(self.config, "actor_structure_mode", "flat_joint_actor")

    def _policy_ratio_mode(self) -> str:
        """Return the canonical PPO policy-ratio aggregation mode (alias modes map to base logic)."""
        alias_to_base_mode = {
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate": (
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate"
            ),
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1": (
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate"
            ),
            "hierarchical_actor_route_step_alignment_gate": (
                "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate"
            ),
        }
        return alias_to_base_mode.get(self.config.policy_ratio_mode, self.config.policy_ratio_mode)

    def _uses_blockwise_surrogate_mean(self) -> bool:
        """Return whether actor updates should use blockwise PPO surrogates."""
        return self._policy_ratio_mode() == "blockwise_surrogate_mean"

    def _uses_blockwise_weighted_surrogate_mean(self) -> bool:
        """Return whether actor updates use blockwise surrogates with activity weights."""
        return self._policy_ratio_mode() == "blockwise_weighted_surrogate_mean"

    def _uses_blockwise_scaled_advantage_surrogate_mean(self) -> bool:
        """Return whether actor updates use blockwise surrogates with block-scaled advantages."""
        return self._policy_ratio_mode() == "blockwise_scaled_advantage_surrogate_mean"

    def _uses_blockwise_value_scaled_advantage_surrogate_mean(self) -> bool:
        """Return whether actor updates use learned block-value scales on top of global A."""
        return self._policy_ratio_mode() == "blockwise_value_scaled_advantage_surrogate_mean"

    def _uses_blockwise_td_advantage_surrogate_mean(self) -> bool:
        """Return whether actor updates use block-specific TD-style advantages."""
        return self._policy_ratio_mode() == "blockwise_td_advantage_surrogate_mean"

    def _uses_blockwise_delta_cost_td_advantage_surrogate_mean(self) -> bool:
        """Return whether actor updates use delta-cost block-specific TD advantages."""
        return self._policy_ratio_mode() == "blockwise_delta_cost_td_advantage_surrogate_mean"

    def _uses_blockwise_action_conditioned_td_advantage_surrogate_mean(self) -> bool:
        """Return whether actor updates use action-conditioned TD-style block advantages."""
        return self._policy_ratio_mode() == "blockwise_action_conditioned_td_advantage_surrogate_mean"

    def _uses_blockwise_action_conditioned_path_value_td_advantage_surrogate_mean(self) -> bool:
        """Return whether actor updates use path-specific block values inside action-conditioned TD."""
        return (
            self._policy_ratio_mode()
            == "blockwise_action_conditioned_path_value_td_advantage_surrogate_mean"
        )

    def _uses_blockwise_path_specific_td_supervised_advantage_surrogate_mean(self) -> bool:
        """Return whether path-specific block values receive direct path TD supervision."""
        return (
            self._policy_ratio_mode()
            == "blockwise_path_specific_td_supervised_advantage_surrogate_mean"
        )

    def _uses_theta_route_split_advantage_surrogate(self) -> bool:
        """Return whether block-internal actor credit is split into theta vs route sub-advantages."""
        return self._policy_ratio_mode() in {
            "theta_route_split_advantage_surrogate",
            "hierarchical_actor_theta_route",
            "hierarchical_actor_offload_gated_route",
            "hierarchical_actor_hard_offload_route",
            "hierarchical_actor_separate_theta_route_backbones",
            "hierarchical_actor_branchwise_balanced_pg",
            "hierarchical_actor_alternating_branch_pg",
            "hierarchical_actor_dual_optimizer_alternating_branch_pg",
            "hierarchical_actor_factorized_ratio_pg",
            "hierarchical_actor_factorized_conditional_route_pg",
            "hierarchical_actor_true_conditional_route_policy",
            "hierarchical_actor_route_credit_theta_gate",
            "hierarchical_actor_route_credit_theta_soft_weight",
            "hierarchical_actor_route_residual_credit_vectorized",
            "hierarchical_actor_true_conditional_route_candidate_score_credit",
            "hierarchical_actor_theta_candidate_score_credit",
            "hierarchical_actor_theta_route_candidate_score_credit",
            "hierarchical_actor_joint_reward_aligned_credit",
            "hierarchical_actor_joint_td_aligned_credit",
            "hierarchical_actor_factorized_trust_region_pg",
            "hierarchical_actor_factorized_trust_region_pg_route_first",
            "hierarchical_actor_factorized_trust_region_pg_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
        }

    def _uses_hierarchical_actor_theta_route(self) -> bool:
        """Return whether the actor output structure is explicitly hierarchical."""
        return self._actor_structure_mode() in {
            "hierarchical_actor_theta_route",
            "hierarchical_actor_separate_theta_route_backbones",
        }

    def _uses_separate_theta_route_backbones(self) -> bool:
        """Return whether theta and route decisions use disjoint actor backbones."""
        return self._actor_structure_mode() == "hierarchical_actor_separate_theta_route_backbones"

    def _uses_branchwise_balanced_pg(self) -> bool:
        """Return whether theta and route policy gradients use branchwise normalization and balancing."""
        return self._policy_ratio_mode() in {
            "hierarchical_actor_branchwise_balanced_pg",
            "hierarchical_actor_alternating_branch_pg",
            "hierarchical_actor_dual_optimizer_alternating_branch_pg",
            "hierarchical_actor_factorized_ratio_pg",
            "hierarchical_actor_factorized_conditional_route_pg",
            "hierarchical_actor_true_conditional_route_policy",
            "hierarchical_actor_route_credit_theta_gate",
            "hierarchical_actor_route_credit_theta_soft_weight",
            "hierarchical_actor_route_residual_credit_vectorized",
            "hierarchical_actor_true_conditional_route_candidate_score_credit",
            "hierarchical_actor_theta_candidate_score_credit",
            "hierarchical_actor_theta_route_candidate_score_credit",
            "hierarchical_actor_joint_reward_aligned_credit",
            "hierarchical_actor_joint_td_aligned_credit",
            "hierarchical_actor_factorized_trust_region_pg",
            "hierarchical_actor_factorized_trust_region_pg_route_first",
            "hierarchical_actor_factorized_trust_region_pg_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
        }

    def _uses_alternating_branch_pg(self) -> bool:
        """Return whether theta and route branches update in separate optimizer steps."""
        return self._policy_ratio_mode() in {
            "hierarchical_actor_alternating_branch_pg",
            "hierarchical_actor_dual_optimizer_alternating_branch_pg",
            "hierarchical_actor_factorized_ratio_pg",
            "hierarchical_actor_factorized_conditional_route_pg",
            "hierarchical_actor_true_conditional_route_policy",
            "hierarchical_actor_route_credit_theta_gate",
            "hierarchical_actor_route_credit_theta_soft_weight",
            "hierarchical_actor_route_residual_credit_vectorized",
            "hierarchical_actor_true_conditional_route_candidate_score_credit",
            "hierarchical_actor_theta_candidate_score_credit",
            "hierarchical_actor_theta_route_candidate_score_credit",
            "hierarchical_actor_joint_reward_aligned_credit",
            "hierarchical_actor_joint_td_aligned_credit",
            "hierarchical_actor_factorized_trust_region_pg",
            "hierarchical_actor_factorized_trust_region_pg_route_first",
            "hierarchical_actor_factorized_trust_region_pg_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
        }

    def _uses_dual_optimizer_alternating_branch_pg(self) -> bool:
        """Return whether alternating branch updates use separate actor optimizer states."""
        return (
            self._policy_ratio_mode()
            == "hierarchical_actor_dual_optimizer_alternating_branch_pg"
        )

    def _uses_factorized_ratio_pg(self) -> bool:
        """Return whether branch summaries and objective diagnostics are fully factorized."""
        return self._policy_ratio_mode() in {
            "hierarchical_actor_factorized_ratio_pg",
            "hierarchical_actor_factorized_trust_region_pg",
            "hierarchical_actor_factorized_trust_region_pg_route_first",
            "hierarchical_actor_factorized_trust_region_pg_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
            "hierarchical_actor_factorized_conditional_route_pg",
            "hierarchical_actor_true_conditional_route_policy",
            "hierarchical_actor_route_credit_theta_gate",
            "hierarchical_actor_route_credit_theta_soft_weight",
            "hierarchical_actor_route_residual_credit_vectorized",
            "hierarchical_actor_true_conditional_route_candidate_score_credit",
            "hierarchical_actor_theta_candidate_score_credit",
            "hierarchical_actor_theta_route_candidate_score_credit",
            "hierarchical_actor_joint_reward_aligned_credit",
            "hierarchical_actor_joint_td_aligned_credit",
        }

    def _uses_factorized_conditional_route_pg(self) -> bool:
        """Return whether route branch uses semantic conditional route-margin log-probs."""
        return self._policy_ratio_mode() == "hierarchical_actor_factorized_conditional_route_pg"

    def _uses_true_conditional_route_policy(self) -> bool:
        """Return whether actor bookkeeping uses a true hierarchical conditional route policy."""
        return self._policy_ratio_mode() in {
            "hierarchical_actor_true_conditional_route_policy",
            "hierarchical_actor_route_credit_theta_gate",
            "hierarchical_actor_route_credit_theta_soft_weight",
            "hierarchical_actor_route_residual_credit_vectorized",
            "hierarchical_actor_true_conditional_route_candidate_score_credit",
            "hierarchical_actor_theta_candidate_score_credit",
            "hierarchical_actor_theta_route_candidate_score_credit",
            "hierarchical_actor_joint_reward_aligned_credit",
            "hierarchical_actor_joint_td_aligned_credit",
        }

    def _uses_route_credit_theta_gate(self) -> bool:
        """Return whether route credit is further gated by positive theta credit."""
        return self._policy_ratio_mode() == "hierarchical_actor_route_credit_theta_gate"

    def _uses_route_credit_theta_soft_weight(self) -> bool:
        """Return whether route credit is reweighted by normalized positive theta credit."""
        return self._policy_ratio_mode() == "hierarchical_actor_route_credit_theta_soft_weight"

    def _uses_route_residual_credit_vectorized(self) -> bool:
        """Return whether route credit uses vectorized selected-minus-expected residual scores."""
        return self._policy_ratio_mode() == "hierarchical_actor_route_residual_credit_vectorized"

    def _uses_true_conditional_route_candidate_score_credit(self) -> bool:
        """Return whether route credit uses candidate-score residuals from existing path scores."""
        return self._policy_ratio_mode() in {
            "hierarchical_actor_true_conditional_route_candidate_score_credit",
            "hierarchical_actor_theta_route_candidate_score_credit",
        }

    def _uses_theta_candidate_score_credit(self) -> bool:
        """Return whether theta credit uses local-vs-best-offload candidate-score residuals."""
        return self._policy_ratio_mode() in {
            "hierarchical_actor_theta_candidate_score_credit",
            "hierarchical_actor_theta_route_candidate_score_credit",
        }

    def _uses_joint_reward_aligned_credit(self) -> bool:
        """Return whether actor training uses reward-aligned joint 3-action residual credit."""
        return self._policy_ratio_mode() == "hierarchical_actor_joint_reward_aligned_credit"

    def _uses_joint_td_aligned_credit(self) -> bool:
        """Return whether actor training uses TD-aligned joint 3-action residual credit."""
        return self._policy_ratio_mode() == "hierarchical_actor_joint_td_aligned_credit"

    def _uses_joint_counterfactual_credit(self) -> bool:
        """Return whether actor training uses a joint 3-action counterfactual credit."""
        return self._uses_joint_reward_aligned_credit() or self._uses_joint_td_aligned_credit()

    def _uses_factorized_trust_region_pg(self) -> bool:
        """Return whether branch-specific KL early-stop control is enabled."""
        return self._policy_ratio_mode() in {
            "hierarchical_actor_factorized_trust_region_pg",
            "hierarchical_actor_factorized_trust_region_pg_route_first",
            "hierarchical_actor_factorized_trust_region_pg_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
        }

    def _uses_route_first_factorized_trust_region_pg(self) -> bool:
        """Return whether trust-region factorized PG uses route-first alternating order."""
        return self._policy_ratio_mode() in {
            "hierarchical_actor_factorized_trust_region_pg_route_first",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
        }

    def _uses_coupled_stop_factorized_trust_region_pg(self) -> bool:
        """Return whether route KL early-stop also freezes theta updates in the same PPO epoch."""
        return self._policy_ratio_mode() in {
            "hierarchical_actor_factorized_trust_region_pg_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
        }

    def _uses_theta_floor_coupled_stop_factorized_trust_region_pg(self) -> bool:
        """Return whether coupled-stop is guarded by a minimum theta-update floor per PPO epoch."""
        return (
            self._policy_ratio_mode()
            == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor"
        )

    def _uses_severity_gate_coupled_stop_factorized_trust_region_pg(self) -> bool:
        """Return whether coupled-stop theta-freeze requires route KL to exceed a severity gate."""
        return (
            self._policy_ratio_mode()
            == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate"
        )

    def _uses_severity_alignment_gate_coupled_stop_factorized_trust_region_pg(self) -> bool:
        """Return whether coupled-stop additionally requires low route-step alignment before freezing theta."""
        return (
            self.config.policy_ratio_mode
            == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_alignment_gate"
        )

    def _uses_warmup1_severity_gate_coupled_stop_factorized_trust_region_pg(self) -> bool:
        """Return whether coupled-stop severity gate is delayed by one outer training epoch."""
        return (
            self.config.policy_ratio_mode
            == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate_warmup1"
        )

    def _uses_route_confident_mask_factorized_trust_region_pg(self) -> bool:
        """Return whether route updates require higher-confidence theta routing masks."""
        return (
            self._policy_ratio_mode()
            == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask"
        )

    def _uses_route_update_cap_factorized_trust_region_pg(self) -> bool:
        """Return whether route updates are capped per PPO epoch under alternating branch updates."""
        return (
            self._policy_ratio_mode()
            == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap"
        )

    def _uses_epochwise_branch_order_factorized_trust_region_pg(self) -> bool:
        """Return whether alternating branch updates are scheduled by PPO epoch phases."""
        return (
            self._policy_ratio_mode()
            == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise"
        )

    def _uses_route_counterfactual_adv_factorized_trust_region_pg(self) -> bool:
        """Return whether route branch uses counterfactual route advantage signals."""
        return (
            self._policy_ratio_mode()
            == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv"
        )

    def _uses_route_alignment_gate_factorized_trust_region_pg(self) -> bool:
        """Return whether route updates are accepted only when step-level alignment meets a threshold."""
        return self.config.policy_ratio_mode in {
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
            "hierarchical_actor_route_step_alignment_gate",
        }

    def _uses_offload_gated_route_surrogate(self) -> bool:
        """Return whether route updates are weighted by detached offload relevance."""
        return self._policy_ratio_mode() == "hierarchical_actor_offload_gated_route"

    def _uses_hard_offload_route_surrogate(self) -> bool:
        """Return whether route updates are restricted to detached hard offload masks."""
        return self._policy_ratio_mode() in {
            "hierarchical_actor_hard_offload_route",
            "hierarchical_actor_separate_theta_route_backbones",
            "hierarchical_actor_branchwise_balanced_pg",
            "hierarchical_actor_alternating_branch_pg",
            "hierarchical_actor_dual_optimizer_alternating_branch_pg",
            "hierarchical_actor_factorized_ratio_pg",
            "hierarchical_actor_factorized_conditional_route_pg",
            "hierarchical_actor_true_conditional_route_policy",
            "hierarchical_actor_route_credit_theta_gate",
            "hierarchical_actor_route_credit_theta_soft_weight",
            "hierarchical_actor_route_residual_credit_vectorized",
            "hierarchical_actor_true_conditional_route_candidate_score_credit",
            "hierarchical_actor_theta_candidate_score_credit",
            "hierarchical_actor_theta_route_candidate_score_credit",
            "hierarchical_actor_joint_reward_aligned_credit",
            "hierarchical_actor_factorized_trust_region_pg",
            "hierarchical_actor_factorized_trust_region_pg_route_first",
            "hierarchical_actor_factorized_trust_region_pg_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
        }

    def _route_mask_threshold(self) -> float:
        """Return the hard offload threshold used for route conditioning."""
        base_threshold = 0.5
        if not self._uses_route_confident_mask_factorized_trust_region_pg():
            return base_threshold
        margin = float(max(0.0, getattr(self.config, "route_confident_mask_margin", 0.0)))
        return min(0.999, base_threshold + margin)

    def _uses_blockwise_action_conditioned_td_reward(self) -> bool:
        """Return whether block TD immediate rewards use action-conditioned path costs."""
        return (
            self._uses_blockwise_action_conditioned_td_advantage_surrogate_mean()
            or self._uses_blockwise_action_conditioned_path_value_td_advantage_surrogate_mean()
            or self._uses_blockwise_path_specific_td_supervised_advantage_surrogate_mean()
            or self._uses_theta_route_split_advantage_surrogate()
        )

    def _uses_blockwise_action_conditioned_path_value_bootstrap(self) -> bool:
        """Return whether block TD bootstrap values aggregate path-specific block-path heads."""
        return (
            self._uses_blockwise_action_conditioned_path_value_td_advantage_surrogate_mean()
            or self._uses_blockwise_path_specific_td_supervised_advantage_surrogate_mean()
            or self._uses_theta_route_split_advantage_surrogate()
        )

    def _uses_blockwise_pseudolocal_advantage_surrogate_mean(self) -> bool:
        """Return whether actor updates use pseudo-local block-specific advantages."""
        return self._policy_ratio_mode() == "blockwise_pseudolocal_advantage_surrogate_mean"

    def _uses_blockwise_td_style_advantage_surrogate_mean(self) -> bool:
        """Return whether actor updates use a TD-style block advantage definition."""
        return (
            self._uses_blockwise_td_advantage_surrogate_mean()
            or self._uses_blockwise_delta_cost_td_advantage_surrogate_mean()
            or self._uses_blockwise_action_conditioned_td_advantage_surrogate_mean()
            or self._uses_blockwise_action_conditioned_path_value_td_advantage_surrogate_mean()
            or self._uses_blockwise_path_specific_td_supervised_advantage_surrogate_mean()
            or self._uses_theta_route_split_advantage_surrogate()
        )

    def _uses_blockwise_policy_surrogate(self) -> bool:
        """Return whether actor updates are aggregated at the semantic action-block level."""
        return self._policy_ratio_mode() in {
            "blockwise_surrogate_mean",
            "blockwise_weighted_surrogate_mean",
            "blockwise_scaled_advantage_surrogate_mean",
            "blockwise_value_scaled_advantage_surrogate_mean",
            "blockwise_td_advantage_surrogate_mean",
            "blockwise_delta_cost_td_advantage_surrogate_mean",
            "blockwise_action_conditioned_td_advantage_surrogate_mean",
            "blockwise_action_conditioned_path_value_td_advantage_surrogate_mean",
            "blockwise_path_specific_td_supervised_advantage_surrogate_mean",
            "theta_route_split_advantage_surrogate",
            "hierarchical_actor_theta_route",
            "hierarchical_actor_offload_gated_route",
            "hierarchical_actor_hard_offload_route",
            "hierarchical_actor_separate_theta_route_backbones",
            "hierarchical_actor_branchwise_balanced_pg",
            "hierarchical_actor_alternating_branch_pg",
            "hierarchical_actor_dual_optimizer_alternating_branch_pg",
            "hierarchical_actor_factorized_ratio_pg",
            "hierarchical_actor_factorized_conditional_route_pg",
            "hierarchical_actor_true_conditional_route_policy",
            "hierarchical_actor_true_conditional_route_candidate_score_credit",
            "hierarchical_actor_theta_candidate_score_credit",
            "hierarchical_actor_theta_route_candidate_score_credit",
            "hierarchical_actor_joint_reward_aligned_credit",
            "hierarchical_actor_factorized_trust_region_pg",
            "hierarchical_actor_factorized_trust_region_pg_route_first",
            "hierarchical_actor_factorized_trust_region_pg_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv",
            "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate",
            "blockwise_pseudolocal_advantage_surrogate_mean",
        }

    def _action_path_count(self) -> int:
        """Return the number of semantic paths per block: local plus each base station route."""
        if self.critic_state_layout is None:
            return 1
        return max(1, 1 + int(self.critic_state_layout.get("base_station_count", 0)))

    def _action_block_slices(self) -> list[slice]:
        """Return semantic action blocks used by block-level ratio aggregation."""
        if self.critic_state_layout is None:
            return [slice(0, self.action_dim)]

        base_station_count = int(self.critic_state_layout.get("base_station_count", 0))
        sensor_task_count = int(self.critic_state_layout.get("sensor_task_count", 0))
        block_size = 1 + max(base_station_count, 0)
        expected_dim = sensor_task_count * block_size
        if block_size <= 0 or sensor_task_count <= 0 or expected_dim != self.action_dim:
            return [slice(0, self.action_dim)]

        return [
            slice(block_index * block_size, (block_index + 1) * block_size)
            for block_index in range(sensor_task_count)
        ]

    def describe_action_block_slices(self) -> list[dict[str, int | str]]:
        """Describe action blocks for experiment logging and diagnosis summaries."""
        block_slices = self._action_block_slices()
        has_dispatch_blocks = len(block_slices) > 1
        return [
            {
                "block_index": int(block_index),
                "start": int(block_slice.start),
                "stop_exclusive": int(block_slice.stop),
                "size": int(block_slice.stop - block_slice.start),
                "semantic": (
                    "dispatch_decision_block" if has_dispatch_blocks else "full_action_vector"
                ),
            }
            for block_index, block_slice in enumerate(block_slices)
        ]

    def describe_actor_structure(self) -> dict[str, object]:
        """Describe the active actor structure for experiment metadata."""
        description = dict(self.network.describe_actor_structure())
        description["action_block_count"] = int(self.action_block_count)
        description["action_path_count"] = int(self.action_path_count)
        description["action_dim"] = int(self.action_dim)
        return description

    def _block_sum_log_prob_components(self, log_prob_components: torch.Tensor) -> torch.Tensor:
        """Sum per-dimension log-prob terms inside each semantic action block."""
        block_slices = self._action_block_slices()
        return torch.stack(
            [log_prob_components[..., block_slice].sum(dim=-1) for block_slice in block_slices],
            dim=-1,
        )

    def _block_theta_route_log_prob_components(
        self,
        log_prob_components: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split block log-prob terms into theta-only and route-only parts."""
        block_slices = self._action_block_slices()
        theta_components = []
        route_components = []
        for block_slice in block_slices:
            block_terms = log_prob_components[..., block_slice]
            theta_components.append(block_terms[..., 0])
            route_components.append(block_terms[..., 1:].sum(dim=-1))
        return torch.stack(theta_components, dim=-1), torch.stack(route_components, dim=-1)

    def _block_route_tensors(self, flat_tensor: torch.Tensor) -> torch.Tensor:
        """Reshape flat per-action tensors into per-block route tensors."""
        block_slices = self._action_block_slices()
        return torch.stack([flat_tensor[..., block_slice][..., 1:] for block_slice in block_slices], dim=-2)

    def _route_margin_distribution_params(
        self,
        action_means: torch.Tensor,
        action_stds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build semantic route-margin Gaussian parameters from route-logit coordinates."""
        route_means = self._block_route_tensors(action_means)
        route_stds = self._block_route_tensors(action_stds)
        if route_means.shape[-1] != 2:
            raise ValueError(
                "hierarchical_actor_factorized_conditional_route_pg requires exactly 2 route logits per block"
            )
        margin_mean = route_means[..., 0] - route_means[..., 1]
        margin_std = torch.sqrt(route_stds[..., 0].square() + route_stds[..., 1].square() + 1e-8)
        return margin_mean, margin_std

    def _route_margin_log_probs(
        self,
        actions: torch.Tensor,
        action_means: torch.Tensor,
        action_stds: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate semantic route log-probs from the BS1-vs-BS2 logit margin."""
        route_actions = self._block_route_tensors(actions)
        if route_actions.shape[-1] != 2:
            raise ValueError(
                "hierarchical_actor_factorized_conditional_route_pg requires exactly 2 route logits per block"
            )
        margin_actions = route_actions[..., 0] - route_actions[..., 1]
        margin_mean, margin_std = self._route_margin_distribution_params(
            action_means,
            action_stds,
        )
        return torch.distributions.Normal(margin_mean, margin_std).log_prob(margin_actions)

    def _block_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Reshape flat actions into semantic per-block tensors."""
        return torch.stack(
            [actions[..., block_slice] for block_slice in self._action_block_slices()],
            dim=1,
        )

    def _conditional_route_active_mask(self, actions: torch.Tensor) -> torch.Tensor:
        """Return the hard offload-active mask used by the true conditional route policy."""
        theta, _ = self._block_theta_and_route_probabilities(actions)
        return theta > self._route_mask_threshold()

    def _true_conditional_theta_policy_terms(
        self,
        actions: torch.Tensor,
        action_means: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build Bernoulli local-vs-offload policy terms from the theta branch mean logits."""
        block_actions = self._block_actions(actions)
        block_action_means = self._block_actions(action_means)
        theta_logits = block_action_means[..., 0]
        offload_active_mask = self._conditional_route_active_mask(actions)
        offload_decisions = offload_active_mask.to(dtype=theta_logits.dtype)
        theta_dist = torch.distributions.Bernoulli(logits=theta_logits)
        theta_log_probs = theta_dist.log_prob(offload_decisions)
        offload_probs = torch.sigmoid(theta_logits)
        local_probs = 1.0 - offload_probs
        selected_probs = torch.where(offload_active_mask, offload_probs, 1.0 - offload_probs)
        return {
            "log_probs": theta_log_probs,
            "selected_probs": selected_probs,
            "offload_probs": offload_probs,
            "local_probs": local_probs,
            "entropy": theta_dist.entropy(),
            "offload_active_mask": offload_active_mask,
        }

    def _true_conditional_route_policy_terms(
        self,
        actions: torch.Tensor,
        action_means: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build conditional categorical BS-routing policy terms on active offload samples."""
        route_logits = self._block_route_tensors(action_means)
        route_actions = self._block_route_tensors(actions)
        if route_logits.shape[-1] <= 0:
            raise ValueError("true conditional route policy requires at least one route logit")
        route_dist = torch.distributions.Categorical(logits=route_logits)
        selected_route_indices = route_actions.argmax(dim=-1)
        route_log_probs = route_dist.log_prob(selected_route_indices)
        route_probs = route_dist.probs
        route_selected_probs = route_dist.probs.gather(
            dim=-1,
            index=selected_route_indices.unsqueeze(-1),
        ).squeeze(-1)
        route_active_mask = self._conditional_route_active_mask(actions)
        return {
            "log_probs": route_log_probs,
            "probs": route_probs,
            "selected_probs": route_selected_probs,
            "selected_indices": selected_route_indices,
            "entropy": route_dist.entropy(),
            "active_mask": route_active_mask,
        }

    def _true_conditional_joint_policy_terms(
        self,
        actions: torch.Tensor,
        action_means: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build full 3-action joint policy terms from the hierarchical theta/route policy."""
        theta_terms = self._true_conditional_theta_policy_terms(actions, action_means)
        route_terms = self._true_conditional_route_policy_terms(actions, action_means)
        route_active_mask = theta_terms["offload_active_mask"]
        joint_log_probs = theta_terms["log_probs"] + route_active_mask.to(
            dtype=theta_terms["log_probs"].dtype
        ) * route_terms["log_probs"]
        joint_probs = torch.stack(
            [
                theta_terms["local_probs"],
                theta_terms["offload_probs"] * route_terms["probs"][..., 0],
                theta_terms["offload_probs"] * route_terms["probs"][..., 1],
            ],
            dim=-1,
        )
        selected_joint_indices = torch.where(
            route_active_mask,
            route_terms["selected_indices"] + 1,
            torch.zeros_like(route_terms["selected_indices"]),
        )
        selected_joint_probs = joint_probs.gather(
            dim=-1,
            index=selected_joint_indices.unsqueeze(-1),
        ).squeeze(-1)
        return {
            "log_probs": joint_log_probs,
            "probs": joint_probs,
            "selected_probs": selected_joint_probs,
            "selected_indices": selected_joint_indices,
            "offload_active_mask": route_active_mask,
            "theta_terms": theta_terms,
            "route_terms": route_terms,
        }

    def _block_split_log_prob_components(self, log_prob_components: torch.Tensor) -> torch.Tensor:
        """Build the equal-weight theta/route block log-prob summary used by split-advantage mode."""
        theta_components, route_components = self._block_theta_route_log_prob_components(
            log_prob_components
        )
        return 0.5 * (theta_components + route_components)

    def _theta_action_indices(self) -> list[int]:
        """Return the raw action indices corresponding to theta dimensions."""
        return [int(block_slice.start) for block_slice in self._action_block_slices()]

    def _route_action_indices(self) -> list[int]:
        """Return the raw action indices corresponding to route-logit dimensions."""
        route_indices: list[int] = []
        for block_slice in self._action_block_slices():
            route_indices.extend(range(int(block_slice.start) + 1, int(block_slice.stop)))
        return route_indices

    def _parameter_list_grad_norm(self, parameters: list[nn.Parameter]) -> float:
        """Compute the L2 norm of gradients over a parameter list."""
        if not parameters:
            return 0.0
        total = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            grad_norm = float(parameter.grad.detach().norm().item())
            total += grad_norm * grad_norm
        return float(total ** 0.5)

    def _linear_output_slice_grad_norm(self, output_indices: list[int]) -> float:
        """Compute gradient norm for selected output rows of the flat joint actor head."""
        if self.network.actor_mean is None or not output_indices:
            return 0.0
        total = 0.0
        weight_grad = self.network.actor_mean.weight.grad
        bias_grad = self.network.actor_mean.bias.grad
        log_std_grad = self.network.actor_log_std.grad
        if weight_grad is not None:
            total += float(weight_grad[output_indices].detach().pow(2).sum().item())
        if bias_grad is not None:
            total += float(bias_grad[output_indices].detach().pow(2).sum().item())
        if log_std_grad is not None:
            total += float(log_std_grad[output_indices].detach().pow(2).sum().item())
        return float(total ** 0.5)

    def _theta_head_grad_norm(self) -> float:
        """Compute the current theta-head gradient norm for diagnostics."""
        if self._uses_hierarchical_actor_theta_route():
            return self._parameter_list_grad_norm(self.network.actor_theta_head_parameters())
        return self._linear_output_slice_grad_norm(self._theta_action_indices())

    def _route_head_grad_norm(self) -> float:
        """Compute the current route-head gradient norm for diagnostics."""
        if self._uses_hierarchical_actor_theta_route():
            return self._parameter_list_grad_norm(self.network.actor_route_head_parameters())
        return self._linear_output_slice_grad_norm(self._route_action_indices())

    def _theta_backbone_grad_norm(self) -> float:
        """Compute the current theta-backbone gradient norm for diagnostics."""
        if self._uses_hierarchical_actor_theta_route():
            return self._parameter_list_grad_norm(self.network.actor_theta_backbone_parameters())
        return 0.0

    def _route_backbone_grad_norm(self) -> float:
        """Compute the current route-backbone gradient norm for diagnostics."""
        if self._uses_hierarchical_actor_theta_route():
            return self._parameter_list_grad_norm(self.network.actor_route_backbone_parameters())
        return 0.0

    def _parameter_list_delta_norm(
        self,
        before_parameters: list[torch.Tensor],
        after_parameters: list[torch.Tensor],
    ) -> float:
        """Compute the L2 norm of parameter deltas between two snapshots."""
        total = 0.0
        for before_parameter, after_parameter in zip(before_parameters, after_parameters):
            delta_norm = float((after_parameter - before_parameter).norm().item())
            total += delta_norm * delta_norm
        return float(total ** 0.5)

    def _route_training_drift_trace_buffer(self) -> list[dict[str, Any]] | None:
        """Return the optional route-step trace buffer when dynamic tracing is enabled."""
        trace_buffer = getattr(self, "route_training_drift_trace_rows", None)
        return trace_buffer if isinstance(trace_buffer, list) else None

    def _append_route_training_drift_trace_row(self, row: dict[str, Any]) -> None:
        """Append one route-step trace row when route drift tracing is enabled."""
        trace_buffer = self._route_training_drift_trace_buffer()
        if trace_buffer is not None:
            trace_buffer.append(row)

    def _critic_training_drift_trace_buffer(self) -> list[dict[str, Any]] | None:
        """Return the optional critic-drift trace buffer when minibatch tracing is enabled."""
        trace_buffer = getattr(self, "critic_training_drift_trace_rows", None)
        return trace_buffer if isinstance(trace_buffer, list) else None

    def _append_critic_training_drift_trace_row(self, row: dict[str, Any]) -> None:
        """Append one critic-drift trace row when tracing is enabled."""
        trace_buffer = self._critic_training_drift_trace_buffer()
        if trace_buffer is not None:
            trace_buffer.append(row)

    def _critic_training_drift_probe_payload(self) -> dict[str, torch.Tensor] | None:
        """Return the optional fixed probe payload for critic-drift tracing."""
        payload = getattr(self, "critic_training_drift_probe_payload", None)
        return payload if isinstance(payload, dict) else None

    def _critic_training_drift_heldout_payload(self) -> dict[str, torch.Tensor] | None:
        """Return the optional fixed held-out payload for critic-drift tracing."""
        payload = getattr(self, "critic_training_drift_heldout_payload", None)
        return payload if isinstance(payload, dict) else None

    def _critic_blended_fixed_heldout_payload(self) -> dict[str, torch.Tensor] | None:
        """Return the optional fixed held-out payload for blended critic loss."""
        payload = getattr(self, "critic_blended_fixed_heldout_payload", None)
        return payload if isinstance(payload, dict) else None

    def _critic_step_acceptance_config(self) -> dict[str, Any] | None:
        """Return the optional critic-step monitor / acceptance configuration."""
        config = getattr(self, "critic_step_acceptance_config", None)
        return config if isinstance(config, dict) else None

    def _critic_step_acceptance_trace_buffer(self) -> list[dict[str, Any]] | None:
        """Return the optional critic-step monitor trace buffer."""
        trace_buffer = getattr(self, "critic_step_acceptance_trace_rows", None)
        return trace_buffer if isinstance(trace_buffer, list) else None

    def _append_critic_step_acceptance_trace_row(self, row: dict[str, Any]) -> None:
        """Append one critic-step monitor row when the trace buffer is enabled."""
        trace_buffer = self._critic_step_acceptance_trace_buffer()
        if trace_buffer is not None:
            trace_buffer.append(row)

    def _freeze_actor_training_updates(self) -> bool:
        """Return whether actor optimizer steps should be skipped for tracing."""
        return bool(getattr(self, "freeze_actor_training_updates", False))

    def _critic_training_internal_stage_trace_enabled(self) -> bool:
        """Return whether critic-step tracing should include internal sub-stages."""
        return bool(getattr(self, "critic_training_internal_stage_trace", False))

    def _critic_optimizer_ablation_request(self) -> dict[str, Any] | None:
        """Return an optional request to capture a pre-step critic snapshot."""
        request = getattr(self, "critic_optimizer_ablation_request", None)
        return request if isinstance(request, dict) else None

    def _zero_parameter_list_grads(self, parameters: list[nn.Parameter]) -> None:
        """Zero gradients for a selected parameter list while leaving other grads intact."""
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.zero_()

    def _snapshot_parameter_list(self, parameters: list[nn.Parameter]) -> list[torch.Tensor]:
        """Create detached copies for parameters that must stay frozen across an optimizer step."""
        return [parameter.detach().clone() for parameter in parameters]

    def _restore_parameter_list(
        self,
        parameters: list[nn.Parameter],
        snapshots: list[torch.Tensor],
    ) -> None:
        """Restore frozen parameters after an optimizer step to block optimizer-state drift."""
        for parameter, snapshot in zip(parameters, snapshots):
            parameter.data.copy_(snapshot)

    def _critic_trace_loss_and_semantics(
        self,
        critic_inputs: torch.Tensor,
        value_targets: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
        prediction_target_mean: torch.Tensor,
        prediction_target_std: torch.Tensor,
    ) -> tuple[dict[str, float], float]:
        """Evaluate critic loss and semantic correlations for a fixed payload."""
        with torch.no_grad():
            value_predictions = self._trace_raw_value_predictions_from_critic_inputs(
                critic_inputs,
                prediction_target_mean,
                prediction_target_std,
            )
            critic_features = self.network.critic_features_from_critic_input(critic_inputs)
            normalized_values = self.network.critic_head(critic_features).squeeze(-1)
            if self.config.value_target_mode == "popart_return_norm":
                critic_predictions_for_loss = normalized_values
            else:
                raw_values = self.network.value_from_critic_input(critic_inputs).squeeze(-1)
                if self.config.value_target_mode in {
                    "normalized_return",
                    "running_return_norm",
                }:
                    critic_predictions_for_loss = (
                        raw_values - prediction_target_mean
                    ) / (prediction_target_std + 1e-8)
                else:
                    critic_predictions_for_loss = raw_values
            semantics = self._critic_trace_semantic_summary(
                value_predictions,
                value_targets,
                returns,
                advantages,
                rewards,
            )
            critic_loss = float(
                self._compute_value_loss(
                    critic_predictions_for_loss,
                    value_targets,
                ).item()
            )
        return semantics, critic_loss

    def _snapshot_actor_log_std_indices(self, output_indices: list[int]) -> torch.Tensor | None:
        """Snapshot selected actor-log-std coordinates for exact branch freezing."""
        if not output_indices:
            return None
        return self.network.actor_log_std.detach()[output_indices].clone()

    def _restore_actor_log_std_indices(
        self,
        output_indices: list[int],
        snapshot: torch.Tensor | None,
    ) -> None:
        """Restore frozen actor-log-std coordinates after a branch-specific optimizer step."""
        if snapshot is None or not output_indices:
            return
        self.network.actor_log_std.data[output_indices] = snapshot

    def _theta_actor_optimizer(self) -> torch.optim.Optimizer:
        """Return the optimizer responsible for theta-branch actor updates."""
        if self._uses_dual_optimizer_alternating_branch_pg():
            if self.theta_actor_optimizer is None:
                raise RuntimeError("Theta actor optimizer is not initialized")
            return self.theta_actor_optimizer
        return self.actor_optimizer

    def _route_actor_optimizer(self) -> torch.optim.Optimizer:
        """Return the optimizer responsible for route-branch actor updates."""
        if self._uses_dual_optimizer_alternating_branch_pg():
            if self.route_actor_optimizer is None:
                raise RuntimeError("Route actor optimizer is not initialized")
            return self.route_actor_optimizer
        return self.actor_optimizer

    def _zero_actor_log_std_grad_indices(self, output_indices: list[int]) -> None:
        """Zero selected actor-log-std coordinates so only one branch updates at a time."""
        log_std_grad = self.network.actor_log_std.grad
        if log_std_grad is None or not output_indices:
            return
        log_std_grad[output_indices] = 0.0

    def _apply_theta_only_gradient_mask(self) -> None:
        """Keep gradients only on theta-side actor parameters for alternating updates."""
        self._zero_parameter_list_grads(self.network.actor_route_backbone_parameters())
        self._zero_parameter_list_grads(self.network.actor_route_head_parameters())
        self._zero_actor_log_std_grad_indices(self._route_action_indices())

    def _apply_route_only_gradient_mask(self) -> None:
        """Keep gradients only on route-side actor parameters for alternating updates."""
        self._zero_parameter_list_grads(self.network.actor_theta_backbone_parameters())
        self._zero_parameter_list_grads(self.network.actor_theta_head_parameters())
        self._zero_actor_log_std_grad_indices(self._theta_action_indices())

    def _branch_entropy_from_components(
        self,
        entropy_components: torch.Tensor,
        action_indices: list[int],
    ) -> torch.Tensor:
        """Aggregate Gaussian entropy over selected raw-action coordinates."""
        if not action_indices:
            return entropy_components.new_zeros(())
        return entropy_components[..., action_indices].sum(dim=-1).mean()

    def _aggregate_log_prob_components(self, log_prob_components: torch.Tensor) -> torch.Tensor:
        """Aggregate per-dimension log-prob terms according to the configured ratio mode."""
        mode = self._policy_ratio_mode()
        if mode == "current_joint_sum_ratio":
            return log_prob_components.sum(dim=-1)
        if mode == "mean_logprob_ratio":
            return log_prob_components.mean(dim=-1)
        if mode == "block_mean_ratio":
            return self._block_sum_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "blockwise_surrogate_mean":
            return self._block_sum_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "blockwise_scaled_advantage_surrogate_mean":
            return self._block_sum_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "blockwise_value_scaled_advantage_surrogate_mean":
            return self._block_sum_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "blockwise_td_advantage_surrogate_mean":
            return self._block_sum_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "blockwise_delta_cost_td_advantage_surrogate_mean":
            return self._block_sum_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "blockwise_action_conditioned_td_advantage_surrogate_mean":
            return self._block_sum_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "blockwise_action_conditioned_path_value_td_advantage_surrogate_mean":
            return self._block_sum_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "blockwise_path_specific_td_supervised_advantage_surrogate_mean":
            return self._block_sum_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "theta_route_split_advantage_surrogate":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_theta_route":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_offload_gated_route":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_hard_offload_route":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_separate_theta_route_backbones":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_branchwise_balanced_pg":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_alternating_branch_pg":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_dual_optimizer_alternating_branch_pg":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_ratio_pg":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_trust_region_pg":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_conditional_route_pg":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_true_conditional_route_policy":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_route_credit_theta_gate":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_route_credit_theta_soft_weight":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_route_residual_credit_vectorized":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_true_conditional_route_candidate_score_credit":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_theta_candidate_score_credit":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_theta_route_candidate_score_credit":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_joint_reward_aligned_credit":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_joint_td_aligned_credit":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_trust_region_pg_route_first":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_trust_region_pg_coupled_stop":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_theta_floor":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_confident_mask":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_cap":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_epochwise":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_route_counterfactual_adv":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_alignment_gate":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "hierarchical_actor_factorized_trust_region_pg_route_first_coupled_stop_severity_gate":
            return self._block_split_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "blockwise_pseudolocal_advantage_surrogate_mean":
            return self._block_sum_log_prob_components(log_prob_components).mean(dim=-1)
        if mode == "blockwise_weighted_surrogate_mean":
            raise ValueError(
                "blockwise_weighted_surrogate_mean requires state-dependent aggregation;"
                " use _aggregate_log_prob_components_with_states instead."
            )
        raise ValueError(f"Unsupported policy_ratio_mode: {mode}")

    def _block_activity_scores(self, states: torch.Tensor) -> torch.Tensor:
        """Build nonnegative per-block local activity from current normalized load and queue state."""
        block_count = len(self._action_block_slices())
        if self.critic_state_layout is None:
            return states.new_ones((states.shape[0], block_count))

        slices = self._state_slices()
        sensor_count = int(self.critic_state_layout["sensor_count"])
        task_count = int(self.critic_state_layout["task_count"])

        workload_block = states[:, slices["workload"]].reshape(-1, sensor_count, task_count)
        access_queue_block = states[:, slices["access_queue"]].reshape(-1, sensor_count, 1)
        virtual_queue_block = states[:, slices["virtual_queue"]].reshape(
            -1,
            sensor_count,
            task_count,
        )
        local_queue_block = states[:, slices["local_queue"]].reshape(-1, sensor_count, task_count)

        # Minimal block relevance: current normalized load plus the directly aligned queue pressure.
        block_activity = (
            workload_block
            + virtual_queue_block
            + local_queue_block
            + access_queue_block.expand(-1, -1, task_count)
        ).reshape(-1, block_count)
        return torch.clamp(block_activity, min=0.0)

    def _block_activity_weights(self, states: torch.Tensor) -> torch.Tensor:
        """Normalize local activity into sum-to-one block weights."""
        block_activity = self._block_activity_scores(states)
        block_count = max(block_activity.shape[-1], 1)
        eps = 1e-6
        return (block_activity + eps) / (
            block_activity.sum(dim=-1, keepdim=True) + eps * float(block_count)
        )

    def _block_advantage_scales(self, states: torch.Tensor) -> torch.Tensor:
        """Normalize local activity into mean-one block scales for advantage rescaling."""
        block_activity = self._block_activity_scores(states)
        stabilized_activity = block_activity + 1e-6
        return stabilized_activity / stabilized_activity.mean(dim=-1, keepdim=True)

    def _block_value_scores_and_scales(
        self,
        critic_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict learned per-block value scores and convert them into mean-one nonnegative scales."""
        with torch.no_grad():
            block_value_scores = self.network.block_value_scores_from_critic_input(critic_inputs)
            positive_scores = nn.functional.softplus(block_value_scores) + 1e-6
            block_value_scales = positive_scores / positive_scores.mean(dim=-1, keepdim=True)
        return block_value_scores, block_value_scales

    def _block_local_costs(self, states: torch.Tensor) -> torch.Tensor:
        """Build a simple per-block immediate local cost proxy from current local workload/queue state."""
        return torch.clamp(self._block_activity_scores(states), min=0.0)

    def _block_path_costs(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build path-specific block costs for local and each base-station route."""
        if self.critic_state_layout is None:
            block_count = len(self._action_block_slices())
            local_costs = states.new_zeros((states.shape[0], block_count))
            bs_costs = states.new_zeros((states.shape[0], block_count, 0))
            return local_costs.unsqueeze(-1), local_costs, bs_costs

        slices = self._state_slices()
        sensor_count = int(self.critic_state_layout["sensor_count"])
        task_count = int(self.critic_state_layout["task_count"])
        base_station_count = int(self.critic_state_layout["base_station_count"])

        workload_block = states[:, slices["workload"]].reshape(-1, sensor_count, task_count)
        access_queue_block = states[:, slices["access_queue"]].reshape(-1, sensor_count, 1)
        virtual_queue_block = states[:, slices["virtual_queue"]].reshape(
            -1,
            sensor_count,
            task_count,
        )
        bs_queue_block = states[:, slices["bs_queue"]].reshape(-1, base_station_count, task_count)
        local_queue_block = states[:, slices["local_queue"]].reshape(-1, sensor_count, task_count)

        cost_local = workload_block + virtual_queue_block + local_queue_block
        cost_bs = (
            workload_block.unsqueeze(1)
            + virtual_queue_block.unsqueeze(1)
            + access_queue_block.unsqueeze(1).expand(-1, base_station_count, -1, task_count)
            + bs_queue_block.unsqueeze(2).expand(-1, -1, sensor_count, -1)
        )
        cost_local_flat = cost_local.reshape(-1, sensor_count * task_count)
        cost_bs_flat = cost_bs.permute(0, 2, 3, 1).reshape(
            -1,
            sensor_count * task_count,
            base_station_count,
        )
        path_costs = torch.cat([cost_local_flat.unsqueeze(-1), cost_bs_flat], dim=-1)
        return path_costs, cost_local_flat, cost_bs_flat

    def _block_theta_and_route_probabilities(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode raw block actions into theta and conditional route probabilities."""
        block_slices = self._action_block_slices()
        block_actions = torch.stack(
            [actions[..., block_slice] for block_slice in block_slices],
            dim=1,
        )
        theta = torch.sigmoid(block_actions[..., 0])
        route_logits = block_actions[..., 1:]
        route_probs = torch.softmax(route_logits, dim=-1)
        return theta, route_probs

    def _block_action_path_probabilities(self, actions: torch.Tensor) -> torch.Tensor:
        """Decode raw action blocks into local/BS path probabilities using the simulator semantics."""
        theta, route_probs = self._block_theta_and_route_probabilities(actions)
        local_prob = 1.0 - theta
        remote_probs = theta.unsqueeze(-1) * route_probs
        return torch.cat([local_prob.unsqueeze(-1), remote_probs], dim=-1)

    def _block_path_value_predictions(
        self,
        critic_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-block per-path values from critic inputs."""
        return self.network.block_path_value_scores_from_critic_input(critic_inputs)

    def _block_path_value_predictions_from_features(
        self,
        critic_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-block per-path values from cached critic features."""
        return self.network.block_path_value_scores_from_features(critic_features)

    def _aggregate_block_path_values(
        self,
        block_path_values: torch.Tensor,
        block_path_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate per-path values into an action-conditioned block scalar value."""
        return (block_path_probs * block_path_values).sum(dim=-1)

    def _block_path_td_rewards(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build path-specific immediate rewards and path-specific costs for each block-path pair."""
        block_path_costs_now, _, _ = self._block_path_costs(states)
        block_path_costs_next, _, _ = self._block_path_costs(next_states)
        block_path_rewards = block_path_costs_now - block_path_costs_next
        return block_path_rewards, block_path_costs_now, block_path_costs_next

    def _block_td_reward_terms(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Build block cost terms and the configured TD immediate reward."""
        block_local_cost_now = self._block_local_costs(states)
        block_local_cost_next = self._block_local_costs(next_states)
        (
            block_path_costs_now,
            block_path_cost_local_now,
            block_path_cost_bs_now,
        ) = self._block_path_costs(states)
        (
            block_path_costs_next,
            _,
            _,
        ) = self._block_path_costs(next_states)
        block_path_probs = None
        block_action_conditioned_cost_now = block_local_cost_now
        block_action_conditioned_cost_next = block_local_cost_next
        if actions is not None:
            block_path_probs = self._block_action_path_probabilities(actions)
            block_action_conditioned_cost_now = (
                block_path_probs * block_path_costs_now
            ).sum(dim=-1)
            block_action_conditioned_cost_next = (
                block_path_probs * block_path_costs_next
            ).sum(dim=-1)
        block_delta_cost = block_action_conditioned_cost_next - block_action_conditioned_cost_now
        if self._uses_blockwise_action_conditioned_td_reward():
            block_local_rewards = block_action_conditioned_cost_now - block_action_conditioned_cost_next
        elif self._uses_blockwise_delta_cost_td_advantage_surrogate_mean():
            block_local_rewards = block_local_cost_now - block_local_cost_next
        else:
            block_local_rewards = -block_local_cost_now
        return (
            block_local_cost_now,
            block_local_cost_next,
            block_path_cost_local_now,
            block_path_cost_bs_now,
            block_action_conditioned_cost_now,
            block_action_conditioned_cost_next,
            block_delta_cost,
            block_local_rewards,
            block_path_probs,
        )

    def _block_td_value_signal(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        actions: torch.Tensor | None = None,
        critic_inputs: torch.Tensor | None = None,
        next_critic_inputs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute current block values, TD targets, and TD advantages for each semantic block."""
        if critic_inputs is None:
            critic_inputs, _, _ = self._prepare_critic_inputs(states, update_stats=False)
        if next_critic_inputs is None:
            next_critic_inputs, _, _ = self._prepare_critic_inputs(next_states, update_stats=False)

        with torch.no_grad():
            block_path_rewards, _, _ = self._block_path_td_rewards(states, next_states)
            (
                *_block_reward_prefix,
                block_local_rewards,
                block_path_probs,
            ) = self._block_td_reward_terms(
                states,
                next_states,
                actions=actions,
            )
            if self._uses_blockwise_action_conditioned_path_value_bootstrap():
                if block_path_probs is None:
                    raise ValueError(
                        "actions are required for action-conditioned path-value TD advantages"
                    )
                block_path_value_preds = self._block_path_value_predictions(critic_inputs)
                next_block_path_value_preds = self._block_path_value_predictions(next_critic_inputs)
                block_value_preds = self._aggregate_block_path_values(
                    block_path_value_preds,
                    block_path_probs,
                )
                next_block_value_preds = self._aggregate_block_path_values(
                    next_block_path_value_preds,
                    block_path_probs,
                )
            else:
                block_value_preds = self.network.block_value_scores_from_critic_input(critic_inputs)
                next_block_value_preds = self.network.block_value_scores_from_critic_input(
                    next_critic_inputs
                )
                block_path_value_preds = block_value_preds.unsqueeze(-1).expand(
                    -1,
                    -1,
                    self.action_path_count,
                )
                next_block_path_value_preds = next_block_value_preds.unsqueeze(-1).expand(
                    -1,
                    -1,
                    self.action_path_count,
                )
            not_done_mask = 1.0 - dones.unsqueeze(-1)
            block_td_targets = (
                block_local_rewards
                + self.config.gamma * not_done_mask * next_block_value_preds
            )
            block_td_advantages = block_td_targets - block_value_preds
            block_path_td_targets = (
                block_path_rewards
                + self.config.gamma * not_done_mask.unsqueeze(-1) * next_block_path_value_preds
            )
        return {
            "block_value_preds": block_value_preds,
            "next_block_value_preds": next_block_value_preds,
            "block_td_targets": block_td_targets,
            "block_td_advantages": block_td_advantages,
            "block_path_td_rewards": block_path_rewards,
            "block_path_td_targets": block_path_td_targets,
            "block_path_value_preds": block_path_value_preds,
            "next_block_path_value_preds": next_block_path_value_preds,
            "block_action_conditioned_value_preds": block_value_preds,
            "block_action_conditioned_value_next_preds": next_block_value_preds,
        }

    def _block_pseudolocal_advantage_coefficients(self, states: torch.Tensor) -> torch.Tensor:
        """Build centered, bounded block coefficients from current local activity only."""
        block_activity = self._block_activity_scores(states)
        centered_activity = block_activity - block_activity.mean(dim=-1, keepdim=True)
        activity_std = centered_activity.std(dim=-1, keepdim=True, unbiased=False)
        normalized_activity = centered_activity / (activity_std + 1e-6)
        bounded_activity = torch.tanh(normalized_activity)
        return bounded_activity / (bounded_activity.abs().mean(dim=-1, keepdim=True) + 1e-6)

    def _theta_route_split_advantages(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        actions: torch.Tensor,
        critic_inputs: torch.Tensor | None = None,
        next_critic_inputs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build separate theta-vs-offload and BS1-vs-BS2 actor credit signals inside each block."""
        td_value_signal = self._block_td_value_signal(
            states,
            next_states,
            dones,
            actions=actions,
            critic_inputs=critic_inputs,
            next_critic_inputs=next_critic_inputs,
        )
        theta, route_probs = self._block_theta_and_route_probabilities(actions)
        path_td_advantages = (
            td_value_signal["block_path_td_targets"] - td_value_signal["block_path_value_preds"]
        )
        local_td_advantage = path_td_advantages[..., 0]
        bs1_td_advantage = (
            path_td_advantages[..., 1]
            if path_td_advantages.shape[-1] > 1
            else torch.zeros_like(local_td_advantage)
        )
        bs2_td_advantage = (
            path_td_advantages[..., 2]
            if path_td_advantages.shape[-1] > 2
            else torch.zeros_like(local_td_advantage)
        )
        route_bs1_prob = (
            route_probs[..., 0] if route_probs.shape[-1] > 0 else torch.zeros_like(local_td_advantage)
        )
        route_bs2_prob = (
            route_probs[..., 1] if route_probs.shape[-1] > 1 else torch.zeros_like(local_td_advantage)
        )
        offload_td_advantage = (
            route_bs1_prob * bs1_td_advantage + route_bs2_prob * bs2_td_advantage
        )
        theta_advantages = offload_td_advantage - local_td_advantage
        route_advantages = bs1_td_advantage - bs2_td_advantage
        if self._uses_route_counterfactual_adv_factorized_trust_region_pg():
            remote_td_advantages = path_td_advantages[..., 1:]
            if remote_td_advantages.shape[-1] > 0:
                selected_route_indices = route_probs.detach().argmax(dim=-1, keepdim=True)
                selected_remote_advantages = remote_td_advantages.gather(
                    dim=-1,
                    index=selected_route_indices,
                ).squeeze(-1)
                mean_remote_advantages = remote_td_advantages.mean(dim=-1)
                route_advantages = selected_remote_advantages - mean_remote_advantages
            else:
                route_advantages = torch.zeros_like(local_td_advantage)
        route_gates = theta.detach()
        route_masks = (route_gates > self._route_mask_threshold()).to(route_gates.dtype)
        route_mask_counts = route_masks.sum(dim=-1)

        current_path_values = td_value_signal["block_path_value_preds"]
        local_values = current_path_values[..., 0]
        bs1_values = (
            current_path_values[..., 1]
            if current_path_values.shape[-1] > 1
            else torch.zeros_like(local_values)
        )
        bs2_values = (
            current_path_values[..., 2]
            if current_path_values.shape[-1] > 2
            else torch.zeros_like(local_values)
        )
        offload_values = route_bs1_prob * bs1_values + route_bs2_prob * bs2_values
        return {
            "theta_advantages": theta_advantages,
            "route_advantages": route_advantages,
            "route_gates": route_gates,
            "route_masks": route_masks,
            "route_mask_counts": route_mask_counts,
            "offload_td_advantage": offload_td_advantage,
            "local_td_advantage": local_td_advantage,
            "bs1_td_advantage": bs1_td_advantage,
            "bs2_td_advantage": bs2_td_advantage,
            "offload_vs_local_value_gap": offload_values - local_values,
            "bs1_vs_bs2_value_gap": bs1_values - bs2_values,
            "td_value_signal": td_value_signal,
        }

    def _block_surrogate_advantages(
        self,
        advantages: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
        critic_inputs: torch.Tensor | None = None,
        next_states: torch.Tensor | None = None,
        next_critic_inputs: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the actual per-block advantages and their nonnegative summary weights."""
        block_count = len(self._action_block_slices())
        if self._uses_blockwise_pseudolocal_advantage_surrogate_mean():
            block_coefficients = self._block_pseudolocal_advantage_coefficients(states)
            block_advantages = advantages.unsqueeze(-1) * block_coefficients
            block_advantage_weights = block_advantages.abs()
        elif self._uses_blockwise_td_style_advantage_surrogate_mean():
            if next_states is None or dones is None:
                raise ValueError("next_states and dones are required for blockwise TD advantages")
            td_value_signal = self._block_td_value_signal(
                states,
                next_states,
                dones,
                actions=actions,
                critic_inputs=critic_inputs,
                next_critic_inputs=next_critic_inputs,
            )
            block_advantages = td_value_signal["block_td_advantages"]
            block_advantage_weights = block_advantages.abs() + 1e-6
        elif self._uses_blockwise_value_scaled_advantage_surrogate_mean():
            if critic_inputs is None:
                critic_inputs, _, _ = self._prepare_critic_inputs(states, update_stats=False)
            _, block_advantage_weights = self._block_value_scores_and_scales(critic_inputs)
            block_advantages = advantages.unsqueeze(-1) * block_advantage_weights
        elif self._uses_blockwise_scaled_advantage_surrogate_mean():
            block_advantage_weights = self._block_advantage_scales(states)
            block_advantages = advantages.unsqueeze(-1) * block_advantage_weights
        else:
            block_advantages = advantages.unsqueeze(-1).expand(-1, block_count)
            block_advantage_weights = torch.ones_like(block_advantages)
        return block_advantages, block_advantage_weights

    def _masked_tensor_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the mean over masked values, or zero when the mask is empty."""
        mask_bool = mask.to(dtype=torch.bool)
        if not bool(mask_bool.any().item()):
            return values.new_zeros(())
        return values[mask_bool].mean()

    def _masked_tensor_std(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the std over masked values, or zero when the mask is empty."""
        mask_bool = mask.to(dtype=torch.bool)
        if not bool(mask_bool.any().item()):
            return values.new_zeros(())
        return values[mask_bool].std(unbiased=False)

    def _sample_masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return per-sample masked means over the last dimension, falling back to zero if empty."""
        mask_float = mask.to(dtype=values.dtype)
        masked_sum = (values * mask_float).sum(dim=-1)
        masked_count = mask_float.sum(dim=-1)
        return torch.where(
            masked_count > 0.0,
            masked_sum / (masked_count + 1e-8),
            torch.zeros_like(masked_sum),
        )

    def _branchwise_normalize_theta_route_advantages(
        self,
        theta_advantages: torch.Tensor,
        route_advantages: torch.Tensor,
        route_masks: torch.Tensor,
        *,
        eps: float = 1e-8,
        min_active_count: int = 2,
    ) -> dict[str, torch.Tensor | float]:
        """Normalize theta and route advantages on their own branch-specific supports."""
        theta_mean = theta_advantages.mean()
        theta_std = theta_advantages.std(unbiased=False)
        theta_advantages_norm = (theta_advantages - theta_mean) / (theta_std + eps)

        route_masks_bool = route_masks > 0.5
        active_route_count = int(route_masks_bool.sum().item())
        route_advantages_norm = torch.zeros_like(route_advantages)
        route_mean = route_advantages.new_zeros(())
        route_std = route_advantages.new_zeros(())
        route_used_fallback = False

        if active_route_count >= min_active_count:
            active_route_advantages = route_advantages[route_masks_bool]
            route_mean = active_route_advantages.mean()
            route_std = active_route_advantages.std(unbiased=False)
            if float(route_std.item()) > eps:
                route_advantages_norm[route_masks_bool] = (
                    active_route_advantages - route_mean
                ) / (route_std + eps)
            else:
                route_advantages_norm[route_masks_bool] = active_route_advantages
                route_used_fallback = True
        else:
            route_advantages_norm[route_masks_bool] = route_advantages[route_masks_bool]
            route_used_fallback = True

        return {
            "theta_advantages_norm": theta_advantages_norm,
            "route_advantages_norm": route_advantages_norm,
            "theta_adv_norm_mean": theta_advantages_norm.mean(),
            "theta_adv_norm_std": theta_advantages_norm.std(unbiased=False),
            "route_adv_norm_mean": self._masked_tensor_mean(route_advantages_norm, route_masks_bool),
            "route_adv_norm_std": self._masked_tensor_std(route_advantages_norm, route_masks_bool),
            "theta_adv_mean": theta_mean,
            "theta_adv_std": theta_std,
            "route_adv_mean": route_mean,
            "route_adv_std": route_std,
            "route_active_count": route_advantages.new_tensor(float(active_route_count)),
            "route_used_fallback": float(route_used_fallback),
        }

    def _mean_route_surrogate_over_active_blocks(
        self,
        route_surrogate: torch.Tensor,
        route_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Average route surrogate only over active route blocks, or return zero if none are active."""
        return self._mean_route_surrogate_over_mask(route_surrogate, route_masks)

    def _mean_route_surrogate_over_mask(
        self,
        route_surrogate: torch.Tensor,
        route_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Average route surrogate only over masked route samples, or return zero if the mask is empty."""
        active_route_mask = route_masks > 0.5
        if not bool(active_route_mask.any().item()):
            return route_surrogate.new_zeros(())
        return route_surrogate[active_route_mask].mean()

    def _route_credit_theta_gate_support(
        self,
        theta_advantages: torch.Tensor,
        route_masks: torch.Tensor,
        *,
        min_effective_count: int = 2,
    ) -> dict[str, torch.Tensor | float]:
        """Build the effective route-credit support after optional positive-theta gating."""
        route_active_mask = route_masks > 0.5
        theta_positive_mask = theta_advantages > 0.0
        route_credit_gate = theta_positive_mask.to(dtype=route_masks.dtype)
        effective_mask = route_active_mask & theta_positive_mask
        effective_count = int(effective_mask.sum().item())
        active_count = int(route_active_mask.sum().item())
        used_fallback = False
        if (
            self._uses_route_credit_theta_gate()
            and active_count > 0
            and effective_count < min_effective_count
        ):
            effective_mask = route_active_mask
            effective_count = active_count
            used_fallback = True
        effective_fraction = float(effective_mask.float().mean().item())
        return {
            "route_credit_gate": route_credit_gate,
            "effective_mask": effective_mask.to(dtype=route_masks.dtype),
            "effective_count": float(effective_count),
            "effective_fraction": effective_fraction,
            "fallback_triggered": float(used_fallback),
        }

    def _route_credit_theta_soft_weight_terms(
        self,
        theta_advantages_norm: torch.Tensor,
        route_masks: torch.Tensor,
        *,
        eps: float = 0.1,
        max_weight: float = 3.0,
    ) -> dict[str, torch.Tensor | float]:
        """Build continuous route-credit weights from normalized positive theta credit."""
        route_active_mask = route_masks > 0.5
        route_credit_weight_raw = torch.zeros_like(theta_advantages_norm)
        route_credit_weight = torch.zeros_like(theta_advantages_norm)
        if bool(route_active_mask.any().item()):
            active_theta_adv_norm = theta_advantages_norm[route_active_mask]
            active_raw = torch.relu(active_theta_adv_norm) + eps
            active_weight = active_raw / (active_raw.mean() + 1e-8)
            active_weight = torch.clamp(active_weight, max=max_weight)
            route_credit_weight_raw[route_active_mask] = active_raw
            route_credit_weight[route_active_mask] = active_weight
        return {
            "raw": route_credit_weight_raw,
            "weight": route_credit_weight,
            "effective_mask": route_masks,
            "effective_fraction": float(route_active_mask.float().mean().item()),
            "effective_count": float(route_active_mask.sum().item()),
            "weight_mean": self._masked_mean(route_credit_weight, route_active_mask),
            "weight_std": float(
                self._masked_tensor_std(route_credit_weight, route_active_mask).item()
            ),
            "weight_min": float(
                route_credit_weight[route_active_mask].min().item()
            )
            if bool(route_active_mask.any().item())
            else 0.0,
            "weight_max": float(
                route_credit_weight[route_active_mask].max().item()
            )
            if bool(route_active_mask.any().item())
            else 0.0,
        }

    def _two_route_score_vector_from_scalar(
        self,
        route_score_scalar: torch.Tensor,
    ) -> torch.Tensor:
        """Lift the current 2-route scalar credit into a vector form compatible with residual-score logic."""
        return torch.stack(
            [
                0.5 * route_score_scalar,
                -0.5 * route_score_scalar,
            ],
            dim=-1,
        )

    def _route_residual_credit_vectorized_terms(
        self,
        route_score_scalar: torch.Tensor,
        route_selected_indices: torch.Tensor,
        route_policy_probs: torch.Tensor,
        route_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor | float]:
        """Build selected-minus-expected residual route credit from a vectorized score interface."""
        route_active_mask = route_masks > 0.5
        route_score_vector = self._two_route_score_vector_from_scalar(route_score_scalar)
        if route_score_vector.shape[-1] != route_policy_probs.shape[-1]:
            raise ValueError(
                "route score vector width must match conditional route policy width"
            )
        selected_route_score = route_score_vector.gather(
            dim=-1,
            index=route_selected_indices.unsqueeze(-1),
        ).squeeze(-1)
        expected_route_score = (route_policy_probs * route_score_vector).sum(dim=-1)
        residual_route_credit = selected_route_score - expected_route_score
        return {
            "route_score_vector": route_score_vector,
            "selected_score": selected_route_score,
            "expected_score": expected_route_score,
            "residual_credit": residual_route_credit,
            "mean": self._masked_mean(residual_route_credit, route_active_mask),
            "std": float(
                self._masked_tensor_std(residual_route_credit, route_active_mask).item()
            ),
            "min": float(residual_route_credit[route_active_mask].min().item())
            if bool(route_active_mask.any().item())
            else 0.0,
            "max": float(residual_route_credit[route_active_mask].max().item())
            if bool(route_active_mask.any().item())
            else 0.0,
            "expected_score_mean": self._masked_mean(expected_route_score, route_active_mask),
            "selected_score_mean": self._masked_mean(selected_route_score, route_active_mask),
            "score_vector_mean_abs": self._masked_mean(
                route_score_vector.abs().mean(dim=-1),
                route_active_mask,
            ),
            "score_vector_std": float(
                self._masked_tensor_std(
                    route_score_vector.reshape(-1, route_score_vector.shape[-1]).reshape(-1),
                    route_active_mask.unsqueeze(-1).expand_as(route_score_vector).reshape(-1),
                ).item()
            ),
        }

    def _route_candidate_score_credit_terms(
        self,
        states: torch.Tensor,
        route_selected_indices: torch.Tensor,
        route_policy_probs: torch.Tensor,
        route_masks: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> dict[str, torch.Tensor | float]:
        """Build route residual credit directly from candidate path scores for future masked-categorical reuse."""
        _, _, block_path_cost_bs_now = self._block_path_costs(states)
        route_active_mask = route_masks > 0.5
        candidate_score_vector = -block_path_cost_bs_now
        route_true_gap = candidate_score_vector[..., 0] - candidate_score_vector[..., 1]
        actual_bs1_decision = route_selected_indices == 0
        candidate_score_mean = candidate_score_vector.mean(dim=-1, keepdim=True)
        candidate_score_std = candidate_score_vector.std(dim=-1, keepdim=True, unbiased=False)
        normalized_candidate_score_vector = (
            candidate_score_vector - candidate_score_mean
        ) / (candidate_score_std + eps)
        selected_score = normalized_candidate_score_vector.gather(
            dim=-1,
            index=route_selected_indices.unsqueeze(-1),
        ).squeeze(-1)
        expected_score = (route_policy_probs * normalized_candidate_score_vector).sum(dim=-1)
        residual_credit = selected_score - expected_score
        expanded_active_mask = route_active_mask.unsqueeze(-1).expand_as(
            normalized_candidate_score_vector
        )
        route_true_gap_positive = route_true_gap > 0.0
        route_true_gap_negative = route_true_gap < 0.0
        route_decision_agreement = (
            route_true_gap_positive == actual_bs1_decision
        ).to(dtype=normalized_candidate_score_vector.dtype)
        actual_bs1_float = actual_bs1_decision.to(dtype=normalized_candidate_score_vector.dtype)
        return {
            "candidate_score_vector": normalized_candidate_score_vector,
            "selected_score": selected_score,
            "expected_score": expected_score,
            "residual_credit": residual_credit,
            "candidate_score_mean": float(
                self._masked_tensor_mean(
                    normalized_candidate_score_vector.reshape(-1),
                    expanded_active_mask.reshape(-1),
                ).item()
            ),
            "candidate_score_std": float(
                self._masked_tensor_std(
                    normalized_candidate_score_vector.reshape(-1),
                    expanded_active_mask.reshape(-1),
                ).item()
            ),
            "residual_mean": self._masked_mean(residual_credit, route_active_mask),
            "residual_std": float(
                self._masked_tensor_std(residual_credit, route_active_mask).item()
            ),
            "residual_min": float(residual_credit[route_active_mask].min().item())
            if bool(route_active_mask.any().item())
            else 0.0,
            "residual_max": float(residual_credit[route_active_mask].max().item())
            if bool(route_active_mask.any().item())
            else 0.0,
            "expected_score_mean": self._masked_mean(expected_score, route_active_mask),
            "selected_score_mean": self._masked_mean(selected_score, route_active_mask),
            "route_decision_agreement_ratio": self._masked_mean(
                route_decision_agreement,
                route_active_mask,
            ),
            "actual_bs1_rate_when_route_true_gap_positive": self._masked_mean(
                actual_bs1_float,
                route_active_mask & route_true_gap_positive,
            ),
            "actual_bs1_rate_when_route_true_gap_negative": self._masked_mean(
                actual_bs1_float,
                route_active_mask & route_true_gap_negative,
            ),
        }

    def _theta_candidate_score_credit_terms(
        self,
        states: torch.Tensor,
        theta_offload_decisions: torch.Tensor,
        theta_old_offload_probs: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> dict[str, torch.Tensor | float]:
        """Build theta residual credit from local-vs-best-offload candidate scores."""
        _, block_path_cost_local_now, block_path_cost_bs_now = self._block_path_costs(states)
        local_score = -block_path_cost_local_now
        best_offload_score = (-block_path_cost_bs_now).max(dim=-1).values
        candidate_score_vector = torch.stack(
            [local_score, best_offload_score],
            dim=-1,
        )
        candidate_score_mean = candidate_score_vector.mean(dim=-1, keepdim=True)
        candidate_score_std = candidate_score_vector.std(dim=-1, keepdim=True, unbiased=False)
        normalized_candidate_score_vector = (
            candidate_score_vector - candidate_score_mean
        ) / (candidate_score_std + eps)
        theta_selected_indices = theta_offload_decisions.to(dtype=torch.long)
        selected_score = normalized_candidate_score_vector.gather(
            dim=-1,
            index=theta_selected_indices.unsqueeze(-1),
        ).squeeze(-1)
        expected_score = (
            (1.0 - theta_old_offload_probs) * normalized_candidate_score_vector[..., 0]
            + theta_old_offload_probs * normalized_candidate_score_vector[..., 1]
        )
        residual_credit = selected_score - expected_score
        theta_true_gap = best_offload_score - local_score
        actual_offload_float = theta_offload_decisions.to(dtype=theta_true_gap.dtype)
        theta_true_gap_positive = theta_true_gap > 0.0
        theta_true_gap_negative = theta_true_gap < 0.0
        offload_decision_agreement = (
            theta_true_gap_positive == theta_offload_decisions
        ).to(dtype=theta_true_gap.dtype)
        return {
            "candidate_score_vector": normalized_candidate_score_vector,
            "selected_score": selected_score,
            "expected_score": expected_score,
            "residual_credit": residual_credit,
            "candidate_score_mean": float(normalized_candidate_score_vector.mean().item()),
            "candidate_score_std": float(
                normalized_candidate_score_vector.std(unbiased=False).item()
            ),
            "residual_mean": float(residual_credit.mean().item()),
            "residual_std": float(residual_credit.std(unbiased=False).item()),
            "residual_min": float(residual_credit.min().item()),
            "residual_max": float(residual_credit.max().item()),
            "expected_score_mean": float(expected_score.mean().item()),
            "selected_score_mean": float(selected_score.mean().item()),
            "offload_decision_agreement_ratio": float(
                offload_decision_agreement.mean().item()
            ),
            "actual_offload_rate_when_theta_true_gap_positive": self._masked_mean(
                actual_offload_float,
                theta_true_gap_positive,
            ),
            "actual_offload_rate_when_theta_true_gap_negative": self._masked_mean(
                actual_offload_float,
                theta_true_gap_negative,
            ),
        }

    def _joint_reward_aligned_credit_terms(
        self,
        joint_candidate_scores: torch.Tensor,
        joint_selected_indices: torch.Tensor,
        joint_old_policy_probs: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> dict[str, torch.Tensor | float]:
        """Build joint 3-action residual credit from reward-aligned counterfactual scores."""
        candidate_score_mean = joint_candidate_scores.mean(dim=-1, keepdim=True)
        candidate_score_std = joint_candidate_scores.std(dim=-1, keepdim=True, unbiased=False)
        normalized_candidate_scores = (
            joint_candidate_scores - candidate_score_mean
        ) / (candidate_score_std + eps)
        selected_score = normalized_candidate_scores.gather(
            dim=-1,
            index=joint_selected_indices.unsqueeze(-1),
        ).squeeze(-1)
        expected_score = (joint_old_policy_probs * normalized_candidate_scores).sum(dim=-1)
        residual_credit = selected_score - expected_score
        best_joint_indices = normalized_candidate_scores.argmax(dim=-1)
        joint_action_agreement = (
            best_joint_indices == joint_selected_indices
        ).to(dtype=normalized_candidate_scores.dtype)
        selected_is_local = joint_selected_indices == 0
        selected_is_bs1 = joint_selected_indices == 1
        selected_is_bs2 = joint_selected_indices == 2
        best_is_local = best_joint_indices == 0
        best_is_bs1 = best_joint_indices == 1
        best_is_bs2 = best_joint_indices == 2
        return {
            "candidate_score_vector": normalized_candidate_scores,
            "selected_score": selected_score,
            "expected_score": expected_score,
            "residual_credit": residual_credit,
            "best_joint_indices": best_joint_indices,
            "candidate_score_mean": float(normalized_candidate_scores.mean().item()),
            "candidate_score_std": float(
                normalized_candidate_scores.std(unbiased=False).item()
            ),
            "selected_score_mean": float(selected_score.mean().item()),
            "expected_score_mean": float(expected_score.mean().item()),
            "residual_mean": float(residual_credit.mean().item()),
            "residual_std": float(residual_credit.std(unbiased=False).item()),
            "residual_min": float(residual_credit.min().item()),
            "residual_max": float(residual_credit.max().item()),
            "joint_action_decision_agreement_ratio_under_reward_aligned": float(
                joint_action_agreement.mean().item()
            ),
            "actual_local_rate_when_reward_aligned_best_is_local": self._masked_mean(
                selected_is_local.to(dtype=normalized_candidate_scores.dtype),
                best_is_local,
            ),
            "actual_bs1_rate_when_reward_aligned_best_is_bs1": self._masked_mean(
                selected_is_bs1.to(dtype=normalized_candidate_scores.dtype),
                best_is_bs1,
            ),
            "actual_bs2_rate_when_reward_aligned_best_is_bs2": self._masked_mean(
                selected_is_bs2.to(dtype=normalized_candidate_scores.dtype),
                best_is_bs2,
            ),
        }

    def _theta_branch_ppo_terms(
        self,
        new_log_prob_components: torch.Tensor,
        old_log_prob_components: torch.Tensor,
        theta_advantages: torch.Tensor,
        *,
        actions: torch.Tensor | None = None,
        new_action_means: torch.Tensor | None = None,
        old_action_means: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build theta-branch PPO terms without constructing any joint surrogate."""
        if self._uses_true_conditional_route_policy():
            if actions is None or new_action_means is None or old_action_means is None:
                raise ValueError(
                    "true conditional route policy requires actions and old/new action means"
                )
            new_theta_log_probs = self._true_conditional_theta_policy_terms(
                actions,
                new_action_means,
            )["log_probs"]
            old_theta_log_probs = self._true_conditional_theta_policy_terms(
                actions,
                old_action_means,
            )["log_probs"]
        else:
            new_theta_log_probs, _ = self._block_theta_route_log_prob_components(
                new_log_prob_components
            )
            old_theta_log_probs, _ = self._block_theta_route_log_prob_components(
                old_log_prob_components
            )
        theta_ratio = torch.exp(new_theta_log_probs - old_theta_log_probs)
        clipped_theta_ratio = torch.clamp(
            theta_ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        )
        theta_surrogate = torch.min(
            theta_ratio * theta_advantages,
            clipped_theta_ratio * theta_advantages,
        )
        return {
            "new_log_probs": new_theta_log_probs,
            "old_log_probs": old_theta_log_probs,
            "ratio": theta_ratio,
            "surrogate": theta_surrogate,
            "loss": -theta_surrogate.mean(),
        }

    def _route_branch_ppo_terms(
        self,
        actions: torch.Tensor,
        new_log_prob_components: torch.Tensor,
        old_log_prob_components: torch.Tensor,
        route_advantages: torch.Tensor,
        route_masks: torch.Tensor,
        *,
        new_action_means: torch.Tensor | None = None,
        new_action_stds: torch.Tensor | None = None,
        old_action_means: torch.Tensor | None = None,
        old_action_stds: torch.Tensor | None = None,
        route_gates: torch.Tensor | None = None,
        route_loss_masks: torch.Tensor | None = None,
        route_used_fallback: bool = False,
    ) -> dict[str, torch.Tensor | bool]:
        """Build route-branch PPO terms only on active route blocks."""
        if self._uses_true_conditional_route_policy():
            if new_action_means is None or old_action_means is None:
                raise ValueError(
                    "true conditional route policy requires old/new action means"
                )
            new_route_log_probs = self._true_conditional_route_policy_terms(
                actions,
                new_action_means,
            )["log_probs"]
            old_route_log_probs = self._true_conditional_route_policy_terms(
                actions,
                old_action_means,
            )["log_probs"]
        elif self._uses_factorized_conditional_route_pg():
            if (
                new_action_means is None
                or new_action_stds is None
                or old_action_means is None
                or old_action_stds is None
            ):
                raise ValueError(
                    "conditional route PG requires old/new action means and stds"
                )
            new_route_log_probs = self._route_margin_log_probs(
                actions,
                new_action_means,
                new_action_stds,
            )
            old_route_log_probs = self._route_margin_log_probs(
                actions,
                old_action_means,
                old_action_stds,
            )
        else:
            _, new_route_log_probs = self._block_theta_route_log_prob_components(
                new_log_prob_components
            )
            _, old_route_log_probs = self._block_theta_route_log_prob_components(
                old_log_prob_components
            )
        route_ratio = torch.exp(new_route_log_probs - old_route_log_probs)
        clipped_route_ratio = torch.clamp(
            route_ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        )
        route_surrogate = torch.min(
            route_ratio * route_advantages,
            clipped_route_ratio * route_advantages,
        )
        route_surrogate_for_loss = route_surrogate
        if self._uses_hard_offload_route_surrogate():
            route_surrogate_for_loss = route_masks * route_surrogate_for_loss
        if self._uses_offload_gated_route_surrogate():
            if route_gates is None:
                raise ValueError("route_gates are required for gated route PPO terms")
            route_surrogate_for_loss = route_gates * route_surrogate_for_loss
        route_loss_mask = route_masks if route_loss_masks is None else route_loss_masks
        route_has_active_blocks = bool((route_loss_mask > 0.5).any().item())
        route_loss = route_surrogate.new_zeros(())
        if route_has_active_blocks:
            route_loss = -self._mean_route_surrogate_over_mask(
                route_surrogate_for_loss,
                route_loss_mask,
            )
        return {
            "new_log_probs": new_route_log_probs,
            "old_log_probs": old_route_log_probs,
            "ratio": route_ratio,
            "surrogate": route_surrogate,
            "surrogate_for_loss": route_surrogate_for_loss,
            "loss": route_loss,
            "has_active_blocks": route_has_active_blocks,
            "loss_mask": route_loss_mask,
            "used_fallback": bool(route_used_fallback),
        }

    def _aggregate_log_prob_components_with_states(
        self,
        log_prob_components: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate log-prob terms, using state-dependent block weights when configured."""
        if not self._uses_blockwise_weighted_surrogate_mean():
            return self._aggregate_log_prob_components(log_prob_components)

        block_log_probs = self._block_sum_log_prob_components(log_prob_components)
        block_weights = self._block_activity_weights(states)
        return (block_weights * block_log_probs).sum(dim=-1)

    def _raw_state_block_slices(self) -> dict[str, slice]:
        """Build slices that match the actual raw state assembly order."""
        if self.critic_state_layout is None:
            raise ValueError("critic_state_layout is required for actor raw-state slicing")

        sensor_task_count = int(self.critic_state_layout["sensor_task_count"])
        access_queue_count = int(self.critic_state_layout["access_queue_count"])
        virtual_queue_count = int(self.critic_state_layout["virtual_queue_count"])
        bs_queue_count = int(self.critic_state_layout["bs_queue_count"])
        local_queue_count = int(self.critic_state_layout["local_queue_count"])
        reachable_mask_count = int(self.critic_state_layout["reachable_mask_count"])
        access_gain_count = int(self.critic_state_layout["access_gain_count"])
        compute_rate_count = int(self.critic_state_layout["compute_rate_count"])

        cursor = 0
        sensor_task_core_slice = slice(cursor, cursor + 4 * sensor_task_count)
        cursor = sensor_task_core_slice.stop
        access_queue_slice = slice(cursor, cursor + access_queue_count)
        cursor = access_queue_slice.stop
        virtual_queue_slice = slice(cursor, cursor + virtual_queue_count)
        cursor = virtual_queue_slice.stop
        bs_queue_slice = slice(cursor, cursor + bs_queue_count)
        cursor = bs_queue_slice.stop
        local_queue_slice = slice(cursor, cursor + local_queue_count)
        cursor = local_queue_slice.stop
        reachable_mask_slice = slice(cursor, cursor + reachable_mask_count)
        cursor = reachable_mask_slice.stop
        access_gain_slice = slice(cursor, cursor + access_gain_count)
        cursor = access_gain_slice.stop
        compute_rate_slice = slice(cursor, cursor + compute_rate_count)
        return {
            "sensor_task_core": sensor_task_core_slice,
            "access_queue": access_queue_slice,
            "virtual_queue": virtual_queue_slice,
            "bs_queue": bs_queue_slice,
            "local_queue": local_queue_slice,
            "reachable_mask": reachable_mask_slice,
            "access_gain": access_gain_slice,
            "compute_rate": compute_rate_slice,
        }

    def _slice_indices(self, value_slice: slice) -> np.ndarray:
        """Convert a Python slice into explicit indices."""
        return np.arange(value_slice.start, value_slice.stop, dtype=np.int64)

    def _static_actor_raw_indices(self) -> np.ndarray:
        """Return the dense-static raw-state indices that are candidate pruning targets."""
        raw_slices = self._raw_state_block_slices()
        static_names = ("reachable_mask", "access_gain", "compute_rate")
        return np.concatenate(
            [self._slice_indices(raw_slices[name]) for name in static_names],
            axis=0,
        )

    def _selected_actor_raw_indices(self) -> np.ndarray:
        """Return the raw-state indices that are kept for the actor."""
        if not self._uses_normalized_augmented_actor_input():
            return np.arange(self.state_dim, dtype=np.int64)

        raw_slices = self._raw_state_block_slices()
        mode = self._actor_raw_input_mode()
        if mode == "baseline_actor_raw":
            excluded_blocks: set[str] = set()
        elif mode == "prune_compute18_actor_raw":
            excluded_blocks = {"compute_rate"}
        elif mode == "prune_all_static42_actor_raw":
            excluded_blocks = {"reachable_mask", "access_gain", "compute_rate"}
        else:
            raise ValueError(f"Unsupported actor_raw_input_mode: {mode}")

        selected_blocks = [
            self._slice_indices(block_slice)
            for block_name, block_slice in raw_slices.items()
            if block_name not in excluded_blocks
        ]
        return np.concatenate(selected_blocks, axis=0) if selected_blocks else np.zeros(0, dtype=np.int64)

    def _actor_raw_input_dim(self) -> int:
        """Return the actor raw-state dimension after raw-block pruning."""
        return int(self._selected_actor_raw_indices().size)

    def describe_actor_raw_layout(self) -> dict[str, Any]:
        """Describe raw-state block slices and actor input dimensions for analysis scripts."""
        raw_slices = self._raw_state_block_slices()
        selected_indices = self._selected_actor_raw_indices()
        static_indices = self._static_actor_raw_indices()
        blocks = {}
        for name, block_slice in raw_slices.items():
            blocks[name] = {
                "start": int(block_slice.start),
                "stop_exclusive": int(block_slice.stop),
                "dim": int(block_slice.stop - block_slice.start),
                "start_1based": int(block_slice.start + 1),
                "end_1based": int(block_slice.stop),
            }
        return {
            "actor_raw_input_mode": self._actor_raw_input_mode(),
            "state_dim": int(self.state_dim),
            "actor_raw_dim": int(selected_indices.size),
            "actor_input_total_dim": int(self.actor_input_dim),
            "actor_input_derived_dim": int(self.actor_derived_feature_dim),
            "actor_raw_pruned_dim_count": int(self.state_dim - selected_indices.size),
            "selected_raw_indices": selected_indices.tolist(),
            "static_raw_indices": static_indices.tolist(),
            "raw_state_blocks": blocks,
        }

    def _state_slices(self) -> dict[str, slice]:
        """Build semantic slices for the normalized state vector."""
        if self.critic_state_layout is None:
            raise ValueError("critic_state_layout is required for augmented actor/critic inputs")

        sensor_task_count = self.critic_state_layout["sensor_task_count"]
        access_queue_count = self.critic_state_layout["access_queue_count"]
        virtual_queue_count = self.critic_state_layout["virtual_queue_count"]
        bs_queue_count = self.critic_state_layout["bs_queue_count"]
        local_queue_count = self.critic_state_layout["local_queue_count"]
        reachable_mask_count = self.critic_state_layout["reachable_mask_count"]
        access_gain_count = self.critic_state_layout["access_gain_count"]
        compute_rate_count = self.critic_state_layout["compute_rate_count"]

        cursor = 0
        lambda_slice = slice(cursor, cursor + sensor_task_count)
        cursor += sensor_task_count
        data_rate_slice = slice(cursor, cursor + sensor_task_count)
        cursor += sensor_task_count
        workload_slice = slice(cursor, cursor + sensor_task_count)
        cursor += sensor_task_count
        predicted_slice = slice(cursor, cursor + sensor_task_count)
        cursor += sensor_task_count
        access_queue_slice = slice(cursor, cursor + access_queue_count)
        cursor += access_queue_count
        virtual_queue_slice = slice(cursor, cursor + virtual_queue_count)
        cursor += virtual_queue_count
        bs_queue_slice = slice(cursor, cursor + bs_queue_count)
        cursor += bs_queue_count
        local_queue_slice = slice(cursor, cursor + local_queue_count)
        cursor += local_queue_count
        reachable_mask_slice = slice(cursor, cursor + reachable_mask_count)
        cursor += reachable_mask_count
        access_gain_slice = slice(cursor, cursor + access_gain_count)
        cursor += access_gain_count
        compute_rate_slice = slice(cursor, cursor + compute_rate_count)
        return {
            "lambda": lambda_slice,
            "data_rate": data_rate_slice,
            "workload": workload_slice,
            "predicted": predicted_slice,
            "access_queue": access_queue_slice,
            "virtual_queue": virtual_queue_slice,
            "bs_queue": bs_queue_slice,
            "local_queue": local_queue_slice,
            "reachable_mask": reachable_mask_slice,
            "access_gain": access_gain_slice,
            "compute_rate": compute_rate_slice,
        }

    def _shared_augmented_feature_dim(self) -> int:
        """Return the shared derived-feature dimension reused by actor and critic."""
        slices = self._state_slices()
        centered_dims = (
            (slices["lambda"].stop - slices["lambda"].start)
            + (slices["workload"].stop - slices["workload"].start)
            + (slices["predicted"].stop - slices["predicted"].start)
            + (slices["access_queue"].stop - slices["access_queue"].start)
            + (slices["virtual_queue"].stop - slices["virtual_queue"].start)
            + (slices["bs_queue"].stop - slices["bs_queue"].start)
            + (slices["local_queue"].stop - slices["local_queue"].start)
            + (slices["access_gain"].stop - slices["access_gain"].start)
            + (slices["compute_rate"].stop - slices["compute_rate"].start)
        )
        aggregate_dim = 14
        return centered_dims + aggregate_dim

    def _actor_extra_feature_dim(self) -> int:
        """Return the extra actor-only policy feature dimension."""
        if self.critic_state_layout is None:
            return 0
        sensor_count = int(self.critic_state_layout["sensor_count"])
        aggregate_dim = 10
        return 2 * sensor_count + aggregate_dim

    def _critic_derived_feature_dim(self) -> int:
        """Return the derived-feature dimension used by critic-only input augmentation."""
        if not self._uses_normalized_augmented_critic_input():
            return 0
        return self._shared_augmented_feature_dim()

    def _actor_derived_feature_dim(self) -> int:
        """Return the derived-feature dimension used by actor-only input augmentation."""
        if not self._uses_normalized_augmented_actor_input():
            return 0
        return self._shared_augmented_feature_dim() + self._actor_extra_feature_dim()

    def _running_actor_input_stats_tensors(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create actor-input running stats tensors on the same device and dtype."""
        if self.actor_input_running_mean is None or self.actor_input_running_var is None:
            feature_dim = reference.shape[-1]
            return (
                torch.zeros(feature_dim, dtype=reference.dtype, device=reference.device),
                torch.ones(feature_dim, dtype=reference.dtype, device=reference.device),
            )
        return (
            torch.tensor(
                self.actor_input_running_mean,
                dtype=reference.dtype,
                device=reference.device,
            ),
            torch.tensor(
                np.sqrt(np.maximum(self.actor_input_running_var, 1e-8)),
                dtype=reference.dtype,
                device=reference.device,
            ),
        )

    def _update_running_actor_input_stats(self, actor_inputs: torch.Tensor) -> None:
        """Update running mean and variance for actor-only processed inputs."""
        batch_mean = actor_inputs.mean(dim=0).detach().cpu().numpy()
        batch_var = actor_inputs.var(dim=0, unbiased=False).detach().cpu().numpy()
        batch_count = float(actor_inputs.shape[0])

        if self.actor_input_running_mean is None or self.actor_input_running_var is None:
            self.actor_input_running_mean = batch_mean
            self.actor_input_running_var = np.maximum(batch_var, 1e-8)
            self.actor_input_count = batch_count
            return

        delta = batch_mean - self.actor_input_running_mean
        total_count = self.actor_input_count + batch_count
        mean = self.actor_input_running_mean + delta * batch_count / total_count
        m_a = self.actor_input_running_var * self.actor_input_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.actor_input_count * batch_count / total_count

        self.actor_input_running_mean = mean
        self.actor_input_running_var = np.maximum(m2 / total_count, 1e-8)
        self.actor_input_count = total_count

    def _running_critic_input_stats_tensors(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create critic-input running stats tensors on the same device and dtype."""
        if self.critic_input_running_mean is None or self.critic_input_running_var is None:
            feature_dim = reference.shape[-1]
            return (
                torch.zeros(feature_dim, dtype=reference.dtype, device=reference.device),
                torch.ones(feature_dim, dtype=reference.dtype, device=reference.device),
            )
        return (
            torch.tensor(
                self.critic_input_running_mean,
                dtype=reference.dtype,
                device=reference.device,
            ),
            torch.tensor(
                np.sqrt(np.maximum(self.critic_input_running_var, 1e-8)),
                dtype=reference.dtype,
                device=reference.device,
            ),
        )

    def _update_running_critic_input_stats(self, critic_inputs: torch.Tensor) -> None:
        """Update running mean and variance for critic-only processed inputs."""
        batch_mean = critic_inputs.mean(dim=0).detach().cpu().numpy()
        batch_var = critic_inputs.var(dim=0, unbiased=False).detach().cpu().numpy()
        batch_count = float(critic_inputs.shape[0])

        if self.critic_input_running_mean is None or self.critic_input_running_var is None:
            self.critic_input_running_mean = batch_mean
            self.critic_input_running_var = np.maximum(batch_var, 1e-8)
            self.critic_input_count = batch_count
            return

        delta = batch_mean - self.critic_input_running_mean
        total_count = self.critic_input_count + batch_count
        mean = self.critic_input_running_mean + delta * batch_count / total_count
        m_a = self.critic_input_running_var * self.critic_input_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * self.critic_input_count * batch_count / total_count

        self.critic_input_running_mean = mean
        self.critic_input_running_var = np.maximum(m2 / total_count, 1e-8)
        self.critic_input_count = total_count

    def _augment_critic_input(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build critic-only augmented features from the raw normalized state vector."""
        if not self._uses_normalized_augmented_critic_input():
            return states, states.new_zeros((states.shape[0], 0))

        slices = self._state_slices()
        lambda_block = states[:, slices["lambda"]]
        workload_block = states[:, slices["workload"]]
        predicted_block = states[:, slices["predicted"]]
        access_queue_block = states[:, slices["access_queue"]]
        virtual_queue_block = states[:, slices["virtual_queue"]]
        bs_queue_block = states[:, slices["bs_queue"]]
        local_queue_block = states[:, slices["local_queue"]]
        access_gain_block = states[:, slices["access_gain"]]
        compute_rate_block = states[:, slices["compute_rate"]]

        centered_blocks = [
            lambda_block - lambda_block.mean(dim=1, keepdim=True),
            workload_block - workload_block.mean(dim=1, keepdim=True),
            predicted_block - predicted_block.mean(dim=1, keepdim=True),
            access_queue_block - access_queue_block.mean(dim=1, keepdim=True),
            virtual_queue_block - virtual_queue_block.mean(dim=1, keepdim=True),
            bs_queue_block - bs_queue_block.mean(dim=1, keepdim=True),
            local_queue_block - local_queue_block.mean(dim=1, keepdim=True),
            access_gain_block - access_gain_block.mean(dim=1, keepdim=True),
            compute_rate_block - compute_rate_block.mean(dim=1, keepdim=True),
        ]

        all_queue = torch.cat(
            [access_queue_block, virtual_queue_block, bs_queue_block, local_queue_block],
            dim=1,
        )
        predicted_mean = predicted_block.mean(dim=1)
        compute_rate_mean = compute_rate_block.mean(dim=1)
        access_queue_mean = access_queue_block.mean(dim=1)
        virtual_queue_mean = virtual_queue_block.mean(dim=1)
        bs_queue_mean = bs_queue_block.mean(dim=1)
        local_queue_mean = local_queue_block.mean(dim=1)
        aggregate_features = torch.stack(
            [
                lambda_block.mean(dim=1),
                workload_block.mean(dim=1),
                predicted_mean,
                predicted_block.max(dim=1).values,
                access_queue_mean,
                virtual_queue_mean,
                bs_queue_mean,
                local_queue_mean,
                access_gain_block.mean(dim=1),
                compute_rate_mean,
                predicted_mean - compute_rate_mean,
                local_queue_mean - bs_queue_mean,
                virtual_queue_mean - access_queue_mean,
                all_queue.mean(dim=1),
            ],
            dim=1,
        )
        derived = torch.cat([*centered_blocks, aggregate_features], dim=1)
        return torch.cat([states, derived], dim=1), derived

    def _build_actor_policy_features(self, states: torch.Tensor) -> torch.Tensor:
        """Build a compact actor-only feature pack around load, queue, and bottleneck ratios."""
        slices = self._state_slices()
        if self.critic_state_layout is None:
            raise ValueError("critic_state_layout is required for actor policy features")

        sensor_count = int(self.critic_state_layout["sensor_count"])
        task_count = int(self.critic_state_layout["task_count"])
        compute_unit_count = int(self.critic_state_layout["compute_rate_count"]) // max(task_count, 1)
        eps = 1e-6

        lambda_block = states[:, slices["lambda"]].reshape(-1, sensor_count, task_count)
        data_rate_block = states[:, slices["data_rate"]].reshape(-1, sensor_count, task_count)
        workload_block = states[:, slices["workload"]].reshape(-1, sensor_count, task_count)
        predicted_block = states[:, slices["predicted"]].reshape(-1, sensor_count, task_count)
        access_queue_block = states[:, slices["access_queue"]]
        virtual_queue_block = states[:, slices["virtual_queue"]].reshape(-1, sensor_count, task_count)
        bs_queue_block = states[:, slices["bs_queue"]]
        local_queue_block = states[:, slices["local_queue"]].reshape(-1, sensor_count, task_count)
        access_gain_block = states[:, slices["access_gain"]]
        compute_rate_block = states[:, slices["compute_rate"]].reshape(-1, compute_unit_count, task_count)

        sensor_queue_pressure = (
            access_queue_block
            + virtual_queue_block.mean(dim=2)
            + local_queue_block.mean(dim=2)
        )
        sensor_queue_pressure_centered = sensor_queue_pressure - sensor_queue_pressure.mean(
            dim=1,
            keepdim=True,
        )

        sensor_load_delta = predicted_block.mean(dim=2) - lambda_block.mean(dim=2)
        sensor_load_delta_centered = sensor_load_delta - sensor_load_delta.mean(
            dim=1,
            keepdim=True,
        )

        data_rate_mean = data_rate_block.mean(dim=(1, 2))
        data_rate_max = data_rate_block.reshape(states.shape[0], -1).max(dim=1).values
        access_queue_mean = access_queue_block.mean(dim=1)
        access_gain_mean = access_gain_block.mean(dim=1)
        virtual_queue_mean = virtual_queue_block.mean(dim=(1, 2))
        local_queue_mean = local_queue_block.mean(dim=(1, 2))
        bs_queue_mean = bs_queue_block.mean(dim=1)
        predicted_mean = predicted_block.mean(dim=(1, 2))
        lambda_mean = lambda_block.mean(dim=(1, 2))
        workload_mean = workload_block.mean(dim=(1, 2))
        compute_rate_mean = compute_rate_block.mean(dim=(1, 2))
        predicted_task_mean = predicted_block.mean(dim=1)
        lambda_task_mean = lambda_block.mean(dim=1)
        compute_rate_task_mean = compute_rate_block.mean(dim=1)

        aggregate_features = torch.stack(
            [
                data_rate_mean,
                data_rate_max,
                data_rate_mean / (access_gain_mean + eps),
                workload_mean / (compute_rate_mean + eps),
                predicted_mean / (compute_rate_mean + eps),
                access_queue_mean / (access_gain_mean + eps),
                virtual_queue_mean / (access_queue_mean + eps),
                local_queue_mean / (bs_queue_mean + eps),
                (predicted_task_mean - lambda_task_mean).mean(dim=1),
                (predicted_task_mean - compute_rate_task_mean).mean(dim=1),
            ],
            dim=1,
        )

        return torch.cat(
            [
                sensor_queue_pressure_centered,
                sensor_load_delta_centered,
                aggregate_features,
            ],
            dim=1,
        )

    def _select_actor_raw_input(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        """Select the actor-visible raw-state blocks and summarize their variability."""
        selected_indices = self._selected_actor_raw_indices()
        if (
            selected_indices.size == self.state_dim
            and np.array_equal(selected_indices, np.arange(self.state_dim, dtype=np.int64))
        ):
            selected_states = states
        else:
            index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=states.device)
            selected_states = states.index_select(dim=1, index=index_tensor)

        raw_dim_std = selected_states.std(dim=0, unbiased=False)
        zero_var_mask = raw_dim_std <= 1e-8
        if self.critic_state_layout is not None:
            static_index_set = set(self._static_actor_raw_indices().tolist())
            static_mask = torch.tensor(
                [int(index in static_index_set) for index in selected_indices],
                dtype=torch.bool,
                device=states.device,
            )
            static_zero_var_count = int((zero_var_mask & static_mask).sum().item())
        else:
            static_zero_var_count = 0

        diagnostics: dict[str, float | int] = {
            "actor_raw_dim": int(selected_states.shape[-1]),
            "actor_raw_pruned_dim_count": int(self.state_dim - selected_states.shape[-1]),
            "actor_raw_dim_std_mean": float(raw_dim_std.mean().item()),
            "actor_raw_dim_std_min": float(raw_dim_std.min().item()),
            "actor_raw_zero_var_dim_count": int(zero_var_mask.sum().item()),
            "actor_raw_static_zero_var_dim_count": static_zero_var_count,
        }
        return selected_states, diagnostics

    def _augment_actor_input(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
        """Build actor-only augmented features from the raw normalized state vector."""
        actor_raw_inputs, actor_raw_diagnostics = self._select_actor_raw_input(states)
        if not self._uses_normalized_augmented_actor_input():
            return actor_raw_inputs, states.new_zeros((states.shape[0], 0)), actor_raw_diagnostics
        _, shared_features = self._augment_critic_input(states)
        actor_policy_features = self._build_actor_policy_features(states)
        derived = torch.cat([shared_features, actor_policy_features], dim=1)
        return torch.cat([actor_raw_inputs, derived], dim=1), derived, actor_raw_diagnostics

    def _prepare_actor_inputs(
        self,
        states: torch.Tensor,
        update_stats: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        """Prepare actor-only inputs and summarize their statistics."""
        augmented_inputs, derived_features, actor_raw_diagnostics = self._augment_actor_input(states)
        if self._uses_normalized_augmented_actor_input():
            if update_stats:
                self._update_running_actor_input_stats(augmented_inputs)
            input_mean, input_std = self._running_actor_input_stats_tensors(augmented_inputs)
            normalized_inputs = (augmented_inputs - input_mean) / (input_std + 1e-8)
            clipped_inputs = torch.clamp(normalized_inputs, -5.0, 5.0)
            clip_fraction = float((clipped_inputs.ne(normalized_inputs)).float().mean().item())
            processed_inputs = clipped_inputs
        else:
            processed_inputs = augmented_inputs
            clip_fraction = 0.0

        input_dim_std = processed_inputs.std(dim=0, unbiased=False)
        diagnostics = {
            "actor_input_mean": float(processed_inputs.mean().item()),
            "actor_input_std": float(processed_inputs.std(unbiased=False).item()),
            "actor_input_dim_std_mean": float(input_dim_std.mean().item()),
            "actor_input_dim_std_min": float(input_dim_std.min().item()),
            "actor_input_clip_fraction": clip_fraction,
            "actor_input_derived_dim": int(self.actor_derived_feature_dim),
            "actor_input_total_dim": int(processed_inputs.shape[-1]),
            "actor_derived_feature_mean": float(
                derived_features.mean().item() if derived_features.numel() > 0 else 0.0
            ),
            "actor_derived_feature_std": float(
                derived_features.std(unbiased=False).item() if derived_features.numel() > 0 else 0.0
            ),
        }
        diagnostics.update(actor_raw_diagnostics)
        return processed_inputs, diagnostics, derived_features

    def _prepare_critic_inputs(
        self,
        states: torch.Tensor,
        update_stats: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        """Prepare critic-only inputs and summarize their statistics."""
        augmented_inputs, derived_features = self._augment_critic_input(states)
        if self._uses_normalized_augmented_critic_input():
            if update_stats:
                self._update_running_critic_input_stats(augmented_inputs)
            input_mean, input_std = self._running_critic_input_stats_tensors(augmented_inputs)
            normalized_inputs = (augmented_inputs - input_mean) / (input_std + 1e-8)
            clipped_inputs = torch.clamp(normalized_inputs, -5.0, 5.0)
            clip_fraction = float((clipped_inputs.ne(normalized_inputs)).float().mean().item())
            processed_inputs = clipped_inputs
        else:
            processed_inputs = augmented_inputs
            clip_fraction = 0.0

        input_dim_std = processed_inputs.std(dim=0, unbiased=False)
        diagnostics = {
            "critic_input_mean": float(processed_inputs.mean().item()),
            "critic_input_std": float(processed_inputs.std(unbiased=False).item()),
            "critic_input_dim_std_mean": float(input_dim_std.mean().item()),
            "critic_input_dim_std_min": float(input_dim_std.min().item()),
            "critic_input_clip_fraction": clip_fraction,
            "critic_input_derived_dim": int(self.critic_derived_feature_dim),
            "critic_input_total_dim": int(processed_inputs.shape[-1]),
            "critic_derived_feature_mean": float(
                derived_features.mean().item() if derived_features.numel() > 0 else 0.0
            ),
            "critic_derived_feature_std": float(
                derived_features.std(unbiased=False).item() if derived_features.numel() > 0 else 0.0
            ),
        }
        return processed_inputs, diagnostics, derived_features

    def _current_running_return_std(self) -> float:
        """Return the current running return standard deviation."""
        return float(np.sqrt(max(self.running_return_var, 1e-8)))

    def _running_stats_tensors(
        self,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create running-stat tensors on the same device/dtype as the reference tensor."""
        return (
            torch.tensor(self.running_return_mean, dtype=reference.dtype, device=reference.device),
            torch.tensor(self._current_running_return_std(), dtype=reference.dtype, device=reference.device),
        )

    def select_action_batch(
        self,
        states: np.ndarray,
        deterministic: bool = False,
        return_policy_cache: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, np.ndarray],
    ]:
        """Batch actor/value inference for one or more states."""
        state_array = np.asarray(states, dtype=np.float32)
        if state_array.ndim == 1:
            state_array = np.expand_dims(state_array, axis=0)
        state_tensor = torch.as_tensor(state_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actor_inputs, _, _ = self._prepare_actor_inputs(state_tensor, update_stats=False)
            distribution = self.network.policy_from_actor_input(actor_inputs)
            value_tensor = self._value_from_state_tensor(state_tensor).squeeze(-1)
            action_tensor = distribution.mean if deterministic else distribution.sample()
            log_prob_components_tensor = distribution.log_prob(action_tensor)
            log_prob_tensor = self._aggregate_log_prob_components_with_states(
                log_prob_components_tensor,
                state_tensor,
            )

        action_array = action_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        log_prob_array = log_prob_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        value_array = value_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        if not return_policy_cache:
            return action_array, log_prob_array, value_array

        policy_cache = {
            "log_prob_components": log_prob_components_tensor.detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False),
            "action_mean": distribution.mean.detach().cpu().numpy().astype(np.float32, copy=False),
            "action_std": distribution.stddev.detach().cpu().numpy().astype(np.float32, copy=False),
        }
        return action_array, log_prob_array, value_array, policy_cache

    def select_action_with_info(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float, dict[str, np.ndarray]]:
        """Single-state wrapper that also returns cached policy outputs for rollout reuse."""
        action_batch, log_prob_batch, value_batch, policy_cache = self.select_action_batch(
            state,
            deterministic=deterministic,
            return_policy_cache=True,
        )
        single_cache = {
            key: np.asarray(value[0], dtype=np.float32).copy()
            for key, value in policy_cache.items()
        }
        return (
            np.asarray(action_batch[0], dtype=np.float32).copy(),
            float(log_prob_batch[0]),
            float(value_batch[0]),
            single_cache,
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """Backward-compatible single-state actor API."""
        action, log_prob, value, _policy_cache = self.select_action_with_info(
            state,
            deterministic=deterministic,
        )
        return action, log_prob, value

    def evaluate_value(self, state: np.ndarray) -> float:
        """Estimate the state value."""
        return float(self.evaluate_value_batch(state)[0])

    def evaluate_value_batch(self, states: np.ndarray) -> np.ndarray:
        """Batch critic inference for one or more states."""
        state_array = np.asarray(states, dtype=np.float32)
        if state_array.ndim == 1:
            state_array = np.expand_dims(state_array, axis=0)
        state_tensor = torch.as_tensor(state_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            value = self._value_from_state_tensor(state_tensor).squeeze(-1)
        return value.detach().cpu().numpy().astype(np.float32, copy=False)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        next_state: np.ndarray | None = None,
        joint_reward_aligned_scores: np.ndarray | None = None,
        joint_td_aligned_scores: np.ndarray | None = None,
        policy_cache: dict[str, np.ndarray] | None = None,
        buffer: PPOBuffer | None = None,
    ) -> None:
        """Store one environment transition."""
        if policy_cache is None:
            state_tensor = torch.as_tensor(
                np.asarray(state, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action_tensor = torch.as_tensor(
                np.asarray(action, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            with torch.no_grad():
                actor_inputs, _, _ = self._prepare_actor_inputs(state_tensor, update_stats=False)
                distribution = self.network.policy_from_actor_input(actor_inputs)
                log_prob_components = (
                    distribution.log_prob(action_tensor).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
                )
                action_mean = distribution.mean.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
                action_std = distribution.stddev.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        else:
            log_prob_components = np.asarray(
                policy_cache["log_prob_components"],
                dtype=np.float32,
            )
            action_mean = np.asarray(policy_cache["action_mean"], dtype=np.float32)
            action_std = np.asarray(policy_cache["action_std"], dtype=np.float32)
        target_buffer = buffer if buffer is not None else self.buffer
        target_buffer.store_transition(
            state,
            action,
            log_prob,
            log_prob_components,
            action_mean,
            action_std,
            reward,
            done,
            value,
            next_state=(
                np.asarray(next_state, dtype=np.float32)
                if next_state is not None
                else None
            ),
            joint_reward_aligned_scores=(
                np.asarray(joint_reward_aligned_scores, dtype=np.float32)
                if joint_reward_aligned_scores is not None
                else None
            ),
            joint_td_aligned_scores=(
                np.asarray(joint_td_aligned_scores, dtype=np.float32)
                if joint_td_aligned_scores is not None
                else None
            ),
        )

    def store_transition_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_states: np.ndarray | None = None,
        joint_reward_aligned_scores: np.ndarray | None = None,
        joint_td_aligned_scores: np.ndarray | None = None,
        policy_cache: dict[str, np.ndarray] | None = None,
        buffers: list[PPOBuffer] | None = None,
    ) -> None:
        """Batch wrapper over the rollout buffer for vectorized sampling."""
        state_array = np.asarray(states, dtype=np.float32)
        action_array = np.asarray(actions, dtype=np.float32)
        log_prob_array = np.asarray(log_probs, dtype=np.float32)
        reward_array = np.asarray(rewards, dtype=np.float32)
        done_array = np.asarray(dones, dtype=np.float32)
        value_array = np.asarray(values, dtype=np.float32)
        next_state_array = (
            np.asarray(next_states, dtype=np.float32)
            if next_states is not None
            else None
        )
        reward_score_array = (
            np.asarray(joint_reward_aligned_scores, dtype=np.float32)
            if joint_reward_aligned_scores is not None
            else None
        )
        td_score_array = (
            np.asarray(joint_td_aligned_scores, dtype=np.float32)
            if joint_td_aligned_scores is not None
            else None
        )
        cache_arrays = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in (policy_cache or {}).items()
        }
        for index in range(state_array.shape[0]):
            single_cache = (
                {key: value[index] for key, value in cache_arrays.items()}
                if cache_arrays
                else None
            )
            self.store_transition(
                state=state_array[index],
                action=action_array[index],
                log_prob=float(log_prob_array[index]),
                reward=float(reward_array[index]),
                done=bool(done_array[index]),
                value=float(value_array[index]),
                next_state=(
                    next_state_array[index]
                    if next_state_array is not None
                    else None
                ),
                joint_reward_aligned_scores=(
                    reward_score_array[index]
                    if reward_score_array is not None
                    else None
                ),
                joint_td_aligned_scores=(
                    td_score_array[index]
                    if td_score_array is not None
                    else None
                ),
                policy_cache=single_cache,
                buffer=(buffers[index] if buffers is not None else None),
            )

    def finish_trajectory(self, last_value: float) -> None:
        """Finish the current trajectory and compute GAE."""
        self.buffer.finish_trajectory(last_value, self.config.gamma, self.config.gae_lambda)

    def _update_running_return_stats(
        self,
        returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update running return statistics and return old/new tensors."""
        old_mean, old_std = self._running_stats_tensors(returns)
        batch_mean = float(returns.mean().item())
        batch_var = float(returns.var(unbiased=False).item())
        batch_count = float(returns.numel())

        if self.running_return_count == 0.0:
            self.running_return_mean = batch_mean
            self.running_return_var = max(batch_var, 1e-8)
            self.running_return_count = batch_count
        else:
            delta = batch_mean - self.running_return_mean
            total_count = self.running_return_count + batch_count
            mean = self.running_return_mean + delta * batch_count / total_count
            m_a = self.running_return_var * self.running_return_count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + (delta**2) * self.running_return_count * batch_count / total_count

            self.running_return_mean = mean
            self.running_return_var = max(m2 / total_count, 1e-8)
            self.running_return_count = total_count

        running_mean = torch.tensor(
            self.running_return_mean,
            dtype=returns.dtype,
            device=returns.device,
        )
        running_std = torch.tensor(
            float(np.sqrt(max(self.running_return_var, 1e-8))),
            dtype=returns.dtype,
            device=returns.device,
        )
        return old_mean, old_std, running_mean, running_std

    def _value_from_state_tensor(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Estimate raw values for arbitrary states, respecting PopArt if enabled."""
        critic_inputs, _, _ = self._prepare_critic_inputs(state_tensor, update_stats=False)
        if self.config.value_target_mode == "popart_return_norm":
            mean, std = self._running_stats_tensors(state_tensor)
            return self.network.value_from_critic_input(critic_inputs, mean, std)
        return self.network.value_from_critic_input(critic_inputs)

    def _compute_value_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the critic loss for the configured value loss mode."""
        if self.config.value_loss_mode == "mse":
            return nn.functional.mse_loss(predictions, targets)
        if self.config.value_loss_mode == "huber":
            return nn.functional.huber_loss(predictions, targets, delta=1.0)
        raise ValueError(f"Unsupported value_loss_mode: {self.config.value_loss_mode}")

    def _compute_value_diagnostics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        """Summarize how well critic predictions match the chosen value targets."""
        prediction_std = predictions.std(unbiased=False)
        target_std = targets.std(unbiased=False)
        target_var = targets.var(unbiased=False)

        if float(target_var.item()) <= 1e-8:
            explained_variance = 0.0
        else:
            residual_var = (targets - predictions).var(unbiased=False)
            explained_variance = float((1.0 - residual_var / (target_var + 1e-8)).item())

        if float(prediction_std.item()) <= 1e-8 or float(target_std.item()) <= 1e-8:
            prediction_target_corr = 0.0
        else:
            centered_predictions = predictions - predictions.mean()
            centered_targets = targets - targets.mean()
            covariance = (centered_predictions * centered_targets).mean()
            prediction_target_corr = float(
                (covariance / (prediction_std * target_std + 1e-8)).item()
            )

        prediction_std_over_target_std = float(
            (prediction_std / (target_std + 1e-8)).item()
        )
        return {
            "value_explained_variance": explained_variance,
            "prediction_target_corr": prediction_target_corr,
            "prediction_std_over_target_std": prediction_std_over_target_std,
        }

    def _grad_norm(self, params: list[nn.Parameter]) -> float:
        """Compute the global L2 norm of parameter gradients."""
        grad_norm_sq = 0.0
        for param in params:
            if param.grad is None:
                continue
            grad_norm_sq += float(param.grad.detach().pow(2).sum().item())
        return float(np.sqrt(grad_norm_sq))

    def _selected_action_histogram(self, actions: torch.Tensor) -> dict[str, int]:
        """Bucket sampled continuous actions for lightweight rollout diagnostics."""
        flat_actions = actions.detach().cpu().numpy().reshape(-1)
        bins = np.asarray([-np.inf, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, np.inf], dtype=np.float32)
        labels = [
            "lt_neg_2",
            "neg_2_to_neg_1",
            "neg_1_to_neg_0p5",
            "neg_0p5_to_0",
            "zero_to_0p5",
            "0p5_to_1",
            "1_to_2",
            "gt_2",
        ]
        counts, _ = np.histogram(flat_actions, bins=bins)
        return {label: int(count) for label, count in zip(labels, counts, strict=True)}

    def _safe_correlation(self, lhs: torch.Tensor, rhs: torch.Tensor) -> float:
        """Compute a stable correlation coefficient, returning 0 when variance collapses."""
        if lhs.numel() < 2 or rhs.numel() < 2:
            return 0.0
        centered_lhs = lhs - lhs.mean()
        centered_rhs = rhs - rhs.mean()
        lhs_std = centered_lhs.std(unbiased=False)
        rhs_std = centered_rhs.std(unbiased=False)
        if float(lhs_std.item()) <= 1e-8 or float(rhs_std.item()) <= 1e-8:
            return 0.0
        correlation = (centered_lhs * centered_rhs).mean() / (lhs_std * rhs_std + 1e-8)
        return float(correlation.item())

    def _safe_spearman_correlation(self, lhs: torch.Tensor, rhs: torch.Tensor) -> float:
        """Compute a stable Spearman correlation coefficient, returning 0 when variance collapses."""
        if lhs.numel() < 2 or rhs.numel() < 2:
            return 0.0
        lhs_np = lhs.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
        rhs_np = rhs.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
        if lhs_np.size < 2 or rhs_np.size < 2:
            return 0.0
        lhs_ranks = np.argsort(np.argsort(lhs_np, kind="mergesort"), kind="mergesort").astype(
            np.float32,
            copy=False,
        )
        rhs_ranks = np.argsort(np.argsort(rhs_np, kind="mergesort"), kind="mergesort").astype(
            np.float32,
            copy=False,
        )
        lhs_tensor = torch.tensor(lhs_ranks, dtype=lhs.dtype, device=lhs.device)
        rhs_tensor = torch.tensor(rhs_ranks, dtype=rhs.dtype, device=rhs.device)
        return self._safe_correlation(lhs_tensor, rhs_tensor)

    def _value_targets_from_returns_with_stats(
        self,
        returns: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> torch.Tensor:
        """Project raw returns into the value-target space used by the current critic objective."""
        if self.config.value_target_mode == "raw_return":
            return returns
        if self.config.value_target_mode in {
            "normalized_return",
            "running_return_norm",
            "popart_return_norm",
        }:
            return (returns - target_mean) / (target_std + 1e-8)
        raise ValueError(f"Unsupported value_target_mode: {self.config.value_target_mode}")

    def _trace_raw_value_predictions_from_critic_inputs(
        self,
        critic_inputs: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw-space critic predictions for tracing without mutating running stats."""
        if self.config.value_target_mode == "popart_return_norm":
            normalized_predictions = self.network.normalized_value_from_critic_input(
                critic_inputs
            ).squeeze(-1)
            return normalized_predictions * target_std + target_mean
        return self.network.value_from_critic_input(critic_inputs).squeeze(-1)

    def _critic_trace_semantic_summary(
        self,
        value_predictions: torch.Tensor,
        value_targets: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        one_step_rewards: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Summarize critic semantics on one batch/probe slice for drift tracing."""
        summary = {
            "value_pred_mean": float(value_predictions.mean().item()),
            "value_pred_std": float(value_predictions.std(unbiased=False).item()),
            "value_target_mean": float(value_targets.mean().item()),
            "value_target_std": float(value_targets.std(unbiased=False).item()),
            "return_mean": float(returns.mean().item()),
            "return_std": float(returns.std(unbiased=False).item()),
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std(unbiased=False).item()),
            "pearson_value_vs_value_target": self._safe_correlation(
                value_predictions,
                value_targets,
            ),
            "spearman_value_vs_value_target": self._safe_spearman_correlation(
                value_predictions,
                value_targets,
            ),
            "pearson_value_vs_return": self._safe_correlation(value_predictions, returns),
            "spearman_value_vs_return": self._safe_spearman_correlation(
                value_predictions,
                returns,
            ),
            "pearson_value_vs_advantage": self._safe_correlation(
                value_predictions,
                advantages,
            ),
            "spearman_value_vs_advantage": self._safe_spearman_correlation(
                value_predictions,
                advantages,
            ),
            "pearson_value_vs_one_step_reward": 0.0,
            "spearman_value_vs_one_step_reward": 0.0,
        }
        if one_step_rewards is not None:
            summary["pearson_value_vs_one_step_reward"] = self._safe_correlation(
                value_predictions,
                one_step_rewards,
            )
            summary["spearman_value_vs_one_step_reward"] = self._safe_spearman_correlation(
                value_predictions,
                one_step_rewards,
            )
        return summary

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> float:
        """Return the mean over a boolean mask, falling back to 0 for empty selections."""
        if bool(mask.any().item()):
            return float(values[mask].mean().item())
        return 0.0

    def _masked_fraction(self, mask: torch.Tensor, reference: torch.Tensor) -> float:
        """Return the masked fraction relative to the reference tensor length."""
        if reference.numel() == 0:
            return 0.0
        return float(mask.float().mean().item())

    def _summary_stats(self, values: torch.Tensor) -> dict[str, float | int]:
        """Summarize a 1-D tensor for lightweight logging."""
        if values.numel() == 0:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        return {
            "count": int(values.numel()),
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
        }

    def _advantage_bucket_masks(
        self,
        raw_advantages: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Partition rollout samples into high/mid/near-zero/negative advantage buckets."""
        near_zero_threshold = max(
            float(raw_advantages.std(unbiased=False).item())
            * float(self.config.policy_update_near_zero_adv_std_scale),
            1e-8,
        )
        positive_large = raw_advantages[raw_advantages > near_zero_threshold]
        if positive_large.numel() > 0:
            high_positive_threshold = max(
                float(
                    torch.quantile(
                        positive_large,
                        float(self.config.policy_update_bucket_quantile),
                    ).item()
                ),
                near_zero_threshold,
            )
        else:
            high_positive_threshold = float("inf")

        high_positive_mask = (
            raw_advantages >= high_positive_threshold
            if np.isfinite(high_positive_threshold)
            else torch.zeros_like(raw_advantages, dtype=torch.bool)
        )
        mid_positive_mask = (
            (raw_advantages > near_zero_threshold)
            & (
                raw_advantages < high_positive_threshold
                if np.isfinite(high_positive_threshold)
                else torch.zeros_like(raw_advantages, dtype=torch.bool)
            )
        )
        near_zero_mask = raw_advantages.abs() <= near_zero_threshold
        negative_mask = raw_advantages < -near_zero_threshold
        return (
            {
                "high_positive": high_positive_mask,
                "mid_positive": mid_positive_mask,
                "near_zero": near_zero_mask,
                "negative": negative_mask,
            },
            {
                "high_positive_threshold": float(high_positive_threshold)
                if np.isfinite(high_positive_threshold)
                else 0.0,
                "near_zero_threshold": float(near_zero_threshold),
            },
        )

    def _ratio_summary_by_adv_sign(
        self,
        ratios: torch.Tensor,
        raw_advantages: torch.Tensor,
    ) -> str:
        """Serialize ratio statistics for positive/negative-advantage samples."""
        summary = {
            "positive": self._summary_stats(ratios[raw_advantages > 0.0]),
            "negative": self._summary_stats(ratios[raw_advantages < 0.0]),
        }
        return json.dumps(summary, ensure_ascii=False, sort_keys=True)

    def train(self) -> dict[str, Any]:
        """Run PPO updates."""
        if len(self.buffer) == 0:
            action_block_layout = self.describe_action_block_slices()
            action_block_count = len(action_block_layout)
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "policy_entropy": 0.0,
                "raw_return_mean": 0.0,
                "raw_return_std": 0.0,
                "value_target_mean": 0.0,
                "value_target_std": 0.0,
                "advantage_mean": 0.0,
                "advantage_std": 0.0,
                "normalized_advantage_mean": 0.0,
                "normalized_advantage_std": 0.0,
                "positive_advantage_ratio": 0.0,
                "negative_advantage_ratio": 0.0,
                "delta_log_prob_selected_action_mean": 0.0,
                "delta_log_prob_selected_action_std": 0.0,
                "selected_action_prob_gain_mean": 0.0,
                "advantage_action_alignment": 0.0,
                "high_advantage_action_prob_gain": 0.0,
                "mid_advantage_action_prob_gain": 0.0,
                "low_advantage_action_prob_gain": 0.0,
                "negative_advantage_action_prob_gain": 0.0,
                "advantage_bucket_high_positive_threshold": 0.0,
                "advantage_bucket_near_zero_threshold": 0.0,
                "ratio_mean": 1.0,
                "ratio_std": 0.0,
                "ratio_min": 1.0,
                "ratio_max": 1.0,
                "clip_fraction": 0.0,
                "positive_adv_clip_fraction": 0.0,
                "negative_adv_clip_fraction": 0.0,
                "approx_kl": 0.0,
                "entropy_loss": 0.0,
                "policy_loss": 0.0,
                "ratio_by_adv_sign": json.dumps({}, ensure_ascii=False, sort_keys=True),
                "policy_surrogate_mode": self._policy_ratio_mode(),
                "policy_ratio_mode": self.config.policy_ratio_mode,
                "actor_structure_mode": self._actor_structure_mode(),
                "action_dim": int(self.action_dim),
                "action_block_count": int(action_block_count),
                "action_block_slices": json.dumps(
                    action_block_layout,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "logprob_delta_sum_mean": 0.0,
                "logprob_delta_sum_std": 0.0,
                "logprob_delta_mean_mean": 0.0,
                "logprob_delta_mean_std": 0.0,
                "block_logprob_delta_mean": 0.0,
                "block_logprob_delta_std": 0.0,
                "block_weight_entropy": 0.0,
                "block_weight_max_mean": 0.0,
                "top_k_block_weight_share": 0.0,
                "block_adv_scale_mean": 1.0,
                "block_adv_scale_std": 0.0,
                "block_adv_scale_max": 1.0,
                "block_adv_scale_min": 1.0,
                "block_adv_scale_entropy": 0.0,
                "top_k_block_adv_scale_share": 0.0,
                "block_value_score_mean": 0.0,
                "block_value_score_std": 0.0,
                "block_value_score_max": 0.0,
                "block_value_score_min": 0.0,
                "block_value_scale_mean": 1.0,
                "block_value_scale_std": 0.0,
                "block_value_scale_max": 1.0,
                "block_value_scale_min": 1.0,
                "block_value_scale_entropy": 0.0,
                "top_k_block_value_scale_share": 0.0,
                "block_value_pred_mean": 0.0,
                "block_value_pred_std": 0.0,
                "block_value_pred_max": 0.0,
                "block_value_pred_min": 0.0,
                "block_path_value_local_mean": 0.0,
                "block_path_value_local_std": 0.0,
                "block_path_value_bs1_mean": 0.0,
                "block_path_value_bs1_std": 0.0,
                "block_path_value_bs2_mean": 0.0,
                "block_path_value_bs2_std": 0.0,
                "path_value_local_mean": 0.0,
                "path_value_local_std": 0.0,
                "path_value_bs1_mean": 0.0,
                "path_value_bs1_std": 0.0,
                "path_value_bs2_mean": 0.0,
                "path_value_bs2_std": 0.0,
                "path_td_target_local_mean": 0.0,
                "path_td_target_local_std": 0.0,
                "path_td_target_bs1_mean": 0.0,
                "path_td_target_bs1_std": 0.0,
                "path_td_target_bs2_mean": 0.0,
                "path_td_target_bs2_std": 0.0,
                "theta_advantage_mean": 0.0,
                "theta_advantage_std": 0.0,
                "theta_advantage_max": 0.0,
                "theta_advantage_min": 0.0,
                "route_advantage_mean": 0.0,
                "route_advantage_std": 0.0,
                "route_advantage_max": 0.0,
                "route_advantage_min": 0.0,
                "theta_adv_norm_mean": 0.0,
                "theta_adv_norm_std": 0.0,
                "route_adv_norm_mean": 0.0,
                "route_adv_norm_std": 0.0,
                "route_adv_norm_active_count": 0.0,
                "route_adv_norm_used_fallback": 0.0,
                "theta_advantage_alignment": 0.0,
                "route_advantage_alignment": 0.0,
                "theta_selected_action_prob_gain": 0.0,
                "route_selected_action_prob_gain": 0.0,
                "theta_negative_adv_prob_gain": 0.0,
                "route_negative_adv_prob_gain": 0.0,
                "theta_clip_fraction": 0.0,
                "route_clip_fraction": 0.0,
                "theta_ratio_mean": 0.0,
                "theta_ratio_std": 0.0,
                "theta_ratio_max": 0.0,
                "route_ratio_mean": 0.0,
                "route_ratio_std": 0.0,
                "route_ratio_max": 0.0,
                "offload_active_fraction": 0.0,
                "route_logprob_active_fraction": 0.0,
                "theta_positive_fraction": 0.0,
                "route_credit_gate_fraction": 0.0,
                "route_credit_effective_fraction": 0.0,
                "route_credit_effective_count": 0.0,
                "route_credit_weight_mean": 1.0,
                "route_credit_weight_std": 0.0,
                "route_credit_weight_min": 1.0,
                "route_credit_weight_max": 1.0,
                "theta_candidate_score_mean": 0.0,
                "theta_candidate_score_std": 0.0,
                "theta_selected_score_mean": 0.0,
                "theta_expected_score_mean": 0.0,
                "theta_residual_credit_mean": 0.0,
                "theta_residual_credit_std": 0.0,
                "theta_residual_credit_min": 0.0,
                "theta_residual_credit_max": 0.0,
                "route_candidate_score_mean": 0.0,
                "route_candidate_score_std": 0.0,
                "route_residual_credit_mean": 0.0,
                "route_residual_credit_std": 0.0,
                "route_residual_credit_min": 0.0,
                "route_residual_credit_max": 0.0,
                "route_expected_score_mean": 0.0,
                "route_selected_score_mean": 0.0,
                "route_decision_agreement_ratio": 0.0,
                "actual_bs1_rate_when_route_true_gap_positive": 0.0,
                "actual_bs1_rate_when_route_true_gap_negative": 0.0,
                "route_score_vector_mean_abs": 0.0,
                "route_score_vector_std": 0.0,
                "route_credit_fallback_trigger_count": 0,
                "route_credit_fallback_rate": 0.0,
                "theta_old_logprob_mean": 0.0,
                "theta_new_logprob_mean": 0.0,
                "route_old_logprob_mean": 0.0,
                "route_new_logprob_mean": 0.0,
                "offload_decision_agreement_ratio": 0.0,
                "actual_offload_rate_when_theta_true_gap_positive": 0.0,
                "actual_offload_rate_when_theta_true_gap_negative": 0.0,
                "theta_approx_kl": 0.0,
                "route_approx_kl": 0.0,
                "conditional_policy_consistency_score": 0.0,
                "theta_loss_mean": 0.0,
                "route_loss_mean": 0.0,
                "theta_route_loss_ratio": 0.0,
                "route_gate_mean": 0.0,
                "route_gate_std": 0.0,
                "route_gate_min": 0.0,
                "route_gate_max": 0.0,
                "route_gate_active_fraction": 0.0,
                "route_mask_mean": 0.0,
                "route_mask_active_fraction": 0.0,
                "route_mask_count_mean": 0.0,
                "route_mask_count_std": 0.0,
                "theta_head_grad_norm": 0.0,
                "route_head_grad_norm": 0.0,
                "theta_backbone_grad_norm": 0.0,
                "route_backbone_grad_norm": 0.0,
                "theta_only_step_kl": 0.0,
                "route_only_step_kl": 0.0,
                "theta_only_step_prob_gain": 0.0,
                "route_only_step_prob_gain": 0.0,
                "theta_after_route_shift": 0.0,
                "route_after_theta_shift": 0.0,
                "theta_route_feature_correlation": 0.0,
                "theta_route_head_correlation": 0.0,
                "offload_rate_mean": 0.0,
                "bs1_vs_bs2_entropy": 0.0,
                "offload_vs_local_value_gap_mean": 0.0,
                "offload_vs_local_value_gap_std": 0.0,
                "bs1_vs_bs2_value_gap_mean": 0.0,
                "bs1_vs_bs2_value_gap_std": 0.0,
                "block_action_conditioned_value_now_mean": 0.0,
                "block_action_conditioned_value_now_std": 0.0,
                "block_action_conditioned_value_next_mean": 0.0,
                "block_action_conditioned_value_next_std": 0.0,
                "block_path_cost_local_mean": 0.0,
                "block_path_cost_local_std": 0.0,
                "block_path_cost_bs1_mean": 0.0,
                "block_path_cost_bs1_std": 0.0,
                "block_path_cost_bs2_mean": 0.0,
                "block_path_cost_bs2_std": 0.0,
                "block_local_cost_now_mean": 0.0,
                "block_local_cost_now_std": 0.0,
                "block_local_cost_next_mean": 0.0,
                "block_local_cost_next_std": 0.0,
                "block_action_conditioned_cost_now_mean": 0.0,
                "block_action_conditioned_cost_now_std": 0.0,
                "block_action_conditioned_cost_next_mean": 0.0,
                "block_action_conditioned_cost_next_std": 0.0,
                "block_delta_cost_mean": 0.0,
                "block_delta_cost_std": 0.0,
                "block_delta_cost_max": 0.0,
                "block_delta_cost_min": 0.0,
                "block_td_reward_mean": 0.0,
                "block_td_reward_std": 0.0,
                "block_td_reward_max": 0.0,
                "block_td_reward_min": 0.0,
                "block_td_target_mean": 0.0,
                "block_td_target_std": 0.0,
                "block_td_target_max": 0.0,
                "block_td_target_min": 0.0,
                "block_td_advantage_mean": 0.0,
                "block_td_advantage_std": 0.0,
                "block_td_advantage_max": 0.0,
                "block_td_advantage_min": 0.0,
                "block_td_advantage_entropy": 0.0,
                "top_k_block_td_advantage_share": 0.0,
                "block_advantage_mean": 0.0,
                "block_advantage_std": 0.0,
                "block_advantage_max": 0.0,
                "block_advantage_min": 0.0,
                "block_advantage_entropy": 0.0,
                "top_k_block_advantage_share": 0.0,
                "active_block_count_mean": 0.0,
                "block_ratio_proxy_mean": 1.0,
                "block_ratio_proxy_std": 0.0,
                "block_clip_fraction_mean": 0.0,
                "block_clip_fraction_std": 0.0,
                "block_positive_adv_clip_fraction_mean": 0.0,
                "block_negative_adv_clip_fraction_mean": 0.0,
                "block_ratio_mean": 1.0,
                "block_ratio_std": 0.0,
                "block_ratio_max": 1.0,
                "block_surrogate_mean": 0.0,
                "block_surrogate_std": 0.0,
                "per_block_selected_action_prob_gain_mean": 0.0,
                "per_block_selected_action_prob_gain_std": 0.0,
                "sum_to_mean_scale_ratio": float(self.action_dim),
                "sum_to_blockmean_scale_ratio": float(max(action_block_count, 1)),
                "mean_to_blockmean_scale_ratio": float(
                    max(action_block_count, 1) / max(self.action_dim, 1)
                ),
                "running_return_mean": self.running_return_mean,
                "running_return_std": self._current_running_return_std(),
                "popart_mean": self.running_return_mean,
                "popart_std": self._current_running_return_std(),
                "actor_grad_norm": 0.0,
                "action_distribution_std": 0.0,
                "action_prob_std": 0.0,
                "policy_std_mean": 0.0,
                "policy_logit_std": 0.0,
                "policy_confidence_mean": 0.0,
                "selected_action_histogram": json.dumps({}, ensure_ascii=False, sort_keys=True),
                "critic_raw_prediction_mean": 0.0,
                "critic_raw_prediction_std": 0.0,
                "critic_normalized_prediction_mean": 0.0,
                "critic_normalized_prediction_std": 0.0,
                "value_explained_variance": 0.0,
                "prediction_target_corr": 0.0,
                "prediction_std_over_target_std": 0.0,
                "constant_baseline_mse": 0.0,
                "critic_prediction_mse": 0.0,
                "critic_vs_constant_mse_gain": 0.0,
                "constant_baseline_huber": 0.0,
                "critic_prediction_huber": 0.0,
                "critic_vs_constant_huber_gain": 0.0,
                "critic_hidden_feature_mean": 0.0,
                "critic_hidden_feature_std": 0.0,
                "critic_hidden_feature_dim_std_mean": 0.0,
                "critic_hidden_feature_dim_std_min": 0.0,
                "critic_head_weight_norm": 0.0,
                "critic_head_bias_mean": 0.0,
                "critic_backbone_grad_norm": 0.0,
                "critic_head_grad_norm": 0.0,
                "actor_input_mean": 0.0,
                "actor_input_std": 0.0,
                "actor_input_dim_std_mean": 0.0,
                "actor_input_dim_std_min": 0.0,
                "actor_input_clip_fraction": 0.0,
                "actor_input_derived_dim": int(self.actor_derived_feature_dim),
                "actor_input_total_dim": int(self.actor_input_dim),
                "actor_derived_feature_mean": 0.0,
                "actor_derived_feature_std": 0.0,
                "actor_raw_dim": int(self.actor_raw_dim),
                "actor_raw_pruned_dim_count": int(self.state_dim - self.actor_raw_dim),
                "actor_raw_dim_std_mean": 0.0,
                "actor_raw_dim_std_min": 0.0,
                "actor_raw_zero_var_dim_count": 0,
                "actor_raw_static_zero_var_dim_count": 0,
                "critic_input_mean": 0.0,
                "critic_input_std": 0.0,
                "critic_input_dim_std_mean": 0.0,
                "critic_input_dim_std_min": 0.0,
                "critic_input_clip_fraction": 0.0,
                "critic_input_derived_dim": int(self.critic_derived_feature_dim),
                "critic_input_total_dim": int(self.critic_input_dim),
                "critic_derived_feature_mean": 0.0,
                "critic_derived_feature_std": 0.0,
                "diagnostic_advantage_bucket_rows": [],
                "diagnostic_block_logprob_rows": [],
                "diagnostic_block_surrogate_rows": [],
                "diagnostic_block_adv_scale_rows": [],
                "diagnostic_block_value_scale_rows": [],
                "diagnostic_block_td_value_rows": [],
                "diagnostic_block_delta_cost_rows": [],
                "diagnostic_block_advantage_rows": [],
                "diagnostic_block_weight_rows": [],
                "diagnostic_theta_route_split_rows": [],
                "diagnostic_value_targets": [],
                "diagnostic_prediction_values": [],
                "diagnostic_constant_predictions": [],
                "diagnostic_raw_predictions": [],
            }

        train_epoch_index = int(self.policy_update_call_count)
        self.policy_update_call_count += 1

        data = self.buffer.as_tensors(self.device)
        states = data["states"]
        next_states = data["next_states"]
        actions = data["actions"]
        old_log_probs = data["old_log_probs"]
        old_log_prob_components = data["old_log_prob_components"]
        old_action_means = data["old_action_means"]
        old_action_stds = data["old_action_stds"]
        advantages = data["advantages"]
        returns = data["returns"]
        dones = data["dones"]
        joint_reward_aligned_scores = data.get("joint_reward_aligned_scores")
        joint_td_aligned_scores = data.get("joint_td_aligned_scores")
        one_step_rewards = torch.tensor(
            self.buffer.rewards,
            dtype=torch.float32,
            device=self.device,
        )

        raw_advantages = advantages.clone()
        raw_advantage_mean = raw_advantages.mean()
        raw_advantage_std = raw_advantages.std(unbiased=False)
        positive_advantage_ratio = float((raw_advantages > 0.0).float().mean().item())
        negative_advantage_ratio = float((raw_advantages < 0.0).float().mean().item())
        advantages = (raw_advantages - raw_advantage_mean) / (raw_advantage_std + 1e-8)
        normalized_advantage_mean = advantages.mean()
        normalized_advantage_std = advantages.std(unbiased=False)
        raw_return_mean = returns.mean()
        raw_return_std = returns.std(unbiased=False)
        critic_training_drift_trace_enabled = (
            self._critic_training_drift_trace_buffer() is not None
        )
        critic_training_internal_stage_trace_enabled = (
            critic_training_drift_trace_enabled
            and self._critic_training_internal_stage_trace_enabled()
        )
        critic_training_drift_probe_payload = self._critic_training_drift_probe_payload()
        critic_training_drift_heldout_payload = self._critic_training_drift_heldout_payload()
        critic_training_drift_minibatch_id = 0
        freeze_actor_training_updates = self._freeze_actor_training_updates()
        pre_update_target_mean = raw_return_mean
        pre_update_target_std = raw_return_std
        critic_param_delta_norm_after_popart_value = 0.0
        critic_backbone_delta_norm_after_popart_value = 0.0
        critic_head_delta_norm_after_popart_value = 0.0

        if self.config.value_target_mode == "raw_return":
            target_mean = raw_return_mean
            target_std = raw_return_std
            value_targets = returns
        elif self.config.value_target_mode == "normalized_return":
            target_mean = raw_return_mean
            target_std = raw_return_std
            value_targets = (returns - target_mean) / (target_std + 1e-8)
        elif self.config.value_target_mode == "running_return_norm":
            (
                pre_update_target_mean,
                pre_update_target_std,
                target_mean,
                target_std,
            ) = self._update_running_return_stats(returns)
            value_targets = (returns - target_mean) / (target_std + 1e-8)
        elif self.config.value_target_mode == "popart_return_norm":
            (
                pre_update_target_mean,
                pre_update_target_std,
                target_mean,
                target_std,
            ) = self._update_running_return_stats(returns)
            if critic_training_internal_stage_trace_enabled:
                critic_params_before_popart = self._snapshot_parameter_list(self.critic_params)
                critic_backbone_params_before_popart = self._snapshot_parameter_list(
                    list(self.network.critic_backbone.parameters())
                )
                critic_head_params_before_popart = self._snapshot_parameter_list(
                    list(self.network.critic_head.parameters())
                )
            self.network.popart_rescale(
                pre_update_target_mean,
                pre_update_target_std,
                target_mean,
                target_std,
            )
            if critic_training_internal_stage_trace_enabled:
                critic_params_after_popart = self._snapshot_parameter_list(self.critic_params)
                critic_backbone_params_after_popart = self._snapshot_parameter_list(
                    list(self.network.critic_backbone.parameters())
                )
                critic_head_params_after_popart = self._snapshot_parameter_list(
                    list(self.network.critic_head.parameters())
                )
                critic_param_delta_norm_after_popart_value = self._parameter_list_delta_norm(
                    critic_params_before_popart,
                    critic_params_after_popart,
                )
                critic_backbone_delta_norm_after_popart_value = (
                    self._parameter_list_delta_norm(
                        critic_backbone_params_before_popart,
                        critic_backbone_params_after_popart,
                    )
                )
                critic_head_delta_norm_after_popart_value = self._parameter_list_delta_norm(
                    critic_head_params_before_popart,
                    critic_head_params_after_popart,
                )
            value_targets = (returns - target_mean) / (target_std + 1e-8)
        else:
            raise ValueError(
                f"Unsupported value_target_mode: {self.config.value_target_mode}"
            )

        pre_update_value_targets = self._value_targets_from_returns_with_stats(
            returns,
            pre_update_target_mean,
            pre_update_target_std,
        )

        value_target_mean = value_targets.mean()
        value_target_std = value_targets.std(unbiased=False)
        full_actor_inputs, actor_input_diagnostics, _ = self._prepare_actor_inputs(
            states,
            update_stats=False,
        )
        full_critic_inputs, critic_input_diagnostics, _ = self._prepare_critic_inputs(
            states,
            update_stats=self._uses_normalized_augmented_critic_input(),
        )
        full_next_critic_inputs, _, _ = self._prepare_critic_inputs(
            next_states,
            update_stats=False,
        )

        critic_blended_value_loss_enabled = bool(
            getattr(self, "critic_blended_value_loss_enabled", False)
        )
        critic_blended_current_weight = float(
            getattr(self, "critic_blended_current_weight", 1.0)
        )
        critic_blended_heldout_weight = float(
            getattr(self, "critic_blended_heldout_weight", 0.0)
        )
        critic_blended_heldout_batch_count = int(
            max(1, getattr(self, "critic_blended_heldout_batch_count", 1))
        )
        critic_blended_fixed_heldout_payload = self._critic_blended_fixed_heldout_payload()
        blended_heldout_indices = None
        blended_heldout_states = None
        blended_heldout_critic_inputs = None
        blended_heldout_value_targets = None
        all_train_indices = torch.arange(states.size(0), device=self.device)
        if critic_blended_value_loss_enabled:
            mini_batch_size = int(self.config.mini_batch_size)
            if critic_blended_fixed_heldout_payload is not None:
                blended_heldout_states = critic_blended_fixed_heldout_payload["states"]
                blended_heldout_critic_inputs, _, _ = self._prepare_critic_inputs(
                    blended_heldout_states,
                    update_stats=False,
                )
                blended_heldout_value_targets = self._value_targets_from_returns_with_stats(
                    critic_blended_fixed_heldout_payload["returns"],
                    target_mean,
                    target_std,
                )
                heldout_count = (
                    blended_heldout_critic_inputs.size(0)
                    // mini_batch_size
                ) * mini_batch_size
                if heldout_count >= mini_batch_size:
                    blended_heldout_critic_inputs = blended_heldout_critic_inputs[:heldout_count]
                    blended_heldout_value_targets = blended_heldout_value_targets[:heldout_count]
                else:
                    critic_blended_value_loss_enabled = False
            else:
                max_heldout_count = max(0, int(states.size(0)) - mini_batch_size)
                requested_heldout_count = mini_batch_size * critic_blended_heldout_batch_count
                heldout_count = min(max_heldout_count, requested_heldout_count)
                heldout_count = (heldout_count // mini_batch_size) * mini_batch_size
                if heldout_count >= mini_batch_size:
                    heldout_generator = torch.Generator(device=self.device)
                    heldout_generator.manual_seed(31000 + int(self.policy_update_call_count))
                    heldout_perm = torch.randperm(
                        states.size(0),
                        generator=heldout_generator,
                        device=self.device,
                    )
                    blended_heldout_indices = heldout_perm[:heldout_count]
                    heldout_mask = torch.ones(states.size(0), dtype=torch.bool, device=self.device)
                    heldout_mask[blended_heldout_indices] = False
                    all_train_indices = all_train_indices[heldout_mask]
                    blended_heldout_states = states[blended_heldout_indices]
                    blended_heldout_critic_inputs = full_critic_inputs[blended_heldout_indices]
                    blended_heldout_value_targets = value_targets[blended_heldout_indices]
                else:
                    critic_blended_value_loss_enabled = False

        critic_step_acceptance_config = self._critic_step_acceptance_config()
        critic_step_monitor_enabled = bool(
            critic_step_acceptance_config is not None
            and critic_step_acceptance_config.get("enabled", False)
        )
        critic_step_enforce_enabled = bool(
            critic_step_monitor_enabled
            and critic_step_acceptance_config.get("enforce", False)
        )
        critic_step_current_loss_improve_epsilon = float(
            0.0
            if critic_step_acceptance_config is None
            else critic_step_acceptance_config.get("current_loss_improve_epsilon", 1e-6)
        )
        critic_step_heldout_loss_tolerance = float(
            0.0
            if critic_step_acceptance_config is None
            else critic_step_acceptance_config.get("heldout_loss_tolerance", 1e-4)
        )
        critic_step_probe_pearson_tolerance = float(
            0.0
            if critic_step_acceptance_config is None
            else critic_step_acceptance_config.get("probe_pearson_tolerance", 1e-4)
        )

        actor_loss_value = 0.0
        critic_loss_value = 0.0
        critic_loss_current_value = 0.0
        critic_loss_heldout_value = 0.0
        entropy_value = 0.0
        actor_grad_norm_value = 0.0
        theta_head_grad_norm_value = 0.0
        route_head_grad_norm_value = 0.0
        theta_backbone_grad_norm_value = 0.0
        route_backbone_grad_norm_value = 0.0
        theta_only_step_kl_value = 0.0
        route_only_step_kl_value = 0.0
        theta_only_step_prob_gain_value = 0.0
        route_only_step_prob_gain_value = 0.0
        theta_after_route_shift_value = 0.0
        route_after_theta_shift_value = 0.0
        route_credit_fallback_trigger_count_value = 0
        route_credit_support_eval_count_value = 0
        theta_early_stop_count_value = 0
        route_early_stop_count_value = 0
        coupled_stop_trigger_count_value = 0
        coupled_stop_blocked_by_theta_floor_count_value = 0
        coupled_stop_blocked_by_severity_gate_count_value = 0
        coupled_stop_blocked_by_alignment_gate_count_value = 0
        coupled_stop_blocked_by_warmup_count_value = 0
        route_update_cap_trigger_count_value = 0
        route_alignment_gate_accept_count_value = 0
        route_alignment_gate_reject_count_value = 0
        route_alignment_gate_score_sum_value = 0.0
        route_alignment_gate_score_count_value = 0
        route_step_alignment_min_value = float("inf")
        route_step_alignment_max_value = float("-inf")
        route_alignment_gate_snapshot_time_ms_value = 0.0
        route_alignment_gate_restore_time_ms_value = 0.0
        route_alignment_gate_snapshot_count_value = 0
        route_alignment_gate_restore_count_value = 0
        theta_update_count_value = 0
        route_update_count_value = 0
        critic_backbone_grad_norm_value = 0.0
        critic_head_grad_norm_value = 0.0
        critic_param_delta_norm_value = 0.0
        critic_backbone_delta_norm_value = 0.0
        critic_head_delta_norm_value = 0.0
        critic_backbone_preconditioner_active_count_value = 0
        critic_backbone_preconditioner_clipped_count_value = 0
        critic_backbone_preconditioner_active_sum_before_clip_value = 0.0
        critic_backbone_preconditioner_active_sum_after_clip_value = 0.0
        critic_backbone_preconditioner_active_max_before_clip_value = 0.0
        critic_backbone_preconditioner_active_max_after_clip_value = 0.0
        critic_backbone_preconditioner_cap_value = 0.0
        critic_step_attempt_count_value = 0
        critic_step_accept_count_value = 0
        critic_step_reject_count_value = 0
        critic_step_would_reject_count_value = 0
        critic_step_current_loss_before_sum_value = 0.0
        critic_step_current_loss_after_sum_value = 0.0
        critic_step_heldout_loss_before_sum_value = 0.0
        critic_step_heldout_loss_after_sum_value = 0.0
        critic_step_probe_pearson_before_sum_value = 0.0
        critic_step_probe_pearson_after_sum_value = 0.0
        critic_step_reject_reason_counts: dict[str, int] = {}
        critic_step_would_reject_reason_counts: dict[str, int] = {}
        coupled_stop_warmup_epochs = int(max(0, getattr(self.config, "coupled_stop_warmup_epochs", 0)))
        coupled_stop_is_active_this_train_epoch = train_epoch_index >= coupled_stop_warmup_epochs
        route_training_drift_trace_enabled = self._route_training_drift_trace_buffer() is not None
        route_training_drift_minibatch_id = 0
        actor_grad_norm_trace_current = 0.0
        critic_grad_norm_trace_current = 0.0
        actor_param_delta_norm_current = 0.0
        critic_param_delta_norm_current = 0.0
        actor_backbone_delta_norm_current = 0.0
        actor_head_delta_norm_current = 0.0
        critic_backbone_delta_norm_current = 0.0
        critic_head_delta_norm_current = 0.0

        def _append_critic_drift_stage_payload(
            trace_stage: str,
            trace_update_epoch: int,
            trace_minibatch_id: int,
            trace_scope: str,
            critic_inputs_for_trace: torch.Tensor,
            value_targets_for_trace: torch.Tensor,
            returns_for_trace: torch.Tensor,
            advantages_for_trace: torch.Tensor,
            rewards_for_trace: torch.Tensor,
            prediction_target_mean: torch.Tensor,
            prediction_target_std: torch.Tensor,
            actor_loss_scalar: float,
            critic_loss_scalar: float,
        ) -> None:
            if not critic_training_drift_trace_enabled:
                return
            with torch.no_grad():
                batch_semantics, batch_critic_loss_trace = self._critic_trace_loss_and_semantics(
                    critic_inputs_for_trace,
                    value_targets_for_trace,
                    returns_for_trace,
                    advantages_for_trace,
                    rewards_for_trace,
                    prediction_target_mean,
                    prediction_target_std,
                )
                probe_metrics: dict[str, float] = {
                    "probe_critic_loss": 0.0,
                    "probe_value_pred_mean": 0.0,
                    "probe_value_pred_std": 0.0,
                    "probe_value_target_mean": 0.0,
                    "probe_value_target_std": 0.0,
                    "probe_return_mean": 0.0,
                    "probe_return_std": 0.0,
                    "probe_advantage_mean": 0.0,
                    "probe_advantage_std": 0.0,
                    "probe_pearson_value_vs_value_target": 0.0,
                    "probe_spearman_value_vs_value_target": 0.0,
                    "probe_pearson_value_vs_return": 0.0,
                    "probe_spearman_value_vs_return": 0.0,
                    "probe_pearson_value_vs_advantage": 0.0,
                    "probe_spearman_value_vs_advantage": 0.0,
                    "probe_pearson_value_vs_one_step_reward": 0.0,
                    "probe_spearman_value_vs_one_step_reward": 0.0,
                }
                heldout_metrics: dict[str, float] = {
                    "heldout_critic_loss": 0.0,
                    "heldout_value_pred_mean": 0.0,
                    "heldout_value_pred_std": 0.0,
                    "heldout_value_target_mean": 0.0,
                    "heldout_value_target_std": 0.0,
                    "heldout_return_mean": 0.0,
                    "heldout_return_std": 0.0,
                    "heldout_advantage_mean": 0.0,
                    "heldout_advantage_std": 0.0,
                    "heldout_pearson_value_vs_value_target": 0.0,
                    "heldout_spearman_value_vs_value_target": 0.0,
                    "heldout_pearson_value_vs_return": 0.0,
                    "heldout_spearman_value_vs_return": 0.0,
                    "heldout_pearson_value_vs_advantage": 0.0,
                    "heldout_spearman_value_vs_advantage": 0.0,
                    "heldout_pearson_value_vs_one_step_reward": 0.0,
                    "heldout_spearman_value_vs_one_step_reward": 0.0,
                }
                if critic_training_drift_probe_payload is not None:
                    probe_states = critic_training_drift_probe_payload["states"]
                    probe_returns = critic_training_drift_probe_payload["returns"]
                    probe_advantages = critic_training_drift_probe_payload["advantages"]
                    probe_rewards = critic_training_drift_probe_payload["rewards"]
                    probe_value_targets = self._value_targets_from_returns_with_stats(
                        probe_returns,
                        prediction_target_mean,
                        prediction_target_std,
                    )
                    probe_critic_inputs, _, _ = self._prepare_critic_inputs(
                        probe_states,
                        update_stats=False,
                    )
                    probe_semantics, probe_critic_loss_trace = self._critic_trace_loss_and_semantics(
                        probe_critic_inputs,
                        probe_value_targets,
                        probe_returns,
                        probe_advantages,
                        probe_rewards,
                        prediction_target_mean,
                        prediction_target_std,
                    )
                    probe_metrics = {"probe_critic_loss": probe_critic_loss_trace}
                    probe_metrics.update(
                        {f"probe_{key}": value for key, value in probe_semantics.items()}
                    )
                if critic_training_drift_heldout_payload is not None:
                    heldout_states = critic_training_drift_heldout_payload["states"]
                    heldout_returns = critic_training_drift_heldout_payload["returns"]
                    heldout_advantages = critic_training_drift_heldout_payload["advantages"]
                    heldout_rewards = critic_training_drift_heldout_payload["rewards"]
                    heldout_value_targets = self._value_targets_from_returns_with_stats(
                        heldout_returns,
                        prediction_target_mean,
                        prediction_target_std,
                    )
                    heldout_critic_inputs, _, _ = self._prepare_critic_inputs(
                        heldout_states,
                        update_stats=False,
                    )
                    heldout_semantics, heldout_critic_loss_trace = self._critic_trace_loss_and_semantics(
                        heldout_critic_inputs,
                        heldout_value_targets,
                        heldout_returns,
                        heldout_advantages,
                        heldout_rewards,
                        prediction_target_mean,
                        prediction_target_std,
                    )
                    heldout_metrics = {"heldout_critic_loss": heldout_critic_loss_trace}
                    heldout_metrics.update(
                        {
                            f"heldout_{key}": value
                            for key, value in heldout_semantics.items()
                        }
                    )
                self._append_critic_training_drift_trace_row(
                    {
                        "mode": self.config.policy_ratio_mode,
                        "epoch": int(train_epoch_index),
                        "update_epoch": int(trace_update_epoch),
                        "minibatch_id": int(trace_minibatch_id),
                        "trace_stage": trace_stage,
                        "trace_scope": trace_scope,
                        "critic_loss": float(batch_critic_loss_trace),
                        "actor_loss": float(actor_loss_scalar),
                        "critic_grad_norm": float(critic_grad_norm_trace_current),
                        "actor_grad_norm": float(actor_grad_norm_trace_current),
                        "critic_param_delta_norm": float(critic_param_delta_norm_current),
                        "actor_param_delta_norm": float(actor_param_delta_norm_current),
                        "critic_backbone_delta_norm": float(
                            critic_backbone_delta_norm_current
                        ),
                        "critic_head_delta_norm": float(critic_head_delta_norm_current),
                        "actor_backbone_delta_norm": float(actor_backbone_delta_norm_current),
                        "actor_head_delta_norm": float(actor_head_delta_norm_current),
                        "critic_param_delta_norm_after_popart": float(
                            critic_param_delta_norm_after_popart_value
                        ),
                        "critic_backbone_delta_norm_after_popart": float(
                            critic_backbone_delta_norm_after_popart_value
                        ),
                        "critic_head_delta_norm_after_popart": float(
                            critic_head_delta_norm_after_popart_value
                        ),
                        "critic_param_delta_norm_after_optimizer_step": float(
                            critic_param_delta_norm_current
                        ),
                        "popart_mean": float(self.running_return_mean),
                        "popart_std": float(self._current_running_return_std()),
                        **batch_semantics,
                        **heldout_metrics,
                        **probe_metrics,
                    }
                )

        if critic_training_internal_stage_trace_enabled:
            _append_critic_drift_stage_payload(
                trace_stage="before_popart_update",
                trace_update_epoch=-1,
                trace_minibatch_id=-1,
                trace_scope="preloop",
                critic_inputs_for_trace=full_critic_inputs,
                value_targets_for_trace=pre_update_value_targets,
                returns_for_trace=returns,
                advantages_for_trace=raw_advantages,
                rewards_for_trace=one_step_rewards,
                prediction_target_mean=pre_update_target_mean,
                prediction_target_std=pre_update_target_std,
                actor_loss_scalar=0.0,
                critic_loss_scalar=0.0,
            )
            _append_critic_drift_stage_payload(
                trace_stage="after_popart_update",
                trace_update_epoch=-1,
                trace_minibatch_id=-1,
                trace_scope="preloop",
                critic_inputs_for_trace=full_critic_inputs,
                value_targets_for_trace=pre_update_value_targets,
                returns_for_trace=returns,
                advantages_for_trace=raw_advantages,
                rewards_for_trace=one_step_rewards,
                prediction_target_mean=target_mean,
                prediction_target_std=target_std,
                actor_loss_scalar=0.0,
                critic_loss_scalar=0.0,
            )
            _append_critic_drift_stage_payload(
                trace_stage="after_target_build",
                trace_update_epoch=-1,
                trace_minibatch_id=-1,
                trace_scope="preloop",
                critic_inputs_for_trace=full_critic_inputs,
                value_targets_for_trace=value_targets,
                returns_for_trace=returns,
                advantages_for_trace=raw_advantages,
                rewards_for_trace=one_step_rewards,
                prediction_target_mean=target_mean,
                prediction_target_std=target_std,
                actor_loss_scalar=0.0,
                critic_loss_scalar=0.0,
            )

        for update_epoch_index in range(self.config.update_epochs):
            theta_branch_stopped = False
            route_branch_stopped = False
            theta_update_count_this_epoch = 0
            route_update_count_this_epoch = 0
            epochwise_branch_order = (
                self._uses_epochwise_branch_order_factorized_trust_region_pg()
            )
            route_phase_epochs = int(np.ceil(self.config.update_epochs / 2.0))
            epochwise_route_phase = (
                epochwise_branch_order and update_epoch_index < route_phase_epochs
            )
            epochwise_theta_phase = (
                epochwise_branch_order and update_epoch_index >= route_phase_epochs
            )
            permutation = all_train_indices[
                torch.randperm(all_train_indices.size(0), device=self.device)
            ]
            for start in range(0, all_train_indices.size(0), self.config.mini_batch_size):
                batch_index = permutation[start : start + self.config.mini_batch_size]
                current_route_trace_minibatch_id = int(route_training_drift_minibatch_id)
                route_training_drift_minibatch_id += 1
                batch_states = states[batch_index]
                batch_next_states = next_states[batch_index]
                batch_actor_inputs = full_actor_inputs[batch_index]
                batch_critic_inputs = full_critic_inputs[batch_index]
                batch_next_critic_inputs = full_next_critic_inputs[batch_index]
                batch_actions = actions[batch_index]
                batch_old_log_probs = old_log_probs[batch_index]
                batch_old_log_prob_components = old_log_prob_components[batch_index]
                batch_old_action_means = old_action_means[batch_index]
                batch_old_action_stds = old_action_stds[batch_index]
                batch_advantages = advantages[batch_index]
                batch_raw_advantages = raw_advantages[batch_index]
                batch_returns = returns[batch_index]
                batch_value_targets = value_targets[batch_index]
                batch_rewards = one_step_rewards[batch_index]
                batch_dones = dones[batch_index]
                batch_joint_reward_aligned_scores = (
                    joint_reward_aligned_scores[batch_index]
                    if joint_reward_aligned_scores is not None
                    else None
                )
                batch_joint_td_aligned_scores = (
                    joint_td_aligned_scores[batch_index]
                    if joint_td_aligned_scores is not None
                    else None
                )
                current_critic_trace_minibatch_id = int(critic_training_drift_minibatch_id)
                critic_training_drift_minibatch_id += 1

                distribution = self.network.policy_from_actor_input(batch_actor_inputs)
                new_log_prob_components = distribution.log_prob(batch_actions)
                new_action_means = distribution.mean
                new_action_stds = distribution.stddev
                new_log_probs = self._aggregate_log_prob_components_with_states(
                    new_log_prob_components,
                    batch_states,
                )
                entropy_components = distribution.entropy()
                entropy = entropy_components.sum(dim=-1).mean()
                theta_entropy = self._branch_entropy_from_components(
                    entropy_components,
                    self._theta_action_indices(),
                )
                route_entropy = self._branch_entropy_from_components(
                    entropy_components,
                    self._route_action_indices(),
                )
                critic_features = self.network.critic_features_from_critic_input(
                    batch_critic_inputs
                )
                normalized_values = self.network.critic_head(critic_features).squeeze(-1)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                theta_loss = ratio.new_zeros(())
                route_loss = ratio.new_zeros(())
                theta_loss_for_update = theta_loss
                route_loss_for_update = route_loss
                split_advantages = None
                new_theta_log_probs = None
                new_route_log_probs = None
                old_theta_log_probs = None
                old_route_log_probs = None
                new_joint_log_probs = None
                old_joint_log_probs = None
                theta_advantages_for_surrogate = None
                route_advantages_for_surrogate = None
                joint_advantages_for_surrogate = None
                route_masks = None
                route_loss_masks = None
                route_used_fallback_for_batch = False
                route_credit_weight_for_surrogate = None
                route_residual_credit_for_surrogate = None
                route_candidate_score_credit_for_surrogate = None
                theta_candidate_score_credit_for_surrogate = None
                joint_reward_aligned_credit_for_surrogate = None
                route_surrogate_for_loss = None
                if self._uses_blockwise_policy_surrogate():
                    if self._uses_theta_route_split_advantage_surrogate():
                        new_theta_log_probs, new_route_log_probs = (
                            self._block_theta_route_log_prob_components(new_log_prob_components)
                        )
                        old_theta_log_probs, old_route_log_probs = (
                            self._block_theta_route_log_prob_components(
                                batch_old_log_prob_components
                            )
                        )
                        theta_ratio = torch.exp(new_theta_log_probs - old_theta_log_probs)
                        route_ratio = torch.exp(new_route_log_probs - old_route_log_probs)
                        split_advantages = self._theta_route_split_advantages(
                            batch_states,
                            batch_next_states,
                            batch_dones,
                            batch_actions,
                            critic_inputs=batch_critic_inputs,
                            next_critic_inputs=batch_next_critic_inputs,
                        )
                        theta_advantages = split_advantages["theta_advantages"]
                        route_advantages = split_advantages["route_advantages"]
                        branchwise_normalization = (
                            self._branchwise_normalize_theta_route_advantages(
                                theta_advantages,
                                route_advantages,
                                split_advantages["route_masks"],
                            )
                            if self._uses_branchwise_balanced_pg()
                            else None
                        )
                        theta_advantages_for_surrogate = (
                            branchwise_normalization["theta_advantages_norm"]
                            if branchwise_normalization is not None
                            else theta_advantages
                        )
                        route_advantages_for_surrogate = (
                            branchwise_normalization["route_advantages_norm"]
                            if branchwise_normalization is not None
                            else route_advantages
                        )
                        route_masks = split_advantages["route_masks"]
                        route_loss_masks = route_masks
                        route_credit_weight_for_surrogate = torch.ones_like(route_masks)
                        if self._uses_theta_candidate_score_credit():
                            theta_old_policy_terms_for_credit = (
                                self._true_conditional_theta_policy_terms(
                                    batch_actions,
                                    batch_old_action_means,
                                )
                            )
                            theta_candidate_score_credit_terms = (
                                self._theta_candidate_score_credit_terms(
                                    batch_states,
                                    theta_old_policy_terms_for_credit[
                                        "offload_active_mask"
                                    ],
                                    theta_old_policy_terms_for_credit["offload_probs"].detach(),
                                )
                            )
                            theta_candidate_score_credit_for_surrogate = (
                                theta_candidate_score_credit_terms["residual_credit"]
                            )
                            theta_advantages_for_surrogate = (
                                theta_candidate_score_credit_for_surrogate
                            )
                        if self._uses_route_credit_theta_gate():
                            route_credit_support = self._route_credit_theta_gate_support(
                                theta_advantages,
                                route_masks,
                            )
                            route_loss_masks = route_credit_support["effective_mask"]
                            route_used_fallback_for_batch = bool(
                                route_credit_support["fallback_triggered"]
                            )
                        elif self._uses_route_credit_theta_soft_weight():
                            if branchwise_normalization is None:
                                raise ValueError(
                                    "route credit theta soft weight requires branchwise normalized theta advantages"
                                )
                            route_credit_weight_terms = (
                                self._route_credit_theta_soft_weight_terms(
                                    branchwise_normalization["theta_advantages_norm"],
                                    route_masks,
                                )
                            )
                            route_credit_weight_for_surrogate = route_credit_weight_terms["weight"]
                            route_advantages_for_surrogate = (
                                route_credit_weight_for_surrogate * route_advantages_for_surrogate
                            )
                        elif self._uses_route_residual_credit_vectorized():
                            route_old_policy_terms_for_credit = (
                                self._true_conditional_route_policy_terms(
                                    batch_actions,
                                    batch_old_action_means,
                                )
                            )
                            route_residual_credit_terms = (
                                self._route_residual_credit_vectorized_terms(
                                    route_advantages_for_surrogate,
                                    route_old_policy_terms_for_credit["selected_indices"],
                                    route_old_policy_terms_for_credit["probs"].detach(),
                                    route_masks,
                                )
                            )
                            route_residual_credit_for_surrogate = (
                                route_residual_credit_terms["residual_credit"]
                            )
                            route_advantages_for_surrogate = route_residual_credit_for_surrogate
                        elif self._uses_true_conditional_route_candidate_score_credit():
                            route_old_policy_terms_for_credit = (
                                self._true_conditional_route_policy_terms(
                                    batch_actions,
                                    batch_old_action_means,
                                )
                            )
                            route_candidate_score_credit_terms = (
                                self._route_candidate_score_credit_terms(
                                    batch_states,
                                    route_old_policy_terms_for_credit["selected_indices"],
                                    route_old_policy_terms_for_credit["probs"].detach(),
                                    route_masks,
                                )
                            )
                            route_candidate_score_credit_for_surrogate = (
                                route_candidate_score_credit_terms["residual_credit"]
                            )
                            route_advantages_for_surrogate = (
                                route_candidate_score_credit_for_surrogate
                            )
                        if self._uses_joint_reward_aligned_credit() or self._uses_joint_td_aligned_credit():
                            batch_joint_credit_scores = (
                                batch_joint_td_aligned_scores
                                if self._uses_joint_td_aligned_credit()
                                else batch_joint_reward_aligned_scores
                            )
                            if batch_joint_credit_scores is None:
                                raise ValueError(
                                    "joint credit mode requires rollout counterfactual scores"
                                )
                            joint_old_policy_terms_for_credit = (
                                self._true_conditional_joint_policy_terms(
                                    batch_actions,
                                    batch_old_action_means,
                                )
                            )
                            joint_reward_aligned_credit_terms = (
                                self._joint_reward_aligned_credit_terms(
                                    batch_joint_credit_scores,
                                    joint_old_policy_terms_for_credit["selected_indices"],
                                    joint_old_policy_terms_for_credit["probs"].detach(),
                                )
                            )
                            joint_reward_aligned_credit_for_surrogate = (
                                joint_reward_aligned_credit_terms["residual_credit"]
                            )
                            joint_advantages_for_surrogate = (
                                joint_reward_aligned_credit_for_surrogate
                            )
                            old_joint_log_probs = joint_old_policy_terms_for_credit["log_probs"]
                            new_joint_policy_terms = self._true_conditional_joint_policy_terms(
                                batch_actions,
                                new_action_means,
                            )
                            new_joint_log_probs = new_joint_policy_terms["log_probs"]
                            joint_ratio = torch.exp(new_joint_log_probs - old_joint_log_probs)
                            joint_surrogate = torch.min(
                                joint_ratio * joint_advantages_for_surrogate,
                                torch.clamp(
                                    joint_ratio,
                                    1.0 - self.config.clip_epsilon,
                                    1.0 + self.config.clip_epsilon,
                                )
                                * joint_advantages_for_surrogate,
                            )
                            theta_loss = -joint_surrogate.mean()
                            route_loss = -self._mean_route_surrogate_over_active_blocks(
                                joint_surrogate,
                                route_masks,
                            )
                            theta_loss_for_update = theta_loss
                            route_loss_for_update = route_loss
                            actor_loss = 0.5 * theta_loss + 0.5 * route_loss
                        elif self._uses_factorized_ratio_pg():
                            theta_branch_terms = self._theta_branch_ppo_terms(
                                new_log_prob_components,
                                batch_old_log_prob_components,
                                theta_advantages_for_surrogate,
                                actions=batch_actions,
                                new_action_means=new_action_means,
                                old_action_means=batch_old_action_means,
                            )
                            route_branch_terms = self._route_branch_ppo_terms(
                                batch_actions,
                                new_log_prob_components,
                                batch_old_log_prob_components,
                                route_advantages_for_surrogate,
                                route_masks,
                                new_action_means=new_action_means,
                                new_action_stds=new_action_stds,
                                old_action_means=batch_old_action_means,
                                old_action_stds=batch_old_action_stds,
                                route_gates=split_advantages["route_gates"],
                                route_loss_masks=route_loss_masks,
                                route_used_fallback=route_used_fallback_for_batch,
                            )
                            new_theta_log_probs = theta_branch_terms["new_log_probs"]
                            new_route_log_probs = route_branch_terms["new_log_probs"]
                            old_theta_log_probs = theta_branch_terms["old_log_probs"]
                            old_route_log_probs = route_branch_terms["old_log_probs"]
                            theta_loss = theta_branch_terms["loss"]
                            route_loss = route_branch_terms["loss"]
                            theta_loss_for_update = theta_loss
                            route_loss_for_update = route_loss
                            actor_loss = 0.5 * theta_loss + 0.5 * route_loss
                        else:
                            theta_surrogate = torch.min(
                                theta_ratio * theta_advantages_for_surrogate,
                                torch.clamp(
                                    theta_ratio,
                                    1.0 - self.config.clip_epsilon,
                                    1.0 + self.config.clip_epsilon,
                                )
                                * theta_advantages_for_surrogate,
                            )
                            route_surrogate = torch.min(
                                route_ratio * route_advantages_for_surrogate,
                                torch.clamp(
                                    route_ratio,
                                    1.0 - self.config.clip_epsilon,
                                    1.0 + self.config.clip_epsilon,
                                )
                                * route_advantages_for_surrogate,
                            )
                            route_surrogate_for_loss = route_surrogate
                            if self._uses_hard_offload_route_surrogate():
                                active_route_count = route_masks.sum(dim=-1, keepdim=True)
                                route_surrogate = (
                                    route_masks
                                    * route_surrogate
                                    * (
                                        float(route_surrogate.shape[-1])
                                        / (active_route_count + 1e-8)
                                    )
                                )
                                route_surrogate_for_loss = route_masks * route_surrogate_for_loss
                            if self._uses_offload_gated_route_surrogate():
                                route_surrogate = (
                                    split_advantages["route_gates"] * route_surrogate
                                )
                                route_surrogate_for_loss = (
                                    split_advantages["route_gates"] * route_surrogate_for_loss
                                )
                            block_surrogate = 0.5 * (theta_surrogate + route_surrogate)
                            if self._uses_branchwise_balanced_pg():
                                theta_loss = -theta_surrogate.mean()
                                route_loss = -self._mean_route_surrogate_over_mask(
                                    route_surrogate_for_loss,
                                    route_loss_masks,
                                )
                                theta_loss_for_update = theta_loss
                                route_loss_for_update = route_loss
                                actor_loss = 0.5 * theta_loss + 0.5 * route_loss
                            else:
                                actor_loss = -block_surrogate.mean()
                    else:
                        new_block_log_probs = self._block_sum_log_prob_components(
                            new_log_prob_components
                        )
                        old_block_log_probs = self._block_sum_log_prob_components(
                            batch_old_log_prob_components
                        )
                        block_ratio = torch.exp(new_block_log_probs - old_block_log_probs)
                        block_advantages, _ = self._block_surrogate_advantages(
                            batch_advantages,
                            batch_states,
                            actions=batch_actions,
                            critic_inputs=batch_critic_inputs,
                            next_states=batch_next_states,
                            next_critic_inputs=batch_next_critic_inputs,
                            dones=batch_dones,
                        )
                        unclipped = block_ratio * block_advantages
                        clipped = torch.clamp(
                            block_ratio,
                            1.0 - self.config.clip_epsilon,
                            1.0 + self.config.clip_epsilon,
                        ) * block_advantages
                        block_surrogate = torch.min(unclipped, clipped)
                        if self._uses_blockwise_weighted_surrogate_mean():
                            block_weights = self._block_activity_weights(batch_states)
                            actor_loss = -(block_weights * block_surrogate).sum(dim=-1).mean()
                        else:
                            actor_loss = -block_surrogate.mean()
                else:
                    unclipped = ratio * batch_advantages
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    ) * batch_advantages
                    actor_loss = -torch.min(unclipped, clipped).mean()
                if self.config.value_target_mode == "popart_return_norm":
                    critic_predictions = normalized_values
                else:
                    raw_values = self.network.value_from_critic_input(batch_critic_inputs).squeeze(-1)
                    if self.config.value_target_mode in {"normalized_return", "running_return_norm"}:
                        critic_predictions = (raw_values - target_mean) / (target_std + 1e-8)
                    else:
                        critic_predictions = raw_values
                critic_loss_current = self._compute_value_loss(
                    critic_predictions,
                    batch_value_targets,
                )
                critic_loss_heldout = critic_loss_current.new_zeros(())
                critic_loss = critic_loss_current
                if (
                    critic_blended_value_loss_enabled
                    and blended_heldout_critic_inputs is not None
                    and blended_heldout_value_targets is not None
                    and blended_heldout_critic_inputs.size(0) >= int(self.config.mini_batch_size)
                ):
                    heldout_batch_count = max(
                        1,
                        blended_heldout_critic_inputs.size(0) // int(self.config.mini_batch_size),
                    )
                    heldout_batch_slot = (
                        int(current_critic_trace_minibatch_id) % heldout_batch_count
                    )
                    heldout_start = heldout_batch_slot * int(self.config.mini_batch_size)
                    heldout_end = heldout_start + int(self.config.mini_batch_size)
                    heldout_batch_critic_inputs = blended_heldout_critic_inputs[
                        heldout_start:heldout_end
                    ]
                    heldout_batch_value_targets = blended_heldout_value_targets[
                        heldout_start:heldout_end
                    ]
                    heldout_critic_features = self.network.critic_features_from_critic_input(
                        heldout_batch_critic_inputs
                    )
                    heldout_normalized_values = self.network.critic_head(
                        heldout_critic_features
                    ).squeeze(-1)
                    if self.config.value_target_mode == "popart_return_norm":
                        heldout_critic_predictions = heldout_normalized_values
                    else:
                        heldout_raw_values = self.network.value_from_critic_input(
                            heldout_batch_critic_inputs
                        ).squeeze(-1)
                        if self.config.value_target_mode in {
                            "normalized_return",
                            "running_return_norm",
                        }:
                            heldout_critic_predictions = (
                                heldout_raw_values - target_mean
                            ) / (target_std + 1e-8)
                        else:
                            heldout_critic_predictions = heldout_raw_values
                    critic_loss_heldout = self._compute_value_loss(
                        heldout_critic_predictions,
                        heldout_batch_value_targets,
                    )
                    critic_loss = (
                        critic_blended_current_weight * critic_loss_current
                        + critic_blended_heldout_weight * critic_loss_heldout
                    )
                if self._uses_blockwise_value_scaled_advantage_surrogate_mean():
                    block_value_scores = self.network.block_value_scores_from_features(
                        critic_features.detach()
                    )
                    block_value_targets = batch_value_targets.unsqueeze(-1).expand_as(
                        block_value_scores
                    )
                    block_value_aux_loss = self._compute_value_loss(
                        block_value_scores,
                        block_value_targets,
                    )
                elif self._uses_blockwise_td_style_advantage_surrogate_mean():
                    with torch.no_grad():
                        *_, block_local_rewards, block_path_probs = self._block_td_reward_terms(
                            batch_states,
                            batch_next_states,
                            actions=batch_actions,
                        )
                        block_path_rewards, _, _ = self._block_path_td_rewards(
                            batch_states,
                            batch_next_states,
                        )
                    if self._uses_blockwise_action_conditioned_path_value_bootstrap():
                        if block_path_probs is None:
                            raise ValueError(
                                "batch_actions are required for path-value TD auxiliary loss"
                            )
                        current_block_path_value_preds = (
                            self._block_path_value_predictions_from_features(
                                critic_features.detach()
                            )
                        )
                        current_block_value_preds = self._aggregate_block_path_values(
                            current_block_path_value_preds,
                            block_path_probs,
                        )
                        with torch.no_grad():
                            next_block_path_value_preds = self._block_path_value_predictions(
                                batch_next_critic_inputs
                            )
                            next_block_value_preds = self._aggregate_block_path_values(
                                next_block_path_value_preds,
                                block_path_probs,
                            )
                            block_td_targets = (
                                block_local_rewards
                                + self.config.gamma
                                * (1.0 - batch_dones.unsqueeze(-1))
                                * next_block_value_preds
                            )
                            block_path_td_targets = (
                                block_path_rewards
                                + self.config.gamma
                                * (1.0 - batch_dones.unsqueeze(-1).unsqueeze(-1))
                                * next_block_path_value_preds
                            )
                        if self._uses_blockwise_path_specific_td_supervised_advantage_surrogate_mean():
                            block_value_aux_loss = self._compute_value_loss(
                                current_block_path_value_preds,
                                block_path_td_targets,
                            )
                        else:
                            block_value_aux_loss = self._compute_value_loss(
                                current_block_value_preds,
                                block_td_targets,
                            )
                    else:
                        current_block_value_preds = self.network.block_value_scores_from_features(
                            critic_features.detach()
                        )
                        with torch.no_grad():
                            next_block_value_preds = self.network.block_value_scores_from_critic_input(
                                batch_next_critic_inputs
                            )
                            block_td_targets = (
                                block_local_rewards
                                + self.config.gamma
                                * (1.0 - batch_dones.unsqueeze(-1))
                                * next_block_value_preds
                            )
                        block_value_aux_loss = self._compute_value_loss(
                            current_block_value_preds,
                            block_td_targets,
                        )
                else:
                    block_value_aux_loss = critic_loss.new_zeros(())

                actor_objective = actor_loss - self.config.entropy_coeff * entropy
                critic_objective = self.config.value_coeff * critic_loss + block_value_aux_loss
                actor_grad_norm_trace_current = 0.0
                critic_grad_norm_trace_current = 0.0
                actor_param_delta_norm_current = 0.0
                critic_param_delta_norm_current = 0.0
                actor_backbone_delta_norm_current = 0.0
                actor_head_delta_norm_current = 0.0
                critic_backbone_delta_norm_current = 0.0
                critic_head_delta_norm_current = 0.0

                def _append_critic_drift_stage(trace_stage: str) -> None:
                    _append_critic_drift_stage_payload(
                        trace_stage=trace_stage,
                        trace_update_epoch=int(update_epoch_index),
                        trace_minibatch_id=int(current_critic_trace_minibatch_id),
                        trace_scope="minibatch",
                        critic_inputs_for_trace=batch_critic_inputs,
                        value_targets_for_trace=batch_value_targets,
                        returns_for_trace=batch_returns,
                        advantages_for_trace=batch_raw_advantages,
                        rewards_for_trace=batch_rewards,
                        prediction_target_mean=target_mean,
                        prediction_target_std=target_std,
                        actor_loss_scalar=float(actor_loss.item()),
                        critic_loss_scalar=float(critic_loss.item()),
                    )

                _append_critic_drift_stage("before_actor")

                actor_params_before_step = self._snapshot_parameter_list(self.actor_params)
                actor_backbone_params_before_step = self._snapshot_parameter_list(
                    self.network.actor_backbone_parameters()
                )
                actor_head_params_before_step = self._snapshot_parameter_list(
                    self.network.actor_main_head_parameters()
                    + self.network.actor_route_head_parameters()
                    + [self.network.actor_log_std]
                )

                if freeze_actor_training_updates:
                    pass
                elif self._uses_alternating_branch_pg():
                    route_first_order = self._uses_route_first_factorized_trust_region_pg()
                    theta_step_happened_before_route_step_current = False
                    theta_param_delta_norm_before_route_step_current = 0.0

                    def _run_theta_only_step() -> None:
                        nonlocal actor_grad_norm_value
                        nonlocal actor_grad_norm_trace_current
                        nonlocal theta_head_grad_norm_value
                        nonlocal theta_backbone_grad_norm_value
                        nonlocal theta_update_count_value
                        nonlocal theta_update_count_this_epoch
                        nonlocal theta_only_step_kl_value
                        nonlocal theta_only_step_prob_gain_value
                        nonlocal theta_branch_stopped
                        nonlocal theta_early_stop_count_value
                        nonlocal route_after_theta_shift_value
                        nonlocal theta_step_happened_before_route_step_current
                        nonlocal theta_param_delta_norm_before_route_step_current

                        if theta_branch_stopped:
                            return

                        theta_step_distribution = self.network.policy_from_actor_input(
                            batch_actor_inputs
                        )
                        theta_step_log_prob_components = theta_step_distribution.log_prob(
                            batch_actions
                        )
                        if self._uses_true_conditional_route_policy():
                            theta_step_theta_terms = self._true_conditional_theta_policy_terms(
                                batch_actions,
                                theta_step_distribution.mean,
                            )
                            theta_step_route_terms = self._true_conditional_route_policy_terms(
                                batch_actions,
                                theta_step_distribution.mean,
                            )
                            theta_step_theta_entropy = theta_step_theta_terms["entropy"].mean()
                            theta_step_theta_log_probs = theta_step_theta_terms["log_probs"]
                            theta_step_route_log_probs = theta_step_route_terms["log_probs"]
                        else:
                            theta_step_entropy_components = theta_step_distribution.entropy()
                            theta_step_theta_entropy = self._branch_entropy_from_components(
                                theta_step_entropy_components,
                                self._theta_action_indices(),
                            )
                            theta_step_theta_log_probs, theta_step_route_log_probs = (
                                self._block_theta_route_log_prob_components(
                                    theta_step_log_prob_components
                                )
                            )
                        if self._uses_factorized_conditional_route_pg():
                            theta_step_route_log_probs = self._route_margin_log_probs(
                                batch_actions,
                                theta_step_distribution.mean,
                                theta_step_distribution.stddev,
                            )
                        theta_step_joint_log_probs = None
                        if self._uses_joint_counterfactual_credit():
                            theta_step_joint_log_probs = (
                                self._true_conditional_joint_policy_terms(
                                    batch_actions,
                                    theta_step_distribution.mean,
                                )["log_probs"]
                            )

                        theta_loss_for_step = theta_loss_for_update
                        if (
                            self._uses_joint_counterfactual_credit()
                            and old_joint_log_probs is not None
                            and joint_advantages_for_surrogate is not None
                            and theta_step_joint_log_probs is not None
                        ):
                            theta_step_ratio = torch.exp(
                                theta_step_joint_log_probs - old_joint_log_probs
                            )
                            theta_step_surrogate = torch.min(
                                theta_step_ratio * joint_advantages_for_surrogate,
                                torch.clamp(
                                    theta_step_ratio,
                                    1.0 - self.config.clip_epsilon,
                                    1.0 + self.config.clip_epsilon,
                                )
                                * joint_advantages_for_surrogate,
                            )
                            theta_loss_for_step = -theta_step_surrogate.mean()
                        elif (
                            self._uses_theta_route_split_advantage_surrogate()
                            and old_theta_log_probs is not None
                            and theta_advantages_for_surrogate is not None
                        ):
                            if self._uses_factorized_ratio_pg():
                                theta_step_terms = self._theta_branch_ppo_terms(
                                    theta_step_log_prob_components,
                                    batch_old_log_prob_components,
                                    theta_advantages_for_surrogate,
                                    actions=batch_actions,
                                    new_action_means=theta_step_distribution.mean,
                                    old_action_means=batch_old_action_means,
                                )
                                theta_loss_for_step = theta_step_terms["loss"]
                            else:
                                theta_step_ratio = torch.exp(
                                    theta_step_theta_log_probs - old_theta_log_probs
                                )
                                theta_step_surrogate = torch.min(
                                    theta_step_ratio * theta_advantages_for_surrogate,
                                    torch.clamp(
                                        theta_step_ratio,
                                        1.0 - self.config.clip_epsilon,
                                        1.0 + self.config.clip_epsilon,
                                    )
                                    * theta_advantages_for_surrogate,
                                )
                                theta_loss_for_step = -theta_step_surrogate.mean()

                        theta_objective = (
                            theta_loss_for_step
                            - self.config.entropy_coeff * theta_step_theta_entropy
                        )
                        frozen_route_params = self._snapshot_parameter_list(
                            self.network.actor_route_backbone_parameters()
                            + self.network.actor_route_head_parameters()
                        )
                        frozen_route_log_std = self._snapshot_actor_log_std_indices(
                            self._route_action_indices()
                        )
                        theta_params_before_step = self._snapshot_parameter_list(
                            self.network.actor_theta_backbone_parameters()
                            + self.network.actor_theta_head_parameters()
                        )
                        theta_log_std_before_step = self._snapshot_actor_log_std_indices(
                            self._theta_action_indices()
                        )
                        theta_optimizer = self._theta_actor_optimizer()
                        theta_optimizer.zero_grad()
                        theta_objective.backward()
                        self._apply_theta_only_gradient_mask()
                        current_actor_grad_norm = self._grad_norm(self.actor_params)
                        actor_grad_norm_value += current_actor_grad_norm
                        actor_grad_norm_trace_current = max(
                            actor_grad_norm_trace_current,
                            float(current_actor_grad_norm),
                        )
                        theta_head_grad_norm_value += self._theta_head_grad_norm()
                        theta_backbone_grad_norm_value += self._theta_backbone_grad_norm()
                        torch.nn.utils.clip_grad_norm_(self.actor_params, self.config.max_grad_norm)
                        theta_optimizer.step()
                        theta_update_count_value += 1
                        theta_update_count_this_epoch += 1
                        theta_step_happened_before_route_step_current = True
                        theta_params_after_step = self._snapshot_parameter_list(
                            self.network.actor_theta_backbone_parameters()
                            + self.network.actor_theta_head_parameters()
                        )
                        theta_param_delta_norm_before_route_step_current = (
                            self._parameter_list_delta_norm(
                                theta_params_before_step,
                                theta_params_after_step,
                            )
                        )
                        if (
                            theta_log_std_before_step is not None
                            and self._theta_action_indices()
                        ):
                            theta_param_delta_norm_before_route_step_current += float(
                                (
                                    self.network.actor_log_std.detach()[
                                        self._theta_action_indices()
                                    ]
                                    - theta_log_std_before_step
                                )
                                .norm()
                                .item()
                            )
                        self._restore_parameter_list(
                            self.network.actor_route_backbone_parameters()
                            + self.network.actor_route_head_parameters(),
                            frozen_route_params,
                        )
                        self._restore_actor_log_std_indices(
                            self._route_action_indices(),
                            frozen_route_log_std,
                        )

                        with torch.no_grad():
                            post_theta_distribution = self.network.policy_from_actor_input(
                                batch_actor_inputs
                            )
                            post_theta_log_prob_components = post_theta_distribution.log_prob(
                                batch_actions
                            )
                            if self._uses_true_conditional_route_policy():
                                post_theta_theta_log_probs = (
                                    self._true_conditional_theta_policy_terms(
                                        batch_actions,
                                        post_theta_distribution.mean,
                                    )["log_probs"]
                                )
                                post_theta_route_log_probs = (
                                    self._true_conditional_route_policy_terms(
                                        batch_actions,
                                        post_theta_distribution.mean,
                                    )["log_probs"]
                                )
                            else:
                                post_theta_theta_log_probs, post_theta_route_log_probs = (
                                    self._block_theta_route_log_prob_components(
                                        post_theta_log_prob_components
                                    )
                                )
                            if self._uses_factorized_conditional_route_pg():
                                post_theta_route_log_probs = self._route_margin_log_probs(
                                    batch_actions,
                                    post_theta_distribution.mean,
                                    post_theta_distribution.stddev,
                                )
                            if self._uses_joint_counterfactual_credit():
                                post_theta_joint_log_probs = (
                                    self._true_conditional_joint_policy_terms(
                                        batch_actions,
                                        post_theta_distribution.mean,
                                    )["log_probs"]
                                )
                                theta_only_step_kl_value += float(
                                    (
                                        theta_step_joint_log_probs
                                        - post_theta_joint_log_probs
                                    )
                                    .mean()
                                    .item()
                                )
                                theta_only_step_prob_gain_value += float(
                                    (
                                        torch.exp(
                                            post_theta_joint_log_probs
                                            - theta_step_joint_log_probs
                                        )
                                        - 1.0
                                    )
                                    .mean()
                                    .item()
                                )
                            else:
                                theta_only_step_kl_value += float(
                                    (theta_step_theta_log_probs - post_theta_theta_log_probs)
                                    .mean()
                                    .item()
                                )
                                theta_only_step_prob_gain_value += float(
                                    (
                                        torch.exp(
                                            post_theta_theta_log_probs
                                            - theta_step_theta_log_probs
                                        )
                                        - 1.0
                                    )
                                    .mean()
                                    .item()
                                )
                            if (
                                self._uses_factorized_trust_region_pg()
                                and old_theta_log_probs is not None
                            ):
                                theta_branch_approx_kl = float(
                                    (
                                        old_theta_log_probs - post_theta_theta_log_probs
                                    )
                                    .mean()
                                    .item()
                                )
                                if theta_branch_approx_kl > self.config.theta_kl_target:
                                    theta_branch_stopped = True
                                    theta_early_stop_count_value += 1
                            route_after_theta_shift_delta = (
                                post_theta_route_log_probs - theta_step_route_log_probs
                            ).abs()
                            if self._uses_factorized_ratio_pg() and route_masks is not None:
                                route_after_theta_shift_value += self._masked_mean(
                                    route_after_theta_shift_delta,
                                    route_masks > 0.5,
                                )
                            else:
                                route_after_theta_shift_value += float(
                                    route_after_theta_shift_delta.mean().item()
                                )

                    def _run_route_only_step() -> None:
                        nonlocal actor_grad_norm_value
                        nonlocal actor_grad_norm_trace_current
                        nonlocal route_head_grad_norm_value
                        nonlocal route_backbone_grad_norm_value
                        nonlocal route_update_count_value
                        nonlocal route_update_count_this_epoch
                        nonlocal route_only_step_kl_value
                        nonlocal route_only_step_prob_gain_value
                        nonlocal theta_after_route_shift_value
                        nonlocal route_branch_stopped
                        nonlocal route_early_stop_count_value
                        nonlocal theta_branch_stopped
                        nonlocal theta_update_count_this_epoch
                        nonlocal coupled_stop_trigger_count_value
                        nonlocal coupled_stop_blocked_by_theta_floor_count_value
                        nonlocal coupled_stop_blocked_by_severity_gate_count_value
                        nonlocal coupled_stop_blocked_by_alignment_gate_count_value
                        nonlocal coupled_stop_blocked_by_warmup_count_value
                        nonlocal route_update_cap_trigger_count_value
                        nonlocal route_alignment_gate_accept_count_value
                        nonlocal route_alignment_gate_reject_count_value
                        nonlocal route_alignment_gate_score_sum_value
                        nonlocal route_alignment_gate_score_count_value
                        nonlocal route_step_alignment_min_value
                        nonlocal route_step_alignment_max_value
                        nonlocal route_alignment_gate_snapshot_time_ms_value
                        nonlocal route_alignment_gate_restore_time_ms_value
                        nonlocal route_alignment_gate_snapshot_count_value
                        nonlocal route_alignment_gate_restore_count_value
                        nonlocal route_credit_fallback_trigger_count_value
                        nonlocal route_credit_support_eval_count_value
                        nonlocal theta_step_happened_before_route_step_current
                        nonlocal theta_param_delta_norm_before_route_step_current

                        if route_branch_stopped or route_masks is None:
                            return
                        if self._uses_joint_counterfactual_credit():
                            if (
                                old_joint_log_probs is None
                                or joint_advantages_for_surrogate is None
                            ):
                                return
                        elif (
                            old_route_log_probs is None
                            or route_advantages_for_surrogate is None
                        ):
                            return

                        route_step_blocked_by_trust_region = False
                        route_step_blocked_by_coupled_stop = False
                        route_step_blocked_by_other_guard = False
                        optimizer_step_executed = False
                        counted_into_route_update_count = False
                        counted_route_head_grad_norm = 0.0
                        counted_route_backbone_grad_norm = 0.0
                        route_param_delta_norm = 0.0

                        if self._uses_route_update_cap_factorized_trust_region_pg():
                            route_update_cap_per_epoch = int(
                                max(
                                    0,
                                    getattr(self.config, "route_update_cap_per_epoch", 0),
                                )
                            )
                            if (
                                route_update_cap_per_epoch > 0
                                and route_update_count_this_epoch >= route_update_cap_per_epoch
                            ):
                                route_branch_stopped = True
                                route_update_cap_trigger_count_value += 1
                                route_step_blocked_by_other_guard = True
                                return

                        route_step_distribution = self.network.policy_from_actor_input(
                            batch_actor_inputs
                        )
                        route_step_log_prob_components = route_step_distribution.log_prob(
                            batch_actions
                        )
                        if self._uses_true_conditional_route_policy():
                            route_step_theta_log_probs = self._true_conditional_theta_policy_terms(
                                batch_actions,
                                route_step_distribution.mean,
                            )["log_probs"]
                            route_step_route_terms = self._true_conditional_route_policy_terms(
                                batch_actions,
                                route_step_distribution.mean,
                            )
                            route_step_route_log_probs = route_step_route_terms["log_probs"]
                            route_step_route_entropy = self._masked_tensor_mean(
                                route_step_route_terms["entropy"],
                                route_step_route_terms["active_mask"],
                            )
                        else:
                            route_step_theta_log_probs, route_step_route_log_probs = (
                                self._block_theta_route_log_prob_components(
                                    route_step_log_prob_components
                                )
                            )
                            route_step_entropy_components = route_step_distribution.entropy()
                            route_step_route_entropy = self._branch_entropy_from_components(
                                route_step_entropy_components,
                                self._route_action_indices(),
                            )
                        if self._uses_factorized_conditional_route_pg():
                            route_step_route_log_probs = self._route_margin_log_probs(
                                batch_actions,
                                route_step_distribution.mean,
                                route_step_distribution.stddev,
                            )
                        route_step_joint_log_probs = None
                        if self._uses_joint_counterfactual_credit():
                            route_step_joint_log_probs = (
                                self._true_conditional_joint_policy_terms(
                                    batch_actions,
                                    route_step_distribution.mean,
                                )["log_probs"]
                            )
                        route_has_active_blocks = bool((route_masks > 0.5).any().item())
                        if self._uses_joint_counterfactual_credit():
                            route_step_ratio = torch.exp(
                                route_step_joint_log_probs - old_joint_log_probs
                            )
                            route_step_surrogate = torch.min(
                                route_step_ratio * joint_advantages_for_surrogate,
                                torch.clamp(
                                    route_step_ratio,
                                    1.0 - self.config.clip_epsilon,
                                    1.0 + self.config.clip_epsilon,
                                )
                                * joint_advantages_for_surrogate,
                            )
                            route_only_objective = (
                                -self._mean_route_surrogate_over_active_blocks(
                                    route_step_surrogate,
                                    route_masks,
                                )
                                - self.config.entropy_coeff * route_step_route_entropy
                            )
                        elif self._uses_factorized_ratio_pg():
                            route_credit_support_eval_count_value += 1
                            route_step_terms = self._route_branch_ppo_terms(
                                batch_actions,
                                route_step_log_prob_components,
                                batch_old_log_prob_components,
                                route_advantages_for_surrogate,
                                route_masks,
                                new_action_means=route_step_distribution.mean,
                                new_action_stds=route_step_distribution.stddev,
                                old_action_means=batch_old_action_means,
                                old_action_stds=batch_old_action_stds,
                                route_gates=(
                                    split_advantages["route_gates"]
                                    if split_advantages is not None
                                    else None
                                ),
                                route_loss_masks=route_loss_masks,
                                route_used_fallback=route_used_fallback_for_batch,
                            )
                            route_has_active_blocks = bool(route_step_terms["has_active_blocks"])
                            if bool(route_step_terms.get("used_fallback", False)):
                                route_credit_fallback_trigger_count_value += 1
                            route_only_objective = (
                                route_step_terms["loss"]
                                - self.config.entropy_coeff * route_step_route_entropy
                            )
                        else:
                            route_step_ratio = torch.exp(
                                route_step_route_log_probs - old_route_log_probs
                            )
                            route_step_surrogate = torch.min(
                                route_step_ratio * route_advantages_for_surrogate,
                                torch.clamp(
                                    route_step_ratio,
                                    1.0 - self.config.clip_epsilon,
                                    1.0 + self.config.clip_epsilon,
                                )
                                * route_advantages_for_surrogate,
                            )
                            route_step_surrogate_for_loss = route_step_surrogate
                            if self._uses_hard_offload_route_surrogate():
                                route_step_surrogate_for_loss = (
                                    route_masks * route_step_surrogate_for_loss
                                )
                            if self._uses_offload_gated_route_surrogate():
                                route_step_surrogate_for_loss = (
                                    split_advantages["route_gates"]
                                    * route_step_surrogate_for_loss
                                )
                            route_only_objective = (
                                -self._mean_route_surrogate_over_active_blocks(
                                    route_step_surrogate_for_loss,
                                    route_masks,
                                )
                                - self.config.entropy_coeff * route_step_route_entropy
                            )
                        route_approx_kl_precheck = 0.0
                        route_clip_fraction_precheck = 0.0
                        route_old_logprob_mean_precheck = 0.0
                        route_new_logprob_mean_precheck = 0.0
                        route_ratio_mean_precheck = 0.0
                        route_effective_advantage_mean_precheck = 0.0
                        route_effective_advantage_std_precheck = 0.0
                        offload_active_count_precheck = int((route_masks > 0.5).sum().item())
                        offload_active_fraction_precheck = float(
                            (route_masks > 0.5).float().mean().item()
                        )
                        route_logprob_active_fraction_precheck = offload_active_fraction_precheck
                        bare_route_head_grad_norm_precheck = 0.0
                        bare_route_backbone_grad_norm_precheck = 0.0
                        route_step_index_within_epoch = int(
                            route_credit_support_eval_count_value + 1
                        )
                        if self._uses_joint_counterfactual_credit():
                            active_route_mask_precheck = route_masks > 0.5
                            route_approx_kl_precheck = self._masked_mean(
                                old_joint_log_probs - route_step_joint_log_probs,
                                active_route_mask_precheck,
                            )
                            route_clip_fraction_precheck = self._masked_mean(
                                (
                                    (route_step_ratio - 1.0).abs()
                                    > self.config.clip_epsilon
                                ).float(),
                                active_route_mask_precheck,
                            )
                            route_old_logprob_mean_precheck = self._masked_mean(
                                old_joint_log_probs,
                                active_route_mask_precheck,
                            )
                            route_new_logprob_mean_precheck = self._masked_mean(
                                route_step_joint_log_probs,
                                active_route_mask_precheck,
                            )
                            route_ratio_mean_precheck = self._masked_mean(
                                route_step_ratio,
                                active_route_mask_precheck,
                            )
                            route_effective_advantage_mean_precheck = self._masked_mean(
                                joint_advantages_for_surrogate,
                                active_route_mask_precheck,
                            )
                            route_effective_advantage_std_precheck = float(
                                self._masked_tensor_std(
                                    joint_advantages_for_surrogate,
                                    active_route_mask_precheck,
                                ).item()
                            )
                        elif self._uses_factorized_ratio_pg():
                            active_route_mask_precheck = route_loss_masks > 0.5
                            route_approx_kl_precheck = self._masked_mean(
                                route_step_terms["old_log_probs"] - route_step_terms["new_log_probs"],
                                active_route_mask_precheck,
                            )
                            route_clip_fraction_precheck = self._masked_mean(
                                (
                                    (route_step_terms["ratio"] - 1.0).abs()
                                    > self.config.clip_epsilon
                                ).float(),
                                active_route_mask_precheck,
                            )
                            route_old_logprob_mean_precheck = self._masked_mean(
                                route_step_terms["old_log_probs"],
                                active_route_mask_precheck,
                            )
                            route_new_logprob_mean_precheck = self._masked_mean(
                                route_step_terms["new_log_probs"],
                                active_route_mask_precheck,
                            )
                            route_ratio_mean_precheck = self._masked_mean(
                                route_step_terms["ratio"],
                                active_route_mask_precheck,
                            )
                            route_effective_advantage_mean_precheck = self._masked_mean(
                                route_advantages_for_surrogate,
                                active_route_mask_precheck,
                            )
                            route_effective_advantage_std_precheck = float(
                                self._masked_tensor_std(
                                    route_advantages_for_surrogate,
                                    active_route_mask_precheck,
                                ).item()
                            )
                        if route_has_active_blocks:
                            frozen_theta_params = self._snapshot_parameter_list(
                                self.network.actor_theta_backbone_parameters()
                                + self.network.actor_theta_head_parameters()
                            )
                            frozen_theta_log_std = self._snapshot_actor_log_std_indices(
                                self._theta_action_indices()
                            )
                            frozen_route_params = self._snapshot_parameter_list(
                                self.network.actor_route_backbone_parameters()
                                + self.network.actor_route_head_parameters()
                            )
                            frozen_route_log_std = self._snapshot_actor_log_std_indices(
                                self._route_action_indices()
                            )
                            route_params_before_step = self._snapshot_parameter_list(
                                self.network.actor_route_backbone_parameters()
                                + self.network.actor_route_head_parameters()
                            )
                            route_log_std_before_step = self._snapshot_actor_log_std_indices(
                                self._route_action_indices()
                            )
                            route_optimizer = self._route_actor_optimizer()
                            route_optimizer_state_before_step: dict[str, Any] | None = None
                            if self._uses_route_alignment_gate_factorized_trust_region_pg():
                                snapshot_start = time.perf_counter()
                                route_optimizer_state_before_step = copy.deepcopy(
                                    route_optimizer.state_dict()
                                )
                                route_alignment_gate_snapshot_time_ms_value += (
                                    time.perf_counter() - snapshot_start
                                ) * 1000.0
                                route_alignment_gate_snapshot_count_value += 1
                            route_optimizer.zero_grad()
                            route_only_objective.backward()
                            self._apply_route_only_gradient_mask()
                            current_actor_grad_norm = self._grad_norm(self.actor_params)
                            actor_grad_norm_value += current_actor_grad_norm
                            actor_grad_norm_trace_current = max(
                                actor_grad_norm_trace_current,
                                float(current_actor_grad_norm),
                            )
                            bare_route_head_grad_norm_precheck = self._route_head_grad_norm()
                            bare_route_backbone_grad_norm_precheck = (
                                self._route_backbone_grad_norm()
                            )
                            route_head_grad_norm_value += bare_route_head_grad_norm_precheck
                            route_backbone_grad_norm_value += (
                                bare_route_backbone_grad_norm_precheck
                            )
                            counted_route_head_grad_norm = bare_route_head_grad_norm_precheck
                            counted_route_backbone_grad_norm = (
                                bare_route_backbone_grad_norm_precheck
                            )
                            torch.nn.utils.clip_grad_norm_(
                                self.actor_params,
                                self.config.max_grad_norm,
                            )
                            route_optimizer.step()
                            optimizer_step_executed = True
                            self._restore_parameter_list(
                                self.network.actor_theta_backbone_parameters()
                                + self.network.actor_theta_head_parameters(),
                                frozen_theta_params,
                            )
                            self._restore_actor_log_std_indices(
                                self._theta_action_indices(),
                                frozen_theta_log_std,
                            )

                            with torch.no_grad():
                                post_route_distribution = self.network.policy_from_actor_input(
                                    batch_actor_inputs
                                )
                                post_route_log_prob_components = post_route_distribution.log_prob(
                                    batch_actions
                                )
                                if self._uses_true_conditional_route_policy():
                                    post_route_theta_log_probs = (
                                        self._true_conditional_theta_policy_terms(
                                            batch_actions,
                                            post_route_distribution.mean,
                                        )["log_probs"]
                                    )
                                    post_route_route_log_probs = (
                                        self._true_conditional_route_policy_terms(
                                            batch_actions,
                                            post_route_distribution.mean,
                                        )["log_probs"]
                                    )
                                else:
                                    post_route_theta_log_probs, post_route_route_log_probs = (
                                        self._block_theta_route_log_prob_components(
                                            post_route_log_prob_components
                                        )
                                    )
                                if self._uses_factorized_conditional_route_pg():
                                    post_route_route_log_probs = self._route_margin_log_probs(
                                        batch_actions,
                                        post_route_distribution.mean,
                                        post_route_distribution.stddev,
                                    )
                                route_alignment_delta = (
                                    post_route_route_log_probs - route_step_route_log_probs
                                )
                                if self._uses_factorized_ratio_pg():
                                    route_step_alignment_score = self._masked_mean(
                                        route_advantages_for_surrogate
                                        * route_alignment_delta,
                                        route_loss_masks > 0.5,
                                    )
                                else:
                                    route_step_alignment_score = float(
                                        (
                                            route_advantages_for_surrogate
                                            * route_alignment_delta
                                        )
                                        .mean()
                                        .item()
                                    )
                                route_alignment_gate_score_sum_value += float(
                                    route_step_alignment_score
                                )
                                route_alignment_gate_score_count_value += 1
                                route_step_alignment_min_value = min(
                                    route_step_alignment_min_value,
                                    float(route_step_alignment_score),
                                )
                                route_step_alignment_max_value = max(
                                    route_step_alignment_max_value,
                                    float(route_step_alignment_score),
                                )

                                if self._uses_route_alignment_gate_factorized_trust_region_pg():
                                    route_alignment_gate_threshold = float(
                                        getattr(
                                            self.config,
                                            "route_alignment_gate_threshold",
                                            0.0,
                                        )
                                    )
                                    if route_step_alignment_score < route_alignment_gate_threshold:
                                        self._restore_parameter_list(
                                            self.network.actor_route_backbone_parameters()
                                            + self.network.actor_route_head_parameters(),
                                            frozen_route_params,
                                        )
                                        self._restore_actor_log_std_indices(
                                            self._route_action_indices(),
                                            frozen_route_log_std,
                                        )
                                        if route_optimizer_state_before_step is not None:
                                            restore_start = time.perf_counter()
                                            route_optimizer.load_state_dict(
                                                route_optimizer_state_before_step
                                            )
                                            route_alignment_gate_restore_time_ms_value += (
                                                time.perf_counter() - restore_start
                                            ) * 1000.0
                                            route_alignment_gate_restore_count_value += 1
                                        route_alignment_gate_reject_count_value += 1
                                        return

                                route_alignment_gate_accept_count_value += 1
                                route_update_count_value += 1
                                route_update_count_this_epoch += 1
                                counted_into_route_update_count = True
                                route_params_after_step = self._snapshot_parameter_list(
                                    self.network.actor_route_backbone_parameters()
                                    + self.network.actor_route_head_parameters()
                                )
                                route_param_delta_norm = self._parameter_list_delta_norm(
                                    route_params_before_step,
                                    route_params_after_step,
                                )
                                if (
                                    route_log_std_before_step is not None
                                    and self._route_action_indices()
                                ):
                                    route_param_delta_norm += float(
                                        (
                                            self.network.actor_log_std.detach()[
                                                self._route_action_indices()
                                            ]
                                            - route_log_std_before_step
                                        )
                                        .norm()
                                        .item()
                                    )

                                if self._uses_joint_counterfactual_credit():
                                    post_route_joint_log_probs = (
                                        self._true_conditional_joint_policy_terms(
                                            batch_actions,
                                            post_route_distribution.mean,
                                        )["log_probs"]
                                    )
                                    route_only_step_kl_delta = (
                                        route_step_joint_log_probs
                                        - post_route_joint_log_probs
                                    )
                                    route_only_step_prob_gain = (
                                        torch.exp(
                                            post_route_joint_log_probs
                                            - route_step_joint_log_probs
                                        )
                                        - 1.0
                                    )
                                    route_only_step_kl_value += self._masked_mean(
                                        route_only_step_kl_delta,
                                        route_masks > 0.5,
                                    )
                                    route_only_step_prob_gain_value += self._masked_mean(
                                        route_only_step_prob_gain,
                                        route_masks > 0.5,
                                    )
                                else:
                                    route_only_step_kl_delta = (
                                        route_step_route_log_probs - post_route_route_log_probs
                                    )
                                    route_only_step_prob_gain = (
                                        torch.exp(
                                            post_route_route_log_probs
                                            - route_step_route_log_probs
                                        )
                                        - 1.0
                                    )
                                if self._uses_factorized_ratio_pg():
                                    route_only_step_kl_value += self._masked_mean(
                                        route_only_step_kl_delta,
                                        route_loss_masks > 0.5,
                                    )
                                    route_only_step_prob_gain_value += self._masked_mean(
                                        route_only_step_prob_gain,
                                        route_loss_masks > 0.5,
                                    )
                                elif not self._uses_joint_counterfactual_credit():
                                    route_only_step_kl_value += float(
                                        route_only_step_kl_delta.mean().item()
                                    )
                                    route_only_step_prob_gain_value += float(
                                        route_only_step_prob_gain.mean().item()
                                    )
                                theta_after_route_shift_value += float(
                                    (
                                        post_route_theta_log_probs
                                        - route_step_theta_log_probs
                                    )
                                    .abs()
                                    .mean()
                                    .item()
                                )
                                if self._uses_factorized_trust_region_pg():
                                    route_branch_approx_kl = self._masked_mean(
                                        old_route_log_probs - post_route_route_log_probs,
                                        route_loss_masks > 0.5,
                                    )
                                    if route_branch_approx_kl > self.config.route_kl_target:
                                        route_step_blocked_by_trust_region = True
                                        route_branch_stopped = True
                                        route_early_stop_count_value += 1
                                        if (
                                            self._uses_coupled_stop_factorized_trust_region_pg()
                                            and not theta_branch_stopped
                                        ):
                                            min_theta_updates_for_coupled_stop = int(
                                                max(
                                                    0,
                                                    getattr(
                                                        self.config,
                                                        "coupled_stop_min_theta_updates_per_epoch",
                                                        0,
                                                    ),
                                                )
                                            )
                                            severity_factor = float(
                                                max(
                                                    1.0,
                                                    getattr(
                                                        self.config,
                                                        "coupled_stop_severity_factor",
                                                        1.0,
                                                    ),
                                                )
                                            )
                                            severity_freeze_threshold = (
                                                float(self.config.route_kl_target)
                                                * severity_factor
                                            )
                                            alignment_freeze_threshold = float(
                                                getattr(
                                                    self.config,
                                                    "route_alignment_gate_threshold",
                                                    0.0,
                                                )
                                            )
                                            if not coupled_stop_is_active_this_train_epoch:
                                                coupled_stop_blocked_by_warmup_count_value += 1
                                            elif (
                                                self._uses_severity_gate_coupled_stop_factorized_trust_region_pg()
                                                and route_branch_approx_kl
                                                <= severity_freeze_threshold
                                            ):
                                                coupled_stop_blocked_by_severity_gate_count_value += 1
                                            elif (
                                                self._uses_severity_alignment_gate_coupled_stop_factorized_trust_region_pg()
                                                and route_step_alignment_score
                                                >= alignment_freeze_threshold
                                            ):
                                                coupled_stop_blocked_by_alignment_gate_count_value += 1
                                            elif (
                                                self._uses_theta_floor_coupled_stop_factorized_trust_region_pg()
                                                and theta_update_count_this_epoch
                                                < min_theta_updates_for_coupled_stop
                                            ):
                                                coupled_stop_blocked_by_theta_floor_count_value += 1
                                            else:
                                                route_step_blocked_by_coupled_stop = True
                                                theta_branch_stopped = True
                                                coupled_stop_trigger_count_value += 1
                        else:
                            route_step_blocked_by_other_guard = True

                        if route_training_drift_trace_enabled:
                            self._append_route_training_drift_trace_row(
                                {
                                    "mode": self.config.policy_ratio_mode,
                                    "epoch": int(train_epoch_index),
                                    "update_epoch": int(update_epoch_index),
                                    "minibatch_id": int(current_route_trace_minibatch_id),
                                    "route_step_index_within_epoch": int(
                                        route_step_index_within_epoch
                                    ),
                                    "offload_active_count": int(offload_active_count_precheck),
                                    "offload_active_fraction": float(
                                        offload_active_fraction_precheck
                                    ),
                                    "route_logprob_active_fraction": float(
                                        route_logprob_active_fraction_precheck
                                    ),
                                    "route_effective_advantage_mean": float(
                                        route_effective_advantage_mean_precheck
                                    ),
                                    "route_effective_advantage_std": float(
                                        route_effective_advantage_std_precheck
                                    ),
                                    "route_loss": float(route_only_objective.item()),
                                    "route_old_logprob_mean": float(
                                        route_old_logprob_mean_precheck
                                    ),
                                    "route_new_logprob_mean": float(
                                        route_new_logprob_mean_precheck
                                    ),
                                    "route_ratio_mean": float(route_ratio_mean_precheck),
                                    "route_approx_kl_precheck": float(
                                        route_approx_kl_precheck
                                    ),
                                    "route_clip_fraction_precheck": float(
                                        route_clip_fraction_precheck
                                    ),
                                    "bare_route_head_grad_norm": float(
                                        bare_route_head_grad_norm_precheck
                                    ),
                                    "bare_route_backbone_grad_norm": float(
                                        bare_route_backbone_grad_norm_precheck
                                    ),
                                    "route_step_blocked_by_trust_region": float(
                                        1.0 if route_step_blocked_by_trust_region else 0.0
                                    ),
                                    "route_step_blocked_by_coupled_stop": float(
                                        1.0 if route_step_blocked_by_coupled_stop else 0.0
                                    ),
                                    "route_step_blocked_by_other_guard": float(
                                        1.0 if route_step_blocked_by_other_guard else 0.0
                                    ),
                                    "optimizer_step_executed": float(
                                        1.0 if optimizer_step_executed else 0.0
                                    ),
                                    "route_param_delta_norm": float(route_param_delta_norm),
                                    "counted_into_route_update_count": float(
                                        1.0 if counted_into_route_update_count else 0.0
                                    ),
                                    "counted_route_head_grad_norm": float(
                                        counted_route_head_grad_norm
                                    ),
                                    "counted_route_backbone_grad_norm": float(
                                        counted_route_backbone_grad_norm
                                    ),
                                    "theta_step_happened_before_this_route_step": float(
                                        1.0
                                        if theta_step_happened_before_route_step_current
                                        else 0.0
                                    ),
                                    "theta_param_delta_norm_before_route_step": float(
                                        theta_param_delta_norm_before_route_step_current
                                    ),
                                }
                            )

                    if epochwise_branch_order:
                        if epochwise_route_phase:
                            _run_route_only_step()
                        elif epochwise_theta_phase:
                            _run_theta_only_step()
                    elif route_first_order:
                        _run_route_only_step()
                        _run_theta_only_step()
                    else:
                        _run_theta_only_step()
                        _run_route_only_step()
                else:
                    self.actor_optimizer.zero_grad()
                    actor_objective.backward()
                    current_actor_grad_norm = self._grad_norm(self.actor_params)
                    actor_grad_norm_value += current_actor_grad_norm
                    actor_grad_norm_trace_current = max(
                        actor_grad_norm_trace_current,
                        float(current_actor_grad_norm),
                    )
                    theta_head_grad_norm_value += self._theta_head_grad_norm()
                    route_head_grad_norm_value += self._route_head_grad_norm()
                    theta_backbone_grad_norm_value += self._theta_backbone_grad_norm()
                    route_backbone_grad_norm_value += self._route_backbone_grad_norm()
                    torch.nn.utils.clip_grad_norm_(self.actor_params, self.config.max_grad_norm)
                    self.actor_optimizer.step()

                actor_params_after_step = self._snapshot_parameter_list(self.actor_params)
                actor_backbone_params_after_step = self._snapshot_parameter_list(
                    self.network.actor_backbone_parameters()
                )
                actor_head_params_after_step = self._snapshot_parameter_list(
                    self.network.actor_main_head_parameters()
                    + self.network.actor_route_head_parameters()
                    + [self.network.actor_log_std]
                )
                actor_param_delta_norm_current = self._parameter_list_delta_norm(
                    actor_params_before_step,
                    actor_params_after_step,
                )
                actor_backbone_delta_norm_current = self._parameter_list_delta_norm(
                    actor_backbone_params_before_step,
                    actor_backbone_params_after_step,
                )
                actor_head_delta_norm_current = self._parameter_list_delta_norm(
                    actor_head_params_before_step,
                    actor_head_params_after_step,
                )
                _append_critic_drift_stage("after_actor_before_critic")
                if critic_training_internal_stage_trace_enabled:
                    _append_critic_drift_stage("before_critic")

                critic_params_before_step = self._snapshot_parameter_list(self.critic_params)
                critic_backbone_params_before_step = self._snapshot_parameter_list(
                    list(self.network.critic_backbone.parameters())
                )
                critic_head_params_before_step = self._snapshot_parameter_list(
                    list(self.network.critic_head.parameters())
                )
                self.critic_optimizer.zero_grad()
                critic_objective.backward()
                critic_grad_norm_trace_current = self._grad_norm(self.critic_params)
                current_critic_backbone_grad_norm = self._grad_norm(
                    list(self.network.critic_backbone.parameters())
                )
                current_critic_head_grad_norm = self._grad_norm(
                    list(self.network.critic_head.parameters())
                )
                critic_backbone_grad_norm_value += current_critic_backbone_grad_norm
                critic_head_grad_norm_value += current_critic_head_grad_norm
                torch.nn.utils.clip_grad_norm_(self.critic_params, self.config.max_grad_norm)
                if critic_training_internal_stage_trace_enabled:
                    _append_critic_drift_stage("after_backward_before_step")
                critic_optimizer_ablation_request = self._critic_optimizer_ablation_request()
                if (
                    critic_optimizer_ablation_request is not None
                    and int(critic_optimizer_ablation_request.get("epoch", -1))
                    == int(train_epoch_index)
                    and int(critic_optimizer_ablation_request.get("update_epoch", -1))
                    == int(update_epoch_index)
                    and int(critic_optimizer_ablation_request.get("minibatch_id", -1))
                    == int(current_critic_trace_minibatch_id)
                ):
                    self.critic_optimizer_ablation_capture = {
                        "epoch": int(train_epoch_index),
                        "update_epoch": int(update_epoch_index),
                        "minibatch_id": int(current_critic_trace_minibatch_id),
                        "critic_loss": float(critic_loss.item()),
                        "actor_loss": float(actor_loss.item()),
                        "batch_index": batch_index.detach().clone(),
                        "batch_states": batch_states.detach().clone(),
                        "batch_next_states": batch_next_states.detach().clone(),
                        "batch_actions": batch_actions.detach().clone(),
                        "batch_dones": batch_dones.detach().clone(),
                        "target_mean": target_mean.detach().clone(),
                        "target_std": target_std.detach().clone(),
                        "batch_critic_inputs": batch_critic_inputs.detach().clone(),
                        "batch_value_targets": batch_value_targets.detach().clone(),
                        "batch_returns": batch_returns.detach().clone(),
                        "batch_raw_advantages": batch_raw_advantages.detach().clone(),
                        "batch_rewards": batch_rewards.detach().clone(),
                        "batch_next_critic_inputs": batch_next_critic_inputs.detach().clone(),
                        "critic_optimizer_state_dict": copy.deepcopy(
                            self.critic_optimizer.state_dict()
                        ),
                    }
                    raise CriticOptimizerAblationCaptured(
                        "Captured critic optimizer ablation snapshot before optimizer.step()."
                    )
                critic_step_optimizer_state_before_step = None
                critic_step_current_loss_before = float(critic_loss_current.item())
                critic_step_current_loss_after = critic_step_current_loss_before
                critic_step_heldout_loss_before = 0.0
                critic_step_heldout_loss_after = 0.0
                critic_step_probe_pearson_before = 0.0
                critic_step_probe_pearson_after = 0.0
                critic_step_current_improved = True
                critic_step_heldout_degraded = False
                critic_step_probe_degraded = False
                critic_step_would_reject = False
                critic_step_accepted = True
                critic_step_reject_reason_labels: list[str] = []
                attempted_critic_param_delta_norm_current = 0.0
                attempted_critic_backbone_delta_norm_current = 0.0
                attempted_critic_head_delta_norm_current = 0.0
                if critic_step_monitor_enabled:
                    critic_step_attempt_count_value += 1
                    critic_step_current_loss_before_sum_value += critic_step_current_loss_before
                    if critic_step_enforce_enabled:
                        critic_step_optimizer_state_before_step = copy.deepcopy(
                            self.critic_optimizer.state_dict()
                        )
                    if critic_training_drift_heldout_payload is not None:
                        heldout_gate_value_targets = self._value_targets_from_returns_with_stats(
                            critic_training_drift_heldout_payload["returns"],
                            target_mean,
                            target_std,
                        )
                        heldout_gate_critic_inputs, _, _ = self._prepare_critic_inputs(
                            critic_training_drift_heldout_payload["states"],
                            update_stats=False,
                        )
                        _heldout_semantics_before, critic_step_heldout_loss_before = (
                            self._critic_trace_loss_and_semantics(
                                heldout_gate_critic_inputs,
                                heldout_gate_value_targets,
                                critic_training_drift_heldout_payload["returns"],
                                critic_training_drift_heldout_payload["advantages"],
                                critic_training_drift_heldout_payload["rewards"],
                                target_mean,
                                target_std,
                            )
                        )
                    critic_step_heldout_loss_before_sum_value += critic_step_heldout_loss_before
                    if critic_training_drift_probe_payload is not None:
                        probe_gate_value_targets = self._value_targets_from_returns_with_stats(
                            critic_training_drift_probe_payload["returns"],
                            target_mean,
                            target_std,
                        )
                        probe_gate_critic_inputs, _, _ = self._prepare_critic_inputs(
                            critic_training_drift_probe_payload["states"],
                            update_stats=False,
                        )
                        probe_semantics_before, _probe_loss_before = (
                            self._critic_trace_loss_and_semantics(
                                probe_gate_critic_inputs,
                                probe_gate_value_targets,
                                critic_training_drift_probe_payload["returns"],
                                critic_training_drift_probe_payload["advantages"],
                                critic_training_drift_probe_payload["rewards"],
                                target_mean,
                                target_std,
                            )
                        )
                        critic_step_probe_pearson_before = float(
                            probe_semantics_before.get("pearson_value_vs_value_target", 0.0)
                        )
                    critic_step_probe_pearson_before_sum_value += critic_step_probe_pearson_before
                if hasattr(self.critic_optimizer, "set_telemetry_context"):
                    self.critic_optimizer.set_telemetry_context(
                        train_epoch=int(train_epoch_index),
                        update_epoch=int(update_epoch_index),
                        minibatch_id=int(current_critic_trace_minibatch_id),
                    )
                self.critic_optimizer.step()
                critic_params_after_step = self._snapshot_parameter_list(self.critic_params)
                critic_backbone_params_after_step = self._snapshot_parameter_list(
                    list(self.network.critic_backbone.parameters())
                )
                critic_head_params_after_step = self._snapshot_parameter_list(
                    list(self.network.critic_head.parameters())
                )
                critic_param_delta_norm_current = self._parameter_list_delta_norm(
                    critic_params_before_step,
                    critic_params_after_step,
                )
                critic_backbone_delta_norm_current = self._parameter_list_delta_norm(
                    critic_backbone_params_before_step,
                    critic_backbone_params_after_step,
                )
                critic_head_delta_norm_current = self._parameter_list_delta_norm(
                    critic_head_params_before_step,
                    critic_head_params_after_step,
                )
                attempted_critic_param_delta_norm_current = critic_param_delta_norm_current
                attempted_critic_backbone_delta_norm_current = critic_backbone_delta_norm_current
                attempted_critic_head_delta_norm_current = critic_head_delta_norm_current
                if hasattr(self.critic_optimizer, "consume_debug_metrics"):
                    critic_optimizer_debug_metrics = self.critic_optimizer.consume_debug_metrics()
                    if isinstance(critic_optimizer_debug_metrics, dict):
                        active_count = int(
                            critic_optimizer_debug_metrics.get("active_count", 0)
                        )
                        clipped_count = int(
                            critic_optimizer_debug_metrics.get("clipped_count", 0)
                        )
                        critic_backbone_preconditioner_active_count_value += active_count
                        critic_backbone_preconditioner_clipped_count_value += clipped_count
                        critic_backbone_preconditioner_active_sum_before_clip_value += float(
                            critic_optimizer_debug_metrics.get(
                                "active_preconditioner_sum_before_clip",
                                0.0,
                            )
                        )
                        critic_backbone_preconditioner_active_sum_after_clip_value += float(
                            critic_optimizer_debug_metrics.get(
                                "active_preconditioner_sum_after_clip",
                                0.0,
                            )
                        )
                        critic_backbone_preconditioner_active_max_before_clip_value = max(
                            critic_backbone_preconditioner_active_max_before_clip_value,
                            float(
                                critic_optimizer_debug_metrics.get(
                                    "active_preconditioner_max_before_clip",
                                    0.0,
                                )
                            ),
                        )
                        critic_backbone_preconditioner_active_max_after_clip_value = max(
                            critic_backbone_preconditioner_active_max_after_clip_value,
                            float(
                                critic_optimizer_debug_metrics.get(
                                    "active_preconditioner_max_after_clip",
                                    0.0,
                                )
                            ),
                        )
                        critic_backbone_preconditioner_cap_value = max(
                            critic_backbone_preconditioner_cap_value,
                            float(
                                critic_optimizer_debug_metrics.get(
                                    "max_preconditioner",
                                    0.0,
                                )
                            ),
                        )
                if critic_step_monitor_enabled:
                    batch_semantics_after, critic_step_current_loss_after = (
                        self._critic_trace_loss_and_semantics(
                            batch_critic_inputs,
                            batch_value_targets,
                            batch_returns,
                            batch_raw_advantages,
                            batch_rewards,
                            target_mean,
                            target_std,
                        )
                    )
                    if critic_training_drift_heldout_payload is not None:
                        heldout_gate_value_targets = self._value_targets_from_returns_with_stats(
                            critic_training_drift_heldout_payload["returns"],
                            target_mean,
                            target_std,
                        )
                        heldout_gate_critic_inputs, _, _ = self._prepare_critic_inputs(
                            critic_training_drift_heldout_payload["states"],
                            update_stats=False,
                        )
                        _heldout_semantics_after, critic_step_heldout_loss_after = (
                            self._critic_trace_loss_and_semantics(
                                heldout_gate_critic_inputs,
                                heldout_gate_value_targets,
                                critic_training_drift_heldout_payload["returns"],
                                critic_training_drift_heldout_payload["advantages"],
                                critic_training_drift_heldout_payload["rewards"],
                                target_mean,
                                target_std,
                            )
                        )
                    if critic_training_drift_probe_payload is not None:
                        probe_gate_value_targets = self._value_targets_from_returns_with_stats(
                            critic_training_drift_probe_payload["returns"],
                            target_mean,
                            target_std,
                        )
                        probe_gate_critic_inputs, _, _ = self._prepare_critic_inputs(
                            critic_training_drift_probe_payload["states"],
                            update_stats=False,
                        )
                        probe_semantics_after, _probe_loss_after = (
                            self._critic_trace_loss_and_semantics(
                                probe_gate_critic_inputs,
                                probe_gate_value_targets,
                                critic_training_drift_probe_payload["returns"],
                                critic_training_drift_probe_payload["advantages"],
                                critic_training_drift_probe_payload["rewards"],
                                target_mean,
                                target_std,
                            )
                        )
                        critic_step_probe_pearson_after = float(
                            probe_semantics_after.get("pearson_value_vs_value_target", 0.0)
                        )

                    critic_step_current_loss_after_sum_value += critic_step_current_loss_after
                    critic_step_heldout_loss_after_sum_value += critic_step_heldout_loss_after
                    critic_step_probe_pearson_after_sum_value += critic_step_probe_pearson_after

                    critic_step_current_improved = (
                        critic_step_current_loss_after
                        < critic_step_current_loss_before
                        - critic_step_current_loss_improve_epsilon
                    )
                    critic_step_heldout_degraded = (
                        critic_training_drift_heldout_payload is not None
                        and critic_step_heldout_loss_after
                        > critic_step_heldout_loss_before + critic_step_heldout_loss_tolerance
                    )
                    critic_step_probe_degraded = (
                        critic_training_drift_probe_payload is not None
                        and critic_step_probe_pearson_after
                        < critic_step_probe_pearson_before - critic_step_probe_pearson_tolerance
                    )
                    if not critic_step_current_improved:
                        critic_step_reject_reason_labels.append(
                            "current_batch_loss_not_decreased"
                        )
                    if critic_step_heldout_degraded:
                        critic_step_reject_reason_labels.append("heldout_loss_increase")
                    if critic_step_probe_degraded:
                        critic_step_reject_reason_labels.append("probe_target_pearson_drop")
                    critic_step_would_reject = bool(critic_step_reject_reason_labels)
                    if critic_step_would_reject:
                        critic_step_would_reject_count_value += 1
                        for reason_label in critic_step_reject_reason_labels:
                            critic_step_would_reject_reason_counts[reason_label] = (
                                critic_step_would_reject_reason_counts.get(reason_label, 0) + 1
                            )
                    if critic_step_enforce_enabled and critic_step_would_reject:
                        critic_step_accepted = False
                        critic_step_reject_count_value += 1
                        for reason_label in critic_step_reject_reason_labels:
                            critic_step_reject_reason_counts[reason_label] = (
                                critic_step_reject_reason_counts.get(reason_label, 0) + 1
                            )
                        self._restore_parameter_list(
                            self.critic_params,
                            critic_params_before_step,
                        )
                        self._restore_parameter_list(
                            list(self.network.critic_backbone.parameters()),
                            critic_backbone_params_before_step,
                        )
                        self._restore_parameter_list(
                            list(self.network.critic_head.parameters()),
                            critic_head_params_before_step,
                        )
                        if critic_step_optimizer_state_before_step is not None:
                            self.critic_optimizer.load_state_dict(
                                critic_step_optimizer_state_before_step
                            )
                        critic_param_delta_norm_current = 0.0
                        critic_backbone_delta_norm_current = 0.0
                        critic_head_delta_norm_current = 0.0
                    else:
                        critic_step_accept_count_value += 1
                    self._append_critic_step_acceptance_trace_row(
                        {
                            "mode": self.config.policy_ratio_mode,
                            "epoch": int(train_epoch_index),
                            "update_epoch": int(update_epoch_index),
                            "minibatch_id": int(current_critic_trace_minibatch_id),
                            "current_batch_loss_before": float(critic_step_current_loss_before),
                            "current_batch_loss_after_attempt": float(
                                critic_step_current_loss_after
                            ),
                            "heldout_loss_before": float(critic_step_heldout_loss_before),
                            "heldout_loss_after_attempt": float(
                                critic_step_heldout_loss_after
                            ),
                            "probe_target_pearson_before": float(
                                critic_step_probe_pearson_before
                            ),
                            "probe_target_pearson_after_attempt": float(
                                critic_step_probe_pearson_after
                            ),
                            "current_batch_improved": float(
                                1.0 if critic_step_current_improved else 0.0
                            ),
                            "heldout_degraded": float(
                                1.0 if critic_step_heldout_degraded else 0.0
                            ),
                            "probe_degraded": float(
                                1.0 if critic_step_probe_degraded else 0.0
                            ),
                            "would_reject": float(1.0 if critic_step_would_reject else 0.0),
                            "accepted": float(1.0 if critic_step_accepted else 0.0),
                            "rejected": float(0.0 if critic_step_accepted else 1.0),
                            "reject_reasons": "|".join(critic_step_reject_reason_labels),
                            "attempted_critic_param_delta_norm": float(
                                attempted_critic_param_delta_norm_current
                            ),
                            "attempted_critic_backbone_delta_norm": float(
                                attempted_critic_backbone_delta_norm_current
                            ),
                            "attempted_critic_head_delta_norm": float(
                                attempted_critic_head_delta_norm_current
                            ),
                            "effective_critic_param_delta_norm": float(
                                critic_param_delta_norm_current
                            ),
                            "effective_critic_backbone_delta_norm": float(
                                critic_backbone_delta_norm_current
                            ),
                            "effective_critic_head_delta_norm": float(
                                critic_head_delta_norm_current
                            ),
                            "current_loss_improve_epsilon": float(
                                critic_step_current_loss_improve_epsilon
                            ),
                            "heldout_loss_tolerance": float(
                                critic_step_heldout_loss_tolerance
                            ),
                            "probe_pearson_tolerance": float(
                                critic_step_probe_pearson_tolerance
                            ),
                        }
                    )
                critic_param_delta_norm_value += critic_param_delta_norm_current
                critic_backbone_delta_norm_value += critic_backbone_delta_norm_current
                critic_head_delta_norm_value += critic_head_delta_norm_current
                if critic_training_internal_stage_trace_enabled:
                    _append_critic_drift_stage("after_optimizer_step")
                _append_critic_drift_stage("after_critic")

                actor_loss_value += float(actor_loss.item())
                critic_loss_value += float(critic_loss.item())
                critic_loss_current_value += float(critic_loss_current.item())
                critic_loss_heldout_value += float(critic_loss_heldout.item())
                entropy_value += float(entropy.item())

        update_count = max(
            1,
            self.config.update_epochs
            * int(np.ceil(all_train_indices.size(0) / self.config.mini_batch_size)),
        )
        theta_update_denominator = max(1, theta_update_count_value)
        route_update_denominator = max(1, route_update_count_value)
        route_alignment_gate_total_steps = max(
            1,
            route_alignment_gate_accept_count_value + route_alignment_gate_reject_count_value,
        )
        route_alignment_gate_score_denominator = max(1, route_alignment_gate_score_count_value)
        route_step_alignment_min_metric = (
            float(route_step_alignment_min_value)
            if route_alignment_gate_score_count_value > 0
            else 0.0
        )
        route_step_alignment_max_metric = (
            float(route_step_alignment_max_value)
            if route_alignment_gate_score_count_value > 0
            else 0.0
        )
        critic_step_denominator = max(1, critic_step_attempt_count_value)
        with torch.no_grad():
            full_distribution, full_actor_diagnostics = (
                self.network.policy_with_diagnostics_from_actor_input(full_actor_inputs)
            )
            full_critic_features = self.network.critic_features_from_critic_input(full_critic_inputs)
            full_new_log_prob_components = full_distribution.log_prob(actions)
            logprob_delta_components = full_new_log_prob_components - old_log_prob_components
            logprob_delta_sum = logprob_delta_components.sum(dim=-1)
            logprob_delta_mean = logprob_delta_components.mean(dim=-1)
            block_logprob_delta = self._block_sum_log_prob_components(logprob_delta_components)
            theta_logprob_delta, route_logprob_delta = self._block_theta_route_log_prob_components(
                logprob_delta_components
            )
            theta_old_log_probs_summary = None
            theta_new_log_probs_summary = None
            route_old_log_probs_summary = None
            route_new_log_probs_summary = None
            joint_old_log_probs_summary = None
            joint_new_log_probs_summary = None
            route_logprob_active_mask_summary = None
            theta_selected_probs_summary = None
            route_selected_probs_summary = None
            if self._uses_true_conditional_route_policy():
                theta_old_terms_summary = self._true_conditional_theta_policy_terms(
                    actions,
                    old_action_means,
                )
                theta_new_terms_summary = self._true_conditional_theta_policy_terms(
                    actions,
                    full_distribution.mean,
                )
                route_old_terms_summary = self._true_conditional_route_policy_terms(
                    actions,
                    old_action_means,
                )
                route_new_terms_summary = self._true_conditional_route_policy_terms(
                    actions,
                    full_distribution.mean,
                )
                theta_old_log_probs_summary = theta_old_terms_summary["log_probs"]
                theta_new_log_probs_summary = theta_new_terms_summary["log_probs"]
                route_old_log_probs_summary = route_old_terms_summary["log_probs"]
                route_new_log_probs_summary = route_new_terms_summary["log_probs"]
                route_logprob_active_mask_summary = route_new_terms_summary["active_mask"]
                theta_selected_probs_summary = theta_new_terms_summary["selected_probs"]
                route_selected_probs_summary = route_new_terms_summary["selected_probs"]
                theta_logprob_delta = theta_new_log_probs_summary - theta_old_log_probs_summary
                route_logprob_delta = route_new_log_probs_summary - route_old_log_probs_summary
            elif self._uses_factorized_conditional_route_pg():
                old_route_log_probs = self._route_margin_log_probs(
                    actions,
                    old_action_means,
                    old_action_stds,
                )
                new_route_log_probs = self._route_margin_log_probs(
                    actions,
                    full_distribution.mean,
                    full_distribution.stddev,
                )
                route_logprob_delta = new_route_log_probs - old_route_log_probs
            theta_ratio = torch.exp(theta_logprob_delta)
            route_ratio = torch.exp(route_logprob_delta)
            theta_clip_mask = (theta_ratio < (1.0 - self.config.clip_epsilon)) | (
                theta_ratio > (1.0 + self.config.clip_epsilon)
            )
            route_clip_mask = (route_ratio < (1.0 - self.config.clip_epsilon)) | (
                route_ratio > (1.0 + self.config.clip_epsilon)
            )
            if route_logprob_active_mask_summary is not None:
                route_clip_mask = route_clip_mask & route_logprob_active_mask_summary
            if self._uses_theta_route_split_advantage_surrogate():
                block_logprob_delta = self._block_split_log_prob_components(logprob_delta_components)
            block_mean_delta = block_logprob_delta.mean(dim=-1)
            block_ratio_proxy = torch.exp(block_mean_delta)
            block_ratio = torch.exp(block_logprob_delta)
            block_clipped_ratio = torch.clamp(
                block_ratio,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon,
            )
            (
                block_local_cost_now,
                block_local_cost_next,
                block_path_cost_local_now,
                block_path_cost_bs_now,
                block_action_conditioned_cost_now,
                block_action_conditioned_cost_next,
                block_delta_cost,
                block_local_rewards,
                block_path_probs,
            ) = self._block_td_reward_terms(states, next_states, actions=actions)
            block_clip_mask = (block_ratio < (1.0 - self.config.clip_epsilon)) | (
                block_ratio > (1.0 + self.config.clip_epsilon)
            )
            block_advantages, block_advantage_weights = self._block_surrogate_advantages(
                advantages,
                states,
                actions=actions,
                critic_inputs=full_critic_inputs,
                next_states=next_states,
                next_critic_inputs=full_next_critic_inputs,
                dones=dones,
            )
            split_advantages = None
            if self._uses_blockwise_action_conditioned_path_value_bootstrap():
                split_advantages = self._theta_route_split_advantages(
                    states,
                    next_states,
                    dones,
                    actions,
                    critic_inputs=full_critic_inputs,
                    next_critic_inputs=full_next_critic_inputs,
                )
            if self._uses_blockwise_value_scaled_advantage_surrogate_mean():
                block_value_scores, block_value_scales = self._block_value_scores_and_scales(
                    full_critic_inputs
                )
                block_adv_scales = block_value_scales
                block_value_preds = block_value_scores
                block_path_value_preds = block_value_preds.unsqueeze(-1).expand(
                    -1,
                    -1,
                    self.action_path_count,
                )
                next_block_path_value_preds = block_path_value_preds
                block_action_conditioned_value_preds = block_value_preds
                block_action_conditioned_value_next_preds = block_value_preds
                block_path_td_rewards = torch.zeros_like(block_path_value_preds)
                block_path_td_targets = torch.zeros_like(block_path_value_preds)
                block_td_targets = torch.zeros_like(block_ratio)
                block_td_advantages = block_advantages
            elif self._uses_blockwise_td_style_advantage_surrogate_mean():
                td_value_signal = self._block_td_value_signal(
                    states,
                    next_states,
                    dones,
                    actions=actions,
                    critic_inputs=full_critic_inputs,
                    next_critic_inputs=full_next_critic_inputs,
                )
                block_value_preds = td_value_signal["block_value_preds"]
                block_td_targets = td_value_signal["block_td_targets"]
                block_td_advantages = td_value_signal["block_td_advantages"]
                block_path_td_rewards = td_value_signal["block_path_td_rewards"]
                block_path_td_targets = td_value_signal["block_path_td_targets"]
                block_path_value_preds = td_value_signal["block_path_value_preds"]
                next_block_path_value_preds = td_value_signal["next_block_path_value_preds"]
                block_action_conditioned_value_preds = td_value_signal[
                    "block_action_conditioned_value_preds"
                ]
                block_action_conditioned_value_next_preds = td_value_signal[
                    "block_action_conditioned_value_next_preds"
                ]
                block_value_scores = block_value_preds
                block_value_scales = torch.ones_like(block_ratio)
                block_adv_scales = torch.ones_like(block_ratio)
            else:
                block_value_scores = torch.zeros_like(block_ratio)
                block_value_scales = torch.ones_like(block_ratio)
                block_value_preds = torch.zeros_like(block_ratio)
                block_path_value_preds = block_value_preds.unsqueeze(-1).expand(
                    -1,
                    -1,
                    self.action_path_count,
                )
                next_block_path_value_preds = block_path_value_preds
                block_action_conditioned_value_preds = block_value_preds
                block_action_conditioned_value_next_preds = block_value_preds
                block_path_td_rewards = torch.zeros_like(block_path_value_preds)
                block_path_td_targets = torch.zeros_like(block_path_value_preds)
                block_td_targets = torch.zeros_like(block_ratio)
                block_td_advantages = block_advantages
            if self._uses_blockwise_scaled_advantage_surrogate_mean():
                block_adv_scales = block_advantage_weights
            elif self._uses_blockwise_pseudolocal_advantage_surrogate_mean():
                block_adv_scales = self._block_advantage_scales(states)
            elif not self._uses_blockwise_value_scaled_advantage_surrogate_mean():
                if not self._uses_blockwise_td_style_advantage_surrogate_mean():
                    block_adv_scales = torch.ones_like(block_ratio)
            else:
                block_adv_scales = block_value_scales
            theta_advantages = torch.zeros_like(block_ratio)
            route_advantages = torch.zeros_like(block_ratio)
            theta_advantages_norm = torch.zeros_like(block_ratio)
            route_advantages_norm = torch.zeros_like(block_ratio)
            route_gates = torch.zeros_like(block_ratio)
            route_masks = torch.zeros_like(block_ratio)
            route_loss_masks = torch.zeros_like(block_ratio)
            route_credit_gate = torch.zeros_like(block_ratio)
            route_credit_weight = torch.ones_like(block_ratio)
            route_residual_credit = torch.zeros_like(block_ratio)
            route_expected_score = torch.zeros_like(block_ratio)
            route_selected_score = torch.zeros_like(block_ratio)
            route_score_vector = torch.zeros(
                (*block_ratio.shape, 2),
                device=block_ratio.device,
                dtype=block_ratio.dtype,
            )
            route_mask_counts = torch.zeros(block_ratio.shape[0], device=block_ratio.device, dtype=block_ratio.dtype)
            offload_vs_local_value_gap = torch.zeros_like(block_ratio)
            bs1_vs_bs2_value_gap = torch.zeros_like(block_ratio)
            theta_adv_norm_mean = block_ratio.new_zeros(())
            theta_adv_norm_std = block_ratio.new_zeros(())
            route_adv_norm_mean = block_ratio.new_zeros(())
            route_adv_norm_std = block_ratio.new_zeros(())
            route_adv_norm_active_count = block_ratio.new_zeros(())
            route_adv_norm_used_fallback = 0.0
            theta_selected_action_prob_gain = theta_ratio - 1.0
            route_selected_action_prob_gain = route_ratio - 1.0
            offload_active_fraction = 0.0
            route_logprob_active_fraction = 0.0
            theta_positive_fraction = 0.0
            route_credit_gate_fraction = 0.0
            route_credit_effective_fraction = 0.0
            route_credit_effective_count = 0.0
            route_credit_weight_mean = 1.0
            route_credit_weight_std = 0.0
            route_credit_weight_min = 1.0
            route_credit_weight_max = 1.0
            theta_candidate_score_mean = 0.0
            theta_candidate_score_std = 0.0
            theta_selected_score_mean = 0.0
            theta_expected_score_mean = 0.0
            theta_residual_credit_mean = 0.0
            theta_residual_credit_std = 0.0
            theta_residual_credit_min = 0.0
            theta_residual_credit_max = 0.0
            route_candidate_score_mean = 0.0
            route_candidate_score_std = 0.0
            route_residual_credit_mean = 0.0
            route_residual_credit_std = 0.0
            route_residual_credit_min = 0.0
            route_residual_credit_max = 0.0
            route_expected_score_mean = 0.0
            route_selected_score_mean = 0.0
            route_decision_agreement_ratio = 0.0
            actual_bs1_rate_when_route_true_gap_positive = 0.0
            actual_bs1_rate_when_route_true_gap_negative = 0.0
            route_score_vector_mean_abs = 0.0
            route_score_vector_std = 0.0
            theta_old_logprob_mean = 0.0
            theta_new_logprob_mean = 0.0
            route_old_logprob_mean = 0.0
            route_new_logprob_mean = 0.0
            joint_old_logprob_mean = 0.0
            joint_new_logprob_mean = 0.0
            joint_ratio_mean = 0.0
            joint_approx_kl = 0.0
            joint_clip_fraction = 0.0
            joint_selected_score_mean = 0.0
            joint_expected_score_mean = 0.0
            joint_residual_credit_mean = 0.0
            joint_residual_credit_std = 0.0
            joint_residual_credit_min = 0.0
            joint_residual_credit_max = 0.0
            joint_action_decision_agreement_ratio_under_reward_aligned = 0.0
            joint_action_decision_agreement_ratio_under_td_aligned = 0.0
            actual_local_rate_when_reward_aligned_best_is_local = 0.0
            actual_bs1_rate_when_reward_aligned_best_is_bs1 = 0.0
            actual_bs2_rate_when_reward_aligned_best_is_bs2 = 0.0
            actual_local_rate_when_td_aligned_best_is_local = 0.0
            actual_bs1_rate_when_td_aligned_best_is_bs1 = 0.0
            actual_bs2_rate_when_td_aligned_best_is_bs2 = 0.0
            conditional_policy_consistency_score = 0.0
            offload_decision_agreement_ratio = 0.0
            actual_offload_rate_when_theta_true_gap_positive = 0.0
            actual_offload_rate_when_theta_true_gap_negative = 0.0
            theta_ratio_mean = float(theta_ratio.mean().item())
            theta_ratio_std = float(theta_ratio.std(unbiased=False).item())
            theta_ratio_max = float(theta_ratio.max().item())
            route_ratio_mean = 0.0
            route_ratio_std = 0.0
            route_ratio_max = 0.0
            theta_approx_kl = (-theta_logprob_delta).mean()
            route_approx_kl = route_ratio.new_zeros(())
            theta_loss_mean = block_ratio.new_zeros(())
            route_loss_mean = block_ratio.new_zeros(())
            theta_route_loss_ratio = block_ratio.new_zeros(())
            action_theta, action_route_probs = self._block_theta_and_route_probabilities(actions)
            if self._uses_theta_route_split_advantage_surrogate() and split_advantages is not None:
                theta_advantages = split_advantages["theta_advantages"]
                route_advantages = split_advantages["route_advantages"]
                route_gates = split_advantages["route_gates"]
                route_masks = split_advantages["route_masks"]
                route_loss_masks = route_masks
                route_credit_gate = (theta_advantages > 0.0).to(route_masks.dtype)
                route_credit_weight = torch.ones_like(route_masks)
                if self._uses_route_credit_theta_gate():
                    route_credit_support = self._route_credit_theta_gate_support(
                        theta_advantages,
                        route_masks,
                    )
                    route_loss_masks = route_credit_support["effective_mask"]
                    route_credit_gate = route_credit_support["route_credit_gate"]
                    route_credit_effective_fraction = float(
                        route_credit_support["effective_fraction"]
                    )
                    route_credit_effective_count = float(
                        route_credit_support["effective_count"]
                    )
                else:
                    route_credit_effective_fraction = float(route_masks.mean().item())
                    route_credit_effective_count = float(route_masks.sum().item())
                route_mask_counts = split_advantages["route_mask_counts"]
                offload_vs_local_value_gap = split_advantages["offload_vs_local_value_gap"]
                bs1_vs_bs2_value_gap = split_advantages["bs1_vs_bs2_value_gap"]
                branchwise_normalization = (
                    self._branchwise_normalize_theta_route_advantages(
                        theta_advantages,
                        route_advantages,
                        route_masks,
                    )
                    if self._uses_branchwise_balanced_pg()
                    else None
                )
                theta_advantages_norm = (
                    branchwise_normalization["theta_advantages_norm"]
                    if branchwise_normalization is not None
                    else theta_advantages
                )
                route_advantages_norm = (
                    branchwise_normalization["route_advantages_norm"]
                    if branchwise_normalization is not None
                    else route_advantages
                )
                if self._uses_true_conditional_route_policy():
                    theta_old_policy_terms_for_credit = (
                        self._true_conditional_theta_policy_terms(
                            actions,
                            old_action_means,
                        )
                    )
                    theta_candidate_score_credit_terms = (
                        self._theta_candidate_score_credit_terms(
                            states,
                            theta_old_policy_terms_for_credit["offload_active_mask"],
                            theta_old_policy_terms_for_credit["offload_probs"].detach(),
                        )
                    )
                    theta_candidate_score_vector = theta_candidate_score_credit_terms[
                        "candidate_score_vector"
                    ]
                    theta_selected_score = theta_candidate_score_credit_terms[
                        "selected_score"
                    ]
                    theta_expected_score = theta_candidate_score_credit_terms[
                        "expected_score"
                    ]
                    theta_residual_credit = theta_candidate_score_credit_terms[
                        "residual_credit"
                    ]
                    theta_candidate_score_mean = float(
                        theta_candidate_score_credit_terms["candidate_score_mean"]
                    )
                    theta_candidate_score_std = float(
                        theta_candidate_score_credit_terms["candidate_score_std"]
                    )
                    theta_selected_score_mean = float(
                        theta_candidate_score_credit_terms["selected_score_mean"]
                    )
                    theta_expected_score_mean = float(
                        theta_candidate_score_credit_terms["expected_score_mean"]
                    )
                    theta_residual_credit_mean = float(
                        theta_candidate_score_credit_terms["residual_mean"]
                    )
                    theta_residual_credit_std = float(
                        theta_candidate_score_credit_terms["residual_std"]
                    )
                    theta_residual_credit_min = float(
                        theta_candidate_score_credit_terms["residual_min"]
                    )
                    theta_residual_credit_max = float(
                        theta_candidate_score_credit_terms["residual_max"]
                    )
                    offload_decision_agreement_ratio = float(
                        theta_candidate_score_credit_terms[
                            "offload_decision_agreement_ratio"
                        ]
                    )
                    actual_offload_rate_when_theta_true_gap_positive = float(
                        theta_candidate_score_credit_terms[
                            "actual_offload_rate_when_theta_true_gap_positive"
                        ]
                    )
                    actual_offload_rate_when_theta_true_gap_negative = float(
                        theta_candidate_score_credit_terms[
                            "actual_offload_rate_when_theta_true_gap_negative"
                        ]
                    )
                if self._uses_theta_candidate_score_credit():
                    theta_advantages_norm = theta_residual_credit
                if self._uses_true_conditional_route_policy():
                    joint_old_policy_terms_for_credit = self._true_conditional_joint_policy_terms(
                        actions,
                        old_action_means,
                    )
                    joint_new_policy_terms_for_credit = self._true_conditional_joint_policy_terms(
                        actions,
                        full_distribution.mean,
                    )
                    if self._uses_joint_reward_aligned_credit() and joint_reward_aligned_scores is not None:
                        joint_reward_aligned_credit_terms = self._joint_reward_aligned_credit_terms(
                            joint_reward_aligned_scores,
                            joint_old_policy_terms_for_credit["selected_indices"],
                            joint_old_policy_terms_for_credit["probs"].detach(),
                        )
                        joint_old_log_probs_summary = joint_old_policy_terms_for_credit["log_probs"]
                        joint_new_log_probs_summary = joint_new_policy_terms_for_credit["log_probs"]
                        joint_logprob_delta = (
                            joint_new_log_probs_summary - joint_old_log_probs_summary
                        )
                        joint_ratio_summary = torch.exp(joint_logprob_delta)
                        joint_old_logprob_mean = float(joint_old_log_probs_summary.mean().item())
                        joint_new_logprob_mean = float(joint_new_log_probs_summary.mean().item())
                        joint_ratio_mean = float(joint_ratio_summary.mean().item())
                        joint_approx_kl = float((-joint_logprob_delta).mean().item())
                        joint_clip_fraction = float(
                            (
                                (joint_ratio_summary < (1.0 - self.config.clip_epsilon))
                                | (joint_ratio_summary > (1.0 + self.config.clip_epsilon))
                            )
                            .float()
                            .mean()
                            .item()
                        )
                        joint_selected_score_mean = float(
                            joint_reward_aligned_credit_terms["selected_score_mean"]
                        )
                        joint_expected_score_mean = float(
                            joint_reward_aligned_credit_terms["expected_score_mean"]
                        )
                        joint_residual_credit_mean = float(
                            joint_reward_aligned_credit_terms["residual_mean"]
                        )
                        joint_residual_credit_std = float(
                            joint_reward_aligned_credit_terms["residual_std"]
                        )
                        joint_residual_credit_min = float(
                            joint_reward_aligned_credit_terms["residual_min"]
                        )
                        joint_residual_credit_max = float(
                            joint_reward_aligned_credit_terms["residual_max"]
                        )
                        joint_action_decision_agreement_ratio_under_reward_aligned = float(
                            joint_reward_aligned_credit_terms[
                                "joint_action_decision_agreement_ratio_under_reward_aligned"
                            ]
                        )
                        actual_local_rate_when_reward_aligned_best_is_local = float(
                            joint_reward_aligned_credit_terms[
                                "actual_local_rate_when_reward_aligned_best_is_local"
                            ]
                        )
                        actual_bs1_rate_when_reward_aligned_best_is_bs1 = float(
                            joint_reward_aligned_credit_terms[
                                "actual_bs1_rate_when_reward_aligned_best_is_bs1"
                            ]
                        )
                        actual_bs2_rate_when_reward_aligned_best_is_bs2 = float(
                            joint_reward_aligned_credit_terms[
                                "actual_bs2_rate_when_reward_aligned_best_is_bs2"
                            ]
                        )
                    if joint_td_aligned_scores is not None:
                        joint_td_aligned_credit_terms = self._joint_reward_aligned_credit_terms(
                            joint_td_aligned_scores,
                            joint_old_policy_terms_for_credit["selected_indices"],
                            joint_old_policy_terms_for_credit["probs"].detach(),
                        )
                        joint_action_decision_agreement_ratio_under_td_aligned = float(
                            joint_td_aligned_credit_terms[
                                "joint_action_decision_agreement_ratio_under_reward_aligned"
                            ]
                        )
                        actual_local_rate_when_td_aligned_best_is_local = float(
                            joint_td_aligned_credit_terms[
                                "actual_local_rate_when_reward_aligned_best_is_local"
                            ]
                        )
                        actual_bs1_rate_when_td_aligned_best_is_bs1 = float(
                            joint_td_aligned_credit_terms[
                                "actual_bs1_rate_when_reward_aligned_best_is_bs1"
                            ]
                        )
                        actual_bs2_rate_when_td_aligned_best_is_bs2 = float(
                            joint_td_aligned_credit_terms[
                                "actual_bs2_rate_when_reward_aligned_best_is_bs2"
                            ]
                        )
                        if self._uses_joint_td_aligned_credit():
                            joint_old_log_probs_summary = joint_old_policy_terms_for_credit["log_probs"]
                            joint_new_log_probs_summary = joint_new_policy_terms_for_credit["log_probs"]
                            joint_logprob_delta = (
                                joint_new_log_probs_summary - joint_old_log_probs_summary
                            )
                            joint_ratio_summary = torch.exp(joint_logprob_delta)
                            joint_old_logprob_mean = float(joint_old_log_probs_summary.mean().item())
                            joint_new_logprob_mean = float(joint_new_log_probs_summary.mean().item())
                            joint_ratio_mean = float(joint_ratio_summary.mean().item())
                            joint_approx_kl = float((-joint_logprob_delta).mean().item())
                            joint_clip_fraction = float(
                                (
                                    (joint_ratio_summary < (1.0 - self.config.clip_epsilon))
                                    | (joint_ratio_summary > (1.0 + self.config.clip_epsilon))
                                )
                                .float()
                                .mean()
                                .item()
                            )
                            joint_selected_score_mean = float(
                                joint_td_aligned_credit_terms["selected_score_mean"]
                            )
                            joint_expected_score_mean = float(
                                joint_td_aligned_credit_terms["expected_score_mean"]
                            )
                            joint_residual_credit_mean = float(
                                joint_td_aligned_credit_terms["residual_mean"]
                            )
                            joint_residual_credit_std = float(
                                joint_td_aligned_credit_terms["residual_std"]
                            )
                            joint_residual_credit_min = float(
                                joint_td_aligned_credit_terms["residual_min"]
                            )
                            joint_residual_credit_max = float(
                                joint_td_aligned_credit_terms["residual_max"]
                            )
                if self._uses_route_credit_theta_soft_weight():
                    if branchwise_normalization is None:
                        raise ValueError(
                            "route credit theta soft weight requires branchwise normalized theta advantages"
                        )
                    route_credit_weight_terms = self._route_credit_theta_soft_weight_terms(
                        branchwise_normalization["theta_advantages_norm"],
                        route_masks,
                    )
                    route_credit_weight = route_credit_weight_terms["weight"]
                    route_credit_effective_fraction = float(
                        route_credit_weight_terms["effective_fraction"]
                    )
                    route_credit_effective_count = float(
                        route_credit_weight_terms["effective_count"]
                    )
                    route_credit_weight_mean = float(
                        route_credit_weight_terms["weight_mean"]
                    )
                    route_credit_weight_std = float(
                        route_credit_weight_terms["weight_std"]
                    )
                    route_credit_weight_min = float(
                        route_credit_weight_terms["weight_min"]
                    )
                    route_credit_weight_max = float(
                        route_credit_weight_terms["weight_max"]
                    )
                    route_advantages_norm = route_credit_weight * route_advantages_norm
                elif self._uses_route_residual_credit_vectorized():
                    route_old_policy_terms_for_credit = self._true_conditional_route_policy_terms(
                        actions,
                        old_action_means,
                    )
                    route_residual_credit_terms = self._route_residual_credit_vectorized_terms(
                        route_advantages_norm,
                        route_old_policy_terms_for_credit["selected_indices"],
                        route_old_policy_terms_for_credit["probs"].detach(),
                        route_masks,
                    )
                    route_score_vector = route_residual_credit_terms["route_score_vector"]
                    route_selected_score = route_residual_credit_terms["selected_score"]
                    route_expected_score = route_residual_credit_terms["expected_score"]
                    route_residual_credit = route_residual_credit_terms["residual_credit"]
                    route_residual_credit_mean = float(route_residual_credit_terms["mean"])
                    route_residual_credit_std = float(route_residual_credit_terms["std"])
                    route_residual_credit_min = float(route_residual_credit_terms["min"])
                    route_residual_credit_max = float(route_residual_credit_terms["max"])
                    route_expected_score_mean = float(
                        route_residual_credit_terms["expected_score_mean"]
                    )
                    route_selected_score_mean = float(
                        route_residual_credit_terms["selected_score_mean"]
                    )
                    route_score_vector_mean_abs = float(
                        route_residual_credit_terms["score_vector_mean_abs"]
                    )
                    route_score_vector_std = float(
                        route_residual_credit_terms["score_vector_std"]
                    )
                    route_advantages_norm = route_residual_credit
                elif self._uses_true_conditional_route_policy():
                    route_old_policy_terms_for_credit = self._true_conditional_route_policy_terms(
                        actions,
                        old_action_means,
                    )
                    route_candidate_score_credit_terms = (
                        self._route_candidate_score_credit_terms(
                            states,
                            route_old_policy_terms_for_credit["selected_indices"],
                            route_old_policy_terms_for_credit["probs"].detach(),
                            route_masks,
                        )
                    )
                    route_score_vector = route_candidate_score_credit_terms[
                        "candidate_score_vector"
                    ]
                    route_selected_score = route_candidate_score_credit_terms["selected_score"]
                    route_expected_score = route_candidate_score_credit_terms["expected_score"]
                    route_residual_credit = route_candidate_score_credit_terms["residual_credit"]
                    route_candidate_score_mean = float(
                        route_candidate_score_credit_terms["candidate_score_mean"]
                    )
                    route_candidate_score_std = float(
                        route_candidate_score_credit_terms["candidate_score_std"]
                    )
                    route_residual_credit_mean = float(
                        route_candidate_score_credit_terms["residual_mean"]
                    )
                    route_residual_credit_std = float(
                        route_candidate_score_credit_terms["residual_std"]
                    )
                    route_residual_credit_min = float(
                        route_candidate_score_credit_terms["residual_min"]
                    )
                    route_residual_credit_max = float(
                        route_candidate_score_credit_terms["residual_max"]
                    )
                    route_expected_score_mean = float(
                        route_candidate_score_credit_terms["expected_score_mean"]
                    )
                    route_selected_score_mean = float(
                        route_candidate_score_credit_terms["selected_score_mean"]
                    )
                    route_decision_agreement_ratio = float(
                        route_candidate_score_credit_terms["route_decision_agreement_ratio"]
                    )
                    actual_bs1_rate_when_route_true_gap_positive = float(
                        route_candidate_score_credit_terms[
                            "actual_bs1_rate_when_route_true_gap_positive"
                        ]
                    )
                    actual_bs1_rate_when_route_true_gap_negative = float(
                        route_candidate_score_credit_terms[
                            "actual_bs1_rate_when_route_true_gap_negative"
                        ]
                    )
                    route_score_vector_mean_abs = float(route_score_vector.abs().mean().item())
                    route_score_vector_std = float(route_score_vector.std(unbiased=False).item())
                if self._uses_true_conditional_route_candidate_score_credit():
                    route_advantages_norm = route_residual_credit
                if branchwise_normalization is not None:
                    theta_adv_norm_mean = branchwise_normalization["theta_adv_norm_mean"]
                    theta_adv_norm_std = branchwise_normalization["theta_adv_norm_std"]
                    route_adv_norm_mean = branchwise_normalization["route_adv_norm_mean"]
                    route_adv_norm_std = branchwise_normalization["route_adv_norm_std"]
                    route_adv_norm_active_count = branchwise_normalization["route_active_count"]
                    route_adv_norm_used_fallback = float(
                        branchwise_normalization["route_used_fallback"]
                    )
                active_route_mask = route_masks > 0.5
                route_metric_mask = route_loss_masks > 0.5
                offload_active_fraction = float(active_route_mask.float().mean().item())
                route_logprob_active_fraction = float(active_route_mask.float().mean().item())
                theta_positive_fraction = float((theta_advantages > 0.0).float().mean().item())
                route_credit_gate_fraction = float(route_credit_gate.mean().item())
                route_ratio_mean = self._masked_mean(route_ratio, route_metric_mask)
                route_ratio_std = float(self._masked_tensor_std(route_ratio, route_metric_mask).item())
                if bool(route_metric_mask.any().item()):
                    route_ratio_max = float(route_ratio[route_metric_mask].max().item())
                route_approx_kl = self._masked_tensor_mean(-route_logprob_delta, route_metric_mask)
                if theta_old_log_probs_summary is not None and theta_new_log_probs_summary is not None:
                    theta_old_logprob_mean = float(theta_old_log_probs_summary.mean().item())
                    theta_new_logprob_mean = float(theta_new_log_probs_summary.mean().item())
                if route_old_log_probs_summary is not None and route_new_log_probs_summary is not None:
                    route_old_logprob_mean = self._masked_mean(
                        route_old_log_probs_summary,
                        route_metric_mask,
                    )
                    route_new_logprob_mean = self._masked_mean(
                        route_new_log_probs_summary,
                        route_metric_mask,
                    )
                if (
                    theta_selected_probs_summary is not None
                    and route_selected_probs_summary is not None
                ):
                    theta_selected_prob_mean = float(theta_selected_probs_summary.mean().item())
                    route_selected_prob_mean = self._masked_mean(
                        route_selected_probs_summary,
                        active_route_mask,
                    )
                    if bool(active_route_mask.any().item()):
                        conditional_policy_consistency_score = float(
                            0.5 * (theta_selected_prob_mean + route_selected_prob_mean)
                        )
                    else:
                        conditional_policy_consistency_score = float(theta_selected_prob_mean)
                theta_surrogate = torch.min(
                    theta_ratio * theta_advantages_norm,
                    torch.clamp(
                        theta_ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * theta_advantages_norm,
                )
                route_surrogate = torch.min(
                    route_ratio * route_advantages_norm,
                    torch.clamp(
                        route_ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * route_advantages_norm,
                )
                route_surrogate_for_loss = route_surrogate
                if self._uses_hard_offload_route_surrogate():
                    active_route_count = route_masks.sum(dim=-1, keepdim=True)
                    route_surrogate = (
                        route_masks
                        * route_surrogate
                        * (float(route_surrogate.shape[-1]) / (active_route_count + 1e-8))
                    )
                    route_surrogate_for_loss = route_masks * route_surrogate_for_loss
                if self._uses_offload_gated_route_surrogate():
                    route_surrogate = route_gates * route_surrogate
                    route_surrogate_for_loss = route_gates * route_surrogate_for_loss
                block_surrogate = 0.5 * (theta_surrogate + route_surrogate)
                theta_loss_mean = -theta_surrogate.mean()
                route_loss_mean = -self._mean_route_surrogate_over_mask(
                    route_surrogate_for_loss,
                    route_loss_masks,
                )
                theta_route_loss_ratio = route_loss_mean.abs() / (theta_loss_mean.abs() + 1e-8)
                block_clip_mask = theta_clip_mask | route_clip_mask
            else:
                block_surrogate = torch.min(
                    block_ratio * block_advantages,
                    block_clipped_ratio * block_advantages,
                )
            block_weights = self._block_activity_weights(states)
            block_adv_scale_distribution = block_adv_scales / block_adv_scales.sum(
                dim=-1,
                keepdim=True,
            )
            block_advantage_distribution = (block_advantage_weights + 1e-6) / (
                block_advantage_weights.sum(dim=-1, keepdim=True)
                + 1e-6 * float(block_advantage_weights.shape[-1])
            )
            block_value_scale_distribution = block_value_scales / block_value_scales.sum(
                dim=-1,
                keepdim=True,
            )
            block_td_advantage_weights = block_td_advantages.abs() + 1e-6
            block_td_advantage_distribution = block_td_advantage_weights / (
                block_td_advantage_weights.sum(dim=-1, keepdim=True)
            )
            block_selected_action_prob_gain = block_ratio - 1.0
            sample_block_ratio = block_ratio.mean(dim=-1)
            sample_block_selected_action_prob_gain = block_selected_action_prob_gain.mean(dim=-1)
            weighted_block_ratio = (block_weights * block_ratio).sum(dim=-1)
            weighted_block_selected_action_prob_gain = (
                block_weights * block_selected_action_prob_gain
            ).sum(dim=-1)
            weighted_block_clip_fraction = (
                block_weights * block_clip_mask.float()
            ).sum(dim=-1)
            weighted_block_delta = (block_weights * block_logprob_delta).sum(dim=-1)
            route_active_mask = route_masks > 0.5
            route_metric_mask = (
                route_loss_masks > 0.5
                if self._uses_theta_route_split_advantage_surrogate()
                else route_active_mask
            )
            sample_theta_delta = theta_logprob_delta.mean(dim=-1)
            sample_route_delta = self._sample_masked_mean(route_logprob_delta, route_metric_mask)
            sample_theta_ratio = theta_ratio.mean(dim=-1)
            sample_route_ratio = self._sample_masked_mean(route_ratio, route_metric_mask)
            sample_theta_prob_gain = theta_selected_action_prob_gain.mean(dim=-1)
            sample_route_prob_gain = self._sample_masked_mean(
                route_selected_action_prob_gain,
                route_metric_mask,
            )
            sample_theta_clip_fraction = theta_clip_mask.float().mean(dim=-1)
            sample_route_clip_fraction = self._sample_masked_mean(
                route_clip_mask.float(),
                route_metric_mask,
            )
            if self._uses_true_conditional_route_policy():
                route_active_mask_float = route_active_mask.to(dtype=theta_logprob_delta.dtype)
                full_new_log_probs = theta_new_log_probs_summary.mean(dim=-1) + (
                    route_new_log_probs_summary * route_active_mask_float
                ).mean(dim=-1)
                full_old_log_probs_summary = theta_old_log_probs_summary.mean(dim=-1) + (
                    route_old_log_probs_summary * route_active_mask_float
                ).mean(dim=-1)
                delta_log_prob = full_new_log_probs - full_old_log_probs_summary
            else:
                full_new_log_probs = self._aggregate_log_prob_components_with_states(
                    full_new_log_prob_components,
                    states,
                )
                delta_log_prob = self._aggregate_log_prob_components_with_states(
                    logprob_delta_components,
                    states,
                )
            full_ratio = torch.exp(delta_log_prob)
            full_clip_mask = (full_ratio < (1.0 - self.config.clip_epsilon)) | (
                full_ratio > (1.0 + self.config.clip_epsilon)
            )
            selected_action_prob_gain = full_ratio - 1.0
            full_policy_entropy = full_distribution.entropy().sum(dim=-1).mean()
            unclipped_objective = full_ratio * advantages
            clipped_objective = (
                torch.clamp(
                    full_ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )
                * advantages
            )
            full_policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
            full_entropy_loss = -self.config.entropy_coeff * full_policy_entropy
            approx_kl = (old_log_probs - full_new_log_probs).mean()
            ratio_tensor_for_summary = full_ratio
            clip_tensor_for_summary = full_clip_mask.float()
            ratio_by_adv_sign_tensor = full_ratio
            ratio_by_adv_sign_advantages = raw_advantages
            ratio_values_for_buckets = full_ratio
            selected_action_prob_gain_values = selected_action_prob_gain
            clip_fraction_values_for_buckets = full_clip_mask.float()
            if self._uses_blockwise_policy_surrogate():
                if self._uses_blockwise_weighted_surrogate_mean():
                    full_policy_loss = -(block_weights * block_surrogate).sum(dim=-1).mean()
                    approx_kl = (-weighted_block_delta).mean()
                    ratio_values_for_buckets = weighted_block_ratio
                    selected_action_prob_gain_values = weighted_block_selected_action_prob_gain
                    clip_fraction_values_for_buckets = weighted_block_clip_fraction
                else:
                    if self._uses_branchwise_balanced_pg():
                        full_policy_loss = 0.5 * theta_loss_mean + 0.5 * route_loss_mean
                    else:
                        full_policy_loss = -block_surrogate.mean()
                    if self._uses_factorized_ratio_pg():
                        if self._uses_joint_counterfactual_credit():
                            delta_log_prob = (
                                joint_new_log_probs_summary - joint_old_log_probs_summary
                            ).mean(dim=-1)
                            ratio_values_for_buckets = torch.exp(delta_log_prob)
                            selected_action_prob_gain_values = ratio_values_for_buckets - 1.0
                            clip_fraction_values_for_buckets = (
                                (ratio_values_for_buckets < (1.0 - self.config.clip_epsilon))
                                | (ratio_values_for_buckets > (1.0 + self.config.clip_epsilon))
                            ).float()
                            approx_kl = delta_log_prob.new_tensor(joint_approx_kl)
                            ratio_tensor_for_summary = ratio_values_for_buckets
                            clip_tensor_for_summary = clip_fraction_values_for_buckets
                            ratio_by_adv_sign_tensor = ratio_values_for_buckets
                            ratio_by_adv_sign_advantages = raw_advantages
                        elif self._uses_true_conditional_route_policy():
                            sample_route_delta_over_all_blocks = (
                                route_logprob_delta * route_active_mask.to(route_logprob_delta.dtype)
                            ).mean(dim=-1)
                            delta_log_prob = sample_theta_delta + sample_route_delta_over_all_blocks
                            ratio_values_for_buckets = torch.exp(delta_log_prob)
                            selected_action_prob_gain_values = ratio_values_for_buckets - 1.0
                            clip_fraction_values_for_buckets = (
                                (ratio_values_for_buckets < (1.0 - self.config.clip_epsilon))
                                | (ratio_values_for_buckets > (1.0 + self.config.clip_epsilon))
                            ).float()
                            approx_kl = (-delta_log_prob).mean()
                            ratio_tensor_for_summary = ratio_values_for_buckets
                            clip_tensor_for_summary = clip_fraction_values_for_buckets
                            ratio_by_adv_sign_tensor = ratio_values_for_buckets
                            ratio_by_adv_sign_advantages = raw_advantages
                        else:
                            approx_kl = 0.5 * theta_approx_kl + 0.5 * route_approx_kl
                            delta_log_prob = 0.5 * sample_theta_delta + 0.5 * sample_route_delta
                            ratio_values_for_buckets = (
                                0.5 * sample_theta_ratio + 0.5 * sample_route_ratio
                            )
                            selected_action_prob_gain_values = (
                                0.5 * sample_theta_prob_gain + 0.5 * sample_route_prob_gain
                            )
                            clip_fraction_values_for_buckets = (
                                0.5 * sample_theta_clip_fraction
                                + 0.5 * sample_route_clip_fraction
                            )
                            ratio_tensor_for_summary = ratio_values_for_buckets
                            clip_tensor_for_summary = clip_fraction_values_for_buckets
                            ratio_by_adv_sign_tensor = ratio_values_for_buckets
                            ratio_by_adv_sign_advantages = raw_advantages
                    else:
                        approx_kl = (-block_mean_delta).mean()
                        ratio_values_for_buckets = sample_block_ratio
                        selected_action_prob_gain_values = sample_block_selected_action_prob_gain
                        clip_fraction_values_for_buckets = block_clip_mask.float().mean(dim=-1)
                        ratio_tensor_for_summary = block_ratio.reshape(-1)
                        clip_tensor_for_summary = block_clip_mask.float()
                        ratio_by_adv_sign_tensor = block_ratio.reshape(-1)
                        ratio_by_adv_sign_advantages = raw_advantages.unsqueeze(-1).expand_as(
                            block_ratio
                        ).reshape(-1)
            advantage_bucket_masks, advantage_bucket_thresholds = self._advantage_bucket_masks(
                raw_advantages
            )
            advantage_bucket_rows = []
            for bucket_name, bucket_mask in advantage_bucket_masks.items():
                advantage_bucket_rows.append(
                    {
                        "bucket_name": bucket_name,
                        "bucket_sample_count": int(bucket_mask.sum().item()),
                        "bucket_mean_advantage": self._masked_mean(raw_advantages, bucket_mask),
                        "bucket_mean_ratio": self._masked_mean(ratio_values_for_buckets, bucket_mask),
                        "bucket_mean_delta_log_prob_selected_action": self._masked_mean(
                            delta_log_prob,
                            bucket_mask,
                        ),
                        "bucket_mean_selected_action_prob_gain": self._masked_mean(
                            selected_action_prob_gain_values,
                            bucket_mask,
                        ),
                        "bucket_mean_clip_fraction": self._masked_mean(
                            clip_fraction_values_for_buckets,
                            bucket_mask,
                        ),
                    }
                )
            action_block_layout = self.describe_action_block_slices()
            block_logprob_rows = []
            block_surrogate_rows = []
            block_adv_scale_rows = []
            block_value_scale_rows = []
            block_td_value_rows = []
            block_delta_cost_rows = []
            block_advantage_rows = []
            block_weight_rows = []
            theta_route_split_rows = []
            positive_adv_mask = raw_advantages > 0.0
            negative_adv_mask = raw_advantages < 0.0
            positive_block_adv_mask = block_advantages > 0.0
            negative_block_adv_mask = block_advantages < 0.0
            uniform_block_weight = 1.0 / max(len(action_block_layout), 1)
            for block_index, block_meta in enumerate(action_block_layout):
                block_delta = block_logprob_delta[:, block_index]
                block_ratio_column = block_ratio[:, block_index]
                block_clip_column = block_clip_mask[:, block_index].float()
                block_surrogate_column = block_surrogate[:, block_index]
                block_prob_gain_column = block_selected_action_prob_gain[:, block_index]
                block_weight_column = block_weights[:, block_index]
                block_adv_scale_column = block_adv_scales[:, block_index]
                block_value_score_column = block_value_scores[:, block_index]
                block_value_scale_column = block_value_scales[:, block_index]
                block_value_pred_column = block_value_preds[:, block_index]
                block_path_value_local_column = block_path_value_preds[:, block_index, 0]
                block_path_value_bs1_column = (
                    block_path_value_preds[:, block_index, 1]
                    if block_path_value_preds.shape[-1] > 1
                    else torch.zeros_like(block_value_pred_column)
                )
                block_path_value_bs2_column = (
                    block_path_value_preds[:, block_index, 2]
                    if block_path_value_preds.shape[-1] > 2
                    else torch.zeros_like(block_value_pred_column)
                )
                path_td_target_local_column = block_path_td_targets[:, block_index, 0]
                path_td_target_bs1_column = (
                    block_path_td_targets[:, block_index, 1]
                    if block_path_td_targets.shape[-1] > 1
                    else torch.zeros_like(block_value_pred_column)
                )
                path_td_target_bs2_column = (
                    block_path_td_targets[:, block_index, 2]
                    if block_path_td_targets.shape[-1] > 2
                    else torch.zeros_like(block_value_pred_column)
                )
                block_action_conditioned_value_now_column = (
                    block_action_conditioned_value_preds[:, block_index]
                )
                block_action_conditioned_value_next_column = (
                    block_action_conditioned_value_next_preds[:, block_index]
                )
                block_local_cost_now_column = block_local_cost_now[:, block_index]
                block_local_cost_next_column = block_local_cost_next[:, block_index]
                block_path_cost_local_column = block_path_cost_local_now[:, block_index]
                block_path_cost_bs1_column = (
                    block_path_cost_bs_now[:, block_index, 0]
                    if block_path_cost_bs_now.shape[-1] > 0
                    else torch.zeros_like(block_local_cost_now_column)
                )
                block_path_cost_bs2_column = (
                    block_path_cost_bs_now[:, block_index, 1]
                    if block_path_cost_bs_now.shape[-1] > 1
                    else torch.zeros_like(block_local_cost_now_column)
                )
                block_action_conditioned_cost_now_column = (
                    block_action_conditioned_cost_now[:, block_index]
                )
                block_action_conditioned_cost_next_column = (
                    block_action_conditioned_cost_next[:, block_index]
                )
                block_delta_cost_column = block_delta_cost[:, block_index]
                block_td_reward_column = block_local_rewards[:, block_index]
                block_td_target_column = block_td_targets[:, block_index]
                block_td_advantage_column = block_td_advantages[:, block_index]
                block_advantage_column = block_advantages[:, block_index]
                theta_advantage_column = theta_advantages[:, block_index]
                route_advantage_column = route_advantages[:, block_index]
                theta_advantage_norm_column = theta_advantages_norm[:, block_index]
                route_advantage_norm_column = route_advantages_norm[:, block_index]
                route_gate_column = route_gates[:, block_index]
                route_mask_column = route_masks[:, block_index]
                route_loss_mask_column = route_loss_masks[:, block_index]
                route_credit_gate_column = route_credit_gate[:, block_index]
                theta_ratio_column = theta_ratio[:, block_index]
                route_ratio_column = route_ratio[:, block_index]
                theta_prob_gain_column = theta_selected_action_prob_gain[:, block_index]
                route_prob_gain_column = route_selected_action_prob_gain[:, block_index]
                theta_clip_column = theta_clip_mask[:, block_index].float()
                route_clip_column = route_clip_mask[:, block_index].float()
                offload_vs_local_value_gap_column = offload_vs_local_value_gap[:, block_index]
                bs1_vs_bs2_value_gap_column = bs1_vs_bs2_value_gap[:, block_index]
                block_logprob_rows.append(
                    {
                        "block_index": int(block_meta["block_index"]),
                        "block_start": int(block_meta["start"]),
                        "block_stop_exclusive": int(block_meta["stop_exclusive"]),
                        "block_size": int(block_meta["size"]),
                        "block_semantic": str(block_meta["semantic"]),
                        "block_weight_mean": float(block_weight_column.mean().item()),
                        "block_weight_std": float(block_weight_column.std(unbiased=False).item()),
                        "block_weight_max": float(block_weight_column.max().item()),
                        "block_logprob_delta_mean": float(block_delta.mean().item()),
                        "block_logprob_delta_std": float(
                            block_delta.std(unbiased=False).item()
                        ),
                        "block_ratio_proxy_mean": float(block_ratio_column.mean().item()),
                        "block_ratio_proxy_std": float(
                            block_ratio_column.std(unbiased=False).item()
                        ),
                    }
                )
                theta_route_split_rows.append(
                    {
                        "block_index": int(block_meta["block_index"]),
                        "block_start": int(block_meta["start"]),
                        "block_stop_exclusive": int(block_meta["stop_exclusive"]),
                        "block_size": int(block_meta["size"]),
                        "block_semantic": str(block_meta["semantic"]),
                        "policy_surrogate_mode": self._policy_ratio_mode(),
                        "theta_advantage_mean": float(theta_advantage_column.mean().item()),
                        "theta_advantage_std": float(
                            theta_advantage_column.std(unbiased=False).item()
                        ),
                        "theta_advantage_norm_mean": float(
                            theta_advantage_norm_column.mean().item()
                        ),
                        "theta_advantage_norm_std": float(
                            theta_advantage_norm_column.std(unbiased=False).item()
                        ),
                        "theta_advantage_max": float(theta_advantage_column.max().item()),
                        "theta_advantage_min": float(theta_advantage_column.min().item()),
                        "route_advantage_mean": float(route_advantage_column.mean().item()),
                        "route_advantage_std": float(
                            route_advantage_column.std(unbiased=False).item()
                        ),
                        "route_advantage_norm_mean": self._masked_mean(
                            route_advantage_norm_column,
                            route_mask_column > 0.5,
                        ),
                        "route_advantage_norm_std": float(
                            self._masked_tensor_std(
                                route_advantage_norm_column,
                                route_mask_column > 0.5,
                            ).item()
                        ),
                        "route_advantage_max": float(route_advantage_column.max().item()),
                        "route_advantage_min": float(route_advantage_column.min().item()),
                        "route_gate_mean": float(route_gate_column.mean().item()),
                        "route_gate_std": float(route_gate_column.std(unbiased=False).item()),
                        "route_gate_min": float(route_gate_column.min().item()),
                        "route_gate_max": float(route_gate_column.max().item()),
                        "route_gate_active_fraction": float(
                            (route_gate_column > 0.5).float().mean().item()
                        ),
                        "route_credit_gate_fraction": float(
                            route_credit_gate_column.mean().item()
                        ),
                        "route_credit_effective_fraction": float(
                            route_loss_mask_column.mean().item()
                        ),
                        "route_mask_mean": float(route_mask_column.mean().item()),
                        "route_mask_active_fraction": float(route_mask_column.mean().item()),
                        "theta_selected_action_prob_gain_mean": float(
                            theta_prob_gain_column.mean().item()
                        ),
                        "theta_selected_action_prob_gain_std": float(
                            theta_prob_gain_column.std(unbiased=False).item()
                        ),
                        "theta_ratio_mean": float(theta_ratio_column.mean().item()),
                        "theta_ratio_std": float(theta_ratio_column.std(unbiased=False).item()),
                        "theta_ratio_max": float(theta_ratio_column.max().item()),
                        "route_selected_action_prob_gain_mean": float(
                            route_prob_gain_column.mean().item()
                        ),
                        "route_selected_action_prob_gain_std": float(
                            route_prob_gain_column.std(unbiased=False).item()
                        ),
                        "route_ratio_mean": self._masked_mean(
                            route_ratio_column,
                            route_loss_mask_column > 0.5,
                        ),
                        "route_ratio_std": float(
                            self._masked_tensor_std(
                                route_ratio_column,
                                route_loss_mask_column > 0.5,
                            ).item()
                        ),
                        "route_ratio_max": float(
                            route_ratio_column[route_loss_mask_column > 0.5].max().item()
                        )
                        if bool((route_loss_mask_column > 0.5).any().item())
                        else 0.0,
                        "theta_clip_fraction": float(theta_clip_column.mean().item()),
                        "route_clip_fraction": self._masked_mean(
                            route_clip_column,
                            route_loss_mask_column > 0.5,
                        ),
                        "offload_vs_local_value_gap_mean": float(
                            offload_vs_local_value_gap_column.mean().item()
                        ),
                        "offload_vs_local_value_gap_std": float(
                            offload_vs_local_value_gap_column.std(unbiased=False).item()
                        ),
                        "bs1_vs_bs2_value_gap_mean": float(
                            bs1_vs_bs2_value_gap_column.mean().item()
                        ),
                        "bs1_vs_bs2_value_gap_std": float(
                            bs1_vs_bs2_value_gap_column.std(unbiased=False).item()
                        ),
                    }
                )
                block_advantage_rows.append(
                    {
                        "block_index": int(block_meta["block_index"]),
                        "block_start": int(block_meta["start"]),
                        "block_stop_exclusive": int(block_meta["stop_exclusive"]),
                        "block_size": int(block_meta["size"]),
                        "block_semantic": str(block_meta["semantic"]),
                        "policy_surrogate_mode": self._policy_ratio_mode(),
                        "block_advantage_mean": float(block_advantage_column.mean().item()),
                        "block_advantage_std": float(
                            block_advantage_column.std(unbiased=False).item()
                        ),
                        "block_advantage_max": float(block_advantage_column.max().item()),
                        "block_advantage_min": float(block_advantage_column.min().item()),
                        "block_clip_fraction": float(block_clip_column.mean().item()),
                        "per_block_selected_action_prob_gain_mean": float(
                            block_prob_gain_column.mean().item()
                        ),
                        "per_block_selected_action_prob_gain_std": float(
                            block_prob_gain_column.std(unbiased=False).item()
                        ),
                    }
                )
                block_adv_scale_rows.append(
                    {
                        "block_index": int(block_meta["block_index"]),
                        "block_start": int(block_meta["start"]),
                        "block_stop_exclusive": int(block_meta["stop_exclusive"]),
                        "block_size": int(block_meta["size"]),
                        "block_semantic": str(block_meta["semantic"]),
                        "policy_surrogate_mode": self._policy_ratio_mode(),
                        "block_adv_scale_mean": float(block_adv_scale_column.mean().item()),
                        "block_adv_scale_std": float(
                            block_adv_scale_column.std(unbiased=False).item()
                        ),
                        "block_adv_scale_max": float(block_adv_scale_column.max().item()),
                        "block_adv_scale_min": float(block_adv_scale_column.min().item()),
                        "block_clip_fraction": float(block_clip_column.mean().item()),
                        "per_block_selected_action_prob_gain_mean": float(
                            block_prob_gain_column.mean().item()
                        ),
                        "per_block_selected_action_prob_gain_std": float(
                            block_prob_gain_column.std(unbiased=False).item()
                        ),
                    }
                )
                block_value_scale_rows.append(
                    {
                        "block_index": int(block_meta["block_index"]),
                        "block_start": int(block_meta["start"]),
                        "block_stop_exclusive": int(block_meta["stop_exclusive"]),
                        "block_size": int(block_meta["size"]),
                        "block_semantic": str(block_meta["semantic"]),
                        "policy_surrogate_mode": self._policy_ratio_mode(),
                        "block_value_score_mean": float(block_value_score_column.mean().item()),
                        "block_value_score_std": float(
                            block_value_score_column.std(unbiased=False).item()
                        ),
                        "block_value_score_max": float(block_value_score_column.max().item()),
                        "block_value_score_min": float(block_value_score_column.min().item()),
                        "block_value_scale_mean": float(block_value_scale_column.mean().item()),
                        "block_value_scale_std": float(
                            block_value_scale_column.std(unbiased=False).item()
                        ),
                        "block_value_scale_max": float(block_value_scale_column.max().item()),
                        "block_value_scale_min": float(block_value_scale_column.min().item()),
                        "block_clip_fraction": float(block_clip_column.mean().item()),
                        "per_block_selected_action_prob_gain_mean": float(
                            block_prob_gain_column.mean().item()
                        ),
                        "per_block_selected_action_prob_gain_std": float(
                            block_prob_gain_column.std(unbiased=False).item()
                        ),
                    }
                )
                block_td_value_rows.append(
                    {
                        "block_index": int(block_meta["block_index"]),
                        "block_start": int(block_meta["start"]),
                        "block_stop_exclusive": int(block_meta["stop_exclusive"]),
                        "block_size": int(block_meta["size"]),
                        "block_semantic": str(block_meta["semantic"]),
                        "policy_surrogate_mode": self._policy_ratio_mode(),
                        "block_path_cost_local_mean": float(
                            block_path_cost_local_column.mean().item()
                        ),
                        "block_path_cost_local_std": float(
                            block_path_cost_local_column.std(unbiased=False).item()
                        ),
                        "block_path_cost_bs1_mean": float(block_path_cost_bs1_column.mean().item()),
                        "block_path_cost_bs1_std": float(
                            block_path_cost_bs1_column.std(unbiased=False).item()
                        ),
                        "block_path_cost_bs2_mean": float(block_path_cost_bs2_column.mean().item()),
                        "block_path_cost_bs2_std": float(
                            block_path_cost_bs2_column.std(unbiased=False).item()
                        ),
                        "block_local_cost_now_mean": float(
                            block_local_cost_now_column.mean().item()
                        ),
                        "block_local_cost_now_std": float(
                            block_local_cost_now_column.std(unbiased=False).item()
                        ),
                        "block_local_cost_now_max": float(block_local_cost_now_column.max().item()),
                        "block_local_cost_now_min": float(block_local_cost_now_column.min().item()),
                        "block_local_cost_next_mean": float(
                            block_local_cost_next_column.mean().item()
                        ),
                        "block_local_cost_next_std": float(
                            block_local_cost_next_column.std(unbiased=False).item()
                        ),
                        "block_local_cost_next_max": float(
                            block_local_cost_next_column.max().item()
                        ),
                        "block_local_cost_next_min": float(
                            block_local_cost_next_column.min().item()
                        ),
                        "block_action_conditioned_cost_now_mean": float(
                            block_action_conditioned_cost_now_column.mean().item()
                        ),
                        "block_action_conditioned_cost_now_std": float(
                            block_action_conditioned_cost_now_column.std(unbiased=False).item()
                        ),
                        "block_action_conditioned_cost_next_mean": float(
                            block_action_conditioned_cost_next_column.mean().item()
                        ),
                        "block_action_conditioned_cost_next_std": float(
                            block_action_conditioned_cost_next_column.std(unbiased=False).item()
                        ),
                        "block_delta_cost_mean": float(block_delta_cost_column.mean().item()),
                        "block_delta_cost_std": float(
                            block_delta_cost_column.std(unbiased=False).item()
                        ),
                        "block_delta_cost_max": float(block_delta_cost_column.max().item()),
                        "block_delta_cost_min": float(block_delta_cost_column.min().item()),
                        "block_td_reward_mean": float(block_td_reward_column.mean().item()),
                        "block_td_reward_std": float(
                            block_td_reward_column.std(unbiased=False).item()
                        ),
                        "block_td_reward_max": float(block_td_reward_column.max().item()),
                        "block_td_reward_min": float(block_td_reward_column.min().item()),
                        "block_value_pred_mean": float(block_value_pred_column.mean().item()),
                        "block_value_pred_std": float(
                            block_value_pred_column.std(unbiased=False).item()
                        ),
                        "block_value_pred_max": float(block_value_pred_column.max().item()),
                        "block_value_pred_min": float(block_value_pred_column.min().item()),
                        "block_path_value_local_mean": float(
                            block_path_value_local_column.mean().item()
                        ),
                        "block_path_value_local_std": float(
                            block_path_value_local_column.std(unbiased=False).item()
                        ),
                        "block_path_value_bs1_mean": float(
                            block_path_value_bs1_column.mean().item()
                        ),
                        "block_path_value_bs1_std": float(
                            block_path_value_bs1_column.std(unbiased=False).item()
                        ),
                        "block_path_value_bs2_mean": float(
                            block_path_value_bs2_column.mean().item()
                        ),
                        "block_path_value_bs2_std": float(
                            block_path_value_bs2_column.std(unbiased=False).item()
                        ),
                        "path_value_local_mean": float(block_path_value_local_column.mean().item()),
                        "path_value_local_std": float(
                            block_path_value_local_column.std(unbiased=False).item()
                        ),
                        "path_value_bs1_mean": float(block_path_value_bs1_column.mean().item()),
                        "path_value_bs1_std": float(
                            block_path_value_bs1_column.std(unbiased=False).item()
                        ),
                        "path_value_bs2_mean": float(block_path_value_bs2_column.mean().item()),
                        "path_value_bs2_std": float(
                            block_path_value_bs2_column.std(unbiased=False).item()
                        ),
                        "path_td_target_local_mean": float(
                            path_td_target_local_column.mean().item()
                        ),
                        "path_td_target_local_std": float(
                            path_td_target_local_column.std(unbiased=False).item()
                        ),
                        "path_td_target_bs1_mean": float(
                            path_td_target_bs1_column.mean().item()
                        ),
                        "path_td_target_bs1_std": float(
                            path_td_target_bs1_column.std(unbiased=False).item()
                        ),
                        "path_td_target_bs2_mean": float(
                            path_td_target_bs2_column.mean().item()
                        ),
                        "path_td_target_bs2_std": float(
                            path_td_target_bs2_column.std(unbiased=False).item()
                        ),
                        "block_action_conditioned_value_now_mean": float(
                            block_action_conditioned_value_now_column.mean().item()
                        ),
                        "block_action_conditioned_value_now_std": float(
                            block_action_conditioned_value_now_column.std(unbiased=False).item()
                        ),
                        "block_action_conditioned_value_next_mean": float(
                            block_action_conditioned_value_next_column.mean().item()
                        ),
                        "block_action_conditioned_value_next_std": float(
                            block_action_conditioned_value_next_column.std(unbiased=False).item()
                        ),
                        "block_td_target_mean": float(block_td_target_column.mean().item()),
                        "block_td_target_std": float(
                            block_td_target_column.std(unbiased=False).item()
                        ),
                        "block_td_target_max": float(block_td_target_column.max().item()),
                        "block_td_target_min": float(block_td_target_column.min().item()),
                        "block_td_advantage_mean": float(
                            block_td_advantage_column.mean().item()
                        ),
                        "block_td_advantage_std": float(
                            block_td_advantage_column.std(unbiased=False).item()
                        ),
                        "block_td_advantage_max": float(block_td_advantage_column.max().item()),
                        "block_td_advantage_min": float(block_td_advantage_column.min().item()),
                        "block_clip_fraction": float(block_clip_column.mean().item()),
                        "per_block_selected_action_prob_gain_mean": float(
                            block_prob_gain_column.mean().item()
                        ),
                        "per_block_selected_action_prob_gain_std": float(
                            block_prob_gain_column.std(unbiased=False).item()
                        ),
                    }
                )
                block_delta_cost_rows.append(
                    {
                        "block_index": int(block_meta["block_index"]),
                        "block_start": int(block_meta["start"]),
                        "block_stop_exclusive": int(block_meta["stop_exclusive"]),
                        "block_size": int(block_meta["size"]),
                        "block_semantic": str(block_meta["semantic"]),
                        "policy_surrogate_mode": self._policy_ratio_mode(),
                        "block_path_cost_local_mean": float(
                            block_path_cost_local_column.mean().item()
                        ),
                        "block_path_cost_local_std": float(
                            block_path_cost_local_column.std(unbiased=False).item()
                        ),
                        "block_path_cost_bs1_mean": float(block_path_cost_bs1_column.mean().item()),
                        "block_path_cost_bs1_std": float(
                            block_path_cost_bs1_column.std(unbiased=False).item()
                        ),
                        "block_path_cost_bs2_mean": float(block_path_cost_bs2_column.mean().item()),
                        "block_path_cost_bs2_std": float(
                            block_path_cost_bs2_column.std(unbiased=False).item()
                        ),
                        "block_local_cost_now_mean": float(
                            block_local_cost_now_column.mean().item()
                        ),
                        "block_local_cost_now_std": float(
                            block_local_cost_now_column.std(unbiased=False).item()
                        ),
                        "block_local_cost_now_max": float(block_local_cost_now_column.max().item()),
                        "block_local_cost_now_min": float(block_local_cost_now_column.min().item()),
                        "block_local_cost_next_mean": float(
                            block_local_cost_next_column.mean().item()
                        ),
                        "block_local_cost_next_std": float(
                            block_local_cost_next_column.std(unbiased=False).item()
                        ),
                        "block_local_cost_next_max": float(
                            block_local_cost_next_column.max().item()
                        ),
                        "block_local_cost_next_min": float(
                            block_local_cost_next_column.min().item()
                        ),
                        "block_action_conditioned_cost_now_mean": float(
                            block_action_conditioned_cost_now_column.mean().item()
                        ),
                        "block_action_conditioned_cost_now_std": float(
                            block_action_conditioned_cost_now_column.std(unbiased=False).item()
                        ),
                        "block_action_conditioned_cost_next_mean": float(
                            block_action_conditioned_cost_next_column.mean().item()
                        ),
                        "block_action_conditioned_cost_next_std": float(
                            block_action_conditioned_cost_next_column.std(unbiased=False).item()
                        ),
                        "block_delta_cost_mean": float(block_delta_cost_column.mean().item()),
                        "block_delta_cost_std": float(
                            block_delta_cost_column.std(unbiased=False).item()
                        ),
                        "block_delta_cost_max": float(block_delta_cost_column.max().item()),
                        "block_delta_cost_min": float(block_delta_cost_column.min().item()),
                        "block_td_reward_mean": float(block_td_reward_column.mean().item()),
                        "block_td_reward_std": float(
                            block_td_reward_column.std(unbiased=False).item()
                        ),
                        "block_td_reward_max": float(block_td_reward_column.max().item()),
                        "block_td_reward_min": float(block_td_reward_column.min().item()),
                        "block_path_value_local_mean": float(
                            block_path_value_local_column.mean().item()
                        ),
                        "block_path_value_local_std": float(
                            block_path_value_local_column.std(unbiased=False).item()
                        ),
                        "block_path_value_bs1_mean": float(
                            block_path_value_bs1_column.mean().item()
                        ),
                        "block_path_value_bs1_std": float(
                            block_path_value_bs1_column.std(unbiased=False).item()
                        ),
                        "block_path_value_bs2_mean": float(
                            block_path_value_bs2_column.mean().item()
                        ),
                        "block_path_value_bs2_std": float(
                            block_path_value_bs2_column.std(unbiased=False).item()
                        ),
                        "path_value_local_mean": float(block_path_value_local_column.mean().item()),
                        "path_value_local_std": float(
                            block_path_value_local_column.std(unbiased=False).item()
                        ),
                        "path_value_bs1_mean": float(block_path_value_bs1_column.mean().item()),
                        "path_value_bs1_std": float(
                            block_path_value_bs1_column.std(unbiased=False).item()
                        ),
                        "path_value_bs2_mean": float(block_path_value_bs2_column.mean().item()),
                        "path_value_bs2_std": float(
                            block_path_value_bs2_column.std(unbiased=False).item()
                        ),
                        "path_td_target_local_mean": float(
                            path_td_target_local_column.mean().item()
                        ),
                        "path_td_target_local_std": float(
                            path_td_target_local_column.std(unbiased=False).item()
                        ),
                        "path_td_target_bs1_mean": float(
                            path_td_target_bs1_column.mean().item()
                        ),
                        "path_td_target_bs1_std": float(
                            path_td_target_bs1_column.std(unbiased=False).item()
                        ),
                        "path_td_target_bs2_mean": float(
                            path_td_target_bs2_column.mean().item()
                        ),
                        "path_td_target_bs2_std": float(
                            path_td_target_bs2_column.std(unbiased=False).item()
                        ),
                        "block_action_conditioned_value_now_mean": float(
                            block_action_conditioned_value_now_column.mean().item()
                        ),
                        "block_action_conditioned_value_now_std": float(
                            block_action_conditioned_value_now_column.std(unbiased=False).item()
                        ),
                        "block_action_conditioned_value_next_mean": float(
                            block_action_conditioned_value_next_column.mean().item()
                        ),
                        "block_action_conditioned_value_next_std": float(
                            block_action_conditioned_value_next_column.std(unbiased=False).item()
                        ),
                        "block_td_target_mean": float(block_td_target_column.mean().item()),
                        "block_td_target_std": float(
                            block_td_target_column.std(unbiased=False).item()
                        ),
                        "block_td_target_max": float(block_td_target_column.max().item()),
                        "block_td_target_min": float(block_td_target_column.min().item()),
                        "block_td_advantage_mean": float(
                            block_td_advantage_column.mean().item()
                        ),
                        "block_td_advantage_std": float(
                            block_td_advantage_column.std(unbiased=False).item()
                        ),
                        "block_td_advantage_max": float(block_td_advantage_column.max().item()),
                        "block_td_advantage_min": float(block_td_advantage_column.min().item()),
                        "block_clip_fraction": float(block_clip_column.mean().item()),
                        "per_block_selected_action_prob_gain_mean": float(
                            block_prob_gain_column.mean().item()
                        ),
                        "per_block_selected_action_prob_gain_std": float(
                            block_prob_gain_column.std(unbiased=False).item()
                        ),
                    }
                )
                block_surrogate_rows.append(
                    {
                        "block_index": int(block_meta["block_index"]),
                        "block_start": int(block_meta["start"]),
                        "block_stop_exclusive": int(block_meta["stop_exclusive"]),
                        "block_size": int(block_meta["size"]),
                        "block_semantic": str(block_meta["semantic"]),
                        "block_weight_mean": float(block_weight_column.mean().item()),
                        "block_weight_std": float(block_weight_column.std(unbiased=False).item()),
                        "block_weight_max": float(block_weight_column.max().item()),
                        "block_weight_min": float(block_weight_column.min().item()),
                        "block_adv_scale_mean": float(block_adv_scale_column.mean().item()),
                        "block_adv_scale_std": float(
                            block_adv_scale_column.std(unbiased=False).item()
                        ),
                        "block_adv_scale_max": float(block_adv_scale_column.max().item()),
                        "block_adv_scale_min": float(block_adv_scale_column.min().item()),
                        "block_value_score_mean": float(block_value_score_column.mean().item()),
                        "block_value_score_std": float(
                            block_value_score_column.std(unbiased=False).item()
                        ),
                        "block_value_score_max": float(block_value_score_column.max().item()),
                        "block_value_score_min": float(block_value_score_column.min().item()),
                        "block_value_scale_mean": float(block_value_scale_column.mean().item()),
                        "block_value_scale_std": float(
                            block_value_scale_column.std(unbiased=False).item()
                        ),
                        "block_value_scale_max": float(block_value_scale_column.max().item()),
                        "block_value_scale_min": float(block_value_scale_column.min().item()),
                        "block_value_pred_mean": float(block_value_pred_column.mean().item()),
                        "block_value_pred_std": float(
                            block_value_pred_column.std(unbiased=False).item()
                        ),
                        "block_value_pred_max": float(block_value_pred_column.max().item()),
                        "block_value_pred_min": float(block_value_pred_column.min().item()),
                        "block_td_target_mean": float(block_td_target_column.mean().item()),
                        "block_td_target_std": float(
                            block_td_target_column.std(unbiased=False).item()
                        ),
                        "block_td_target_max": float(block_td_target_column.max().item()),
                        "block_td_target_min": float(block_td_target_column.min().item()),
                        "block_td_advantage_mean": float(
                            block_td_advantage_column.mean().item()
                        ),
                        "block_td_advantage_std": float(
                            block_td_advantage_column.std(unbiased=False).item()
                        ),
                        "block_td_advantage_max": float(block_td_advantage_column.max().item()),
                        "block_td_advantage_min": float(block_td_advantage_column.min().item()),
                        "block_advantage_mean": float(block_advantage_column.mean().item()),
                        "block_advantage_std": float(
                            block_advantage_column.std(unbiased=False).item()
                        ),
                        "block_advantage_max": float(block_advantage_column.max().item()),
                        "block_advantage_min": float(block_advantage_column.min().item()),
                        "block_active_rate": float(
                            (
                                block_advantage_column.abs()
                                > block_advantage_weights.mean(dim=-1)
                            ).float().mean().item()
                        ),
                        "block_clip_fraction": float(block_clip_column.mean().item()),
                        "block_positive_adv_clip_fraction": self._masked_mean(
                            block_clip_column,
                            positive_block_adv_mask[:, block_index],
                        ),
                        "block_negative_adv_clip_fraction": self._masked_mean(
                            block_clip_column,
                            negative_block_adv_mask[:, block_index],
                        ),
                        "block_ratio_mean": float(block_ratio_column.mean().item()),
                        "block_ratio_std": float(block_ratio_column.std(unbiased=False).item()),
                        "block_ratio_max": float(block_ratio_column.max().item()),
                        "block_surrogate_mean": float(block_surrogate_column.mean().item()),
                        "block_surrogate_std": float(
                            block_surrogate_column.std(unbiased=False).item()
                        ),
                        "per_block_selected_action_prob_gain_mean": float(
                            block_prob_gain_column.mean().item()
                        ),
                        "per_block_selected_action_prob_gain_std": float(
                            block_prob_gain_column.std(unbiased=False).item()
                        ),
                    }
                )
                block_weight_rows.append(
                    {
                        "block_index": int(block_meta["block_index"]),
                        "block_start": int(block_meta["start"]),
                        "block_stop_exclusive": int(block_meta["stop_exclusive"]),
                        "block_size": int(block_meta["size"]),
                        "block_semantic": str(block_meta["semantic"]),
                        "block_weight_mean": float(block_weight_column.mean().item()),
                        "block_weight_std": float(block_weight_column.std(unbiased=False).item()),
                        "block_weight_max": float(block_weight_column.max().item()),
                        "block_weight_min": float(block_weight_column.min().item()),
                        "block_active_rate": float(
                            (block_weight_column > uniform_block_weight).float().mean().item()
                        ),
                    }
                )
            if self.config.value_target_mode == "popart_return_norm":
                full_normalized_predictions = self.network.normalized_value_from_critic_input(
                    full_critic_inputs
                ).squeeze(-1)
                full_raw_predictions = full_normalized_predictions * target_std + target_mean
            else:
                full_raw_predictions = self.network.value_from_critic_input(full_critic_inputs).squeeze(-1)
                if self.config.value_target_mode in {"normalized_return", "running_return_norm"}:
                    full_normalized_predictions = (full_raw_predictions - target_mean) / (
                        target_std + 1e-8
                    )
                else:
                    full_normalized_predictions = full_raw_predictions

        action_distribution_std = full_distribution.mean.std(unbiased=False)
        action_prob_std = action_distribution_std
        policy_std_mean = full_distribution.stddev.mean()
        policy_logit_std = full_distribution.mean.std(unbiased=False)
        policy_confidence_mean = (
            full_distribution.mean.abs() / (full_distribution.stddev + 1e-8)
        ).mean()
        high_bucket_lookup = {
            row["bucket_name"]: float(row["bucket_mean_selected_action_prob_gain"])
            for row in advantage_bucket_rows
        }
        block_weight_entropy = (
            -(block_weights * torch.log(block_weights + 1e-8)).sum(dim=-1).mean()
        )
        block_weight_max_mean = block_weights.max(dim=-1).values.mean()
        top_k = min(4, block_weights.shape[-1])
        top_k_block_weight_share = torch.topk(block_weights, k=top_k, dim=-1).values.sum(
            dim=-1
        ).mean()
        block_adv_scale_entropy = (
            -(
                block_adv_scale_distribution
                * torch.log(block_adv_scale_distribution + 1e-8)
            ).sum(dim=-1).mean()
        )
        top_k_block_adv_scale_share = torch.topk(
            block_adv_scale_distribution,
            k=top_k,
            dim=-1,
        ).values.sum(dim=-1).mean()
        block_value_scale_entropy = (
            -(
                block_value_scale_distribution
                * torch.log(block_value_scale_distribution + 1e-8)
            ).sum(dim=-1).mean()
        )
        top_k_block_value_scale_share = torch.topk(
            block_value_scale_distribution,
            k=top_k,
            dim=-1,
        ).values.sum(dim=-1).mean()
        block_td_advantage_entropy = (
            -(
                block_td_advantage_distribution
                * torch.log(block_td_advantage_distribution + 1e-8)
            ).sum(dim=-1).mean()
        )
        top_k_block_td_advantage_share = torch.topk(
            block_td_advantage_distribution,
            k=top_k,
            dim=-1,
        ).values.sum(dim=-1).mean()
        block_advantage_entropy = (
            -(
                block_advantage_distribution
                * torch.log(block_advantage_distribution + 1e-8)
            ).sum(dim=-1).mean()
        )
        top_k_block_advantage_share = torch.topk(
            block_advantage_distribution,
            k=top_k,
            dim=-1,
        ).values.sum(dim=-1).mean()
        active_block_count_mean = (
            block_advantage_weights > block_advantage_weights.mean(dim=-1, keepdim=True)
        ).float().sum(dim=-1).mean()
        selected_action_histogram = json.dumps(
            self._selected_action_histogram(actions),
            ensure_ascii=False,
            sort_keys=True,
        )
        route_active_mask_full = route_masks > 0.5
        route_metric_mask_full = (
            route_loss_masks > 0.5
            if self._uses_theta_route_split_advantage_surrogate()
            else route_active_mask_full
        )
        theta_advantage_alignment_value = self._safe_correlation(
            theta_advantages.reshape(-1),
            theta_logprob_delta.reshape(-1),
        )
        theta_adv_to_route_delta_alignment_value = self._safe_correlation(
            theta_advantages.reshape(-1),
            route_logprob_delta.reshape(-1),
        )
        if self._uses_factorized_ratio_pg() and bool(route_metric_mask_full.any().item()):
            route_advantage_alignment_value = self._safe_correlation(
                route_advantages[route_metric_mask_full],
                route_logprob_delta[route_metric_mask_full],
            )
            route_adv_to_theta_delta_alignment_value = self._safe_correlation(
                route_advantages[route_metric_mask_full],
                theta_logprob_delta[route_metric_mask_full],
            )
            route_selected_action_prob_gain_value = self._masked_mean(
                route_selected_action_prob_gain,
                route_metric_mask_full,
            )
            route_negative_adv_prob_gain_value = self._masked_mean(
                route_selected_action_prob_gain,
                (route_advantages < 0.0) & route_metric_mask_full,
            )
            route_clip_fraction_value = self._masked_mean(
                route_clip_mask.float(),
                route_metric_mask_full,
            )
        else:
            route_advantage_alignment_value = self._safe_correlation(
                route_advantages.reshape(-1),
                route_logprob_delta.reshape(-1),
            )
            route_adv_to_theta_delta_alignment_value = self._safe_correlation(
                route_advantages.reshape(-1),
                theta_logprob_delta.reshape(-1),
            )
            route_selected_action_prob_gain_value = float(
                route_selected_action_prob_gain.mean().item()
            )
            route_negative_adv_prob_gain_value = self._masked_mean(
                route_selected_action_prob_gain,
                route_advantages < 0.0,
            )
            route_clip_fraction_value = float(route_clip_mask.float().mean().item())
        theta_selected_action_prob_gain_value = float(theta_selected_action_prob_gain.mean().item())
        theta_negative_adv_prob_gain_value = self._masked_mean(
            theta_selected_action_prob_gain,
            theta_advantages < 0.0,
        )
        theta_clip_fraction_value = float(theta_clip_mask.float().mean().item())
        theta_alignment_margin_over_cross_value = (
            theta_advantage_alignment_value - theta_adv_to_route_delta_alignment_value
        )
        route_alignment_margin_over_cross_value = (
            route_advantage_alignment_value - route_adv_to_theta_delta_alignment_value
        )
        cross_branch_alignment_mean_value = 0.5 * (
            theta_adv_to_route_delta_alignment_value
            + route_adv_to_theta_delta_alignment_value
        )
        critic_raw_prediction_mean = full_raw_predictions.mean()
        critic_raw_prediction_std = full_raw_predictions.std(unbiased=False)
        critic_normalized_prediction_mean = full_normalized_predictions.mean()
        critic_normalized_prediction_std = full_normalized_predictions.std(unbiased=False)
        diagnostics = self._compute_value_diagnostics(full_normalized_predictions, value_targets)
        constant_predictions = torch.full_like(value_targets, value_targets.mean())
        critic_prediction_mse = nn.functional.mse_loss(full_normalized_predictions, value_targets)
        constant_baseline_mse = nn.functional.mse_loss(constant_predictions, value_targets)
        critic_prediction_huber = nn.functional.huber_loss(
            full_normalized_predictions,
            value_targets,
            delta=1.0,
        )
        constant_baseline_huber = nn.functional.huber_loss(
            constant_predictions,
            value_targets,
            delta=1.0,
        )
        hidden_features = full_critic_features
        feature_dim_std = hidden_features.std(dim=0, unbiased=False)
        critic_hidden_feature_mean = hidden_features.mean()
        critic_hidden_feature_std = hidden_features.std(unbiased=False)
        critic_hidden_feature_dim_std_mean = feature_dim_std.mean()
        critic_hidden_feature_dim_std_min = feature_dim_std.min()
        critic_head_weight_norm = self.network.critic_head.weight.detach().norm()
        critic_head_bias_mean = self.network.critic_head.bias.detach().mean()
        block_path_cost_bs1 = (
            block_path_cost_bs_now[..., 0]
            if block_path_cost_bs_now.shape[-1] > 0
            else torch.zeros_like(block_local_cost_now)
        )
        block_path_cost_bs2 = (
            block_path_cost_bs_now[..., 1]
            if block_path_cost_bs_now.shape[-1] > 1
            else torch.zeros_like(block_local_cost_now)
        )
        block_path_value_local = block_path_value_preds[..., 0]
        block_path_value_bs1 = (
            block_path_value_preds[..., 1]
            if block_path_value_preds.shape[-1] > 1
            else torch.zeros_like(block_value_preds)
        )
        block_path_value_bs2 = (
            block_path_value_preds[..., 2]
            if block_path_value_preds.shape[-1] > 2
            else torch.zeros_like(block_value_preds)
        )
        path_td_target_local = block_path_td_targets[..., 0]
        path_td_target_bs1 = (
            block_path_td_targets[..., 1]
            if block_path_td_targets.shape[-1] > 1
            else torch.zeros_like(block_value_preds)
        )
        path_td_target_bs2 = (
            block_path_td_targets[..., 2]
            if block_path_td_targets.shape[-1] > 2
            else torch.zeros_like(block_value_preds)
        )
        actor_theta_outputs = full_actor_diagnostics.get(
            "theta_mean",
            torch.zeros_like(block_ratio),
        )
        theta_backbone_features = full_actor_diagnostics.get(
            "theta_backbone_features",
            torch.zeros(
                block_ratio.shape[0],
                self.config.hidden_size,
                device=block_ratio.device,
                dtype=block_ratio.dtype,
            ),
        )
        route_backbone_features = full_actor_diagnostics.get(
            "route_backbone_features",
            theta_backbone_features,
        )
        actor_route_outputs = full_actor_diagnostics.get(
            "route_mean",
            torch.zeros(
                block_ratio.shape[0],
                block_ratio.shape[1],
                max(self.action_path_count - 1, 0),
                device=block_ratio.device,
                dtype=block_ratio.dtype,
            ),
        )
        if actor_route_outputs.shape[-1] >= 2:
            route_preference = actor_route_outputs[..., 0] - actor_route_outputs[..., 1]
        elif actor_route_outputs.shape[-1] == 1:
            route_preference = actor_route_outputs[..., 0]
        else:
            route_preference = torch.zeros_like(actor_theta_outputs)
        theta_route_head_correlation = self._safe_correlation(
            actor_theta_outputs.reshape(-1),
            route_preference.reshape(-1),
        )
        theta_route_feature_correlation = self._safe_correlation(
            theta_backbone_features.reshape(-1),
            route_backbone_features.reshape(-1),
        )
        offload_rate_mean = float(action_theta.mean().item())
        bs1_vs_bs2_entropy = float(
            (
                -(
                    action_route_probs
                    * torch.log(action_route_probs + 1e-8)
                ).sum(dim=-1)
            ).mean().item()
        )

        # Freeze actor-input normalization within this rollout/update cycle, then
        # fold the collected states into running stats only for the next rollout.
        if self._uses_normalized_augmented_actor_input():
            actor_augmented_inputs, _, _ = self._augment_actor_input(states)
            self._update_running_actor_input_stats(actor_augmented_inputs)

        self.buffer.clear()
        return {
            "actor_loss": actor_loss_value / update_count,
            "critic_loss": critic_loss_value / update_count,
            "critic_loss_current_batch": critic_loss_current_value / update_count,
            "critic_loss_heldout_batch": critic_loss_heldout_value / update_count,
            "critic_blended_value_loss_enabled": float(critic_blended_value_loss_enabled),
            "critic_blended_current_weight": float(critic_blended_current_weight),
            "critic_blended_heldout_weight": float(critic_blended_heldout_weight),
            "critic_blended_heldout_batch_count": float(
                0
                if blended_heldout_critic_inputs is None
                else blended_heldout_critic_inputs.size(0) // int(self.config.mini_batch_size)
            ),
            "entropy": entropy_value / update_count,
            "policy_entropy": entropy_value / update_count,
            "raw_return_mean": float(raw_return_mean.item()),
            "raw_return_std": float(raw_return_std.item()),
            "value_target_mean": float(value_target_mean.item()),
            "value_target_std": float(value_target_std.item()),
            "advantage_mean": float(raw_advantage_mean.item()),
            "advantage_std": float(raw_advantage_std.item()),
            "normalized_advantage_mean": float(normalized_advantage_mean.item()),
            "normalized_advantage_std": float(normalized_advantage_std.item()),
            "positive_advantage_ratio": positive_advantage_ratio,
            "negative_advantage_ratio": negative_advantage_ratio,
            "delta_log_prob_selected_action_mean": float(delta_log_prob.mean().item()),
            "delta_log_prob_selected_action_std": float(delta_log_prob.std(unbiased=False).item()),
            "selected_action_prob_gain_mean": float(
                selected_action_prob_gain_values.mean().item()
            ),
            "advantage_action_alignment": self._safe_correlation(raw_advantages, delta_log_prob),
            "high_advantage_action_prob_gain": float(
                high_bucket_lookup.get("high_positive", 0.0)
            ),
            "mid_advantage_action_prob_gain": float(
                high_bucket_lookup.get("mid_positive", 0.0)
            ),
            "low_advantage_action_prob_gain": float(high_bucket_lookup.get("near_zero", 0.0)),
            "negative_advantage_action_prob_gain": float(
                high_bucket_lookup.get("negative", 0.0)
            ),
            "advantage_bucket_high_positive_threshold": float(
                advantage_bucket_thresholds["high_positive_threshold"]
            ),
            "advantage_bucket_near_zero_threshold": float(
                advantage_bucket_thresholds["near_zero_threshold"]
            ),
            "ratio_mean": float(ratio_tensor_for_summary.mean().item()),
            "ratio_std": float(ratio_tensor_for_summary.std(unbiased=False).item()),
            "ratio_min": float(ratio_tensor_for_summary.min().item()),
            "ratio_max": float(ratio_tensor_for_summary.max().item()),
            "clip_fraction": float(clip_tensor_for_summary.mean().item()),
            "positive_adv_clip_fraction": self._masked_mean(
                clip_fraction_values_for_buckets
                if self._uses_factorized_ratio_pg()
                else (
                    block_clip_mask.float()
                    if self._uses_blockwise_policy_surrogate()
                    else full_clip_mask.float()
                ),
                positive_adv_mask
                if self._uses_factorized_ratio_pg()
                else (
                    (block_advantages > 0.0)
                    if self._uses_blockwise_policy_surrogate()
                    else positive_adv_mask
                ),
            ),
            "negative_adv_clip_fraction": self._masked_mean(
                clip_fraction_values_for_buckets
                if self._uses_factorized_ratio_pg()
                else (
                    block_clip_mask.float()
                    if self._uses_blockwise_policy_surrogate()
                    else full_clip_mask.float()
                ),
                negative_adv_mask
                if self._uses_factorized_ratio_pg()
                else (
                    (block_advantages < 0.0)
                    if self._uses_blockwise_policy_surrogate()
                    else negative_adv_mask
                ),
            ),
            "approx_kl": float(approx_kl.item()),
            "entropy_loss": float(full_entropy_loss.item()),
            "policy_loss": float(full_policy_loss.item()),
            "ratio_by_adv_sign": self._ratio_summary_by_adv_sign(
                ratio_by_adv_sign_tensor,
                ratio_by_adv_sign_advantages,
            ),
            "policy_surrogate_mode": self._policy_ratio_mode(),
            "policy_ratio_mode": self.config.policy_ratio_mode,
            "policy_ratio_mode_raw": self.config.policy_ratio_mode,
            "policy_ratio_mode_resolved": self._policy_ratio_mode(),
            "actor_structure_mode": self._actor_structure_mode(),
            "action_dim": int(self.action_dim),
            "action_block_count": int(len(action_block_layout)),
            "action_block_slices": json.dumps(
                action_block_layout,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "logprob_delta_sum_mean": float(logprob_delta_sum.mean().item()),
            "logprob_delta_sum_std": float(logprob_delta_sum.std(unbiased=False).item()),
            "logprob_delta_mean_mean": float(logprob_delta_mean.mean().item()),
            "logprob_delta_mean_std": float(logprob_delta_mean.std(unbiased=False).item()),
            "block_logprob_delta_mean": float(block_mean_delta.mean().item()),
            "block_logprob_delta_std": float(block_mean_delta.std(unbiased=False).item()),
            "block_weight_entropy": float(block_weight_entropy.item()),
            "block_weight_max_mean": float(block_weight_max_mean.item()),
            "top_k_block_weight_share": float(top_k_block_weight_share.item()),
            "block_adv_scale_mean": float(block_adv_scales.mean().item()),
            "block_adv_scale_std": float(block_adv_scales.std(unbiased=False).item()),
            "block_adv_scale_max": float(block_adv_scales.max().item()),
            "block_adv_scale_min": float(block_adv_scales.min().item()),
            "block_adv_scale_entropy": float(block_adv_scale_entropy.item()),
            "top_k_block_adv_scale_share": float(top_k_block_adv_scale_share.item()),
            "block_value_score_mean": float(block_value_scores.mean().item()),
            "block_value_score_std": float(block_value_scores.std(unbiased=False).item()),
            "block_value_score_max": float(block_value_scores.max().item()),
            "block_value_score_min": float(block_value_scores.min().item()),
            "block_value_scale_mean": float(block_value_scales.mean().item()),
            "block_value_scale_std": float(block_value_scales.std(unbiased=False).item()),
            "block_value_scale_max": float(block_value_scales.max().item()),
            "block_value_scale_min": float(block_value_scales.min().item()),
            "block_value_scale_entropy": float(block_value_scale_entropy.item()),
            "top_k_block_value_scale_share": float(top_k_block_value_scale_share.item()),
            "block_value_pred_mean": float(block_value_preds.mean().item()),
            "block_value_pred_std": float(block_value_preds.std(unbiased=False).item()),
            "block_value_pred_max": float(block_value_preds.max().item()),
            "block_value_pred_min": float(block_value_preds.min().item()),
            "block_path_value_local_mean": float(block_path_value_local.mean().item()),
            "block_path_value_local_std": float(block_path_value_local.std(unbiased=False).item()),
            "block_path_value_bs1_mean": float(block_path_value_bs1.mean().item()),
            "block_path_value_bs1_std": float(block_path_value_bs1.std(unbiased=False).item()),
            "block_path_value_bs2_mean": float(block_path_value_bs2.mean().item()),
            "block_path_value_bs2_std": float(block_path_value_bs2.std(unbiased=False).item()),
            "path_value_local_mean": float(block_path_value_local.mean().item()),
            "path_value_local_std": float(block_path_value_local.std(unbiased=False).item()),
            "path_value_bs1_mean": float(block_path_value_bs1.mean().item()),
            "path_value_bs1_std": float(block_path_value_bs1.std(unbiased=False).item()),
            "path_value_bs2_mean": float(block_path_value_bs2.mean().item()),
            "path_value_bs2_std": float(block_path_value_bs2.std(unbiased=False).item()),
            "path_td_target_local_mean": float(path_td_target_local.mean().item()),
            "path_td_target_local_std": float(path_td_target_local.std(unbiased=False).item()),
            "path_td_target_bs1_mean": float(path_td_target_bs1.mean().item()),
            "path_td_target_bs1_std": float(path_td_target_bs1.std(unbiased=False).item()),
            "path_td_target_bs2_mean": float(path_td_target_bs2.mean().item()),
            "path_td_target_bs2_std": float(path_td_target_bs2.std(unbiased=False).item()),
            "theta_advantage_mean": float(theta_advantages.mean().item()),
            "theta_advantage_std": float(theta_advantages.std(unbiased=False).item()),
            "theta_advantage_max": float(theta_advantages.max().item()),
            "theta_advantage_min": float(theta_advantages.min().item()),
            "route_advantage_mean": float(route_advantages.mean().item()),
            "route_advantage_std": float(route_advantages.std(unbiased=False).item()),
            "route_advantage_max": float(route_advantages.max().item()),
            "route_advantage_min": float(route_advantages.min().item()),
            "theta_adv_norm_mean": float(theta_adv_norm_mean.item()),
            "theta_adv_norm_std": float(theta_adv_norm_std.item()),
            "route_adv_norm_mean": float(route_adv_norm_mean.item()),
            "route_adv_norm_std": float(route_adv_norm_std.item()),
            "route_adv_norm_active_count": float(route_adv_norm_active_count.item()),
            "route_adv_norm_used_fallback": float(route_adv_norm_used_fallback),
            "theta_advantage_alignment": theta_advantage_alignment_value,
            "route_advantage_alignment": route_advantage_alignment_value,
            "theta_adv_to_route_delta_alignment": theta_adv_to_route_delta_alignment_value,
            "route_adv_to_theta_delta_alignment": route_adv_to_theta_delta_alignment_value,
            "theta_alignment_margin_over_cross": theta_alignment_margin_over_cross_value,
            "route_alignment_margin_over_cross": route_alignment_margin_over_cross_value,
            "cross_branch_alignment_mean": cross_branch_alignment_mean_value,
            "theta_selected_action_prob_gain": theta_selected_action_prob_gain_value,
            "route_selected_action_prob_gain": route_selected_action_prob_gain_value,
            "theta_negative_adv_prob_gain": theta_negative_adv_prob_gain_value,
            "route_negative_adv_prob_gain": route_negative_adv_prob_gain_value,
            "theta_clip_fraction": theta_clip_fraction_value,
            "route_clip_fraction": route_clip_fraction_value,
            "theta_ratio_mean": theta_ratio_mean,
            "theta_ratio_std": theta_ratio_std,
            "theta_ratio_max": theta_ratio_max,
            "route_ratio_mean": route_ratio_mean,
            "route_ratio_std": route_ratio_std,
            "route_ratio_max": route_ratio_max,
            "offload_active_fraction": offload_active_fraction,
            "route_logprob_active_fraction": route_logprob_active_fraction,
            "theta_positive_fraction": theta_positive_fraction,
            "route_credit_gate_fraction": route_credit_gate_fraction,
            "route_credit_effective_fraction": route_credit_effective_fraction,
            "route_credit_effective_count": route_credit_effective_count,
            "route_credit_weight_mean": route_credit_weight_mean,
            "route_credit_weight_std": route_credit_weight_std,
            "route_credit_weight_min": route_credit_weight_min,
            "route_credit_weight_max": route_credit_weight_max,
            "theta_candidate_score_mean": theta_candidate_score_mean,
            "theta_candidate_score_std": theta_candidate_score_std,
            "theta_selected_score_mean": theta_selected_score_mean,
            "theta_expected_score_mean": theta_expected_score_mean,
            "theta_residual_credit_mean": theta_residual_credit_mean,
            "theta_residual_credit_std": theta_residual_credit_std,
            "theta_residual_credit_min": theta_residual_credit_min,
            "theta_residual_credit_max": theta_residual_credit_max,
            "route_candidate_score_mean": route_candidate_score_mean,
            "route_candidate_score_std": route_candidate_score_std,
            "route_residual_credit_mean": route_residual_credit_mean,
            "route_residual_credit_std": route_residual_credit_std,
            "route_residual_credit_min": route_residual_credit_min,
            "route_residual_credit_max": route_residual_credit_max,
            "route_expected_score_mean": route_expected_score_mean,
            "route_selected_score_mean": route_selected_score_mean,
            "route_decision_agreement_ratio": route_decision_agreement_ratio,
            "actual_bs1_rate_when_route_true_gap_positive": (
                actual_bs1_rate_when_route_true_gap_positive
            ),
            "actual_bs1_rate_when_route_true_gap_negative": (
                actual_bs1_rate_when_route_true_gap_negative
            ),
            "route_score_vector_mean_abs": route_score_vector_mean_abs,
            "route_score_vector_std": route_score_vector_std,
            "route_credit_fallback_trigger_count": int(route_credit_fallback_trigger_count_value),
            "route_credit_fallback_rate": float(
                route_credit_fallback_trigger_count_value
                / max(route_credit_support_eval_count_value, 1)
            ),
            "theta_old_logprob_mean": theta_old_logprob_mean,
            "theta_new_logprob_mean": theta_new_logprob_mean,
            "route_old_logprob_mean": route_old_logprob_mean,
            "route_new_logprob_mean": route_new_logprob_mean,
            "joint_old_logprob_mean": joint_old_logprob_mean,
            "joint_new_logprob_mean": joint_new_logprob_mean,
            "joint_ratio_mean": joint_ratio_mean,
            "joint_approx_kl": joint_approx_kl,
            "joint_clip_fraction": joint_clip_fraction,
            "joint_selected_score_mean": joint_selected_score_mean,
            "joint_expected_score_mean": joint_expected_score_mean,
            "joint_residual_credit_mean": joint_residual_credit_mean,
            "joint_residual_credit_std": joint_residual_credit_std,
            "joint_residual_credit_min": joint_residual_credit_min,
            "joint_residual_credit_max": joint_residual_credit_max,
            "joint_action_decision_agreement_ratio_under_reward_aligned": (
                joint_action_decision_agreement_ratio_under_reward_aligned
            ),
            "joint_action_decision_agreement_ratio_under_td_aligned": (
                joint_action_decision_agreement_ratio_under_td_aligned
            ),
            "actual_local_rate_when_reward_aligned_best_is_local": (
                actual_local_rate_when_reward_aligned_best_is_local
            ),
            "actual_bs1_rate_when_reward_aligned_best_is_bs1": (
                actual_bs1_rate_when_reward_aligned_best_is_bs1
            ),
            "actual_bs2_rate_when_reward_aligned_best_is_bs2": (
                actual_bs2_rate_when_reward_aligned_best_is_bs2
            ),
            "actual_local_rate_when_td_aligned_best_is_local": (
                actual_local_rate_when_td_aligned_best_is_local
            ),
            "actual_bs1_rate_when_td_aligned_best_is_bs1": (
                actual_bs1_rate_when_td_aligned_best_is_bs1
            ),
            "actual_bs2_rate_when_td_aligned_best_is_bs2": (
                actual_bs2_rate_when_td_aligned_best_is_bs2
            ),
            "offload_decision_agreement_ratio": offload_decision_agreement_ratio,
            "actual_offload_rate_when_theta_true_gap_positive": (
                actual_offload_rate_when_theta_true_gap_positive
            ),
            "actual_offload_rate_when_theta_true_gap_negative": (
                actual_offload_rate_when_theta_true_gap_negative
            ),
            "theta_approx_kl": float(theta_approx_kl.item()),
            "route_approx_kl": float(route_approx_kl.item()),
            "conditional_policy_consistency_score": conditional_policy_consistency_score,
            "theta_loss_mean": float(theta_loss_mean.item()),
            "route_loss_mean": float(route_loss_mean.item()),
            "theta_route_loss_ratio": float(theta_route_loss_ratio.item()),
            "route_gate_mean": float(route_gates.mean().item()),
            "route_gate_std": float(route_gates.std(unbiased=False).item()),
            "route_gate_min": float(route_gates.min().item()),
            "route_gate_max": float(route_gates.max().item()),
            "route_gate_active_fraction": float((route_gates > 0.5).float().mean().item()),
            "route_mask_mean": float(route_masks.mean().item()),
            "route_mask_active_fraction": float(route_masks.mean().item()),
            "route_mask_threshold": float(self._route_mask_threshold()),
            "route_confident_mask_margin": float(
                max(0.0, getattr(self.config, "route_confident_mask_margin", 0.0))
            ),
            "route_mask_count_mean": float(route_mask_counts.mean().item()),
            "route_mask_count_std": float(route_mask_counts.std(unbiased=False).item()),
            "theta_head_grad_norm": theta_head_grad_norm_value / theta_update_denominator,
            "route_head_grad_norm": route_head_grad_norm_value / route_update_denominator,
            "theta_backbone_grad_norm": theta_backbone_grad_norm_value / theta_update_denominator,
            "route_backbone_grad_norm": route_backbone_grad_norm_value / route_update_denominator,
            "theta_kl_target": float(self.config.theta_kl_target),
            "route_kl_target": float(self.config.route_kl_target),
            "theta_early_stop_count": int(theta_early_stop_count_value),
            "route_early_stop_count": int(route_early_stop_count_value),
            "coupled_stop_trigger_count": int(coupled_stop_trigger_count_value),
            "coupled_stop_blocked_by_theta_floor_count": int(
                coupled_stop_blocked_by_theta_floor_count_value
            ),
            "coupled_stop_blocked_by_severity_gate_count": int(
                coupled_stop_blocked_by_severity_gate_count_value
            ),
            "coupled_stop_blocked_by_alignment_gate_count": int(
                coupled_stop_blocked_by_alignment_gate_count_value
            ),
            "coupled_stop_blocked_by_warmup_count": int(
                coupled_stop_blocked_by_warmup_count_value
            ),
            "coupled_stop_min_theta_updates_per_epoch": int(
                max(
                    0,
                    getattr(self.config, "coupled_stop_min_theta_updates_per_epoch", 0),
                )
            ),
            "coupled_stop_warmup_epochs": int(
                max(0, getattr(self.config, "coupled_stop_warmup_epochs", 0))
            ),
            "coupled_stop_is_active_this_train_epoch": float(
                1.0 if coupled_stop_is_active_this_train_epoch else 0.0
            ),
            "coupled_stop_train_epoch_index": int(train_epoch_index),
            "coupled_stop_severity_factor": float(
                max(1.0, getattr(self.config, "coupled_stop_severity_factor", 1.0))
            ),
            "coupled_stop_severity_freeze_kl_threshold": float(
                float(self.config.route_kl_target)
                * max(1.0, getattr(self.config, "coupled_stop_severity_factor", 1.0))
            ),
            "coupled_stop_alignment_freeze_threshold": float(
                getattr(self.config, "route_alignment_gate_threshold", 0.0)
            ),
            "route_update_cap_trigger_count": int(route_update_cap_trigger_count_value),
            "route_update_cap_per_epoch": int(
                max(0, getattr(self.config, "route_update_cap_per_epoch", 0))
            ),
            "route_alignment_gate_threshold": float(
                getattr(self.config, "route_alignment_gate_threshold", 0.0)
            ),
            "route_alignment_gate_accept_count": int(route_alignment_gate_accept_count_value),
            "route_alignment_gate_reject_count": int(route_alignment_gate_reject_count_value),
            "route_alignment_gate_reject_rate": float(
                route_alignment_gate_reject_count_value / route_alignment_gate_total_steps
            ),
            "route_alignment_gate_score_mean": float(
                route_alignment_gate_score_sum_value / route_alignment_gate_score_denominator
            ),
            "route_alignment_gate_snapshot_time_ms": float(
                route_alignment_gate_snapshot_time_ms_value
            ),
            "route_alignment_gate_restore_time_ms": float(
                route_alignment_gate_restore_time_ms_value
            ),
            "route_alignment_gate_snapshot_count": int(
                route_alignment_gate_snapshot_count_value
            ),
            "route_alignment_gate_restore_count": int(route_alignment_gate_restore_count_value),
            "route_step_alignment_mean": float(
                route_alignment_gate_score_sum_value / route_alignment_gate_score_denominator
            ),
            "route_step_alignment_min": route_step_alignment_min_metric,
            "route_step_alignment_max": route_step_alignment_max_metric,
            "route_step_accept_count": int(route_alignment_gate_accept_count_value),
            "route_step_reject_count": int(route_alignment_gate_reject_count_value),
            "route_step_accept_rate": float(
                route_alignment_gate_accept_count_value / route_alignment_gate_total_steps
            ),
            "route_step_reject_rate": float(
                route_alignment_gate_reject_count_value / route_alignment_gate_total_steps
            ),
            "theta_update_count": int(theta_update_count_value),
            "route_update_count": int(route_update_count_value),
            "theta_only_step_kl": theta_only_step_kl_value / theta_update_denominator,
            "route_only_step_kl": route_only_step_kl_value / route_update_denominator,
            "theta_only_step_prob_gain": theta_only_step_prob_gain_value / theta_update_denominator,
            "route_only_step_prob_gain": route_only_step_prob_gain_value / route_update_denominator,
            "theta_after_route_shift": theta_after_route_shift_value / route_update_denominator,
            "route_after_theta_shift": route_after_theta_shift_value / theta_update_denominator,
            "theta_route_feature_correlation": theta_route_feature_correlation,
            "theta_route_head_correlation": theta_route_head_correlation,
            "offload_rate_mean": offload_rate_mean,
            "bs1_vs_bs2_entropy": bs1_vs_bs2_entropy,
            "offload_vs_local_value_gap_mean": float(offload_vs_local_value_gap.mean().item()),
            "offload_vs_local_value_gap_std": float(
                offload_vs_local_value_gap.std(unbiased=False).item()
            ),
            "bs1_vs_bs2_value_gap_mean": float(bs1_vs_bs2_value_gap.mean().item()),
            "bs1_vs_bs2_value_gap_std": float(
                bs1_vs_bs2_value_gap.std(unbiased=False).item()
            ),
            "block_action_conditioned_value_now_mean": float(
                block_action_conditioned_value_preds.mean().item()
            ),
            "block_action_conditioned_value_now_std": float(
                block_action_conditioned_value_preds.std(unbiased=False).item()
            ),
            "block_action_conditioned_value_next_mean": float(
                block_action_conditioned_value_next_preds.mean().item()
            ),
            "block_action_conditioned_value_next_std": float(
                block_action_conditioned_value_next_preds.std(unbiased=False).item()
            ),
            "block_path_cost_local_mean": float(block_path_cost_local_now.mean().item()),
            "block_path_cost_local_std": float(
                block_path_cost_local_now.std(unbiased=False).item()
            ),
            "block_path_cost_bs1_mean": float(block_path_cost_bs1.mean().item()),
            "block_path_cost_bs1_std": float(block_path_cost_bs1.std(unbiased=False).item()),
            "block_path_cost_bs2_mean": float(block_path_cost_bs2.mean().item()),
            "block_path_cost_bs2_std": float(block_path_cost_bs2.std(unbiased=False).item()),
            "block_local_cost_now_mean": float(block_local_cost_now.mean().item()),
            "block_local_cost_now_std": float(block_local_cost_now.std(unbiased=False).item()),
            "block_local_cost_next_mean": float(block_local_cost_next.mean().item()),
            "block_local_cost_next_std": float(block_local_cost_next.std(unbiased=False).item()),
            "block_action_conditioned_cost_now_mean": float(
                block_action_conditioned_cost_now.mean().item()
            ),
            "block_action_conditioned_cost_now_std": float(
                block_action_conditioned_cost_now.std(unbiased=False).item()
            ),
            "block_action_conditioned_cost_next_mean": float(
                block_action_conditioned_cost_next.mean().item()
            ),
            "block_action_conditioned_cost_next_std": float(
                block_action_conditioned_cost_next.std(unbiased=False).item()
            ),
            "block_delta_cost_mean": float(block_delta_cost.mean().item()),
            "block_delta_cost_std": float(block_delta_cost.std(unbiased=False).item()),
            "block_delta_cost_max": float(block_delta_cost.max().item()),
            "block_delta_cost_min": float(block_delta_cost.min().item()),
            "block_td_reward_mean": float(block_local_rewards.mean().item()),
            "block_td_reward_std": float(block_local_rewards.std(unbiased=False).item()),
            "block_td_reward_max": float(block_local_rewards.max().item()),
            "block_td_reward_min": float(block_local_rewards.min().item()),
            "block_td_target_mean": float(block_td_targets.mean().item()),
            "block_td_target_std": float(block_td_targets.std(unbiased=False).item()),
            "block_td_target_max": float(block_td_targets.max().item()),
            "block_td_target_min": float(block_td_targets.min().item()),
            "block_td_advantage_mean": float(block_td_advantages.mean().item()),
            "block_td_advantage_std": float(block_td_advantages.std(unbiased=False).item()),
            "block_td_advantage_max": float(block_td_advantages.max().item()),
            "block_td_advantage_min": float(block_td_advantages.min().item()),
            "block_td_advantage_entropy": float(block_td_advantage_entropy.item()),
            "top_k_block_td_advantage_share": float(top_k_block_td_advantage_share.item()),
            "block_advantage_mean": float(block_advantages.mean().item()),
            "block_advantage_std": float(block_advantages.std(unbiased=False).item()),
            "block_advantage_max": float(block_advantages.max().item()),
            "block_advantage_min": float(block_advantages.min().item()),
            "block_advantage_entropy": float(block_advantage_entropy.item()),
            "top_k_block_advantage_share": float(top_k_block_advantage_share.item()),
            "active_block_count_mean": float(active_block_count_mean.item()),
            "block_ratio_proxy_mean": float(block_ratio_proxy.mean().item()),
            "block_ratio_proxy_std": float(block_ratio_proxy.std(unbiased=False).item()),
            "block_clip_fraction_mean": float(
                block_clip_mask.float().mean(dim=0).mean().item()
            ),
            "block_clip_fraction_std": float(
                block_clip_mask.float().mean(dim=0).std(unbiased=False).item()
            ),
            "block_positive_adv_clip_fraction_mean": float(
                block_clip_mask[positive_adv_mask].float().mean().item()
            )
            if bool(positive_adv_mask.any().item())
            else 0.0,
            "block_negative_adv_clip_fraction_mean": float(
                block_clip_mask[negative_adv_mask].float().mean().item()
            )
            if bool(negative_adv_mask.any().item())
            else 0.0,
            "block_ratio_mean": float(block_ratio.mean().item()),
            "block_ratio_std": float(block_ratio.std(unbiased=False).item()),
            "block_ratio_max": float(block_ratio.max().item()),
            "block_surrogate_mean": float(block_surrogate.mean().item()),
            "block_surrogate_std": float(block_surrogate.std(unbiased=False).item()),
            "per_block_selected_action_prob_gain_mean": float(
                block_selected_action_prob_gain.mean().item()
            ),
            "per_block_selected_action_prob_gain_std": float(
                block_selected_action_prob_gain.std(unbiased=False).item()
            ),
            "sum_to_mean_scale_ratio": float(
                logprob_delta_sum.abs().mean().item()
                / (logprob_delta_mean.abs().mean().item() + 1e-8)
            ),
            "sum_to_blockmean_scale_ratio": float(
                logprob_delta_sum.abs().mean().item()
                / (block_mean_delta.abs().mean().item() + 1e-8)
            ),
            "mean_to_blockmean_scale_ratio": float(
                logprob_delta_mean.abs().mean().item()
                / (block_mean_delta.abs().mean().item() + 1e-8)
            ),
            "running_return_mean": self.running_return_mean,
            "running_return_std": self._current_running_return_std(),
            "popart_mean": self.running_return_mean,
            "popart_std": self._current_running_return_std(),
            "actor_grad_norm": actor_grad_norm_value / update_count,
            "action_distribution_std": float(action_distribution_std.item()),
            "action_prob_std": float(action_prob_std.item()),
            "policy_std_mean": float(policy_std_mean.item()),
            "policy_logit_std": float(policy_logit_std.item()),
            "policy_confidence_mean": float(policy_confidence_mean.item()),
            "selected_action_histogram": selected_action_histogram,
            "critic_raw_prediction_mean": float(critic_raw_prediction_mean.item()),
            "critic_raw_prediction_std": float(critic_raw_prediction_std.item()),
            "critic_normalized_prediction_mean": float(critic_normalized_prediction_mean.item()),
            "critic_normalized_prediction_std": float(critic_normalized_prediction_std.item()),
            "value_explained_variance": diagnostics["value_explained_variance"],
            "prediction_target_corr": diagnostics["prediction_target_corr"],
            "prediction_std_over_target_std": diagnostics["prediction_std_over_target_std"],
            "constant_baseline_mse": float(constant_baseline_mse.item()),
            "critic_prediction_mse": float(critic_prediction_mse.item()),
            "critic_vs_constant_mse_gain": float(
                (constant_baseline_mse - critic_prediction_mse).item()
            ),
            "constant_baseline_huber": float(constant_baseline_huber.item()),
            "critic_prediction_huber": float(critic_prediction_huber.item()),
            "critic_vs_constant_huber_gain": float(
                (constant_baseline_huber - critic_prediction_huber).item()
            ),
            "critic_hidden_feature_mean": float(critic_hidden_feature_mean.item()),
            "critic_hidden_feature_std": float(critic_hidden_feature_std.item()),
            "critic_hidden_feature_dim_std_mean": float(critic_hidden_feature_dim_std_mean.item()),
            "critic_hidden_feature_dim_std_min": float(critic_hidden_feature_dim_std_min.item()),
            "critic_head_weight_norm": float(critic_head_weight_norm.item()),
            "critic_head_bias_mean": float(critic_head_bias_mean.item()),
            "critic_backbone_grad_norm": critic_backbone_grad_norm_value / update_count,
            "critic_head_grad_norm": critic_head_grad_norm_value / update_count,
            "critic_param_delta_norm_after_optimizer_step": (
                critic_param_delta_norm_value / update_count
            ),
            "critic_backbone_delta_norm": critic_backbone_delta_norm_value / update_count,
            "critic_head_delta_norm": critic_head_delta_norm_value / update_count,
            "critic_backbone_preconditioner_cap": float(
                critic_backbone_preconditioner_cap_value
            ),
            "critic_backbone_preconditioner_active_count": float(
                critic_backbone_preconditioner_active_count_value / update_count
            ),
            "critic_backbone_preconditioner_clipped_count": float(
                critic_backbone_preconditioner_clipped_count_value / update_count
            ),
            "critic_backbone_preconditioner_clip_fraction": float(
                critic_backbone_preconditioner_clipped_count_value
                / max(1, critic_backbone_preconditioner_active_count_value)
            ),
            "critic_backbone_preconditioner_active_mean_before_clip": float(
                critic_backbone_preconditioner_active_sum_before_clip_value
                / max(1, critic_backbone_preconditioner_active_count_value)
            ),
            "critic_backbone_preconditioner_active_mean_after_clip": float(
                critic_backbone_preconditioner_active_sum_after_clip_value
                / max(1, critic_backbone_preconditioner_active_count_value)
            ),
            "critic_backbone_preconditioner_active_max_before_clip": float(
                critic_backbone_preconditioner_active_max_before_clip_value
            ),
            "critic_backbone_preconditioner_active_max_after_clip": float(
                critic_backbone_preconditioner_active_max_after_clip_value
            ),
            "critic_step_monitor_enabled": float(
                1.0 if critic_step_monitor_enabled else 0.0
            ),
            "critic_step_acceptance_enforced": float(
                1.0 if critic_step_enforce_enabled else 0.0
            ),
            "critic_step_attempt_count": int(critic_step_attempt_count_value),
            "critic_step_accept_count": int(critic_step_accept_count_value),
            "critic_step_reject_count": int(critic_step_reject_count_value),
            "critic_step_would_reject_count": int(critic_step_would_reject_count_value),
            "critic_step_reject_rate": float(
                critic_step_reject_count_value / critic_step_denominator
            ),
            "critic_step_accept_rate": float(
                critic_step_accept_count_value / critic_step_denominator
            ),
            "critic_step_would_reject_rate": float(
                critic_step_would_reject_count_value / critic_step_denominator
            ),
            "critic_step_current_loss_improve_epsilon": float(
                critic_step_current_loss_improve_epsilon
            ),
            "critic_step_heldout_loss_tolerance": float(
                critic_step_heldout_loss_tolerance
            ),
            "critic_step_probe_pearson_tolerance": float(
                critic_step_probe_pearson_tolerance
            ),
            "critic_step_mean_current_loss_before": float(
                critic_step_current_loss_before_sum_value / critic_step_denominator
            ),
            "critic_step_mean_current_loss_after_attempt": float(
                critic_step_current_loss_after_sum_value / critic_step_denominator
            ),
            "critic_step_mean_heldout_loss_before": float(
                critic_step_heldout_loss_before_sum_value / critic_step_denominator
            ),
            "critic_step_mean_heldout_loss_after_attempt": float(
                critic_step_heldout_loss_after_sum_value / critic_step_denominator
            ),
            "critic_step_mean_probe_target_pearson_before": float(
                critic_step_probe_pearson_before_sum_value / critic_step_denominator
            ),
            "critic_step_mean_probe_target_pearson_after_attempt": float(
                critic_step_probe_pearson_after_sum_value / critic_step_denominator
            ),
            "critic_step_reject_reason_breakdown": json.dumps(
                critic_step_reject_reason_counts,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "critic_step_would_reject_reason_breakdown": json.dumps(
                critic_step_would_reject_reason_counts,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "actor_input_mean": actor_input_diagnostics["actor_input_mean"],
            "actor_input_std": actor_input_diagnostics["actor_input_std"],
            "actor_input_dim_std_mean": actor_input_diagnostics["actor_input_dim_std_mean"],
            "actor_input_dim_std_min": actor_input_diagnostics["actor_input_dim_std_min"],
            "actor_input_clip_fraction": actor_input_diagnostics["actor_input_clip_fraction"],
            "actor_input_derived_dim": actor_input_diagnostics["actor_input_derived_dim"],
            "actor_input_total_dim": actor_input_diagnostics["actor_input_total_dim"],
            "actor_derived_feature_mean": actor_input_diagnostics["actor_derived_feature_mean"],
            "actor_derived_feature_std": actor_input_diagnostics["actor_derived_feature_std"],
            "actor_raw_dim": actor_input_diagnostics["actor_raw_dim"],
            "actor_raw_pruned_dim_count": actor_input_diagnostics["actor_raw_pruned_dim_count"],
            "actor_raw_dim_std_mean": actor_input_diagnostics["actor_raw_dim_std_mean"],
            "actor_raw_dim_std_min": actor_input_diagnostics["actor_raw_dim_std_min"],
            "actor_raw_zero_var_dim_count": actor_input_diagnostics["actor_raw_zero_var_dim_count"],
            "actor_raw_static_zero_var_dim_count": actor_input_diagnostics[
                "actor_raw_static_zero_var_dim_count"
            ],
            "critic_input_mean": critic_input_diagnostics["critic_input_mean"],
            "critic_input_std": critic_input_diagnostics["critic_input_std"],
            "critic_input_dim_std_mean": critic_input_diagnostics["critic_input_dim_std_mean"],
            "critic_input_dim_std_min": critic_input_diagnostics["critic_input_dim_std_min"],
            "critic_input_clip_fraction": critic_input_diagnostics["critic_input_clip_fraction"],
            "critic_input_derived_dim": critic_input_diagnostics["critic_input_derived_dim"],
            "critic_input_total_dim": critic_input_diagnostics["critic_input_total_dim"],
            "critic_derived_feature_mean": critic_input_diagnostics["critic_derived_feature_mean"],
            "critic_derived_feature_std": critic_input_diagnostics["critic_derived_feature_std"],
            "diagnostic_advantage_bucket_rows": advantage_bucket_rows,
            "diagnostic_block_logprob_rows": block_logprob_rows,
            "diagnostic_block_surrogate_rows": block_surrogate_rows,
            "diagnostic_block_adv_scale_rows": block_adv_scale_rows,
            "diagnostic_block_value_scale_rows": block_value_scale_rows,
            "diagnostic_block_td_value_rows": block_td_value_rows,
            "diagnostic_block_delta_cost_rows": block_delta_cost_rows,
            "diagnostic_block_advantage_rows": block_advantage_rows,
            "diagnostic_block_weight_rows": block_weight_rows,
            "diagnostic_theta_route_split_rows": theta_route_split_rows,
            "diagnostic_value_targets": value_targets.detach().cpu().tolist(),
            "diagnostic_prediction_values": full_normalized_predictions.detach().cpu().tolist(),
            "diagnostic_constant_predictions": constant_predictions.detach().cpu().tolist(),
            "diagnostic_raw_predictions": full_raw_predictions.detach().cpu().tolist(),
        }

    def save(self, path: str) -> None:
        """Save model and optimizer states."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "theta_actor_optimizer_state_dict": (
                    self.theta_actor_optimizer.state_dict()
                    if self.theta_actor_optimizer is not None
                    else None
                ),
                "route_actor_optimizer_state_dict": (
                    self.route_actor_optimizer.state_dict()
                    if self.route_actor_optimizer is not None
                    else None
                ),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "running_return_mean": self.running_return_mean,
                "running_return_var": self.running_return_var,
                "running_return_count": self.running_return_count,
                "actor_input_running_mean": (
                    self.actor_input_running_mean.tolist()
                    if self.actor_input_running_mean is not None
                    else None
                ),
                "actor_input_running_var": (
                    self.actor_input_running_var.tolist()
                    if self.actor_input_running_var is not None
                    else None
                ),
                "actor_input_count": self.actor_input_count,
                "critic_input_running_mean": (
                    self.critic_input_running_mean.tolist()
                    if self.critic_input_running_mean is not None
                    else None
                ),
                "critic_input_running_var": (
                    self.critic_input_running_var.tolist()
                    if self.critic_input_running_var is not None
                    else None
                ),
                "critic_input_count": self.critic_input_count,
            },
            save_path,
        )

    def load(self, path: str, load_optimizer: bool = False) -> None:
        """Load model state, with backward-compatible optimizer loading."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        network_state = checkpoint["network_state_dict"]
        if "critic.0.weight" in network_state and "critic_backbone.0.weight" not in network_state:
            remapped_state = dict(network_state)
            key_mapping = {
                "critic.0.weight": "critic_backbone.0.weight",
                "critic.0.bias": "critic_backbone.0.bias",
                "critic.2.weight": "critic_backbone.2.weight",
                "critic.2.bias": "critic_backbone.2.bias",
                "critic.4.weight": "critic_head.weight",
                "critic.4.bias": "critic_head.bias",
            }
            for old_key, new_key in key_mapping.items():
                if old_key in remapped_state:
                    remapped_state[new_key] = remapped_state.pop(old_key)
            network_state = remapped_state
        load_result = self.network.load_state_dict(network_state, strict=False)
        unexpected_keys = [
            key
            for key in load_result.unexpected_keys
            if not (
                key.startswith("critic_block_value_head.")
                or key.startswith("critic_block_path_value_head.")
            )
        ]
        missing_keys = [
            key
            for key in load_result.missing_keys
            if not (
                key.startswith("critic_block_value_head.")
                or key.startswith("critic_block_path_value_head.")
            )
        ]
        if unexpected_keys or missing_keys:
            raise RuntimeError(
                "Failed to load checkpoint with unexpected/missing non-block-value keys: "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )
        self.running_return_mean = float(checkpoint.get("running_return_mean", 0.0))
        self.running_return_var = float(checkpoint.get("running_return_var", 1.0))
        self.running_return_count = float(checkpoint.get("running_return_count", 0.0))
        actor_input_running_mean = checkpoint.get("actor_input_running_mean")
        actor_input_running_var = checkpoint.get("actor_input_running_var")
        self.actor_input_running_mean = (
            np.asarray(actor_input_running_mean, dtype=np.float32)
            if actor_input_running_mean is not None
            else None
        )
        self.actor_input_running_var = (
            np.asarray(actor_input_running_var, dtype=np.float32)
            if actor_input_running_var is not None
            else None
        )
        self.actor_input_count = float(checkpoint.get("actor_input_count", 0.0))
        critic_input_running_mean = checkpoint.get("critic_input_running_mean")
        critic_input_running_var = checkpoint.get("critic_input_running_var")
        self.critic_input_running_mean = (
            np.asarray(critic_input_running_mean, dtype=np.float32)
            if critic_input_running_mean is not None
            else None
        )
        self.critic_input_running_var = (
            np.asarray(critic_input_running_var, dtype=np.float32)
            if critic_input_running_var is not None
            else None
        )
        self.critic_input_count = float(checkpoint.get("critic_input_count", 0.0))
        if not load_optimizer:
            return

        actor_state = checkpoint.get("actor_optimizer_state_dict")
        theta_actor_state = checkpoint.get("theta_actor_optimizer_state_dict")
        route_actor_state = checkpoint.get("route_actor_optimizer_state_dict")
        critic_state = checkpoint.get("critic_optimizer_state_dict")
        if critic_state is not None:
            self.critic_optimizer.load_state_dict(critic_state)
        if actor_state is not None:
            self.actor_optimizer.load_state_dict(actor_state)
        if (
            theta_actor_state is not None
            and self.theta_actor_optimizer is not None
        ):
            self.theta_actor_optimizer.load_state_dict(theta_actor_state)
        elif actor_state is not None and self.theta_actor_optimizer is not None:
            self.theta_actor_optimizer.load_state_dict(actor_state)
        if (
            route_actor_state is not None
            and self.route_actor_optimizer is not None
        ):
            self.route_actor_optimizer.load_state_dict(route_actor_state)
        elif actor_state is not None and self.route_actor_optimizer is not None:
            self.route_actor_optimizer.load_state_dict(actor_state)
        if actor_state is not None and critic_state is not None:
            return

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is None:
            return

        try:
            self.actor_optimizer.load_state_dict(optimizer_state)
            self.critic_optimizer.load_state_dict(optimizer_state)
        except ValueError:
            pass
