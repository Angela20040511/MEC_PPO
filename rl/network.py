"""PPO network definitions."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def build_actor_backbone(input_dim: int, hidden_size: int) -> nn.Sequential:
    """Build the standard two-layer actor backbone used across actor structure modes."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
    )


def build_baseline_critic_backbone(state_dim: int, hidden_size: int) -> nn.Sequential:
    """Build the original critic backbone for baseline comparisons."""
    return nn.Sequential(
        nn.Linear(state_dim, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.Tanh(),
    )


def build_stronger_critic_backbone(state_dim: int, hidden_size: int) -> nn.Sequential:
    """Build a slightly deeper critic with normalization to improve state sensitivity."""
    return nn.Sequential(
        nn.Linear(state_dim, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.Tanh(),
    )


class ActorCritic(nn.Module):
    """Actor-Critic network with a switchable critic backbone."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        initial_log_std: float,
        actor_structure_mode: str = "flat_joint_actor",
        critic_arch_mode: str = "baseline_critic",
        actor_input_dim: int | None = None,
        critic_input_dim: int | None = None,
        actor_block_count: int | None = None,
        actor_path_count: int | None = None,
        critic_block_value_count: int | None = None,
        critic_block_path_count: int | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.actor_structure_mode = actor_structure_mode
        self.critic_arch_mode = critic_arch_mode
        actor_input_dim = actor_input_dim or state_dim
        critic_input_dim = critic_input_dim or state_dim
        actor_block_count = actor_block_count or 1
        actor_path_count = actor_path_count or 1
        critic_block_value_count = critic_block_value_count or 1
        critic_block_path_count = critic_block_path_count or 1
        self.actor_block_count = actor_block_count
        self.actor_path_count = actor_path_count
        self.actor_route_dim = max(actor_path_count - 1, 0)
        self.critic_block_value_count = critic_block_value_count
        self.critic_block_path_count = critic_block_path_count
        self.actor_backbone: nn.Sequential | None = None
        self.actor_theta_backbone: nn.Sequential | None = None
        self.actor_route_backbone: nn.Sequential | None = None
        if self._uses_separate_theta_route_backbones():
            self.actor_theta_backbone = build_actor_backbone(actor_input_dim, hidden_size)
            self.actor_route_backbone = build_actor_backbone(actor_input_dim, hidden_size)
        else:
            self.actor_backbone = build_actor_backbone(actor_input_dim, hidden_size)
        self.actor_mean: nn.Linear | None = None
        self.actor_theta_head: nn.Linear | None = None
        self.actor_route_context: nn.Sequential | None = None
        self.actor_route_head: nn.Linear | None = None
        self.route_uses_detached_theta_context = False
        if self._uses_hierarchical_actor_theta_route():
            expected_action_dim = self.actor_block_count * (1 + self.actor_route_dim)
            if self.actor_block_count <= 0 or self.actor_route_dim <= 0 or expected_action_dim != action_dim:
                raise ValueError(
                    "hierarchical_actor_theta_route requires action_dim to match "
                    "actor_block_count * actor_path_count semantics"
                )
            self.actor_theta_head = nn.Linear(hidden_size, self.actor_block_count)
            self.actor_route_context = nn.Sequential(
                nn.Linear(hidden_size + self.actor_block_count, hidden_size),
                nn.Tanh(),
            )
            self.actor_route_head = nn.Linear(
                hidden_size,
                self.actor_block_count * self.actor_route_dim,
            )
            self.route_uses_detached_theta_context = True
        else:
            self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))

        if critic_arch_mode == "baseline_critic":
            self.critic_backbone = build_baseline_critic_backbone(critic_input_dim, hidden_size)
            self.critic_head = nn.Linear(hidden_size, 1)
            self.critic_block_value_head = nn.Linear(hidden_size, critic_block_value_count)
            self.critic_block_path_value_head = nn.Linear(
                hidden_size,
                critic_block_value_count * critic_block_path_count,
            )
            nn.init.zeros_(self.critic_block_value_head.bias)
            nn.init.zeros_(self.critic_block_path_value_head.bias)
        elif critic_arch_mode == "stronger_critic_backbone":
            self.critic_backbone = build_stronger_critic_backbone(critic_input_dim, hidden_size)
            self.critic_head = nn.Linear(hidden_size, 1)
            nn.init.zeros_(self.critic_head.bias)
            self.critic_block_value_head = nn.Linear(hidden_size, critic_block_value_count)
            self.critic_block_path_value_head = nn.Linear(
                hidden_size,
                critic_block_value_count * critic_block_path_count,
            )
            nn.init.zeros_(self.critic_block_value_head.bias)
            nn.init.zeros_(self.critic_block_path_value_head.bias)
        else:
            raise ValueError(f"Unsupported critic_arch_mode: {critic_arch_mode}")

    def policy(self, state: torch.Tensor) -> Normal:
        """Construct the Gaussian policy for the given state."""
        return self.policy_from_actor_input(state)

    def _uses_hierarchical_actor_theta_route(self) -> bool:
        """Return whether the actor uses an explicit theta-then-route head structure."""
        return self.actor_structure_mode in {
            "hierarchical_actor_theta_route",
            "hierarchical_actor_separate_theta_route_backbones",
        }

    def _uses_separate_theta_route_backbones(self) -> bool:
        """Return whether theta and route decisions use disjoint actor backbones."""
        return self.actor_structure_mode == "hierarchical_actor_separate_theta_route_backbones"

    def actor_features_from_actor_input(self, actor_input: torch.Tensor) -> torch.Tensor:
        """Expose the features driving the theta branch for compatibility diagnostics."""
        return self.actor_theta_features_from_actor_input(actor_input)

    def actor_theta_features_from_actor_input(self, actor_input: torch.Tensor) -> torch.Tensor:
        """Expose features used by the theta decision branch."""
        if self._uses_separate_theta_route_backbones():
            if self.actor_theta_backbone is None:
                raise RuntimeError("Theta backbone is not initialized")
            return self.actor_theta_backbone(actor_input)
        if self.actor_backbone is None:
            raise RuntimeError("Shared actor backbone is not initialized")
        return self.actor_backbone(actor_input)

    def actor_route_backbone_features_from_actor_input(
        self,
        actor_input: torch.Tensor,
    ) -> torch.Tensor:
        """Expose backbone features used by the route decision branch before theta conditioning."""
        if self._uses_separate_theta_route_backbones():
            if self.actor_route_backbone is None:
                raise RuntimeError("Route backbone is not initialized")
            return self.actor_route_backbone(actor_input)
        if self.actor_backbone is None:
            raise RuntimeError("Shared actor backbone is not initialized")
        return self.actor_backbone(actor_input)

    def actor_mean_and_diagnostics_from_actor_features(
        self,
        actor_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode shared actor features into the action mean and lightweight structure diagnostics."""
        if self._uses_separate_theta_route_backbones():
            raise RuntimeError(
                "Separate theta/route backbone actors must be decoded from actor inputs "
                "so each branch can use its own backbone."
            )
        if self._uses_hierarchical_actor_theta_route():
            if (
                self.actor_theta_head is None
                or self.actor_route_context is None
                or self.actor_route_head is None
            ):
                raise RuntimeError("Hierarchical actor heads are not initialized")
            theta_mean = self.actor_theta_head(actor_features)
            theta_context = torch.sigmoid(theta_mean.detach())
            route_context_input = torch.cat([actor_features, theta_context], dim=-1)
            route_features = self.actor_route_context(route_context_input)
            route_mean = self.actor_route_head(route_features).reshape(
                -1,
                self.actor_block_count,
                self.actor_route_dim,
            )
            joint_mean = torch.cat([theta_mean.unsqueeze(-1), route_mean], dim=-1).reshape(
                -1,
                self.action_dim,
            )
            return joint_mean, {
                "theta_mean": theta_mean,
                "theta_context": theta_context,
                "theta_backbone_features": actor_features,
                "route_backbone_features": actor_features,
                "route_mean": route_mean,
                "route_features": route_features,
            }

        if self.actor_mean is None:
            raise RuntimeError("Flat actor head is not initialized")
        joint_mean = self.actor_mean(actor_features)
        diagnostics: dict[str, torch.Tensor] = {}
        expected_action_dim = self.actor_block_count * (1 + self.actor_route_dim)
        if self.actor_block_count > 0 and self.actor_route_dim > 0 and expected_action_dim == self.action_dim:
            reshaped_mean = joint_mean.reshape(-1, self.actor_block_count, 1 + self.actor_route_dim)
            diagnostics["theta_mean"] = reshaped_mean[..., 0]
            diagnostics["theta_context"] = torch.sigmoid(reshaped_mean[..., 0])
            diagnostics["route_mean"] = reshaped_mean[..., 1:]
        return joint_mean, diagnostics

    def actor_mean_and_diagnostics_from_actor_input(
        self,
        actor_input: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode preprocessed actor inputs into the action mean and diagnostics."""
        if self._uses_hierarchical_actor_theta_route():
            if (
                self.actor_theta_head is None
                or self.actor_route_context is None
                or self.actor_route_head is None
            ):
                raise RuntimeError("Hierarchical actor heads are not initialized")
            theta_features = self.actor_theta_features_from_actor_input(actor_input)
            route_backbone_features = self.actor_route_backbone_features_from_actor_input(actor_input)
            theta_mean = self.actor_theta_head(theta_features)
            theta_context = torch.sigmoid(theta_mean.detach())
            route_context_input = torch.cat([route_backbone_features, theta_context], dim=-1)
            route_features = self.actor_route_context(route_context_input)
            route_mean = self.actor_route_head(route_features).reshape(
                -1,
                self.actor_block_count,
                self.actor_route_dim,
            )
            joint_mean = torch.cat([theta_mean.unsqueeze(-1), route_mean], dim=-1).reshape(
                -1,
                self.action_dim,
            )
            diagnostics = {
                "theta_mean": theta_mean,
                "theta_context": theta_context,
                "theta_backbone_features": theta_features,
                "route_backbone_features": route_backbone_features,
                "route_mean": route_mean,
                "route_features": route_features,
            }
            if not self._uses_separate_theta_route_backbones():
                diagnostics["shared_actor_features"] = theta_features
            return joint_mean, diagnostics

        actor_features = self.actor_features_from_actor_input(actor_input)
        joint_mean, diagnostics = self.actor_mean_and_diagnostics_from_actor_features(actor_features)
        diagnostics["shared_actor_features"] = actor_features
        return joint_mean, diagnostics

    def policy_with_diagnostics_from_actor_input(
        self,
        actor_input: torch.Tensor,
    ) -> tuple[Normal, dict[str, torch.Tensor]]:
        """Construct the policy distribution together with head-level actor diagnostics."""
        mean, diagnostics = self.actor_mean_and_diagnostics_from_actor_input(actor_input)
        std = self.actor_log_std.exp().expand_as(mean)
        return Normal(mean, std), diagnostics

    def policy_from_actor_input(self, actor_input: torch.Tensor) -> Normal:
        """Construct the Gaussian policy from preprocessed actor-only inputs."""
        distribution, _ = self.policy_with_diagnostics_from_actor_input(actor_input)
        return distribution

    def actor_main_head_parameters(self) -> list[nn.Parameter]:
        """Return the actor head parameters used for the primary mean decoder."""
        if self._uses_hierarchical_actor_theta_route():
            if self.actor_theta_head is None:
                return []
            return list(self.actor_theta_head.parameters())
        if self.actor_mean is None:
            return []
        return list(self.actor_mean.parameters())

    def actor_theta_head_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to the theta decision head."""
        if self._uses_hierarchical_actor_theta_route():
            if self.actor_theta_head is None:
                return []
            return list(self.actor_theta_head.parameters())
        return []

    def actor_backbone_parameters(self) -> list[nn.Parameter]:
        """Return all actor backbone parameters participating in policy optimization."""
        if self._uses_separate_theta_route_backbones():
            parameters: list[nn.Parameter] = []
            if self.actor_theta_backbone is not None:
                parameters.extend(list(self.actor_theta_backbone.parameters()))
            if self.actor_route_backbone is not None:
                parameters.extend(list(self.actor_route_backbone.parameters()))
            return parameters
        if self.actor_backbone is None:
            return []
        return list(self.actor_backbone.parameters())

    def actor_theta_backbone_parameters(self) -> list[nn.Parameter]:
        """Return backbone parameters directly serving the theta branch."""
        if self._uses_separate_theta_route_backbones():
            if self.actor_theta_backbone is None:
                return []
            return list(self.actor_theta_backbone.parameters())
        if self.actor_backbone is None:
            return []
        return list(self.actor_backbone.parameters())

    def actor_route_backbone_parameters(self) -> list[nn.Parameter]:
        """Return backbone parameters directly serving the route branch."""
        if self._uses_separate_theta_route_backbones():
            if self.actor_route_backbone is None:
                return []
            return list(self.actor_route_backbone.parameters())
        if self.actor_backbone is None:
            return []
        return list(self.actor_backbone.parameters())

    def actor_route_head_parameters(self) -> list[nn.Parameter]:
        """Return parameters belonging to the route decision head stack."""
        if not self._uses_hierarchical_actor_theta_route():
            return []
        route_parameters: list[nn.Parameter] = []
        if self.actor_route_context is not None:
            route_parameters.extend(list(self.actor_route_context.parameters()))
        if self.actor_route_head is not None:
            route_parameters.extend(list(self.actor_route_head.parameters()))
        return route_parameters

    def describe_actor_structure(self) -> dict[str, object]:
        """Describe the active actor output structure for experiment logging."""
        if self._uses_separate_theta_route_backbones():
            return {
                "actor_structure_mode": self.actor_structure_mode,
                "theta_backbone": "two_layer_tanh_backbone",
                "route_backbone": "two_layer_tanh_backbone",
                "theta_head": {
                    "type": "linear",
                    "input": "theta_backbone_features",
                    "output": f"{self.actor_block_count} raw theta logits",
                },
                "route_head": {
                    "type": "mlp_then_linear",
                    "input": "route_backbone_features + detached sigmoid(theta_mean)",
                    "output": (
                        f"{self.actor_block_count} x {self.actor_route_dim} route logits"
                    ),
                },
                "route_uses_detached_theta_context": self.route_uses_detached_theta_context,
                "backbones_separated": True,
            }
        if self._uses_hierarchical_actor_theta_route():
            return {
                "actor_structure_mode": self.actor_structure_mode,
                "shared_backbone": "two_layer_tanh_backbone",
                "theta_head": {
                    "type": "linear",
                    "input": "shared_actor_features",
                    "output": f"{self.actor_block_count} raw theta logits",
                },
                "route_head": {
                    "type": "mlp_then_linear",
                    "input": "shared_actor_features + detached sigmoid(theta_mean)",
                    "output": (
                        f"{self.actor_block_count} x {self.actor_route_dim} route logits"
                    ),
                },
                "route_uses_detached_theta_context": self.route_uses_detached_theta_context,
            }
        return {
            "actor_structure_mode": self.actor_structure_mode,
            "shared_backbone": "two_layer_tanh_backbone",
            "joint_head": {
                "type": "linear",
                "input": "shared_actor_features",
                "output": f"{self.action_dim} flat action means",
            },
        }

    def normalized_value(self, state: torch.Tensor) -> torch.Tensor:
        """Output critic predictions in normalized target space."""
        return self.normalized_value_from_critic_input(state)

    def normalized_value_from_critic_input(self, critic_input: torch.Tensor) -> torch.Tensor:
        """Output critic predictions from preprocessed critic-only inputs."""
        return self.critic_head(self.critic_features_from_critic_input(critic_input))

    def critic_features_from_critic_input(self, critic_input: torch.Tensor) -> torch.Tensor:
        """Expose critic-backbone features for lightweight auxiliary heads."""
        return self.critic_backbone(critic_input)

    def block_value_scores_from_critic_input(self, critic_input: torch.Tensor) -> torch.Tensor:
        """Predict per-block value relevance scores from critic features."""
        features = self.critic_features_from_critic_input(critic_input)
        return self.block_value_scores_from_features(features)

    def block_value_scores_from_features(self, critic_features: torch.Tensor) -> torch.Tensor:
        """Predict per-block value relevance scores from precomputed critic features."""
        return self.critic_block_value_head(critic_features)

    def block_path_value_scores_from_critic_input(self, critic_input: torch.Tensor) -> torch.Tensor:
        """Predict per-block per-path value scores from critic features."""
        features = self.critic_features_from_critic_input(critic_input)
        return self.block_path_value_scores_from_features(features)

    def block_path_value_scores_from_features(self, critic_features: torch.Tensor) -> torch.Tensor:
        """Predict per-block per-path value scores from precomputed critic features."""
        flat_scores = self.critic_block_path_value_head(critic_features)
        return flat_scores.reshape(
            -1,
            self.critic_block_value_count,
            self.critic_block_path_count,
        )

    def value(
        self,
        state: torch.Tensor,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Output state values, optionally mapped back to raw scale."""
        normalized = self.value_from_critic_input(state)
        if mean is None or std is None:
            return normalized
        return normalized * std + mean

    def value_from_critic_input(
        self,
        critic_input: torch.Tensor,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Output values from preprocessed critic-only inputs."""
        normalized = self.normalized_value_from_critic_input(critic_input)
        if mean is None or std is None:
            return normalized
        return normalized * std + mean

    def popart_rescale(
        self,
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        new_mean: torch.Tensor,
        new_std: torch.Tensor,
    ) -> None:
        """Rescale the critic head so raw predictions stay continuous after stats updates."""
        with torch.no_grad():
            head_weight = self.critic_head.weight
            head_bias = self.critic_head.bias
            old_mean = torch.as_tensor(old_mean, dtype=head_weight.dtype, device=head_weight.device)
            old_std = torch.as_tensor(old_std, dtype=head_weight.dtype, device=head_weight.device)
            new_mean = torch.as_tensor(new_mean, dtype=head_weight.dtype, device=head_weight.device)
            new_std = torch.as_tensor(new_std, dtype=head_weight.dtype, device=head_weight.device)

            scale = old_std / new_std
            head_weight.mul_(scale)
            head_bias.mul_(scale)
            head_bias.add_((old_mean - new_mean) / new_std)

    def forward(self, state: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        """Return both the policy distribution and the raw value estimate."""
        return self.policy(state), self.value(state)
