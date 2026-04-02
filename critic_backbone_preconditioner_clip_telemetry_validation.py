import argparse
import json
import math
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

from dense_actor_input_experiment import build_state_layout
from dense_critic_diagnosis import (
    PROBE_STATE_COUNT,
    collect_probe_states,
    dataframe_to_markdown,
    extract_metrics,
    plot_single_run_outputs,
    train_with_diagnosis,
)
from dense_policy_joint_reward_aligned_credit_experiment import (
    JOINT_REWARD_ALIGNED_MODE_DEFINITIONS,
)
from dense_policy_ratio_mode_experiment import (
    FIXED_CRITIC_LEARNING_RATE,
    build_action_block_layout,
    build_policy_ratio_mode_config,
    compute_bucket_slope,
    plot_two_panel_compare,
)
from dense_policy_true_conditional_route_experiment import (
    FIXED_MIN_DELTA,
    FIXED_PATIENCE,
    FIXED_SEED,
    FIXED_START_CHECK_EPOCH,
)
from fine_grained_epoch_early_stop_search import EarlyStoppingConfig, set_global_seeds
from rl.ppo_agent import PPOAgent


POLICY_RATIO_MODE = "hierarchical_actor_joint_reward_aligned_credit"
ACTIVE_GRAD_THRESHOLD = 1e-12
DEFAULT_PRECONDITIONER_CAP = 5e3
TRACKED_HOTSPOT_LAYERS = (
    "critic_backbone.0.weight",
    "critic_backbone.3.weight",
    "critic_backbone.6.weight",
    "critic_backbone.7.weight",
    "critic_backbone.7.bias",
    "critic_backbone.4.bias",
)
LN_FOCUS_CLIP_LAYERS = (
    "critic_backbone.7.weight",
    "critic_backbone.7.bias",
    "critic_backbone.4.bias",
)
DEFAULT_GROUP_SPECS: dict[str, dict[str, Any]] = {
    "baseline_critic_adam": {
        "backbone_preconditioner_cap": None,
        "clip_focus_parameter_names": None,
        "description": (
            "Backbone and head both keep the current Adam rule. Split only for "
            "telemetry and exact backbone/head targeting."
        ),
    },
    "critic_backbone_preconditioner_clip_head_adam": {
        "backbone_preconditioner_cap": DEFAULT_PRECONDITIONER_CAP,
        "clip_focus_parameter_names": None,
        "description": (
            "Keep Adam on both sides, but cap active critic_backbone preconditioner "
            "multipliers across the full backbone."
        ),
    },
    "critic_backbone_preconditioner_clip_ln_focus_head_adam": {
        "backbone_preconditioner_cap": DEFAULT_PRECONDITIONER_CAP,
        "clip_focus_parameter_names": LN_FOCUS_CLIP_LAYERS,
        "description": (
            "Keep Adam on both sides, but cap active critic_backbone preconditioner "
            "multipliers only on the LayerNorm-focused hotspot parameters."
        ),
    },
}
DEFAULT_SELECTED_GROUPS = (
    "baseline_critic_adam",
    "critic_backbone_preconditioner_clip_head_adam",
    "critic_backbone_preconditioner_clip_ln_focus_head_adam",
)
VALIDATION_STEM = "critic_backbone_preconditioner_clip_telemetry_validation"


def _sanitize_layer_name(layer_name: str) -> str:
    return layer_name.replace(".", "_")


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _concat_arrays(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(values, axis=0).astype(np.float32, copy=False)


def _softplus_scalar(value: float) -> float:
    if value > 20.0:
        return float(value)
    if value < -20.0:
        return float(math.exp(value))
    return float(math.log1p(math.exp(value)))


class TelemetryActivePreconditionerAdam(torch.optim.Optimizer):
    """Adam with active preconditioner telemetry and optional clipping."""

    def __init__(
        self,
        named_params: list[tuple[str, torch.nn.Parameter]],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        max_preconditioner: float | None = None,
        layerwise_max_preconditioner_by_name: dict[str, float] | None = None,
        layerwise_soft_reference_preconditioner_by_name: dict[str, float] | None = None,
        active_grad_threshold: float = ACTIVE_GRAD_THRESHOLD,
        clip_focus_parameter_names: tuple[str, ...] | None = None,
        soft_geometry_focus_parameter_names: tuple[str, ...] | None = None,
        soft_geometry_alpha: float = 0.0,
        soft_geometry_beta: float = 8.0,
        tracked_hotspot_layers: tuple[str, ...] = TRACKED_HOTSPOT_LAYERS,
    ) -> None:
        params = [param for _name, param in named_params]
        defaults = {"lr": lr, "betas": betas, "eps": eps}
        super().__init__(params, defaults)
        self.param_name_by_id = {id(param): name for name, param in named_params}
        self.max_preconditioner = (
            float(max_preconditioner) if max_preconditioner is not None else None
        )
        self.layerwise_max_preconditioner_by_name = {
            str(name): float(cap)
            for name, cap in (layerwise_max_preconditioner_by_name or {}).items()
        }
        self.layerwise_soft_reference_preconditioner_by_name = {
            str(name): float(reference)
            for name, reference in (layerwise_soft_reference_preconditioner_by_name or {}).items()
        }
        self.active_grad_threshold = float(active_grad_threshold)
        self.clip_focus_parameter_names = (
            set(clip_focus_parameter_names)
            if clip_focus_parameter_names is not None
            else None
        )
        self.soft_geometry_focus_parameter_names = (
            set(soft_geometry_focus_parameter_names)
            if soft_geometry_focus_parameter_names is not None
            else set(self.layerwise_soft_reference_preconditioner_by_name.keys())
        )
        self.soft_geometry_alpha = float(soft_geometry_alpha)
        self.soft_geometry_beta = float(soft_geometry_beta)
        self.tracked_hotspot_layers = tuple(tracked_hotspot_layers)
        self.current_context = {
            "train_epoch": -1,
            "update_epoch": -1,
            "minibatch_id": -1,
        }
        self.telemetry_step_index = 0
        self.backbone_step_rows: list[dict[str, Any]] = []
        self.layer_step_rows: list[dict[str, Any]] = []
        self.backbone_epoch_storage: dict[int, dict[str, Any]] = {}
        self.layer_epoch_storage: dict[tuple[int, str], dict[str, Any]] = {}

    def set_telemetry_context(
        self,
        train_epoch: int,
        update_epoch: int,
        minibatch_id: int,
    ) -> None:
        self.current_context = {
            "train_epoch": int(train_epoch),
            "update_epoch": int(update_epoch),
            "minibatch_id": int(minibatch_id),
        }

    def _backbone_epoch_bucket(self, epoch: int) -> dict[str, Any]:
        bucket = self.backbone_epoch_storage.get(epoch)
        if bucket is None:
            bucket = {
                "pre_before": [],
                "pre_after": [],
                "active_count": 0,
                "clipped_count": 0,
                "delta_norms": [],
                "soft_scale_factors": [],
                "soft_trigger_active_fractions": [],
                "soft_overflow_stats": [],
            }
            self.backbone_epoch_storage[epoch] = bucket
        return bucket

    def _layer_epoch_bucket(self, epoch: int, layer_name: str) -> dict[str, Any]:
        key = (epoch, layer_name)
        bucket = self.layer_epoch_storage.get(key)
        if bucket is None:
            bucket = {
                "pre_before": [],
                "pre_after": [],
                "active_count": 0,
                "clipped_count": 0,
                "actual_param_delta_norms": [],
                "raw_grad_norms": [],
                "preconditioned_grad_norms": [],
                "soft_scale_factors": [],
                "soft_trigger_active_fractions": [],
                "soft_overflow_stats": [],
            }
            self.layer_epoch_storage[key] = bucket
        return bucket

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = closure() if closure is not None else None
        train_epoch = int(self.current_context.get("train_epoch", -1))
        update_epoch = int(self.current_context.get("update_epoch", -1))
        minibatch_id = int(self.current_context.get("minibatch_id", -1))
        step_index = self.telemetry_step_index
        self.telemetry_step_index += 1

        backbone_pre_before_chunks: list[np.ndarray] = []
        backbone_pre_after_chunks: list[np.ndarray] = []
        backbone_active_count = 0
        backbone_clipped_count = 0
        backbone_delta_norm_sq = 0.0
        backbone_soft_scale_factors: list[float] = []
        backbone_soft_trigger_active_fractions: list[float] = []
        backbone_soft_overflow_stats: list[float] = []

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError(
                        "TelemetryActivePreconditionerAdam does not support sparse gradients."
                    )

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = int(state["step"])

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom_before_clip = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add_(eps)
                preconditioner_before = denom_before_clip.reciprocal()
                active_mask = grad.abs() > self.active_grad_threshold

                param_name = self.param_name_by_id.get(id(param), "")
                param_max_preconditioner = self.layerwise_max_preconditioner_by_name.get(
                    param_name,
                    self.max_preconditioner,
                )
                min_denom = (
                    1.0 / float(param_max_preconditioner)
                    if param_max_preconditioner is not None
                    and float(param_max_preconditioner) > 0.0
                    else None
                )
                clip_target_mask = active_mask
                if self.clip_focus_parameter_names is not None:
                    if param_name in self.clip_focus_parameter_names:
                        clip_target_mask = active_mask
                    else:
                        clip_target_mask = torch.zeros_like(active_mask, dtype=torch.bool)

                denom_after_clip = denom_before_clip
                clipped_mask = torch.zeros_like(active_mask, dtype=torch.bool)
                if min_denom is not None and bool(clip_target_mask.any().item()):
                    clipped_mask = clip_target_mask & (
                        preconditioner_before > float(param_max_preconditioner)
                    )
                    if bool(clipped_mask.any().item()):
                        denom_after_clip = denom_before_clip.clone()
                        denom_after_clip[clipped_mask] = torch.clamp_min(
                            denom_after_clip[clipped_mask],
                            min_denom,
                        )
                preconditioner_after = denom_after_clip.reciprocal()

                step_size = lr / bias_correction1
                param_delta = step_size * exp_avg / denom_after_clip
                layer_soft_reference = self.layerwise_soft_reference_preconditioner_by_name.get(
                    param_name
                )
                layer_soft_scale_factor = 1.0
                layer_soft_trigger_active_fraction = 0.0
                layer_soft_overflow_stat = 0.0
                if (
                    layer_soft_reference is not None
                    and param_name in self.soft_geometry_focus_parameter_names
                    and bool(active_mask.any().item())
                ):
                    active_pre_before_tensor = preconditioner_before[active_mask]
                    layer_active_pre_p95 = float(
                        torch.quantile(active_pre_before_tensor.float(), 0.95).item()
                    )
                    normalized_over_reference = (
                        layer_active_pre_p95 / max(float(layer_soft_reference), 1e-12)
                    ) - 1.0
                    smooth_overflow = _softplus_scalar(
                        self.soft_geometry_beta * normalized_over_reference
                    ) / max(self.soft_geometry_beta, 1e-12)
                    layer_soft_overflow_stat = float(math.log1p(smooth_overflow))
                    layer_soft_scale_factor = float(
                        1.0 / (1.0 + self.soft_geometry_alpha * layer_soft_overflow_stat)
                    )
                    layer_soft_trigger_active_fraction = float(
                        (
                            active_pre_before_tensor > float(layer_soft_reference)
                        ).float().mean().item()
                    )
                    param_delta.mul_(layer_soft_scale_factor)
                    backbone_soft_scale_factors.append(float(layer_soft_scale_factor))
                    backbone_soft_trigger_active_fractions.append(
                        float(layer_soft_trigger_active_fraction)
                    )
                    backbone_soft_overflow_stats.append(float(layer_soft_overflow_stat))
                param.add_(-param_delta)

                if bool(active_mask.any().item()):
                    active_pre_before = (
                        preconditioner_before[active_mask].detach().cpu().float().numpy()
                    )
                    active_pre_after = (
                        preconditioner_after[active_mask].detach().cpu().float().numpy()
                    )
                    backbone_pre_before_chunks.append(active_pre_before)
                    backbone_pre_after_chunks.append(active_pre_after)
                    backbone_active_count += int(active_mask.sum().item())
                    backbone_clipped_count += int(clipped_mask.sum().item())

                backbone_delta_norm_sq += float(param_delta.pow(2).sum().item())

                if param_name in self.tracked_hotspot_layers:
                    layer_active_pre_before = (
                        preconditioner_before[active_mask].detach().cpu().float().numpy()
                        if bool(active_mask.any().item())
                        else np.zeros(0, dtype=np.float32)
                    )
                    layer_active_pre_after = (
                        preconditioner_after[active_mask].detach().cpu().float().numpy()
                        if bool(active_mask.any().item())
                        else np.zeros(0, dtype=np.float32)
                    )
                    layer_active_count = int(active_mask.sum().item())
                    layer_clipped_count = int(clipped_mask.sum().item())
                    layer_actual_param_delta_norm = float(param_delta.norm().item())
                    layer_raw_grad_norm = float(grad.norm().item())
                    layer_preconditioned_grad_norm = float(
                        (grad / denom_after_clip).norm().item()
                    )
                    self.layer_step_rows.append(
                        {
                            "telemetry_step": int(step_index),
                            "train_epoch": int(train_epoch),
                            "update_epoch": int(update_epoch),
                            "minibatch_id": int(minibatch_id),
                            "layer_name": param_name,
                            "layer_active_count": int(layer_active_count),
                            "layer_clipped_active_count": int(layer_clipped_count),
                            "layer_clipped_active_fraction": float(
                                layer_clipped_count / max(1, layer_active_count)
                            ),
                            "layer_active_preconditioner_mean": float(
                                layer_active_pre_before.mean()
                            )
                            if layer_active_count > 0
                            else 0.0,
                            "layer_active_preconditioner_p95": float(
                                np.quantile(layer_active_pre_before, 0.95)
                            )
                            if layer_active_count > 0
                            else 0.0,
                            "layer_active_preconditioner_p99": float(
                                np.quantile(layer_active_pre_before, 0.99)
                            )
                            if layer_active_count > 0
                            else 0.0,
                            "layer_active_preconditioner_max": float(
                                layer_active_pre_before.max()
                            )
                            if layer_active_count > 0
                            else 0.0,
                            "layer_active_preconditioner_mean_after_clip": float(
                                layer_active_pre_after.mean()
                            )
                            if layer_active_count > 0
                            else 0.0,
                            "layer_active_preconditioner_p95_after_clip": float(
                                np.quantile(layer_active_pre_after, 0.95)
                            )
                            if layer_active_count > 0
                            else 0.0,
                            "layer_active_preconditioner_p99_after_clip": float(
                                np.quantile(layer_active_pre_after, 0.99)
                            )
                            if layer_active_count > 0
                            else 0.0,
                            "layer_active_preconditioner_max_after_clip": float(
                                layer_active_pre_after.max()
                            )
                            if layer_active_count > 0
                            else 0.0,
                            "layer_actual_param_delta_norm": float(
                                layer_actual_param_delta_norm
                            ),
                            "layer_raw_grad_norm": float(layer_raw_grad_norm),
                            "layer_preconditioned_grad_norm": float(
                                layer_preconditioned_grad_norm
                            ),
                            "layer_soft_geometry_reference_preconditioner": float(
                                layer_soft_reference or 0.0
                            ),
                            "layer_soft_geometry_scale_factor": float(
                                layer_soft_scale_factor
                            ),
                            "layer_soft_geometry_trigger_active_fraction": float(
                                layer_soft_trigger_active_fraction
                            ),
                            "layer_soft_geometry_overflow_stat": float(
                                layer_soft_overflow_stat
                            ),
                        }
                    )
                    layer_bucket = self._layer_epoch_bucket(train_epoch, param_name)
                    if layer_active_count > 0:
                        layer_bucket["pre_before"].append(layer_active_pre_before)
                        layer_bucket["pre_after"].append(layer_active_pre_after)
                    layer_bucket["active_count"] += int(layer_active_count)
                    layer_bucket["clipped_count"] += int(layer_clipped_count)
                    layer_bucket["actual_param_delta_norms"].append(
                        float(layer_actual_param_delta_norm)
                    )
                    layer_bucket["raw_grad_norms"].append(float(layer_raw_grad_norm))
                    layer_bucket["preconditioned_grad_norms"].append(
                        float(layer_preconditioned_grad_norm)
                    )
                    layer_bucket["soft_scale_factors"].append(float(layer_soft_scale_factor))
                    layer_bucket["soft_trigger_active_fractions"].append(
                        float(layer_soft_trigger_active_fraction)
                    )
                    layer_bucket["soft_overflow_stats"].append(float(layer_soft_overflow_stat))

        backbone_pre_before = _concat_arrays(backbone_pre_before_chunks)
        backbone_pre_after = _concat_arrays(backbone_pre_after_chunks)
        backbone_delta_norm = float(math.sqrt(backbone_delta_norm_sq))
        self.backbone_step_rows.append(
            {
                "telemetry_step": int(step_index),
                "train_epoch": int(train_epoch),
                "update_epoch": int(update_epoch),
                "minibatch_id": int(minibatch_id),
                "final_critic_backbone_delta_norm": float(backbone_delta_norm),
                "backbone_active_count": int(backbone_active_count),
                "backbone_clipped_active_count": int(backbone_clipped_count),
                "backbone_clipped_active_fraction": float(
                    backbone_clipped_count / max(1, backbone_active_count)
                ),
                "backbone_active_preconditioner_mean": float(backbone_pre_before.mean())
                if backbone_active_count > 0
                else 0.0,
                "backbone_active_preconditioner_p95": float(
                    np.quantile(backbone_pre_before, 0.95)
                )
                if backbone_active_count > 0
                else 0.0,
                "backbone_active_preconditioner_p99": float(
                    np.quantile(backbone_pre_before, 0.99)
                )
                if backbone_active_count > 0
                else 0.0,
                "backbone_active_preconditioner_max": float(backbone_pre_before.max())
                if backbone_active_count > 0
                else 0.0,
                "backbone_active_preconditioner_mean_after_clip": float(
                    backbone_pre_after.mean()
                )
                if backbone_active_count > 0
                else 0.0,
                "backbone_active_preconditioner_p95_after_clip": float(
                    np.quantile(backbone_pre_after, 0.95)
                )
                if backbone_active_count > 0
                else 0.0,
                "backbone_active_preconditioner_p99_after_clip": float(
                    np.quantile(backbone_pre_after, 0.99)
                )
                if backbone_active_count > 0
                else 0.0,
                "backbone_active_preconditioner_max_after_clip": float(
                    backbone_pre_after.max()
                )
                if backbone_active_count > 0
                else 0.0,
                "backbone_soft_geometry_scale_factor_mean": _safe_mean(
                    backbone_soft_scale_factors
                )
                if backbone_soft_scale_factors
                else 1.0,
                "backbone_soft_geometry_scale_factor_p50": float(
                    np.quantile(np.asarray(backbone_soft_scale_factors, dtype=np.float32), 0.50)
                )
                if backbone_soft_scale_factors
                else 1.0,
                "backbone_soft_geometry_scale_factor_p95": float(
                    np.quantile(np.asarray(backbone_soft_scale_factors, dtype=np.float32), 0.95)
                )
                if backbone_soft_scale_factors
                else 1.0,
                "backbone_soft_geometry_scale_factor_min": float(
                    min(backbone_soft_scale_factors)
                )
                if backbone_soft_scale_factors
                else 1.0,
                "backbone_soft_geometry_trigger_active_fraction": _safe_mean(
                    backbone_soft_trigger_active_fractions
                ),
                "backbone_soft_geometry_overflow_stat_mean": _safe_mean(
                    backbone_soft_overflow_stats
                ),
                "backbone_soft_geometry_overflow_stat_p95": float(
                    np.quantile(np.asarray(backbone_soft_overflow_stats, dtype=np.float32), 0.95)
                )
                if backbone_soft_overflow_stats
                else 0.0,
            }
        )
        backbone_bucket = self._backbone_epoch_bucket(train_epoch)
        if backbone_active_count > 0:
            backbone_bucket["pre_before"].append(backbone_pre_before)
            backbone_bucket["pre_after"].append(backbone_pre_after)
        backbone_bucket["active_count"] += int(backbone_active_count)
        backbone_bucket["clipped_count"] += int(backbone_clipped_count)
        backbone_bucket["delta_norms"].append(float(backbone_delta_norm))
        backbone_bucket["soft_scale_factors"].extend(backbone_soft_scale_factors)
        backbone_bucket["soft_trigger_active_fractions"].extend(
            backbone_soft_trigger_active_fractions
        )
        backbone_bucket["soft_overflow_stats"].extend(backbone_soft_overflow_stats)
        return loss

    def _build_backbone_epoch_df(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for epoch, bucket in sorted(self.backbone_epoch_storage.items()):
            before = _concat_arrays(bucket["pre_before"])
            after = _concat_arrays(bucket["pre_after"])
            active_count = int(bucket["active_count"])
            clipped_count = int(bucket["clipped_count"])
            rows.append(
                {
                    "epoch": int(epoch),
                    "final_critic_backbone_delta_norm": _safe_mean(
                        bucket["delta_norms"]
                    ),
                    "backbone_active_count": int(active_count),
                    "backbone_clipped_active_count": int(clipped_count),
                    "backbone_clipped_active_fraction": float(
                        clipped_count / max(1, active_count)
                    ),
                    "backbone_active_preconditioner_mean": float(before.mean())
                    if active_count > 0
                    else 0.0,
                    "backbone_active_preconditioner_p95": float(
                        np.quantile(before, 0.95)
                    )
                    if active_count > 0
                    else 0.0,
                    "backbone_active_preconditioner_p99": float(
                        np.quantile(before, 0.99)
                    )
                    if active_count > 0
                    else 0.0,
                    "backbone_active_preconditioner_max": float(before.max())
                    if active_count > 0
                    else 0.0,
                    "backbone_active_preconditioner_mean_after_clip": float(after.mean())
                    if active_count > 0
                    else 0.0,
                    "backbone_active_preconditioner_p95_after_clip": float(
                        np.quantile(after, 0.95)
                    )
                    if active_count > 0
                    else 0.0,
                    "backbone_active_preconditioner_p99_after_clip": float(
                        np.quantile(after, 0.99)
                    )
                    if active_count > 0
                    else 0.0,
                    "backbone_active_preconditioner_max_after_clip": float(after.max())
                    if active_count > 0
                    else 0.0,
                    "backbone_soft_geometry_scale_factor_mean": _safe_mean(
                        bucket["soft_scale_factors"]
                    )
                    if bucket["soft_scale_factors"]
                    else 1.0,
                    "backbone_soft_geometry_scale_factor_p50": float(
                        np.quantile(np.asarray(bucket["soft_scale_factors"], dtype=np.float32), 0.50)
                    )
                    if bucket["soft_scale_factors"]
                    else 1.0,
                    "backbone_soft_geometry_scale_factor_p95": float(
                        np.quantile(np.asarray(bucket["soft_scale_factors"], dtype=np.float32), 0.95)
                    )
                    if bucket["soft_scale_factors"]
                    else 1.0,
                    "backbone_soft_geometry_scale_factor_min": float(
                        min(bucket["soft_scale_factors"])
                    )
                    if bucket["soft_scale_factors"]
                    else 1.0,
                    "backbone_soft_geometry_trigger_active_fraction": _safe_mean(
                        bucket["soft_trigger_active_fractions"]
                    ),
                    "backbone_soft_geometry_overflow_stat_mean": _safe_mean(
                        bucket["soft_overflow_stats"]
                    ),
                    "backbone_soft_geometry_overflow_stat_p95": float(
                        np.quantile(np.asarray(bucket["soft_overflow_stats"], dtype=np.float32), 0.95)
                    )
                    if bucket["soft_overflow_stats"]
                    else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def _build_layer_epoch_df(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for (epoch, layer_name), bucket in sorted(self.layer_epoch_storage.items()):
            before = _concat_arrays(bucket["pre_before"])
            after = _concat_arrays(bucket["pre_after"])
            active_count = int(bucket["active_count"])
            clipped_count = int(bucket["clipped_count"])
            rows.append(
                {
                    "epoch": int(epoch),
                    "layer_name": layer_name,
                    "layer_active_count": int(active_count),
                    "layer_clipped_active_count": int(clipped_count),
                    "layer_clipped_active_fraction": float(
                        clipped_count / max(1, active_count)
                    ),
                    "layer_active_preconditioner_mean": float(before.mean())
                    if active_count > 0
                    else 0.0,
                    "layer_active_preconditioner_p95": float(np.quantile(before, 0.95))
                    if active_count > 0
                    else 0.0,
                    "layer_active_preconditioner_p99": float(np.quantile(before, 0.99))
                    if active_count > 0
                    else 0.0,
                    "layer_active_preconditioner_max": float(before.max())
                    if active_count > 0
                    else 0.0,
                    "layer_active_preconditioner_mean_after_clip": float(after.mean())
                    if active_count > 0
                    else 0.0,
                    "layer_active_preconditioner_p95_after_clip": float(
                        np.quantile(after, 0.95)
                    )
                    if active_count > 0
                    else 0.0,
                    "layer_active_preconditioner_p99_after_clip": float(
                        np.quantile(after, 0.99)
                    )
                    if active_count > 0
                    else 0.0,
                    "layer_active_preconditioner_max_after_clip": float(after.max())
                    if active_count > 0
                    else 0.0,
                    "layer_actual_param_delta_norm": _safe_mean(
                        bucket["actual_param_delta_norms"]
                    ),
                    "layer_raw_grad_norm": _safe_mean(bucket["raw_grad_norms"]),
                    "layer_preconditioned_grad_norm": _safe_mean(
                        bucket["preconditioned_grad_norms"]
                    ),
                    "layer_soft_geometry_scale_factor": _safe_mean(
                        bucket["soft_scale_factors"]
                    )
                    if bucket["soft_scale_factors"]
                    else 1.0,
                    "layer_soft_geometry_scale_factor_min": float(
                        min(bucket["soft_scale_factors"])
                    )
                    if bucket["soft_scale_factors"]
                    else 1.0,
                    "layer_soft_geometry_scale_factor_p50": float(
                        np.quantile(np.asarray(bucket["soft_scale_factors"], dtype=np.float32), 0.50)
                    )
                    if bucket["soft_scale_factors"]
                    else 1.0,
                    "layer_soft_geometry_scale_factor_p95": float(
                        np.quantile(np.asarray(bucket["soft_scale_factors"], dtype=np.float32), 0.95)
                    )
                    if bucket["soft_scale_factors"]
                    else 1.0,
                    "layer_soft_geometry_trigger_active_fraction": _safe_mean(
                        bucket["soft_trigger_active_fractions"]
                    ),
                    "layer_soft_geometry_overflow_stat": _safe_mean(
                        bucket["soft_overflow_stats"]
                    ),
                    "layer_soft_geometry_overflow_stat_p95": float(
                        np.quantile(np.asarray(bucket["soft_overflow_stats"], dtype=np.float32), 0.95)
                    )
                    if bucket["soft_overflow_stats"]
                    else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def finalize_telemetry(self, run_dir: Path) -> dict[str, Any]:
        backbone_step_df = pd.DataFrame(self.backbone_step_rows)
        layer_step_df = pd.DataFrame(self.layer_step_rows)
        backbone_epoch_df = self._build_backbone_epoch_df()
        layer_epoch_df = self._build_layer_epoch_df()

        backbone_step_path = run_dir / "backbone_preconditioner_step_telemetry.csv"
        layer_step_path = run_dir / "hotspot_layer_preconditioner_step_telemetry.csv"
        backbone_epoch_path = run_dir / "backbone_preconditioner_telemetry_by_epoch.csv"
        layer_epoch_path = run_dir / "hotspot_layer_telemetry_by_epoch.csv"
        backbone_step_df.to_csv(backbone_step_path, index=False, encoding="utf-8-sig")
        layer_step_df.to_csv(layer_step_path, index=False, encoding="utf-8-sig")
        backbone_epoch_df.to_csv(backbone_epoch_path, index=False, encoding="utf-8-sig")
        layer_epoch_df.to_csv(layer_epoch_path, index=False, encoding="utf-8-sig")

        final_backbone = (
            backbone_epoch_df.iloc[-1].to_dict() if not backbone_epoch_df.empty else {}
        )
        final_layer_rows = (
            layer_epoch_df[layer_epoch_df["epoch"] == int(layer_epoch_df["epoch"].max())]
            if not layer_epoch_df.empty
            else pd.DataFrame()
        )
        final_layer_summary = {
            str(row.layer_name): {
                key: value
                for key, value in row._asdict().items()
                if key not in {"Index", "layer_name", "epoch"}
            }
            for row in final_layer_rows.itertuples()
        }
        summary_payload = {
            "active_grad_threshold": float(self.active_grad_threshold),
            "max_preconditioner_cap": (
                float(self.max_preconditioner) if self.max_preconditioner is not None else None
            ),
            "layerwise_max_preconditioner_by_name": self.layerwise_max_preconditioner_by_name,
            "layerwise_soft_reference_preconditioner_by_name": (
                self.layerwise_soft_reference_preconditioner_by_name
            ),
            "clip_focus_parameter_names": sorted(self.clip_focus_parameter_names)
            if self.clip_focus_parameter_names is not None
            else [],
            "soft_geometry_focus_parameter_names": sorted(self.soft_geometry_focus_parameter_names),
            "soft_geometry_alpha": float(self.soft_geometry_alpha),
            "soft_geometry_beta": float(self.soft_geometry_beta),
            "tracked_hotspot_layers": list(self.tracked_hotspot_layers),
            "final_backbone_telemetry": final_backbone,
            "final_hotspot_layer_telemetry": final_layer_summary,
            "files": {
                "backbone_step_telemetry": str(backbone_step_path),
                "hotspot_layer_step_telemetry": str(layer_step_path),
                "backbone_epoch_telemetry": str(backbone_epoch_path),
                "hotspot_layer_epoch_telemetry": str(layer_epoch_path),
            },
        }
        summary_path = run_dir / "preconditioner_telemetry_summary.json"
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_payload["summary_path"] = str(summary_path)
        return summary_payload


class SplitCriticOptimizer:
    """Keep backbone/head optimizers separate while delegating telemetry."""

    def __init__(
        self,
        backbone_optimizer: torch.optim.Optimizer,
        head_optimizer: torch.optim.Optimizer,
    ) -> None:
        self.backbone_optimizer = backbone_optimizer
        self.head_optimizer = head_optimizer

    def zero_grad(self) -> None:
        self.backbone_optimizer.zero_grad()
        self.head_optimizer.zero_grad()

    def step(self) -> None:
        self.backbone_optimizer.step()
        self.head_optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "split_optimizer": True,
            "backbone": self.backbone_optimizer.state_dict(),
            "head": self.head_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not isinstance(state_dict, dict) or not state_dict.get("split_optimizer", False):
            raise ValueError("Expected split critic optimizer state dict.")
        self.backbone_optimizer.load_state_dict(state_dict["backbone"])
        self.head_optimizer.load_state_dict(state_dict["head"])

    def set_telemetry_context(
        self,
        train_epoch: int,
        update_epoch: int,
        minibatch_id: int,
    ) -> None:
        if hasattr(self.backbone_optimizer, "set_telemetry_context"):
            self.backbone_optimizer.set_telemetry_context(
                train_epoch=train_epoch,
                update_epoch=update_epoch,
                minibatch_id=minibatch_id,
            )

    def finalize_telemetry(self, run_dir: Path) -> dict[str, Any]:
        if hasattr(self.backbone_optimizer, "finalize_telemetry"):
            return self.backbone_optimizer.finalize_telemetry(run_dir)
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Formal short validation for critic_backbone preconditioner clip telemetry."
        )
    )
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_SELECTED_GROUPS),
        help="Subset of validation groups to run.",
    )
    return parser.parse_args()


def _load_group_logs(summary_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(row.group_name): pd.read_csv(Path(row.run_dir) / "train_logs.csv")
        for row in summary_df.itertuples(index=False)
    }


def _load_group_backbone_epoch_telemetry(
    summary_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    payload: dict[str, pd.DataFrame] = {}
    for row in summary_df.itertuples(index=False):
        path = Path(row.run_dir) / "backbone_preconditioner_telemetry_by_epoch.csv"
        if path.exists():
            payload[str(row.group_name)] = pd.read_csv(path)
    return payload


def _plot_reward_compare(group_logs: dict[str, pd.DataFrame], output_path: Path) -> None:
    plot_two_panel_compare(
        group_logs,
        panel_specs=[
            ("episode_reward", "Episode Reward"),
            ("best_reward_so_far", "Best Reward So Far"),
        ],
        output_path=output_path,
    )


def _plot_advantage_alignment_compare(
    group_logs: dict[str, pd.DataFrame], output_path: Path
) -> None:
    plot_two_panel_compare(
        group_logs,
        panel_specs=[
            ("advantage_action_alignment", "Overall Advantage Alignment"),
            ("theta_advantage_alignment", "Theta Advantage Alignment"),
            ("route_advantage_alignment", "Route Advantage Alignment"),
        ],
        output_path=output_path,
    )


def _plot_critic_health_compare(group_logs: dict[str, pd.DataFrame], output_path: Path) -> None:
    plot_two_panel_compare(
        group_logs,
        panel_specs=[
            ("value_explained_variance", "Value Explained Variance"),
            ("prediction_target_corr", "Prediction Target Corr"),
            ("prediction_std_over_target_std", "Prediction/Target Std Ratio"),
        ],
        output_path=output_path,
    )


def _plot_critic_drift_compare(group_logs: dict[str, pd.DataFrame], output_path: Path) -> None:
    plot_two_panel_compare(
        group_logs,
        panel_specs=[
            ("critic_backbone_grad_norm", "Critic Backbone Grad Norm"),
            ("critic_head_grad_norm", "Critic Head Grad Norm"),
            ("critic_loss", "Critic Loss"),
        ],
        output_path=output_path,
    )


def _plot_backbone_preconditioner_telemetry_compare(
    telemetry_logs: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9, 16), sharex=True)
    panel_specs = [
        ("backbone_active_preconditioner_p95", "Backbone Active Preconditioner P95"),
        ("backbone_active_preconditioner_p99", "Backbone Active Preconditioner P99"),
        ("backbone_clipped_active_fraction", "Backbone Clipped Active Fraction"),
        ("final_critic_backbone_delta_norm", "Backbone Delta Norm"),
    ]
    for axis, (column, title) in zip(axes, panel_specs, strict=True):
        for group_name, df in telemetry_logs.items():
            values = df[column] if column in df.columns else np.zeros(len(df))
            axis.plot(df["epoch"], values, marker="o", label=group_name)
        axis.set_title(title)
        axis.set_ylabel(column)
        axis.legend()
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_hotspot_layer_telemetry_compare(
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    layer_rows: list[dict[str, Any]] = []
    for row in summary_df.itertuples(index=False):
        layer_path = Path(row.run_dir) / "hotspot_layer_telemetry_by_epoch.csv"
        if not layer_path.exists():
            continue
        try:
            layer_df = pd.read_csv(layer_path)
        except pd.errors.EmptyDataError:
            continue
        if layer_df.empty:
            continue
        final_epoch = int(layer_df["epoch"].max())
        final_layer_df = layer_df[layer_df["epoch"] == final_epoch].copy()
        final_layer_df["group_name"] = str(row.group_name)
        layer_rows.extend(final_layer_df.to_dict(orient="records"))
    final_layer_summary = pd.DataFrame(layer_rows)
    if final_layer_summary.empty:
        plt.figure(figsize=(6, 4))
        plt.title("No hotspot telemetry available")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    metrics = [
        ("layer_active_preconditioner_p99", "Layer Active Preconditioner P99"),
        ("layer_clipped_active_fraction", "Layer Clipped Active Fraction"),
        ("layer_actual_param_delta_norm", "Layer Actual Param Delta Norm"),
        ("layer_preconditioned_grad_norm", "Layer Preconditioned Grad Norm"),
    ]
    group_names = list(final_layer_summary["group_name"].unique())
    layer_names = list(final_layer_summary["layer_name"].unique())
    x = np.arange(len(layer_names))
    width = 0.8 / max(1, len(group_names))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4.5 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for axis, (metric_name, title) in zip(axes, metrics, strict=True):
        for group_index, group_name in enumerate(group_names):
            group_df = final_layer_summary[final_layer_summary["group_name"] == group_name]
            value_map = {
                str(layer): float(value)
                for layer, value in zip(
                    group_df["layer_name"],
                    group_df[metric_name],
                    strict=True,
                )
            }
            values = [value_map.get(layer_name, 0.0) for layer_name in layer_names]
            axis.bar(
                x + (group_index - (len(group_names) - 1) / 2.0) * width,
                values,
                width=width,
                label=group_name,
            )
        axis.set_title(title)
        axis.set_ylabel(metric_name)
        axis.legend()
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(layer_names, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_backbone_optimizer(
    named_backbone_params: list[tuple[str, torch.nn.Parameter]],
    group_spec: dict[str, Any],
    tracked_hotspot_layers: tuple[str, ...],
) -> TelemetryActivePreconditionerAdam:
    return TelemetryActivePreconditionerAdam(
        named_params=named_backbone_params,
        lr=FIXED_CRITIC_LEARNING_RATE,
        max_preconditioner=group_spec.get("backbone_preconditioner_cap"),
        layerwise_max_preconditioner_by_name=group_spec.get(
            "layerwise_backbone_preconditioner_caps"
        ),
        active_grad_threshold=ACTIVE_GRAD_THRESHOLD,
        clip_focus_parameter_names=group_spec.get("clip_focus_parameter_names"),
        tracked_hotspot_layers=tracked_hotspot_layers,
    )


def _configure_agent_for_group(
    agent: PPOAgent,
    group_spec: dict[str, Any],
    tracked_hotspot_layers: tuple[str, ...],
) -> None:
    named_backbone_params = [
        (f"critic_backbone.{name}", param)
        for name, param in agent.network.critic_backbone.named_parameters()
    ]
    head_params = (
        list(agent.network.critic_head.parameters())
        + list(agent.network.critic_block_value_head.parameters())
        + list(agent.network.critic_block_path_value_head.parameters())
    )
    backbone_optimizer = _build_backbone_optimizer(
        named_backbone_params,
        group_spec,
        tracked_hotspot_layers,
    )
    head_optimizer = torch.optim.Adam(head_params, lr=FIXED_CRITIC_LEARNING_RATE)
    agent.critic_optimizer = SplitCriticOptimizer(
        backbone_optimizer=backbone_optimizer,
        head_optimizer=head_optimizer,
    )


def _extract_layer_final_metrics(
    telemetry_summary: dict[str, Any],
    layer_name: str,
) -> dict[str, float]:
    layer_payload = telemetry_summary.get("final_hotspot_layer_telemetry", {}).get(layer_name, {})
    return {
        "layer_active_preconditioner_p95": float(
            layer_payload.get("layer_active_preconditioner_p95", 0.0)
        ),
        "layer_active_preconditioner_p99": float(
            layer_payload.get("layer_active_preconditioner_p99", 0.0)
        ),
        "layer_active_preconditioner_p95_after_clip": float(
            layer_payload.get("layer_active_preconditioner_p95_after_clip", 0.0)
        ),
        "layer_active_preconditioner_p99_after_clip": float(
            layer_payload.get("layer_active_preconditioner_p99_after_clip", 0.0)
        ),
        "layer_clipped_active_fraction": float(
            layer_payload.get("layer_clipped_active_fraction", 0.0)
        ),
        "layer_actual_param_delta_norm": float(
            layer_payload.get("layer_actual_param_delta_norm", 0.0)
        ),
        "layer_raw_grad_norm": float(layer_payload.get("layer_raw_grad_norm", 0.0)),
        "layer_preconditioned_grad_norm": float(
            layer_payload.get("layer_preconditioned_grad_norm", 0.0)
        ),
    }


def _build_summary_row(
    run_dir: Path,
    group_name: str,
    group_spec: dict[str, Any],
    telemetry_summary: dict[str, Any],
    tracked_hotspot_layers: tuple[str, ...],
) -> dict[str, Any]:
    analysis_metrics = extract_metrics(run_dir=run_dir)
    logs = pd.read_csv(run_dir / "train_logs.csv")
    final_backbone = telemetry_summary.get("final_backbone_telemetry", {})
    summary_row: dict[str, Any] = {
        "group_name": group_name,
        "policy_ratio_mode": POLICY_RATIO_MODE,
        "best_epoch": int(analysis_metrics["best_epoch"]),
        "best_reward": float(analysis_metrics["best_reward"]),
        "final_reward": float(analysis_metrics["final_reward"]),
        "reward_gap": float(analysis_metrics["reward_gap"]),
        "overall_advantage_action_alignment": float(
            logs.iloc[-1]["advantage_action_alignment"]
        ),
        "theta_advantage_alignment": float(logs.iloc[-1]["theta_advantage_alignment"]),
        "route_advantage_alignment": float(logs.iloc[-1]["route_advantage_alignment"]),
        "value_explained_variance": float(logs.iloc[-1]["value_explained_variance"]),
        "prediction_target_corr": float(logs.iloc[-1]["prediction_target_corr"]),
        "prediction_std_over_target_std": float(
            logs.iloc[-1]["prediction_std_over_target_std"]
        ),
        "target_bucket_prediction_slope": float(compute_bucket_slope(run_dir)),
        "joint_action_decision_agreement_ratio_under_reward_aligned": float(
            logs.iloc[-1]["joint_action_decision_agreement_ratio_under_reward_aligned"]
        ),
        "final_critic_backbone_delta_norm": float(
            final_backbone.get("final_critic_backbone_delta_norm", 0.0)
        ),
        "backbone_active_preconditioner_mean": float(
            final_backbone.get("backbone_active_preconditioner_mean", 0.0)
        ),
        "backbone_active_preconditioner_p95": float(
            final_backbone.get("backbone_active_preconditioner_p95", 0.0)
        ),
        "backbone_active_preconditioner_p99": float(
            final_backbone.get("backbone_active_preconditioner_p99", 0.0)
        ),
        "backbone_active_preconditioner_max": float(
            final_backbone.get("backbone_active_preconditioner_max", 0.0)
        ),
        "backbone_active_preconditioner_mean_after_clip": float(
            final_backbone.get("backbone_active_preconditioner_mean_after_clip", 0.0)
        ),
        "backbone_active_preconditioner_p95_after_clip": float(
            final_backbone.get("backbone_active_preconditioner_p95_after_clip", 0.0)
        ),
        "backbone_active_preconditioner_p99_after_clip": float(
            final_backbone.get("backbone_active_preconditioner_p99_after_clip", 0.0)
        ),
        "backbone_active_preconditioner_max_after_clip": float(
            final_backbone.get("backbone_active_preconditioner_max_after_clip", 0.0)
        ),
        "backbone_clipped_active_fraction": float(
            final_backbone.get("backbone_clipped_active_fraction", 0.0)
        ),
        "backbone_clipped_active_count": float(
            final_backbone.get("backbone_clipped_active_count", 0.0)
        ),
        "critic_backbone_preconditioner_cap": (
            float(group_spec["backbone_preconditioner_cap"])
            if group_spec.get("backbone_preconditioner_cap") is not None
            else 0.0
        ),
        "layerwise_backbone_preconditioner_caps_json": json.dumps(
            group_spec.get("layerwise_backbone_preconditioner_caps", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "critic_backbone_preconditioner_active_threshold": float(ACTIVE_GRAD_THRESHOLD),
        "clip_focus_parameter_names": ",".join(group_spec["clip_focus_parameter_names"])
        if group_spec.get("clip_focus_parameter_names") is not None
        else "all_critic_backbone_parameters",
        "tracked_hotspot_layers": ",".join(tracked_hotspot_layers),
        "critic_head_untouched": True,
        "run_dir": str(run_dir),
    }
    for layer_name in tracked_hotspot_layers:
        layer_metrics = _extract_layer_final_metrics(telemetry_summary, layer_name)
        prefix = _sanitize_layer_name(layer_name)
        for metric_name, metric_value in layer_metrics.items():
            summary_row[f"{prefix}_{metric_name}"] = metric_value

    analysis_metrics.update(summary_row)
    analysis_metrics["preconditioner_telemetry_summary"] = telemetry_summary
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(analysis_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_row


def _apply_config_overrides(
    config: Any,
    training_overrides: dict[str, Any] | None = None,
    ppo_overrides: dict[str, Any] | None = None,
) -> Any:
    training_config = config.training
    ppo_config = config.ppo
    if training_overrides:
        training_config = replace(training_config, **training_overrides)
    if ppo_overrides:
        ppo_config = replace(ppo_config, **ppo_overrides)
    if training_config is config.training and ppo_config is config.ppo:
        return config
    return replace(config, training=training_config, ppo=ppo_config)


def run_experiment(
    seed: int,
    output_root: Path,
    group_names: list[str],
    validation_stem: str = VALIDATION_STEM,
    group_specs: dict[str, dict[str, Any]] = DEFAULT_GROUP_SPECS,
    tracked_hotspot_layers: tuple[str, ...] = TRACKED_HOTSPOT_LAYERS,
    training_overrides: dict[str, Any] | None = None,
    ppo_overrides: dict[str, Any] | None = None,
) -> Path:
    root_dir = output_root / datetime.now().strftime(
        f"{validation_stem}_%Y%m%d_%H%M%S"
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStoppingConfig(
        patience=FIXED_PATIENCE,
        min_delta=FIXED_MIN_DELTA,
        monitor_start_epoch=FIXED_START_CHECK_EPOCH,
    )
    probe_config = build_policy_ratio_mode_config(
        policy_ratio_mode=POLICY_RATIO_MODE,
        seed=seed,
    )
    probe_config = _apply_config_overrides(
        probe_config,
        training_overrides=training_overrides,
        ppo_overrides=ppo_overrides,
    )
    probe_states = collect_probe_states(
        config=probe_config,
        seed=seed,
        probe_count=PROBE_STATE_COUNT,
    )

    selected_groups: list[tuple[str, dict[str, Any]]] = []
    for group_name in group_names:
        if group_name not in group_specs:
            raise ValueError(f"Unknown group name: {group_name}")
        selected_groups.append((group_name, group_specs[group_name]))

    manifest: dict[str, Any] = {
        "root_dir": str(root_dir),
        "seed": int(seed),
        "policy_ratio_mode": POLICY_RATIO_MODE,
        "base_critic_learning_rate": float(FIXED_CRITIC_LEARNING_RATE),
        "probe_state_count": int(PROBE_STATE_COUNT),
        "training_num_epochs": int(probe_config.training.num_epochs),
        "training_time_steps": int(probe_config.training.time_steps),
        "update_epochs": int(probe_config.ppo.update_epochs),
        "training_overrides": dict(training_overrides or {}),
        "ppo_overrides": dict(ppo_overrides or {}),
        "critic_backbone_preconditioner_active_threshold": float(ACTIVE_GRAD_THRESHOLD),
        "tracked_hotspot_layers": list(tracked_hotspot_layers),
        "selected_groups": [group_name for group_name, _group_spec in selected_groups],
        "groups": [],
    }
    summary_rows: list[dict[str, Any]] = []

    for group_name, group_spec in selected_groups:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = root_dir / f"{group_name}_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = build_policy_ratio_mode_config(
            policy_ratio_mode=POLICY_RATIO_MODE,
            seed=seed,
        )
        config = _apply_config_overrides(
            config,
            training_overrides=training_overrides,
            ppo_overrides=ppo_overrides,
        )
        state_layout = build_state_layout(config)
        agent_preview = PPOAgent(
            config=config.ppo,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            critic_state_layout=state_layout,
        )
        _configure_agent_for_group(agent_preview, group_spec, tracked_hotspot_layers)

        (run_dir / "experiment_config.json").write_text(
            json.dumps(
                {
                    "group_name": group_name,
                    "training": asdict(config.training),
                    "ppo": asdict(config.ppo),
                    "dt": asdict(config.dt),
                    "system": asdict(config.system),
                    "critic_state_layout": state_layout,
                    "action_block_layout": build_action_block_layout(config),
                    "actor_structure": agent_preview.describe_actor_structure(),
                    "actor_raw_layout": agent_preview.describe_actor_raw_layout(),
                    "mode_definition": JOINT_REWARD_ALIGNED_MODE_DEFINITIONS[
                        POLICY_RATIO_MODE
                    ],
                    "critic_backbone_preconditioner_active_threshold": float(
                        ACTIVE_GRAD_THRESHOLD
                    ),
                    "critic_backbone_preconditioner_cap": (
                        float(group_spec["backbone_preconditioner_cap"])
                        if group_spec.get("backbone_preconditioner_cap") is not None
                        else None
                    ),
                    "clip_focus_parameter_names": list(
                        group_spec["clip_focus_parameter_names"]
                    )
                    if group_spec.get("clip_focus_parameter_names") is not None
                    else [],
                    "layerwise_backbone_preconditioner_caps": group_spec.get(
                        "layerwise_backbone_preconditioner_caps",
                        {},
                    ),
                    "tracked_hotspot_layers": list(tracked_hotspot_layers),
                    "critic_head_optimizer": {
                        "name": "adam",
                        "learning_rate": float(FIXED_CRITIC_LEARNING_RATE),
                        "untouched": True,
                    },
                    "only_change": group_spec["description"],
                    "early_stopping": asdict(early_stopping),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[train] {group_name} seed={seed} "
            f"cap={group_spec.get('backbone_preconditioner_cap')} "
            f"clip_focus={group_spec.get('clip_focus_parameter_names')} "
            f"layerwise_caps={group_spec.get('layerwise_backbone_preconditioner_caps')} "
            f"-> {run_dir}"
        )
        set_global_seeds(seed)
        agent, _logs, _summary, _payload = train_with_diagnosis(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
            probe_states=probe_states,
            agent_kwargs={"critic_state_layout": state_layout},
            agent_setup_hook=lambda agent_obj, spec=group_spec: _configure_agent_for_group(
                agent_obj,
                spec,
                tracked_hotspot_layers,
            ),
        )
        plot_single_run_outputs(run_dir)
        telemetry_summary = {}
        if hasattr(agent.critic_optimizer, "finalize_telemetry"):
            telemetry_summary = agent.critic_optimizer.finalize_telemetry(run_dir)
        summary_row = _build_summary_row(
            run_dir=run_dir,
            group_name=group_name,
            group_spec=group_spec,
            telemetry_summary=telemetry_summary,
            tracked_hotspot_layers=tracked_hotspot_layers,
        )
        summary_rows.append(summary_row)
        manifest["groups"].append(
            {
                "group_name": group_name,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
                "telemetry_summary": str(run_dir / "preconditioner_telemetry_summary.json"),
                "backbone_preconditioner_cap": (
                    float(group_spec["backbone_preconditioner_cap"])
                    if group_spec.get("backbone_preconditioner_cap") is not None
                    else None
                ),
                "layerwise_backbone_preconditioner_caps": group_spec.get(
                    "layerwise_backbone_preconditioner_caps",
                    {},
                ),
                "clip_focus_parameter_names": list(group_spec["clip_focus_parameter_names"])
                if group_spec.get("clip_focus_parameter_names") is not None
                else [],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = root_dir / f"{validation_stem}_summary.csv"
    summary_json = root_dir / f"{validation_stem}_summary.json"
    summary_md = root_dir / f"{validation_stem}_summary.md"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_columns = [
        "group_name",
        "best_epoch",
        "best_reward",
        "final_reward",
        "reward_gap",
        "overall_advantage_action_alignment",
        "theta_advantage_alignment",
        "route_advantage_alignment",
        "value_explained_variance",
        "prediction_target_corr",
        "prediction_std_over_target_std",
        "target_bucket_prediction_slope",
        "joint_action_decision_agreement_ratio_under_reward_aligned",
        "final_critic_backbone_delta_norm",
        "backbone_active_preconditioner_p95",
        "backbone_active_preconditioner_p99",
        "backbone_clipped_active_fraction",
        "backbone_clipped_active_count",
        "critic_backbone_preconditioner_cap",
        "clip_focus_parameter_names",
    ]
    summary_md.write_text(
        dataframe_to_markdown(summary_df[markdown_columns]),
        encoding="utf-8",
    )

    group_logs = _load_group_logs(summary_df)
    backbone_telemetry_logs = _load_group_backbone_epoch_telemetry(summary_df)
    _plot_reward_compare(group_logs, root_dir / "reward_curve_compare.png")
    _plot_advantage_alignment_compare(
        group_logs,
        root_dir / "advantage_alignment_curve_compare.png",
    )
    _plot_critic_health_compare(group_logs, root_dir / "critic_health_curve_compare.png")
    _plot_critic_drift_compare(group_logs, root_dir / "critic_drift_compare.png")
    _plot_backbone_preconditioner_telemetry_compare(
        backbone_telemetry_logs,
        root_dir / "backbone_preconditioner_telemetry_compare.png",
    )
    _plot_hotspot_layer_telemetry_compare(
        summary_df,
        root_dir / "hotspot_layer_telemetry_compare.png",
    )

    manifest["summary"] = {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "reward_curve_compare": str(root_dir / "reward_curve_compare.png"),
        "advantage_alignment_curve_compare": str(
            root_dir / "advantage_alignment_curve_compare.png"
        ),
        "critic_health_curve_compare": str(root_dir / "critic_health_curve_compare.png"),
        "critic_drift_compare": str(root_dir / "critic_drift_compare.png"),
        "backbone_preconditioner_telemetry_compare": str(
            root_dir / "backbone_preconditioner_telemetry_compare.png"
        ),
        "hotspot_layer_telemetry_compare": str(
            root_dir / "hotspot_layer_telemetry_compare.png"
        ),
    }
    (root_dir / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root_dir


def main() -> None:
    args = parse_args()
    root_dir = run_experiment(
        seed=args.seed,
        output_root=Path(args.output_root),
        group_names=list(args.groups),
    )
    print(
        "Critic backbone preconditioner clip telemetry validation completed. "
        f"Results saved to: {root_dir}"
    )


if __name__ == "__main__":
    main()
