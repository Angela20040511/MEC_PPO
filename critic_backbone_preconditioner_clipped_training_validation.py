import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

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
LN_FOCUS_PARAMETER_NAMES = (
    "critic_backbone.7.weight",
    "critic_backbone.7.bias",
    "critic_backbone.4.bias",
)
DEFAULT_GROUP_SPECS: dict[str, dict[str, Any]] = {
    "baseline_critic_adam": {
        "backbone_preconditioner_cap": None,
        "focus_parameter_names": None,
        "description": (
            "Backbone and head both keep the current Adam rule. Backbone/head are split "
            "only for implementation convenience; with identical Adam hyperparameters this "
            "is equivalent to the baseline optimizer geometry."
        ),
    },
    "critic_backbone_preconditioner_clip_head_adam": {
        "backbone_preconditioner_cap": DEFAULT_PRECONDITIONER_CAP,
        "focus_parameter_names": None,
        "description": (
            "Keep Adam on both critic_backbone and critic_head, but cap only the active "
            "critic_backbone preconditioner multiplier."
        ),
    },
    "critic_backbone_preconditioner_clip_ln_focus_head_adam": {
        "backbone_preconditioner_cap": DEFAULT_PRECONDITIONER_CAP,
        "focus_parameter_names": LN_FOCUS_PARAMETER_NAMES,
        "description": (
            "Keep Adam on both sides, but apply the same backbone preconditioner cap only "
            "to the hottest LayerNorm-focused parameters."
        ),
    },
}
DEFAULT_SELECTED_GROUPS = (
    "baseline_critic_adam",
    "critic_backbone_preconditioner_clip_head_adam",
    "critic_backbone_preconditioner_clip_ln_focus_head_adam",
)


class ActivePreconditionerCappedAdam(torch.optim.Optimizer):
    """Adam with optional clipping on active per-parameter preconditioner multipliers."""

    def __init__(
        self,
        named_params: list[tuple[str, torch.nn.Parameter]],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        max_preconditioner: float | None = None,
        active_grad_threshold: float = ACTIVE_GRAD_THRESHOLD,
        focus_parameter_names: tuple[str, ...] | None = None,
    ) -> None:
        params = [param for _name, param in named_params]
        defaults = {"lr": lr, "betas": betas, "eps": eps}
        super().__init__(params, defaults)
        self.param_name_by_id = {id(param): name for name, param in named_params}
        self.max_preconditioner = (
            float(max_preconditioner) if max_preconditioner is not None else None
        )
        self.active_grad_threshold = float(active_grad_threshold)
        self.focus_parameter_names = (
            set(focus_parameter_names) if focus_parameter_names is not None else None
        )
        self._last_step_metrics = self._empty_metrics()

    def _empty_metrics(self) -> dict[str, float]:
        return {
            "active_count": 0.0,
            "clipped_count": 0.0,
            "active_preconditioner_sum_before_clip": 0.0,
            "active_preconditioner_sum_after_clip": 0.0,
            "active_preconditioner_max_before_clip": 0.0,
            "active_preconditioner_max_after_clip": 0.0,
            "max_preconditioner": float(self.max_preconditioner or 0.0),
        }

    def consume_debug_metrics(self) -> dict[str, float]:
        metrics = dict(self._last_step_metrics)
        self._last_step_metrics = self._empty_metrics()
        return metrics

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = closure() if closure is not None else None
        metrics = self._empty_metrics()
        min_denom = (
            1.0 / float(self.max_preconditioner)
            if self.max_preconditioner is not None and self.max_preconditioner > 0.0
            else None
        )

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("ActivePreconditionerCappedAdam does not support sparse gradients.")

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
                denom = exp_avg_sq.sqrt().div(math.sqrt(bias_correction2)).add_(eps)

                param_name = self.param_name_by_id.get(id(param), "")
                use_focus_filter = (
                    self.focus_parameter_names is not None
                    and param_name not in self.focus_parameter_names
                )
                active_mask = grad.abs() > self.active_grad_threshold
                if use_focus_filter:
                    active_mask = torch.zeros_like(active_mask, dtype=torch.bool)

                if bool(active_mask.any().item()):
                    active_preconditioner_before = denom[active_mask].reciprocal()
                    active_count = int(active_preconditioner_before.numel())
                    metrics["active_count"] += float(active_count)
                    metrics["active_preconditioner_sum_before_clip"] += float(
                        active_preconditioner_before.sum().item()
                    )
                    metrics["active_preconditioner_max_before_clip"] = max(
                        metrics["active_preconditioner_max_before_clip"],
                        float(active_preconditioner_before.max().item()),
                    )

                    if min_denom is not None:
                        clipped_mask = active_preconditioner_before > float(
                            self.max_preconditioner
                        )
                        metrics["clipped_count"] += float(clipped_mask.sum().item())
                        clipped_active_denom = torch.clamp_min(
                            denom[active_mask],
                            min_denom,
                        )
                        denom = denom.clone()
                        denom[active_mask] = clipped_active_denom
                    active_preconditioner_after = denom[active_mask].reciprocal()
                    metrics["active_preconditioner_sum_after_clip"] += float(
                        active_preconditioner_after.sum().item()
                    )
                    metrics["active_preconditioner_max_after_clip"] = max(
                        metrics["active_preconditioner_max_after_clip"],
                        float(active_preconditioner_after.max().item()),
                    )

                step_size = lr / bias_correction1
                param.addcdiv_(exp_avg, denom, value=-step_size)

        self._last_step_metrics = metrics
        return loss


class SplitCriticOptimizer:
    """Keep critic_backbone and critic heads on separate optimizers."""

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

    def consume_debug_metrics(self) -> dict[str, float]:
        if hasattr(self.backbone_optimizer, "consume_debug_metrics"):
            return self.backbone_optimizer.consume_debug_metrics()
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Short formal validation for clipping only critic_backbone Adam "
            "preconditioner geometry on the joint reward-aligned mainline."
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
            ("critic_backbone_delta_norm", "Critic Backbone Delta Norm"),
            (
                "critic_param_delta_norm_after_optimizer_step",
                "Critic Param Delta Norm After Step",
            ),
            (
                "critic_backbone_preconditioner_clip_fraction",
                "Backbone Preconditioner Clip Fraction",
            ),
            (
                "critic_backbone_preconditioner_active_max_after_clip",
                "Backbone Active Preconditioner Max After Clip",
            ),
        ],
        output_path=output_path,
    )


def _build_backbone_optimizer(
    named_backbone_params: list[tuple[str, torch.nn.Parameter]],
    group_spec: dict[str, Any],
) -> ActivePreconditionerCappedAdam:
    return ActivePreconditionerCappedAdam(
        named_params=named_backbone_params,
        lr=FIXED_CRITIC_LEARNING_RATE,
        max_preconditioner=group_spec.get("backbone_preconditioner_cap"),
        active_grad_threshold=ACTIVE_GRAD_THRESHOLD,
        focus_parameter_names=group_spec.get("focus_parameter_names"),
    )


def _configure_agent_for_group(agent: PPOAgent, group_spec: dict[str, Any]) -> None:
    named_backbone_params = list(agent.network.critic_backbone.named_parameters())
    head_params = (
        list(agent.network.critic_head.parameters())
        + list(agent.network.critic_block_value_head.parameters())
        + list(agent.network.critic_block_path_value_head.parameters())
    )
    backbone_optimizer = _build_backbone_optimizer(named_backbone_params, group_spec)
    head_optimizer = torch.optim.Adam(head_params, lr=FIXED_CRITIC_LEARNING_RATE)
    agent.critic_optimizer = SplitCriticOptimizer(
        backbone_optimizer=backbone_optimizer,
        head_optimizer=head_optimizer,
    )


def _final_row_value(logs: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if logs.empty or column not in logs.columns:
        return float(default)
    return float(logs.iloc[-1][column])


def _focus_parameter_string(group_spec: dict[str, Any]) -> str:
    focus_parameter_names = group_spec.get("focus_parameter_names")
    if not focus_parameter_names:
        return "all_critic_backbone_parameters"
    return ",".join(str(name) for name in focus_parameter_names)


def _build_summary_row(
    run_dir: Path,
    group_name: str,
    group_spec: dict[str, Any],
) -> dict[str, Any]:
    analysis_metrics = extract_metrics(run_dir=run_dir)
    logs = pd.read_csv(run_dir / "train_logs.csv")
    summary_row: dict[str, Any] = {
        "group_name": group_name,
        "policy_ratio_mode": POLICY_RATIO_MODE,
        "best_epoch": int(analysis_metrics["best_epoch"]),
        "best_reward": float(analysis_metrics["best_reward"]),
        "final_reward": float(analysis_metrics["final_reward"]),
        "reward_gap": float(analysis_metrics["reward_gap"]),
        "overall_advantage_action_alignment": _final_row_value(
            logs,
            "advantage_action_alignment",
        ),
        "theta_advantage_alignment": _final_row_value(logs, "theta_advantage_alignment"),
        "route_advantage_alignment": _final_row_value(logs, "route_advantage_alignment"),
        "value_explained_variance": _final_row_value(logs, "value_explained_variance"),
        "prediction_target_corr": _final_row_value(logs, "prediction_target_corr"),
        "prediction_std_over_target_std": _final_row_value(
            logs,
            "prediction_std_over_target_std",
        ),
        "target_bucket_prediction_slope": float(compute_bucket_slope(run_dir)),
        "joint_action_decision_agreement_ratio_under_reward_aligned": _final_row_value(
            logs,
            "joint_action_decision_agreement_ratio_under_reward_aligned",
        ),
        "critic_param_delta_norm_after_optimizer_step": _final_row_value(
            logs,
            "critic_param_delta_norm_after_optimizer_step",
        ),
        "critic_backbone_delta_norm": _final_row_value(logs, "critic_backbone_delta_norm"),
        "critic_head_delta_norm": _final_row_value(logs, "critic_head_delta_norm"),
        "critic_backbone_preconditioner_cap": float(
            group_spec["backbone_preconditioner_cap"]
            if group_spec.get("backbone_preconditioner_cap") is not None
            else 0.0
        ),
        "critic_backbone_preconditioner_active_threshold": float(ACTIVE_GRAD_THRESHOLD),
        "critic_backbone_preconditioner_clip_fraction": _final_row_value(
            logs,
            "critic_backbone_preconditioner_clip_fraction",
        ),
        "critic_backbone_preconditioner_active_mean_before_clip": _final_row_value(
            logs,
            "critic_backbone_preconditioner_active_mean_before_clip",
        ),
        "critic_backbone_preconditioner_active_mean_after_clip": _final_row_value(
            logs,
            "critic_backbone_preconditioner_active_mean_after_clip",
        ),
        "critic_backbone_preconditioner_active_max_before_clip": _final_row_value(
            logs,
            "critic_backbone_preconditioner_active_max_before_clip",
        ),
        "critic_backbone_preconditioner_active_max_after_clip": _final_row_value(
            logs,
            "critic_backbone_preconditioner_active_max_after_clip",
        ),
        "focus_parameter_names": _focus_parameter_string(group_spec),
        "critic_head_untouched": True,
        "run_dir": str(run_dir),
    }
    analysis_metrics.update(summary_row)
    (run_dir / "analysis_summary.json").write_text(
        json.dumps(analysis_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_row


def run_experiment(seed: int, output_root: Path, group_names: list[str]) -> Path:
    root_dir = output_root / datetime.now().strftime(
        "critic_backbone_preconditioner_clipped_training_validation_%Y%m%d_%H%M%S"
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
    probe_states = collect_probe_states(
        config=probe_config,
        seed=seed,
        probe_count=PROBE_STATE_COUNT,
    )

    selected_groups: list[tuple[str, dict[str, Any]]] = []
    for group_name in group_names:
        if group_name not in DEFAULT_GROUP_SPECS:
            raise ValueError(f"Unknown group name: {group_name}")
        selected_groups.append((group_name, DEFAULT_GROUP_SPECS[group_name]))

    manifest: dict[str, Any] = {
        "root_dir": str(root_dir),
        "seed": int(seed),
        "policy_ratio_mode": POLICY_RATIO_MODE,
        "base_critic_learning_rate": float(FIXED_CRITIC_LEARNING_RATE),
        "probe_state_count": int(PROBE_STATE_COUNT),
        "training_num_epochs": int(probe_config.training.num_epochs),
        "training_time_steps": int(probe_config.training.time_steps),
        "update_epochs": int(probe_config.ppo.update_epochs),
        "critic_backbone_preconditioner_active_threshold": float(ACTIVE_GRAD_THRESHOLD),
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
        state_layout = build_state_layout(config)
        agent_preview = PPOAgent(
            config=config.ppo,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            critic_state_layout=state_layout,
        )
        _configure_agent_for_group(agent_preview, group_spec)

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
                    "critic_backbone_preconditioner_focus_parameter_names": list(
                        group_spec["focus_parameter_names"]
                    )
                    if group_spec.get("focus_parameter_names") is not None
                    else [],
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
            f"backbone_cap={group_spec.get('backbone_preconditioner_cap')} "
            f"focus={_focus_parameter_string(group_spec)} -> {run_dir}"
        )
        set_global_seeds(seed)
        train_with_diagnosis(
            config=config,
            checkpoint_dir=str(run_dir),
            early_stopping=early_stopping,
            probe_states=probe_states,
            agent_kwargs={"critic_state_layout": state_layout},
            agent_setup_hook=lambda agent, spec=group_spec: _configure_agent_for_group(
                agent,
                spec,
            ),
        )

        plot_single_run_outputs(run_dir)
        summary_row = _build_summary_row(
            run_dir=run_dir,
            group_name=group_name,
            group_spec=group_spec,
        )
        summary_rows.append(summary_row)
        manifest["groups"].append(
            {
                "group_name": group_name,
                "run_dir": str(run_dir),
                "analysis_summary": str(run_dir / "analysis_summary.json"),
                "backbone_preconditioner_cap": (
                    float(group_spec["backbone_preconditioner_cap"])
                    if group_spec.get("backbone_preconditioner_cap") is not None
                    else None
                ),
                "focus_parameter_names": list(group_spec["focus_parameter_names"])
                if group_spec.get("focus_parameter_names") is not None
                else [],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = (
        root_dir / "critic_backbone_preconditioner_clipped_training_validation_summary.csv"
    )
    summary_json = (
        root_dir / "critic_backbone_preconditioner_clipped_training_validation_summary.json"
    )
    summary_md = (
        root_dir / "critic_backbone_preconditioner_clipped_training_validation_summary.md"
    )
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md.write_text(dataframe_to_markdown(summary_df), encoding="utf-8")

    group_logs = _load_group_logs(summary_df)
    _plot_reward_compare(group_logs, root_dir / "reward_curve_compare.png")
    _plot_advantage_alignment_compare(
        group_logs,
        root_dir / "advantage_alignment_curve_compare.png",
    )
    _plot_critic_health_compare(group_logs, root_dir / "critic_health_curve_compare.png")
    _plot_critic_drift_compare(group_logs, root_dir / "critic_drift_compare.png")

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
        "Critic backbone preconditioner clipped training validation completed. "
        f"Results saved to: {root_dir}"
    )


if __name__ == "__main__":
    main()
