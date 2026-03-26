import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from critic_gradient_alignment_audit import (
    _build_global_payload,
    _collect_gradients,
    _main_value_objective_from_payload,
)
from critic_loss_component_ablation import (
    _build_payloads_from_capture_and_rollout,
    _evaluate_all_sets,
    _flatten_tensors,
    _parameter_list_delta_norm,
    _semantic_metrics_from_payload,
)
from critic_optimizer_rule_ablation import _collect_single_rollout_epoch
from dense_actor_input_experiment import build_state_layout
from dense_policy_ratio_mode_experiment import build_policy_ratio_mode_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from joint_training_critic_drift_trace import (
    MODE,
    _collect_fixed_probe_payload,
    _find_current_mainline_checkpoint,
)
from rl.ppo_agent import PPOAgent, CriticOptimizerAblationCaptured
from simulator.simulator import Simulator


DEFAULT_EPOCH = 0
DEFAULT_UPDATE_EPOCH = 2
DEFAULT_MINIBATCH_ID = 8
DEFAULT_NORMS = "1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan matched pure-gradient update norms on the same captured bad-step minibatch "
            "to estimate the semantic stability radius."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument("--update-epoch", type=int, default=DEFAULT_UPDATE_EPOCH)
    parser.add_argument("--minibatch-id", type=int, default=DEFAULT_MINIBATCH_ID)
    parser.add_argument("--target-update-norms", type=str, default=DEFAULT_NORMS)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    return parser.parse_args()


def _parse_norms(norms: str) -> list[float]:
    values = []
    for piece in norms.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(float(piece))
    if not values:
        raise ValueError("At least one target update norm is required.")
    return values


def _apply_small_matched_update(
    base_agent: PPOAgent,
    grads: list[torch.Tensor | None],
    target_update_norm: float,
) -> tuple[PPOAgent, dict[str, float]]:
    updated_agent = copy.deepcopy(base_agent)
    before_params = [parameter.detach().clone() for parameter in base_agent.critic_params]
    before_backbone = [
        parameter.detach().clone() for parameter in base_agent.network.critic_backbone.parameters()
    ]
    before_head = [
        parameter.detach().clone() for parameter in base_agent.network.critic_head.parameters()
    ]

    grad_vector = _flatten_tensors(grads)
    grad_norm = float(grad_vector.norm().item())
    scale = 0.0 if grad_norm <= 1e-12 else float(target_update_norm) / grad_norm
    with torch.no_grad():
        for parameter, grad in zip(updated_agent.critic_params, grads):
            if grad is None:
                continue
            parameter.add_(grad, alpha=-scale)

    after_params = [parameter.detach().clone() for parameter in updated_agent.critic_params]
    after_backbone = [
        parameter.detach().clone()
        for parameter in updated_agent.network.critic_backbone.parameters()
    ]
    after_head = [
        parameter.detach().clone() for parameter in updated_agent.network.critic_head.parameters()
    ]
    delta_metrics = {
        "critic_param_delta_norm": _parameter_list_delta_norm(before_params, after_params),
        "critic_backbone_delta_norm": _parameter_list_delta_norm(before_backbone, after_backbone),
        "critic_head_delta_norm": _parameter_list_delta_norm(before_head, after_head),
        "update_over_grad_ratio": float(
            _parameter_list_delta_norm(before_params, after_params) / max(grad_norm, 1e-12)
        ),
        "scale_factor": float(scale),
        "grad_norm": float(grad_norm),
    }
    return updated_agent, delta_metrics


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Critic Update Radius Threshold Audit",
        "",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        (
            f"- captured minibatch: epoch `{summary['target_epoch']}` / update_epoch "
            f"`{summary['target_update_epoch']}` / minibatch `{summary['target_minibatch_id']}`"
        ),
        f"- target_update_norms: `{summary['target_update_norms']}`",
        f"- interpretation: {summary['interpretation']}",
        "",
        "## Gradient Alignment",
        "",
        "```json",
        json.dumps(summary["gradient_alignment"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Radius Thresholds",
        "",
        "```json",
        json.dumps(summary["radius_thresholds"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    target_update_norms = _parse_norms(args.target_update_norms)
    output_root = Path(args.output_root)
    root_dir = output_root / datetime.now().strftime(
        "critic_update_radius_threshold_audit_%Y%m%d_%H%M%S"
    )
    root_dir.mkdir(parents=True, exist_ok=True)

    set_global_seeds(args.seed)
    config = build_policy_ratio_mode_config(policy_ratio_mode=MODE, seed=args.seed)
    checkpoint_path = _find_current_mainline_checkpoint(output_root)
    state_layout = build_state_layout(config)
    agent = PPOAgent(
        config=config.ppo,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        critic_state_layout=state_layout,
    )
    agent.load(str(checkpoint_path), load_optimizer=True)
    agent.freeze_actor_training_updates = True

    probe_payload = _collect_fixed_probe_payload(agent, config, args.seed)
    simulator = Simulator(config)
    _collect_single_rollout_epoch(agent, simulator, config, epoch=args.epoch)
    full_rollout = agent.buffer.as_tensors(agent.device)
    full_rollout["rewards"] = torch.tensor(
        agent.buffer.rewards,
        dtype=torch.float32,
        device=agent.device,
    )

    agent.critic_optimizer_ablation_request = {
        "epoch": int(args.epoch),
        "update_epoch": int(args.update_epoch),
        "minibatch_id": int(args.minibatch_id),
    }
    try:
        agent.train()
        raise RuntimeError("Expected critic radius-threshold capture did not trigger.")
    except CriticOptimizerAblationCaptured:
        capture = agent.critic_optimizer_ablation_capture

    payloads = _build_payloads_from_capture_and_rollout(
        agent=agent,
        capture=capture,
        full_data=full_rollout,
        probe_payload=probe_payload,
        seed=args.seed,
    )
    payloads["global_batch"] = _build_global_payload(full_rollout)

    before_sets = _evaluate_all_sets(
        agent,
        {
            "current_batch": payloads["current_batch"],
            "held_out_batch": payloads["held_out_batch"],
            "probe": payloads["probe"],
        },
        capture["target_mean"],
        capture["target_std"],
    )
    global_before = _semantic_metrics_from_payload(
        agent,
        payloads["global_batch"],
        capture["target_mean"],
        capture["target_std"],
    )

    base_agent = copy.deepcopy(agent)
    gradient_groups = {
        "current_grad": payloads["current_batch"],
        "global_grad": payloads["global_batch"],
        "blended_grad": None,
    }
    gradient_results = {
        "current_grad": _collect_gradients(
            base_agent,
            payloads["current_batch"],
            capture["target_mean"],
            capture["target_std"],
        ),
        "global_grad": _collect_gradients(
            base_agent,
            payloads["global_batch"],
            capture["target_mean"],
            capture["target_std"],
        ),
    }

    blended_agent = copy.deepcopy(base_agent)
    blended_agent.critic_optimizer.zero_grad()
    current_objective, _ = _main_value_objective_from_payload(
        blended_agent,
        payloads["current_batch"],
        capture["target_mean"],
        capture["target_std"],
    )
    heldout_objective, _ = _main_value_objective_from_payload(
        blended_agent,
        payloads["held_out_batch"],
        capture["target_mean"],
        capture["target_std"],
    )
    blended_objective = 0.5 * current_objective + 0.5 * heldout_objective
    blended_objective.backward()
    blended_grads = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in blended_agent.critic_params
    ]
    gradient_results["blended_grad"] = {
        "grads": blended_grads,
        "grad_vector": _flatten_tensors(blended_grads),
        "raw_grad_norm": float(blended_agent._grad_norm(blended_agent.critic_params)),
    }

    gradient_alignment = {
        "current_vs_global_cosine": float(
            torch.dot(
                gradient_results["current_grad"]["grad_vector"],
                gradient_results["global_grad"]["grad_vector"],
            ).item()
            / max(
                float(gradient_results["current_grad"]["grad_vector"].norm().item())
                * float(gradient_results["global_grad"]["grad_vector"].norm().item()),
                1e-12,
            )
        ),
        "current_vs_blended_cosine": float(
            torch.dot(
                gradient_results["current_grad"]["grad_vector"],
                gradient_results["blended_grad"]["grad_vector"],
            ).item()
            / max(
                float(gradient_results["current_grad"]["grad_vector"].norm().item())
                * float(gradient_results["blended_grad"]["grad_vector"].norm().item()),
                1e-12,
            )
        ),
        "global_vs_blended_cosine": float(
            torch.dot(
                gradient_results["global_grad"]["grad_vector"],
                gradient_results["blended_grad"]["grad_vector"],
            ).item()
            / max(
                float(gradient_results["global_grad"]["grad_vector"].norm().item())
                * float(gradient_results["blended_grad"]["grad_vector"].norm().item()),
                1e-12,
            )
        ),
        "grad_norm_current": gradient_results["current_grad"]["raw_grad_norm"],
        "grad_norm_global": gradient_results["global_grad"]["raw_grad_norm"],
        "grad_norm_blended": gradient_results["blended_grad"]["raw_grad_norm"],
    }

    metrics_rows: list[dict[str, Any]] = []
    extended_before_sets = {
        "current_batch": before_sets["current_batch"],
        "held_out_batch": before_sets["held_out_batch"],
        "probe": before_sets["probe"],
        "global_batch": global_before,
    }

    def _append_rows_for_group(
        group_name: str,
        update_direction: str,
        target_update_norm: float,
        step_type: str,
        set_metrics: dict[str, dict[str, float]],
        grad_norm: float,
        delta_metrics: dict[str, float],
    ) -> None:
        for set_type in ("current_batch", "held_out_batch", "probe", "global_batch"):
            metrics = set_metrics[set_type]
            metrics_rows.append(
                {
                    "group_name": group_name,
                    "update_direction": update_direction,
                    "target_update_norm": float(target_update_norm),
                    "step_type": step_type,
                    "set_type": set_type,
                    **metrics,
                    "grad_norm": float(grad_norm),
                    **delta_metrics,
                }
            )

    _append_rows_for_group(
        "no_op",
        "no_op",
        0.0,
        "before_update",
        extended_before_sets,
        0.0,
        {
            "critic_param_delta_norm": 0.0,
            "critic_backbone_delta_norm": 0.0,
            "critic_head_delta_norm": 0.0,
            "update_over_grad_ratio": 0.0,
            "scale_factor": 0.0,
            "cosine_update_vs_current_grad": 0.0,
            "cosine_update_vs_global_grad": 0.0,
        },
    )
    _append_rows_for_group(
        "no_op",
        "no_op",
        0.0,
        "after_update",
        extended_before_sets,
        0.0,
        {
            "critic_param_delta_norm": 0.0,
            "critic_backbone_delta_norm": 0.0,
            "critic_head_delta_norm": 0.0,
            "update_over_grad_ratio": 0.0,
            "scale_factor": 0.0,
            "cosine_update_vs_current_grad": 0.0,
            "cosine_update_vs_global_grad": 0.0,
        },
    )

    for direction_name in ("current_grad", "global_grad", "blended_grad"):
        grad_bundle = gradient_results[direction_name]
        for target_update_norm in target_update_norms:
            updated_agent, delta_metrics = _apply_small_matched_update(
                base_agent,
                grad_bundle["grads"],
                float(target_update_norm),
            )
            after_sets = _evaluate_all_sets(
                updated_agent,
                {
                    "current_batch": payloads["current_batch"],
                    "held_out_batch": payloads["held_out_batch"],
                    "probe": payloads["probe"],
                },
                capture["target_mean"],
                capture["target_std"],
            )
            after_global = _semantic_metrics_from_payload(
                updated_agent,
                payloads["global_batch"],
                capture["target_mean"],
                capture["target_std"],
            )
            extended_after_sets = {
                "current_batch": after_sets["current_batch"],
                "held_out_batch": after_sets["held_out_batch"],
                "probe": after_sets["probe"],
                "global_batch": after_global,
            }
            row_name = f"{direction_name}_norm_{target_update_norm:.0e}"
            extra = {
                "cosine_update_vs_current_grad": 1.0
                if direction_name == "current_grad"
                else float(
                    torch.dot(
                        grad_bundle["grad_vector"],
                        gradient_results["current_grad"]["grad_vector"],
                    ).item()
                    / max(
                        float(grad_bundle["grad_vector"].norm().item())
                        * float(gradient_results["current_grad"]["grad_vector"].norm().item()),
                        1e-12,
                    )
                ),
                "cosine_update_vs_global_grad": 1.0
                if direction_name == "global_grad"
                else float(
                    torch.dot(
                        grad_bundle["grad_vector"],
                        gradient_results["global_grad"]["grad_vector"],
                    ).item()
                    / max(
                        float(grad_bundle["grad_vector"].norm().item())
                        * float(gradient_results["global_grad"]["grad_vector"].norm().item()),
                        1e-12,
                    )
                ),
            }
            zero_delta = {
                "critic_param_delta_norm": 0.0,
                "critic_backbone_delta_norm": 0.0,
                "critic_head_delta_norm": 0.0,
                "update_over_grad_ratio": 0.0,
                "scale_factor": 0.0,
                **{key: 0.0 for key in extra},
            }
            _append_rows_for_group(
                row_name,
                direction_name,
                target_update_norm,
                "before_update",
                extended_before_sets,
                grad_bundle["raw_grad_norm"],
                zero_delta,
            )
            _append_rows_for_group(
                row_name,
                direction_name,
                target_update_norm,
                "after_update",
                extended_after_sets,
                grad_bundle["raw_grad_norm"],
                {
                    **delta_metrics,
                    **extra,
                },
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        root_dir / "critic_update_radius_threshold_audit_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for set_type, filename in (
        ("current_batch", "critic_update_radius_threshold_audit_current_batch.csv"),
        ("held_out_batch", "critic_update_radius_threshold_audit_heldout_batch.csv"),
        ("probe", "critic_update_radius_threshold_audit_probe.csv"),
    ):
        metrics_df[metrics_df["set_type"] == set_type].to_csv(
            root_dir / filename,
            index=False,
            encoding="utf-8-sig",
        )

    radius_thresholds: dict[str, Any] = {}
    for direction_name in ("current_grad", "global_grad", "blended_grad"):
        direction_rows = metrics_df[
            (metrics_df["update_direction"] == direction_name)
            & (metrics_df["step_type"] == "after_update")
        ].copy()
        thresholds = {
            "first_loss_triad_failure_norm": None,
            "first_probe_target_pearson_drop_norm": None,
            "per_norm": {},
        }
        for target_update_norm in sorted(direction_rows["target_update_norm"].unique()):
            norm_rows = direction_rows[direction_rows["target_update_norm"] == target_update_norm]
            by_set = norm_rows.set_index("set_type")
            current_loss_delta = float(
                by_set.loc["current_batch", "critic_loss"]
                - before_sets["current_batch"]["critic_loss"]
            )
            heldout_loss_delta = float(
                by_set.loc["held_out_batch", "critic_loss"]
                - before_sets["held_out_batch"]["critic_loss"]
            )
            probe_loss_delta = float(
                by_set.loc["probe", "critic_loss"] - before_sets["probe"]["critic_loss"]
            )
            current_target_delta = float(
                by_set.loc["current_batch", "pearson_value_vs_value_target"]
                - before_sets["current_batch"]["pearson_value_vs_value_target"]
            )
            heldout_target_delta = float(
                by_set.loc["held_out_batch", "pearson_value_vs_value_target"]
                - before_sets["held_out_batch"]["pearson_value_vs_value_target"]
            )
            probe_target_delta = float(
                by_set.loc["probe", "pearson_value_vs_value_target"]
                - before_sets["probe"]["pearson_value_vs_value_target"]
            )
            thresholds["per_norm"][f"{target_update_norm:.0e}"] = {
                "current_batch": {
                    "critic_loss_delta": current_loss_delta,
                    "target_pearson_delta": current_target_delta,
                },
                "held_out_batch": {
                    "critic_loss_delta": heldout_loss_delta,
                    "target_pearson_delta": heldout_target_delta,
                },
                "probe": {
                    "critic_loss_delta": probe_loss_delta,
                    "target_pearson_delta": probe_target_delta,
                },
            }
            if (
                thresholds["first_loss_triad_failure_norm"] is None
                and current_loss_delta < 0.0
                and heldout_loss_delta > 0.0
                and probe_loss_delta > 0.0
            ):
                thresholds["first_loss_triad_failure_norm"] = float(target_update_norm)
            if (
                thresholds["first_probe_target_pearson_drop_norm"] is None
                and current_target_delta > 0.0
                and probe_target_delta < 0.0
            ):
                thresholds["first_probe_target_pearson_drop_norm"] = float(target_update_norm)
        radius_thresholds[direction_name] = thresholds

    interpretation = (
        "A semantic stability radius exists if small matched pure-gradient updates keep current, "
        "held-out, and probe roughly aligned, while larger update norms start producing "
        "'current better, global worse'."
    )
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "target_epoch": int(args.epoch),
        "target_update_epoch": int(args.update_epoch),
        "target_minibatch_id": int(args.minibatch_id),
        "target_update_norms": target_update_norms,
        "gradient_alignment": gradient_alignment,
        "radius_thresholds": radius_thresholds,
        "interpretation": interpretation,
    }
    (root_dir / "critic_update_radius_threshold_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, root_dir / "critic_update_radius_threshold_audit_summary.md")


if __name__ == "__main__":
    main()
