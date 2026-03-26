import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from critic_loss_component_ablation import (
    _build_payloads_from_capture_and_rollout,
    _main_value_loss_only,
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
TOPK_COUNT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand the bad-step Adam preconditioner geometry on critic_backbone/head "
            "for the current joint reward-aligned actor mainline."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument("--update-epoch", type=int, default=DEFAULT_UPDATE_EPOCH)
    parser.add_argument("--minibatch-id", type=int, default=DEFAULT_MINIBATCH_ID)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    return parser.parse_args()


def _classify_module(name: str) -> str:
    if name.startswith("critic_backbone."):
        return "critic_backbone"
    if name.startswith("critic_head."):
        return "critic_head"
    if name.startswith("critic_block_value_head."):
        return "critic_block_value_head"
    if name.startswith("critic_block_path_value_head."):
        return "critic_block_path_value_head"
    return "other"


def _layer_name(name: str) -> str:
    if "." not in name:
        return name
    return ".".join(name.split(".")[:-1])


def _named_critic_params(agent: PPOAgent) -> list[tuple[str, torch.nn.Parameter]]:
    named_params: list[tuple[str, torch.nn.Parameter]] = []
    named_params.extend(
        list(agent.network.critic_backbone.named_parameters(prefix="critic_backbone"))
    )
    named_params.extend(list(agent.network.critic_head.named_parameters(prefix="critic_head")))
    named_params.extend(
        list(
            agent.network.critic_block_value_head.named_parameters(
                prefix="critic_block_value_head"
            )
        )
    )
    named_params.extend(
        list(
            agent.network.critic_block_path_value_head.named_parameters(
                prefix="critic_block_path_value_head"
            )
        )
    )
    return named_params


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(torch.dot(left, right).item() / (left_norm * right_norm))


def _flatten_tensor_list(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat([tensor.reshape(-1).detach().cpu().float() for tensor in tensors], dim=0)


def _tensor_distribution_stats(tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.reshape(-1).detach().cpu().float()
    if flat.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "p999": 0.0,
        }
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
        "p999": float(torch.quantile(flat, 0.999).item()),
    }


def _module_tensor_map(
    tensor_rows: list[dict[str, Any]],
    key: str,
) -> dict[str, list[torch.Tensor]]:
    groups: dict[str, list[torch.Tensor]] = {
        "all_critic": [],
        "critic_backbone": [],
        "critic_head": [],
        "critic_block_value_head": [],
        "critic_block_path_value_head": [],
    }
    for row in tensor_rows:
        tensor = row[key]
        module_group = row["module_group"]
        groups["all_critic"].append(tensor)
        if module_group in groups:
            groups[module_group].append(tensor)
    return groups


def _evaluate_payload_loss_and_semantics(
    agent: PPOAgent,
    payload: dict[str, torch.Tensor],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, float]:
    value_targets = agent._value_targets_from_returns_with_stats(
        payload["returns"],
        target_mean,
        target_std,
    )
    critic_inputs, _, _ = agent._prepare_critic_inputs(
        payload["states"],
        update_stats=False,
    )
    objective, _loss_parts = _main_value_loss_only(
        agent,
        critic_inputs,
        value_targets,
        target_mean,
        target_std,
    )
    semantics = _semantic_metrics_from_payload(
        agent,
        payload,
        target_mean,
        target_std,
    )
    return {
        "critic_loss": float(objective.item()),
        **semantics,
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Critic Backbone Preconditioner Detail Audit",
        "",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        (
            f"- captured minibatch: epoch `{summary['target_epoch']}` / update_epoch "
            f"`{summary['target_update_epoch']}` / minibatch `{summary['target_minibatch_id']}`"
        ),
        f"- backbone heavier tail: `{summary['backbone_preconditioner_heavier_tail']}`",
        f"- dominant module: `{summary['dominant_module']}`",
        f"- interpretation: {summary['interpretation']}",
        "",
        "## Module Summary",
        "",
        "```json",
        json.dumps(summary["module_summary"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Actual Step Effects",
        "",
        "```json",
        json.dumps(summary["actual_step_effects"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Top Layers",
        "",
        "```json",
        json.dumps(summary["top_layers"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    root_dir = output_root / datetime.now().strftime(
        "critic_backbone_preconditioner_detail_audit_%Y%m%d_%H%M%S"
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
        raise RuntimeError("Expected critic optimizer capture did not trigger.")
    except CriticOptimizerAblationCaptured:
        capture = agent.critic_optimizer_ablation_capture

    payloads = _build_payloads_from_capture_and_rollout(
        agent=agent,
        capture=capture,
        full_data=full_rollout,
        probe_payload=probe_payload,
        seed=args.seed,
    )
    eval_payloads = {
        "current_batch": payloads["current_batch"],
        "held_out_batch": payloads["held_out_batch"],
        "probe": payloads["probe"],
    }

    base_agent = copy.deepcopy(agent)
    base_agent.critic_optimizer.zero_grad()
    objective, loss_parts = _main_value_loss_only(
        base_agent,
        capture["batch_critic_inputs"],
        capture["batch_value_targets"],
        capture["target_mean"],
        capture["target_std"],
    )
    objective.backward()

    critic_group = base_agent.critic_optimizer.param_groups[0]
    beta1, beta2 = critic_group["betas"]
    learning_rate = float(critic_group["lr"])
    eps = float(critic_group["eps"])
    weight_decay = float(critic_group.get("weight_decay", 0.0))

    tensor_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    named_params = _named_critic_params(base_agent)
    total_actual_delta_norm_sq = 0.0

    for param_index, (name, parameter) in enumerate(named_params):
        grad = parameter.grad
        if grad is None:
            continue
        raw_grad = grad.detach().clone().cpu().float()
        grad_for_adam = grad.detach().clone()
        if weight_decay != 0.0:
            grad_for_adam = grad_for_adam + weight_decay * parameter.detach()

        state = base_agent.critic_optimizer.state[parameter]
        exp_avg = state["exp_avg"].detach().clone().cpu().float()
        exp_avg_sq = state["exp_avg_sq"].detach().clone().cpu().float()
        step_value = state["step"]
        if torch.is_tensor(step_value):
            step_value = int(step_value.item())
        else:
            step_value = int(step_value)
        next_step = step_value + 1

        grad_for_adam_cpu = grad_for_adam.detach().clone().cpu().float()
        exp_avg_t = exp_avg * beta1 + grad_for_adam_cpu * (1.0 - beta1)
        exp_avg_sq_t = exp_avg_sq * beta2 + grad_for_adam_cpu.square() * (1.0 - beta2)
        bias_correction1 = 1.0 - beta1 ** next_step
        bias_correction2 = 1.0 - beta2 ** next_step
        m_hat = exp_avg_t / bias_correction1
        v_hat = exp_avg_sq_t / bias_correction2
        preconditioner = 1.0 / (v_hat.sqrt() + eps)
        preconditioned_grad = raw_grad * preconditioner
        full_adam_direction = m_hat * preconditioner
        actual_param_delta = -learning_rate * full_adam_direction

        actual_delta_norm = float(actual_param_delta.norm().item())
        total_actual_delta_norm_sq += actual_delta_norm ** 2

        flat_preconditioner = preconditioner.reshape(-1)
        top_k = min(TOPK_COUNT, int(flat_preconditioner.numel()))
        top_values, top_indices = torch.topk(flat_preconditioner, k=top_k)
        for rank, (value, flat_index) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
            unravel_index = list(torch.unravel_index(torch.tensor(flat_index), preconditioner.shape))
            topk_rows.append(
                {
                    "module_group": _classify_module(name),
                    "layer_name": _layer_name(name),
                    "tensor_name": name,
                    "param_index": int(param_index),
                    "rank_within_tensor": rank,
                    "flat_index": int(flat_index),
                    "index_tuple": json.dumps([int(idx) for idx in unravel_index], ensure_ascii=False),
                    "preconditioner": float(value),
                    "raw_grad": float(raw_grad.reshape(-1)[flat_index].item()),
                    "m_hat": float(m_hat.reshape(-1)[flat_index].item()),
                    "v_hat": float(v_hat.reshape(-1)[flat_index].item()),
                    "preconditioned_grad": float(preconditioned_grad.reshape(-1)[flat_index].item()),
                    "full_adam_direction": float(full_adam_direction.reshape(-1)[flat_index].item()),
                    "abs_full_adam_direction": float(
                        abs(full_adam_direction.reshape(-1)[flat_index].item())
                    ),
                    "actual_param_delta": float(actual_param_delta.reshape(-1)[flat_index].item()),
                    "abs_actual_param_delta": float(
                        abs(actual_param_delta.reshape(-1)[flat_index].item())
                    ),
                }
            )

        raw_grad_norm = float(raw_grad.norm().item())
        tensor_rows.append(
            {
                "module_group": _classify_module(name),
                "layer_name": _layer_name(name),
                "tensor_name": name,
                "param_index": int(param_index),
                "numel": int(raw_grad.numel()),
                "raw_grad": raw_grad,
                "m_hat": m_hat,
                "v_hat": v_hat,
                "preconditioner": preconditioner,
                "preconditioned_grad": preconditioned_grad,
                "full_adam_direction": full_adam_direction,
                "actual_param_delta": actual_param_delta,
                "raw_grad_norm": raw_grad_norm,
                "m_hat_norm": float(m_hat.norm().item()),
                "v_hat_norm": float(v_hat.norm().item()),
                "preconditioned_grad_norm": float(preconditioned_grad.norm().item()),
                "full_adam_update_norm": float(full_adam_direction.norm().item()),
                "actual_param_delta_norm": actual_delta_norm,
                "update_over_grad_ratio": float(actual_delta_norm / max(raw_grad_norm, 1e-12)),
                "cosine_raw_vs_momentum": _cosine(raw_grad.reshape(-1), m_hat.reshape(-1)),
                "cosine_raw_vs_preconditioned": _cosine(
                    raw_grad.reshape(-1),
                    preconditioned_grad.reshape(-1),
                ),
                "cosine_raw_vs_full_adam": _cosine(
                    raw_grad.reshape(-1),
                    full_adam_direction.reshape(-1),
                ),
                "cosine_momentum_vs_full_adam": _cosine(
                    m_hat.reshape(-1),
                    full_adam_direction.reshape(-1),
                ),
                "cosine_preconditioned_vs_full_adam": _cosine(
                    preconditioned_grad.reshape(-1),
                    full_adam_direction.reshape(-1),
                ),
                **{
                    f"preconditioner_{key}": value
                    for key, value in _tensor_distribution_stats(preconditioner).items()
                },
            }
        )

    total_actual_delta_norm = total_actual_delta_norm_sq ** 0.5
    for row in tensor_rows:
        row["actual_delta_norm_share"] = float(
            row["actual_param_delta_norm"] / max(total_actual_delta_norm, 1e-12)
        )

    layer_stats_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if isinstance(value, (int, float, str))
            }
            for row in tensor_rows
        ]
    ).sort_values(
        ["actual_delta_norm_share", "preconditioner_p999"],
        ascending=[False, False],
    )
    layer_stats_path = root_dir / "critic_backbone_preconditioner_layer_stats.csv"
    layer_stats_df.to_csv(layer_stats_path, index=False, encoding="utf-8-sig")

    topk_df = pd.DataFrame(topk_rows).sort_values(
        ["preconditioner", "abs_full_adam_direction"],
        ascending=[False, False],
    )
    topk_path = root_dir / "critic_backbone_preconditioner_topk.csv"
    topk_df.to_csv(topk_path, index=False, encoding="utf-8-sig")

    module_rows: list[dict[str, Any]] = []
    module_tensor_maps = {
        "raw_grad": _module_tensor_map(tensor_rows, "raw_grad"),
        "m_hat": _module_tensor_map(tensor_rows, "m_hat"),
        "v_hat": _module_tensor_map(tensor_rows, "v_hat"),
        "preconditioner": _module_tensor_map(tensor_rows, "preconditioner"),
        "preconditioned_grad": _module_tensor_map(tensor_rows, "preconditioned_grad"),
        "full_adam_direction": _module_tensor_map(tensor_rows, "full_adam_direction"),
        "actual_param_delta": _module_tensor_map(tensor_rows, "actual_param_delta"),
    }
    for module_name in [
        "all_critic",
        "critic_backbone",
        "critic_head",
        "critic_block_value_head",
        "critic_block_path_value_head",
    ]:
        raw_grad_vec = _flatten_tensor_list(module_tensor_maps["raw_grad"][module_name])
        m_hat_vec = _flatten_tensor_list(module_tensor_maps["m_hat"][module_name])
        v_hat_vec = _flatten_tensor_list(module_tensor_maps["v_hat"][module_name])
        preconditioner_vec = _flatten_tensor_list(module_tensor_maps["preconditioner"][module_name])
        preconditioned_grad_vec = _flatten_tensor_list(
            module_tensor_maps["preconditioned_grad"][module_name]
        )
        full_adam_vec = _flatten_tensor_list(module_tensor_maps["full_adam_direction"][module_name])
        actual_delta_vec = _flatten_tensor_list(module_tensor_maps["actual_param_delta"][module_name])
        module_rows.append(
            {
                "module_group": module_name,
                "numel": int(raw_grad_vec.numel()),
                "raw_grad_norm": float(raw_grad_vec.norm().item()),
                "m_hat_norm": float(m_hat_vec.norm().item()),
                "v_hat_norm": float(v_hat_vec.norm().item()),
                "preconditioned_grad_norm": float(preconditioned_grad_vec.norm().item()),
                "full_adam_update_norm": float(full_adam_vec.norm().item()),
                "actual_param_delta_norm": float(actual_delta_vec.norm().item()),
                "update_over_grad_ratio": float(
                    actual_delta_vec.norm().item() / max(raw_grad_vec.norm().item(), 1e-12)
                ),
                "actual_delta_norm_share": float(
                    actual_delta_vec.norm().item() / max(total_actual_delta_norm, 1e-12)
                ),
                "cosine_raw_vs_momentum": _cosine(raw_grad_vec, m_hat_vec),
                "cosine_raw_vs_preconditioned": _cosine(raw_grad_vec, preconditioned_grad_vec),
                "cosine_raw_vs_full_adam": _cosine(raw_grad_vec, full_adam_vec),
                "cosine_momentum_vs_full_adam": _cosine(m_hat_vec, full_adam_vec),
                "cosine_preconditioned_vs_full_adam": _cosine(preconditioned_grad_vec, full_adam_vec),
                "cosine_full_adam_vs_actual_delta": _cosine(full_adam_vec, actual_delta_vec),
                **{
                    f"preconditioner_{key}": value
                    for key, value in _tensor_distribution_stats(preconditioner_vec).items()
                },
            }
        )
    module_df = pd.DataFrame(module_rows)
    metrics_path = root_dir / "critic_backbone_preconditioner_detail_audit_metrics.csv"
    module_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    distribution_path = root_dir / "critic_backbone_preconditioner_distribution.csv"
    module_df.to_csv(distribution_path, index=False, encoding="utf-8-sig")

    updated_agent = copy.deepcopy(base_agent)
    with torch.no_grad():
        for row in tensor_rows:
            updated_agent.critic_params[int(row["param_index"])].add_(
                row["actual_param_delta"].to(updated_agent.device)
            )

    actual_step_effects: dict[str, Any] = {}
    for set_type, payload in eval_payloads.items():
        before_metrics = _evaluate_payload_loss_and_semantics(
            base_agent,
            payload,
            capture["target_mean"],
            capture["target_std"],
        )
        after_metrics = _evaluate_payload_loss_and_semantics(
            updated_agent,
            payload,
            capture["target_mean"],
            capture["target_std"],
        )
        actual_step_effects[set_type] = {
            "before_critic_loss": before_metrics["critic_loss"],
            "after_critic_loss": after_metrics["critic_loss"],
            "delta_critic_loss": after_metrics["critic_loss"] - before_metrics["critic_loss"],
            "before_target_pearson": before_metrics["pearson_value_vs_value_target"],
            "after_target_pearson": after_metrics["pearson_value_vs_value_target"],
            "delta_target_pearson": (
                after_metrics["pearson_value_vs_value_target"]
                - before_metrics["pearson_value_vs_value_target"]
            ),
        }

    backbone_row = module_df.loc[module_df["module_group"] == "critic_backbone"].iloc[0].to_dict()
    head_row = module_df.loc[module_df["module_group"] == "critic_head"].iloc[0].to_dict()
    top_layers = layer_stats_df.head(10)[
        [
            "module_group",
            "layer_name",
            "tensor_name",
            "actual_delta_norm_share",
            "actual_param_delta_norm",
            "preconditioner_p99",
            "preconditioner_p999",
            "preconditioner_max",
            "cosine_raw_vs_preconditioned",
            "cosine_raw_vs_full_adam",
        ]
    ].to_dict(orient="records")

    backbone_topk = topk_df[topk_df["module_group"] == "critic_backbone"].head(TOPK_COUNT)
    head_topk = topk_df[topk_df["module_group"] == "critic_head"].head(TOPK_COUNT)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "target_epoch": int(args.epoch),
        "target_update_epoch": int(args.update_epoch),
        "target_minibatch_id": int(args.minibatch_id),
        "fixed_rollout_steps": int(config.training.time_steps),
        "fixed_probe_count": int(probe_payload["states"].shape[0]),
        "heldout_count": int(payloads["held_out_batch"]["states"].shape[0]),
        "main_value_loss_only": {
            "main_value_loss": float(loss_parts["main_value_loss"]),
            "aux_loss": float(loss_parts["aux_loss"]),
        },
        "backbone_preconditioner_heavier_tail": bool(
            float(backbone_row["preconditioner_p999"]) > float(head_row["preconditioner_p999"])
        ),
        "dominant_module": "critic_backbone"
        if float(backbone_row["actual_delta_norm_share"]) >= float(head_row["actual_delta_norm_share"])
        else "critic_head",
        "module_summary": {
            row["module_group"]: {
                key: value
                for key, value in row.items()
                if key != "module_group"
            }
            for row in module_rows
        },
        "top_layers": top_layers,
        "top_backbone_preconditioner_dims": backbone_topk.to_dict(orient="records"),
        "top_head_preconditioner_dims": head_topk.to_dict(orient="records"),
        "actual_step_effects": actual_step_effects,
        "interpretation": (
            "Use this audit to decide whether the next formal validation should constrain "
            "critic_backbone preconditioning geometry rather than replace the full optimizer."
        ),
    }

    summary_json_path = root_dir / "critic_backbone_preconditioner_detail_audit_summary.json"
    summary_md_path = root_dir / "critic_backbone_preconditioner_detail_audit_summary.md"
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, summary_md_path)

    print(
        "Critic backbone preconditioner detail audit completed. "
        f"Results saved to: {root_dir}"
    )


if __name__ == "__main__":
    main()
