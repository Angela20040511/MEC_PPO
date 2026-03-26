import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from critic_batch_global_loss_trace import HELDOUT_BATCH_COUNT
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
HELDOUT_SEED_OFFSET = 23000
BLEND_CURRENT_WEIGHT = 0.5
BLEND_HELDOUT_WEIGHT = 0.5
SET_TYPES = ("current_batch", "held_out_batch", "probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate critic loss components on the same captured bad-step minibatch to "
            "check whether current-batch-only value fitting conflicts with held-out/probe semantics."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument("--update-epoch", type=int, default=DEFAULT_UPDATE_EPOCH)
    parser.add_argument("--minibatch-id", type=int, default=DEFAULT_MINIBATCH_ID)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    return parser.parse_args()


def _critic_predictions_and_features(
    agent: PPOAgent,
    critic_inputs: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    critic_features = agent.network.critic_features_from_critic_input(critic_inputs)
    normalized_values = agent.network.critic_head(critic_features).squeeze(-1)
    if agent.config.value_target_mode == "popart_return_norm":
        critic_predictions = normalized_values
    else:
        raw_values = agent.network.value_from_critic_input(critic_inputs).squeeze(-1)
        if agent.config.value_target_mode in {"normalized_return", "running_return_norm"}:
            critic_predictions = (raw_values - target_mean) / (target_std + 1e-8)
        else:
            critic_predictions = raw_values
    return critic_predictions, critic_features


def _main_value_loss_only(
    agent: PPOAgent,
    critic_inputs: torch.Tensor,
    value_targets: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    critic_predictions, _critic_features = _critic_predictions_and_features(
        agent,
        critic_inputs,
        target_mean,
        target_std,
    )
    main_value_loss = agent._compute_value_loss(critic_predictions, value_targets)
    objective = agent.config.value_coeff * main_value_loss
    return objective, {
        "main_value_loss": float(main_value_loss.detach().item()),
        "aux_loss": 0.0,
        "heldout_main_value_loss": 0.0,
    }


def _current_full_critic_objective(
    agent: PPOAgent,
    capture: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_critic_inputs = capture["batch_critic_inputs"]
    batch_value_targets = capture["batch_value_targets"]
    batch_states = capture["batch_states"]
    batch_next_states = capture["batch_next_states"]
    batch_actions = capture["batch_actions"]
    batch_dones = capture["batch_dones"]
    batch_next_critic_inputs = capture["batch_next_critic_inputs"]
    target_mean = capture["target_mean"]
    target_std = capture["target_std"]

    critic_predictions, critic_features = _critic_predictions_and_features(
        agent,
        batch_critic_inputs,
        target_mean,
        target_std,
    )
    critic_loss = agent._compute_value_loss(critic_predictions, batch_value_targets)

    if agent._uses_blockwise_value_scaled_advantage_surrogate_mean():
        block_value_scores = agent.network.block_value_scores_from_features(
            critic_features.detach()
        )
        block_value_targets = batch_value_targets.unsqueeze(-1).expand_as(block_value_scores)
        block_value_aux_loss = agent._compute_value_loss(
            block_value_scores,
            block_value_targets,
        )
    elif agent._uses_blockwise_td_style_advantage_surrogate_mean():
        with torch.no_grad():
            *_, block_local_rewards, block_path_probs = agent._block_td_reward_terms(
                batch_states,
                batch_next_states,
                actions=batch_actions,
            )
            block_path_rewards, _, _ = agent._block_path_td_rewards(
                batch_states,
                batch_next_states,
            )
        if agent._uses_blockwise_action_conditioned_path_value_bootstrap():
            if block_path_probs is None:
                raise ValueError(
                    "Captured batch_actions are required for path-value TD auxiliary loss."
                )
            current_block_path_value_preds = agent._block_path_value_predictions_from_features(
                critic_features.detach()
            )
            current_block_value_preds = agent._aggregate_block_path_values(
                current_block_path_value_preds,
                block_path_probs,
            )
            with torch.no_grad():
                next_block_path_value_preds = agent._block_path_value_predictions(
                    batch_next_critic_inputs
                )
                next_block_value_preds = agent._aggregate_block_path_values(
                    next_block_path_value_preds,
                    block_path_probs,
                )
                block_td_targets = (
                    block_local_rewards
                    + agent.config.gamma
                    * (1.0 - batch_dones.unsqueeze(-1))
                    * next_block_value_preds
                )
                block_path_td_targets = (
                    block_path_rewards
                    + agent.config.gamma
                    * (1.0 - batch_dones.unsqueeze(-1).unsqueeze(-1))
                    * next_block_path_value_preds
                )
            if agent._uses_blockwise_path_specific_td_supervised_advantage_surrogate_mean():
                block_value_aux_loss = agent._compute_value_loss(
                    current_block_path_value_preds,
                    block_path_td_targets,
                )
            else:
                block_value_aux_loss = agent._compute_value_loss(
                    current_block_value_preds,
                    block_td_targets,
                )
        else:
            current_block_value_preds = agent.network.block_value_scores_from_features(
                critic_features.detach()
            )
            with torch.no_grad():
                next_block_value_preds = agent.network.block_value_scores_from_critic_input(
                    batch_next_critic_inputs
                )
                block_td_targets = (
                    block_local_rewards
                    + agent.config.gamma
                    * (1.0 - batch_dones.unsqueeze(-1))
                    * next_block_value_preds
                )
            block_value_aux_loss = agent._compute_value_loss(
                current_block_value_preds,
                block_td_targets,
            )
    else:
        block_value_aux_loss = critic_loss.new_zeros(())

    objective = agent.config.value_coeff * critic_loss + block_value_aux_loss
    return objective, {
        "main_value_loss": float(critic_loss.detach().item()),
        "aux_loss": float(block_value_aux_loss.detach().item()),
        "heldout_main_value_loss": 0.0,
    }


def _blended_current_heldout_objective(
    agent: PPOAgent,
    capture: dict[str, Any],
    heldout_payload: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    current_predictions, _ = _critic_predictions_and_features(
        agent,
        capture["batch_critic_inputs"],
        capture["target_mean"],
        capture["target_std"],
    )
    current_main_value_loss = agent._compute_value_loss(
        current_predictions,
        capture["batch_value_targets"],
    )

    heldout_value_targets = agent._value_targets_from_returns_with_stats(
        heldout_payload["returns"],
        capture["target_mean"],
        capture["target_std"],
    )
    heldout_critic_inputs, _, _ = agent._prepare_critic_inputs(
        heldout_payload["states"],
        update_stats=False,
    )
    heldout_predictions, _ = _critic_predictions_and_features(
        agent,
        heldout_critic_inputs,
        capture["target_mean"],
        capture["target_std"],
    )
    heldout_main_value_loss = agent._compute_value_loss(
        heldout_predictions,
        heldout_value_targets,
    )
    blended_value_loss = (
        BLEND_CURRENT_WEIGHT * current_main_value_loss
        + BLEND_HELDOUT_WEIGHT * heldout_main_value_loss
    )
    objective = agent.config.value_coeff * blended_value_loss
    return objective, {
        "main_value_loss": float(current_main_value_loss.detach().item()),
        "aux_loss": 0.0,
        "heldout_main_value_loss": float(heldout_main_value_loss.detach().item()),
    }


def _semantic_metrics_from_payload(
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
    with torch.no_grad():
        value_predictions = agent._trace_raw_value_predictions_from_critic_inputs(
            critic_inputs,
            target_mean,
            target_std,
        )
        semantics = agent._critic_trace_semantic_summary(
            value_predictions,
            value_targets,
            payload["returns"],
            payload["advantages"],
            payload["rewards"],
        )
        critic_predictions_for_loss, _ = _critic_predictions_and_features(
            agent,
            critic_inputs,
            target_mean,
            target_std,
        )
        main_value_loss = agent._compute_value_loss(
            critic_predictions_for_loss,
            value_targets,
        )
    return {
        "critic_loss": float(main_value_loss.item()),
        **semantics,
    }


def _parameter_list_delta_norm(
    before_params: list[torch.Tensor],
    after_params: list[torch.Tensor],
) -> float:
    if not before_params or not after_params:
        return 0.0
    squared_sum = 0.0
    for before_param, after_param in zip(before_params, after_params):
        diff = after_param.detach() - before_param.detach()
        squared_sum += float(torch.sum(diff * diff).item())
    return float(squared_sum ** 0.5)


def _flatten_tensors(items: list[torch.Tensor | None]) -> torch.Tensor:
    flat: list[torch.Tensor] = []
    for item in items:
        if item is None:
            continue
        flat.append(item.detach().reshape(-1).cpu())
    return torch.cat(flat) if flat else torch.zeros(1, dtype=torch.float32)


def _delta_vector(
    before_params: list[torch.Tensor],
    after_params: list[torch.Tensor],
    grads: list[torch.Tensor | None],
) -> torch.Tensor:
    pieces = []
    for before_param, after_param, grad in zip(before_params, after_params, grads):
        if grad is None:
            continue
        pieces.append((after_param.detach() - before_param.detach()).reshape(-1).cpu())
    return torch.cat(pieces) if pieces else torch.zeros(1, dtype=torch.float32)


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return float(torch.dot(left, right).item() / (left_norm * right_norm))


def _build_same_rollout_heldout_payload(
    full_data: dict[str, torch.Tensor],
    current_batch_index: torch.Tensor,
    seed: int,
    mini_batch_size: int,
) -> dict[str, torch.Tensor]:
    all_indices = torch.arange(full_data["states"].size(0), device=current_batch_index.device)
    current_mask = torch.zeros_like(all_indices, dtype=torch.bool)
    current_mask[current_batch_index.long()] = True
    remaining_indices = all_indices[~current_mask]
    generator = torch.Generator(device=current_batch_index.device)
    generator.manual_seed(seed + HELDOUT_SEED_OFFSET)
    perm = torch.randperm(
        remaining_indices.numel(),
        generator=generator,
        device=current_batch_index.device,
    )
    heldout_count = min(
        remaining_indices.numel(),
        int(mini_batch_size) * HELDOUT_BATCH_COUNT,
    )
    selected = remaining_indices[perm[:heldout_count]]
    return {
        "states": full_data["states"][selected].clone(),
        "returns": full_data["returns"][selected].clone(),
        "advantages": full_data["advantages"][selected].clone(),
        "rewards": full_data["rewards"][selected].clone(),
        "source_step_count": int(full_data["states"].size(0)),
        "heldout_sample_count": int(heldout_count),
        "heldout_batch_count": int(max(1, heldout_count // int(mini_batch_size))),
        "selected_indices": selected.detach().cpu().tolist(),
    }


def _build_payloads_from_capture_and_rollout(
    agent: PPOAgent,
    capture: dict[str, Any],
    full_data: dict[str, torch.Tensor],
    probe_payload: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    heldout_payload = _build_same_rollout_heldout_payload(
        full_data=full_data,
        current_batch_index=capture["batch_index"],
        seed=seed,
        mini_batch_size=int(agent.config.mini_batch_size),
    )
    current_payload = {
        "states": capture["batch_states"],
        "returns": capture["batch_returns"],
        "advantages": capture["batch_raw_advantages"],
        "rewards": capture["batch_rewards"],
    }
    return {
        "current_batch": current_payload,
        "held_out_batch": heldout_payload,
        "probe": probe_payload,
    }


def _evaluate_all_sets(
    agent: PPOAgent,
    payloads: dict[str, dict[str, torch.Tensor]],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, dict[str, float]]:
    return {
        set_type: _semantic_metrics_from_payload(
            agent,
            payloads[set_type],
            target_mean,
            target_std,
        )
        for set_type in SET_TYPES
    }


def _append_rows(
    rows: list[dict[str, Any]],
    group_name: str,
    step_type: str,
    set_metrics: dict[str, dict[str, float]],
    grad_norm: float,
    delta_metrics: dict[str, float],
    objective_info: dict[str, float],
) -> None:
    for set_type in SET_TYPES:
        metrics = set_metrics[set_type]
        update_over_grad_ratio = (
            delta_metrics["critic_param_delta_norm"] / max(float(grad_norm), 1e-12)
            if step_type == "after_update"
            else 0.0
        )
        rows.append(
            {
                "group_name": group_name,
                "step_type": step_type,
                "set_type": set_type,
                **metrics,
                **objective_info,
                "grad_norm": float(grad_norm),
                **delta_metrics,
                "update_over_grad_ratio": float(update_over_grad_ratio),
            }
        )


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Critic Loss Component Ablation",
        "",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        (
            f"- captured minibatch: epoch `{summary['target_epoch']}` / update_epoch "
            f"`{summary['target_update_epoch']}` / minibatch `{summary['target_minibatch_id']}`"
        ),
        f"- interpretation: {summary['interpretation']}",
        "",
        "## Group Comparison",
        "",
        "```json",
        json.dumps(summary["group_comparison"], ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    root_dir = output_root / datetime.now().strftime(
        "critic_loss_component_ablation_%Y%m%d_%H%M%S"
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
        raise RuntimeError("Expected critic loss component ablation capture did not trigger.")
    except CriticOptimizerAblationCaptured:
        capture = agent.critic_optimizer_ablation_capture

    payloads = _build_payloads_from_capture_and_rollout(
        agent=agent,
        capture=capture,
        full_data=full_rollout,
        probe_payload=probe_payload,
        seed=args.seed,
    )

    before_sets = _evaluate_all_sets(
        agent,
        payloads,
        capture["target_mean"],
        capture["target_std"],
    )
    base_agent = copy.deepcopy(agent)
    before_params = [parameter.detach().clone() for parameter in base_agent.critic_params]

    metrics_rows: list[dict[str, Any]] = []

    def _current_full_loss(agent_for_loss: PPOAgent) -> tuple[torch.Tensor, dict[str, float]]:
        return _current_full_critic_objective(agent_for_loss, capture)

    def _main_value_only_loss(agent_for_loss: PPOAgent) -> tuple[torch.Tensor, dict[str, float]]:
        return _main_value_loss_only(
            agent_for_loss,
            capture["batch_critic_inputs"],
            capture["batch_value_targets"],
            capture["target_mean"],
            capture["target_std"],
        )

    def _blended_loss(agent_for_loss: PPOAgent) -> tuple[torch.Tensor, dict[str, float]]:
        return _blended_current_heldout_objective(
            agent_for_loss,
            capture,
            payloads["held_out_batch"],
        )

    def _run_update_group(
        group_name: str,
        objective_builder,
    ) -> None:
        updated_agent = copy.deepcopy(base_agent)
        updated_agent.critic_optimizer.load_state_dict(
            copy.deepcopy(capture["critic_optimizer_state_dict"])
        )
        updated_agent.critic_optimizer.zero_grad()
        objective, before_objective_info = objective_builder(updated_agent)
        objective.backward()
        grad_norm = float(updated_agent._grad_norm(updated_agent.critic_params))
        torch.nn.utils.clip_grad_norm_(
            updated_agent.critic_params,
            updated_agent.config.max_grad_norm,
        )
        updated_agent.critic_optimizer.step()

        after_sets = _evaluate_all_sets(
            updated_agent,
            payloads,
            capture["target_mean"],
            capture["target_std"],
        )
        after_objective, after_objective_info = objective_builder(updated_agent)
        after_params = [parameter.detach().clone() for parameter in updated_agent.critic_params]
        backbone_before = [
            parameter.detach().clone()
            for parameter in base_agent.network.critic_backbone.parameters()
        ]
        backbone_after = [
            parameter.detach().clone()
            for parameter in updated_agent.network.critic_backbone.parameters()
        ]
        head_before = [
            parameter.detach().clone() for parameter in base_agent.network.critic_head.parameters()
        ]
        head_after = [
            parameter.detach().clone()
            for parameter in updated_agent.network.critic_head.parameters()
        ]
        grad_vector = _flatten_tensors(
            [
                parameter.grad.detach().clone() if parameter.grad is not None else None
                for parameter in updated_agent.critic_params
            ]
        )
        update_grads = [
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for parameter in updated_agent.critic_params
        ]
        update_vector = _delta_vector(before_params, after_params, update_grads)
        delta_metrics = {
            "critic_param_delta_norm": _parameter_list_delta_norm(before_params, after_params),
            "critic_backbone_delta_norm": _parameter_list_delta_norm(
                backbone_before,
                backbone_after,
            ),
            "critic_head_delta_norm": _parameter_list_delta_norm(head_before, head_after),
            "cosine_update_vs_grad": _cosine_similarity(update_vector, grad_vector),
        }

        _append_rows(
            rows=metrics_rows,
            group_name=group_name,
            step_type="before_update",
            set_metrics=before_sets,
            grad_norm=grad_norm,
            delta_metrics={
                "critic_param_delta_norm": 0.0,
                "critic_backbone_delta_norm": 0.0,
                "critic_head_delta_norm": 0.0,
                "cosine_update_vs_grad": 0.0,
            },
            objective_info={
                "update_objective": float(objective.detach().item()),
                **before_objective_info,
            },
        )
        _append_rows(
            rows=metrics_rows,
            group_name=group_name,
            step_type="after_update",
            set_metrics=after_sets,
            grad_norm=grad_norm,
            delta_metrics=delta_metrics,
            objective_info={
                "update_objective": float(after_objective.detach().item()),
                **after_objective_info,
            },
        )

    _append_rows(
        rows=metrics_rows,
        group_name="no_op",
        step_type="before_update",
        set_metrics=before_sets,
        grad_norm=0.0,
        delta_metrics={
            "critic_param_delta_norm": 0.0,
            "critic_backbone_delta_norm": 0.0,
            "critic_head_delta_norm": 0.0,
            "cosine_update_vs_grad": 0.0,
        },
        objective_info={
            "update_objective": 0.0,
            "main_value_loss": 0.0,
            "aux_loss": 0.0,
            "heldout_main_value_loss": 0.0,
        },
    )
    _append_rows(
        rows=metrics_rows,
        group_name="no_op",
        step_type="after_update",
        set_metrics=before_sets,
        grad_norm=0.0,
        delta_metrics={
            "critic_param_delta_norm": 0.0,
            "critic_backbone_delta_norm": 0.0,
            "critic_head_delta_norm": 0.0,
            "cosine_update_vs_grad": 0.0,
        },
        objective_info={
            "update_objective": 0.0,
            "main_value_loss": 0.0,
            "aux_loss": 0.0,
            "heldout_main_value_loss": 0.0,
        },
    )

    _run_update_group("current_full_critic_loss", _current_full_loss)
    _run_update_group("main_value_loss_only", _main_value_only_loss)
    _run_update_group("blended_current_heldout_loss", _blended_loss)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(root_dir / "critic_loss_component_ablation_metrics.csv", index=False)
    for set_type, filename in {
        "probe": "critic_loss_component_ablation_probe.csv",
        "current_batch": "critic_loss_component_ablation_current_batch.csv",
        "held_out_batch": "critic_loss_component_ablation_heldout_batch.csv",
    }.items():
        metrics_df[metrics_df["set_type"] == set_type].to_csv(root_dir / filename, index=False)

    def _extract(group_name: str, set_type: str, metric: str) -> dict[str, float]:
        before_row = metrics_df[
            (metrics_df["group_name"] == group_name)
            & (metrics_df["set_type"] == set_type)
            & (metrics_df["step_type"] == "before_update")
        ].iloc[0]
        after_row = metrics_df[
            (metrics_df["group_name"] == group_name)
            & (metrics_df["set_type"] == set_type)
            & (metrics_df["step_type"] == "after_update")
        ].iloc[0]
        return {
            "before": float(before_row[metric]),
            "after": float(after_row[metric]),
            "delta": float(after_row[metric] - before_row[metric]),
        }

    def _group_summary(group_name: str) -> dict[str, Any]:
        current_before = metrics_df[
            (metrics_df["group_name"] == group_name)
            & (metrics_df["set_type"] == "current_batch")
            & (metrics_df["step_type"] == "before_update")
        ].iloc[0]
        current_after = metrics_df[
            (metrics_df["group_name"] == group_name)
            & (metrics_df["set_type"] == "current_batch")
            & (metrics_df["step_type"] == "after_update")
        ].iloc[0]
        return {
            "current_batch": {
                "critic_loss": _extract(group_name, "current_batch", "critic_loss"),
                "target_pearson": _extract(
                    group_name,
                    "current_batch",
                    "pearson_value_vs_value_target",
                ),
                "return_pearson": _extract(
                    group_name,
                    "current_batch",
                    "pearson_value_vs_return",
                ),
                "advantage_pearson": _extract(
                    group_name,
                    "current_batch",
                    "pearson_value_vs_advantage",
                ),
            },
            "held_out_batch": {
                "critic_loss": _extract(group_name, "held_out_batch", "critic_loss"),
                "target_pearson": _extract(
                    group_name,
                    "held_out_batch",
                    "pearson_value_vs_value_target",
                ),
                "return_pearson": _extract(
                    group_name,
                    "held_out_batch",
                    "pearson_value_vs_return",
                ),
                "advantage_pearson": _extract(
                    group_name,
                    "held_out_batch",
                    "pearson_value_vs_advantage",
                ),
            },
            "probe": {
                "critic_loss": _extract(group_name, "probe", "critic_loss"),
                "target_pearson": _extract(
                    group_name,
                    "probe",
                    "pearson_value_vs_value_target",
                ),
                "return_pearson": _extract(
                    group_name,
                    "probe",
                    "pearson_value_vs_return",
                ),
                "advantage_pearson": _extract(
                    group_name,
                    "probe",
                    "pearson_value_vs_advantage",
                ),
            },
            "update_details": {
                "objective_before": float(current_before["update_objective"]),
                "objective_after": float(current_after["update_objective"]),
                "main_value_loss_before": float(current_before["main_value_loss"]),
                "main_value_loss_after": float(current_after["main_value_loss"]),
                "aux_loss_before": float(current_before["aux_loss"]),
                "aux_loss_after": float(current_after["aux_loss"]),
                "heldout_main_value_loss_before": float(
                    current_before["heldout_main_value_loss"]
                ),
                "heldout_main_value_loss_after": float(
                    current_after["heldout_main_value_loss"]
                ),
                "grad_norm": float(current_after["grad_norm"]),
                "critic_param_delta_norm": float(current_after["critic_param_delta_norm"]),
                "critic_backbone_delta_norm": float(current_after["critic_backbone_delta_norm"]),
                "critic_head_delta_norm": float(current_after["critic_head_delta_norm"]),
                "update_over_grad_ratio": float(current_after["update_over_grad_ratio"]),
                "cosine_update_vs_grad": float(current_after["cosine_update_vs_grad"]),
            },
        }

    group_comparison = {
        "no_op": _group_summary("no_op"),
        "current_full_critic_loss": _group_summary("current_full_critic_loss"),
        "main_value_loss_only": _group_summary("main_value_loss_only"),
        "blended_current_heldout_loss": _group_summary("blended_current_heldout_loss"),
    }

    full_probe_drop = group_comparison["current_full_critic_loss"]["probe"]["target_pearson"][
        "delta"
    ]
    main_probe_drop = group_comparison["main_value_loss_only"]["probe"]["target_pearson"][
        "delta"
    ]
    blend_probe_drop = group_comparison["blended_current_heldout_loss"]["probe"][
        "target_pearson"
    ]["delta"]
    full_heldout_drop = group_comparison["current_full_critic_loss"]["held_out_batch"][
        "target_pearson"
    ]["delta"]
    main_heldout_drop = group_comparison["main_value_loss_only"]["held_out_batch"][
        "target_pearson"
    ]["delta"]
    blend_heldout_drop = group_comparison["blended_current_heldout_loss"]["held_out_batch"][
        "target_pearson"
    ]["delta"]
    full_aux_before = group_comparison["current_full_critic_loss"]["update_details"][
        "aux_loss_before"
    ]

    if main_probe_drop < -0.01 and main_heldout_drop < -0.01:
        if full_probe_drop < main_probe_drop - 0.01 or full_heldout_drop < main_heldout_drop - 0.01:
            interpretation = (
                "Main value loss on the current minibatch already creates a batch-fit/global-semantics "
                "conflict, and the extra critic auxiliary loss makes it worse."
            )
        else:
            interpretation = (
                "The main current-batch value loss itself is already sufficient to reproduce "
                "batch-better/global-worse behavior; auxiliary critic losses are not the primary cause."
            )
    elif full_probe_drop < -0.01 and main_probe_drop > full_probe_drop + 0.01:
        interpretation = (
            "The full critic loss is substantially worse than main-value-only, which suggests "
            "extra critic auxiliary components are the main amplifier of the conflict."
        )
    elif blend_probe_drop > max(full_probe_drop, main_probe_drop) + 0.01 and blend_heldout_drop > max(
        full_heldout_drop,
        main_heldout_drop,
    ) + 0.01:
        interpretation = (
            "Blending current and held-out value loss clearly reduces held-out/probe damage, "
            "which points to a current-batch-only local objective being too narrow."
        )
    else:
        interpretation = (
            "Both current-batch-only value fitting and auxiliary critic terms contribute, but the "
            "dominant factor is not fully isolated by this bad-step ablation alone."
        )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "target_epoch": int(args.epoch),
        "target_update_epoch": int(args.update_epoch),
        "target_minibatch_id": int(args.minibatch_id),
        "heldout_source_step_count": int(payloads["held_out_batch"]["source_step_count"]),
        "heldout_sample_count": int(payloads["held_out_batch"]["heldout_sample_count"]),
        "heldout_batch_count": int(payloads["held_out_batch"]["heldout_batch_count"]),
        "blend_current_weight": BLEND_CURRENT_WEIGHT,
        "blend_heldout_weight": BLEND_HELDOUT_WEIGHT,
        "full_loss_aux_before": float(full_aux_before),
        "group_comparison": group_comparison,
        "interpretation": interpretation,
    }

    (root_dir / "critic_loss_component_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, root_dir / "critic_loss_component_ablation_summary.md")


if __name__ == "__main__":
    main()
