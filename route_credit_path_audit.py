import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dense_actor_input_experiment import build_state_layout
from dense_policy_ratio_mode_experiment import build_policy_ratio_mode_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator


BASELINE_MODE = "hierarchical_actor_true_conditional_route_policy"
CANDIDATE_MODE = "hierarchical_actor_true_conditional_route_candidate_score_credit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-minibatch route-credit path audit for true-conditional route policy "
            "vs candidate-score route credit."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help=(
            "Optional checkpoint path used as the shared policy source for rollout collection. "
            "If omitted, the latest baseline checkpoint from the candidate-score experiment is used."
        ),
    )
    return parser.parse_args()


def _find_latest_baseline_checkpoint() -> Path:
    root = Path("checkpoints")
    experiment_roots = sorted(
        path
        for path in root.glob("dense_policy_route_candidate_score_credit_experiment_*")
        if path.is_dir()
    )
    if not experiment_roots:
        raise FileNotFoundError(
            "No dense_policy_route_candidate_score_credit_experiment_* directory found."
        )
    latest_root = experiment_roots[-1]
    baseline_runs = sorted(
        latest_root.glob(f"policy_ratio_{BASELINE_MODE}_*/best_model.pt")
    )
    if not baseline_runs:
        raise FileNotFoundError(
            f"No baseline best_model.pt found under {latest_root}"
        )
    return baseline_runs[-1]


def _build_agent(mode: str, seed: int) -> tuple[PPOAgent, Any]:
    config = build_policy_ratio_mode_config(policy_ratio_mode=mode, seed=seed)
    state_layout = build_state_layout(config)
    agent = PPOAgent(
        config=config.ppo,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        critic_state_layout=state_layout,
    )
    return agent, config


def _collect_rollout(agent: PPOAgent, config: Any, seed: int) -> dict[str, torch.Tensor]:
    simulator = Simulator(config)
    state = simulator.reset(seed=seed)
    done = False

    for _ in range(config.training.time_steps):
        action, log_prob, value, policy_cache = agent.select_action_with_info(state)
        next_state, reward, done, _ = simulator.step(action)
        agent.store_transition(
            state,
            action,
            log_prob,
            reward,
            done,
            value,
            next_state,
            policy_cache=policy_cache,
        )
        state = next_state
        if done:
            break

    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)
    return agent.buffer.as_tensors(agent.device)


def _masked_stats(tensor: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    flat_mask = mask.reshape(-1) > 0.5
    flat_tensor = tensor.reshape(-1)
    if not bool(flat_mask.any().item()):
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    values = flat_tensor[flat_mask]
    return {
        "count": float(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _log_std_grad_norm(agent: PPOAgent, indices: list[int]) -> float:
    grad = agent.network.actor_log_std.grad
    if grad is None or not indices:
        return 0.0
    return float(grad[indices].detach().norm().item())


def _parameter_delta_norm(parameters_before: list[torch.Tensor], parameters_after: list[torch.Tensor]) -> float:
    total = 0.0
    for before, after in zip(parameters_before, parameters_after):
        delta = after.detach() - before.detach()
        total += float(delta.norm().item()) ** 2
    return float(total ** 0.5)


def _snapshot_params(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _route_param_groups(agent: PPOAgent) -> list[torch.nn.Parameter]:
    return (
        agent.network.actor_route_backbone_parameters()
        + agent.network.actor_route_head_parameters()
    )


def _theta_param_groups(agent: PPOAgent) -> list[torch.nn.Parameter]:
    return (
        agent.network.actor_theta_backbone_parameters()
        + agent.network.actor_theta_head_parameters()
    )


def _prepare_batch(agent: PPOAgent, data: dict[str, torch.Tensor], batch_index: torch.Tensor) -> dict[str, torch.Tensor]:
    states = data["states"][batch_index]
    next_states = data["next_states"][batch_index]
    actions = data["actions"][batch_index]
    old_log_probs = data["old_log_probs"][batch_index]
    old_log_prob_components = data["old_log_prob_components"][batch_index]
    old_action_means = data["old_action_means"][batch_index]
    old_action_stds = data["old_action_stds"][batch_index]
    dones = data["dones"][batch_index]

    actor_inputs, _, _ = agent._prepare_actor_inputs(states, update_stats=False)
    critic_inputs, _, _ = agent._prepare_critic_inputs(states, update_stats=False)
    next_critic_inputs, _, _ = agent._prepare_critic_inputs(next_states, update_stats=False)
    distribution = agent.network.policy_from_actor_input(actor_inputs)
    new_log_prob_components = distribution.log_prob(actions)

    split_advantages = agent._theta_route_split_advantages(
        states,
        next_states,
        dones,
        actions,
        critic_inputs=critic_inputs,
        next_critic_inputs=next_critic_inputs,
    )
    branchwise_normalization = agent._branchwise_normalize_theta_route_advantages(
        split_advantages["theta_advantages"],
        split_advantages["route_advantages"],
        split_advantages["route_masks"],
    )
    theta_advantages_for_surrogate = branchwise_normalization["theta_advantages_norm"]
    route_advantages_for_surrogate = branchwise_normalization["route_advantages_norm"]
    route_masks = split_advantages["route_masks"]
    route_loss_masks = route_masks

    route_old_policy_terms_for_credit = agent._true_conditional_route_policy_terms(
        actions,
        old_action_means,
    )

    candidate_terms = None
    if agent._uses_true_conditional_route_candidate_score_credit():
        candidate_terms = agent._route_candidate_score_credit_terms(
            states,
            route_old_policy_terms_for_credit["selected_indices"],
            route_old_policy_terms_for_credit["probs"].detach(),
            route_masks,
        )
        route_advantages_for_surrogate = candidate_terms["residual_credit"]

    theta_terms = agent._true_conditional_theta_policy_terms(
        actions,
        distribution.mean,
    )
    route_terms = agent._true_conditional_route_policy_terms(
        actions,
        distribution.mean,
    )
    entropy = agent._masked_tensor_mean(route_terms["entropy"], route_terms["active_mask"])
    route_branch_terms = agent._route_branch_ppo_terms(
        actions,
        new_log_prob_components,
        old_log_prob_components,
        route_advantages_for_surrogate,
        route_masks,
        new_action_means=distribution.mean,
        new_action_stds=distribution.stddev,
        old_action_means=old_action_means,
        old_action_stds=old_action_stds,
        route_gates=split_advantages["route_gates"],
        route_loss_masks=route_loss_masks,
        route_used_fallback=False,
    )
    route_only_objective = (
        route_branch_terms["loss"] - agent.config.entropy_coeff * entropy
    )

    return {
        "states": states,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "old_log_prob_components": old_log_prob_components,
        "old_action_means": old_action_means,
        "old_action_stds": old_action_stds,
        "actor_inputs": actor_inputs,
        "distribution_mean": distribution.mean,
        "distribution_std": distribution.stddev,
        "new_log_prob_components": new_log_prob_components,
        "split_advantages": split_advantages,
        "theta_advantages_for_surrogate": theta_advantages_for_surrogate,
        "route_advantages_for_surrogate": route_advantages_for_surrogate,
        "route_masks": route_masks,
        "route_loss_masks": route_loss_masks,
        "theta_terms": theta_terms,
        "route_terms": route_terms,
        "route_old_policy_terms_for_credit": route_old_policy_terms_for_credit,
        "route_branch_terms": route_branch_terms,
        "route_only_objective": route_only_objective,
        "candidate_terms": candidate_terms,
    }


def _collect_raw_active_rows(mode: str, prepared: dict[str, torch.Tensor]) -> list[dict[str, float | int | str]]:
    route_loss_mask = prepared["route_loss_masks"] > 0.5
    route_terms = prepared["route_terms"]
    route_branch_terms = prepared["route_branch_terms"]
    selected_indices = route_terms["selected_indices"]

    if prepared["candidate_terms"] is not None:
        candidate_score_vector = prepared["candidate_terms"]["candidate_score_vector"]
        selected_score = prepared["candidate_terms"]["selected_score"]
        expected_score = prepared["candidate_terms"]["expected_score"]
        residual_credit = prepared["candidate_terms"]["residual_credit"]
    else:
        route_adv = prepared["route_advantages_for_surrogate"]
        candidate_score_vector = torch.stack((0.5 * route_adv, -0.5 * route_adv), dim=-1)
        selected_score = candidate_score_vector.gather(
            dim=-1,
            index=selected_indices.unsqueeze(-1),
        ).squeeze(-1)
        expected_score = (
            prepared["route_old_policy_terms_for_credit"]["probs"] * candidate_score_vector
        ).sum(dim=-1)
        residual_credit = route_adv

    rows: list[dict[str, float | int | str]] = []
    active_positions = route_loss_mask.nonzero(as_tuple=False)
    for pos in active_positions:
        sample_index = int(pos[0].item())
        block_index = int(pos[1].item())
        rows.append(
            {
                "mode": mode,
                "sample_index": sample_index,
                "block_index": block_index,
                "selected_route_index": int(selected_indices[sample_index, block_index].item()),
                "candidate_score_0": float(candidate_score_vector[sample_index, block_index, 0].item()),
                "candidate_score_1": float(candidate_score_vector[sample_index, block_index, 1].item()),
                "selected_score": float(selected_score[sample_index, block_index].item()),
                "expected_score": float(expected_score[sample_index, block_index].item()),
                "route_effective_advantage": float(residual_credit[sample_index, block_index].item()),
                "route_old_logprob": float(route_branch_terms["old_log_probs"][sample_index, block_index].item()),
                "route_new_logprob": float(route_branch_terms["new_log_probs"][sample_index, block_index].item()),
                "route_ratio": float(route_branch_terms["ratio"][sample_index, block_index].item()),
                "route_surrogate": float(route_branch_terms["surrogate"][sample_index, block_index].item()),
            }
        )
    return rows


def _run_bare_backward(agent: PPOAgent, prepared: dict[str, torch.Tensor]) -> dict[str, Any]:
    route_optimizer = agent._route_actor_optimizer()
    route_optimizer.zero_grad()
    prepared["route_only_objective"].backward()
    agent._apply_route_only_gradient_mask()

    route_params = _route_param_groups(agent)
    theta_params = _theta_param_groups(agent)
    route_param_grads_all_zero = all(
        parameter.grad is None or float(parameter.grad.detach().abs().sum().item()) == 0.0
        for parameter in route_params
    )

    result = {
        "route_head_grad_norm": agent._route_head_grad_norm(),
        "route_backbone_grad_norm": agent._route_backbone_grad_norm(),
        "route_logstd_grad_norm": _log_std_grad_norm(agent, agent._route_action_indices()),
        "route_param_grads_all_zero": bool(route_param_grads_all_zero),
        "theta_head_grad_norm": agent._theta_head_grad_norm(),
        "theta_backbone_grad_norm": agent._theta_backbone_grad_norm(),
        "theta_logstd_grad_norm": _log_std_grad_norm(agent, agent._theta_action_indices()),
        "only_route_branch_received_gradient": bool(
            agent._route_head_grad_norm() > 0.0
            or agent._route_backbone_grad_norm() > 0.0
            or _log_std_grad_norm(agent, agent._route_action_indices()) > 0.0
        )
        and bool(
            agent._theta_head_grad_norm() == 0.0
            and agent._theta_backbone_grad_norm() == 0.0
            and _log_std_grad_norm(agent, agent._theta_action_indices()) == 0.0
        ),
    }
    route_optimizer.zero_grad()
    return result


def _run_full_control_step(agent: PPOAgent, prepared: dict[str, torch.Tensor]) -> dict[str, Any]:
    route_loss_mask = prepared["route_loss_masks"] > 0.5
    route_has_active_blocks = bool(route_loss_mask.any().item())
    route_branch_stopped = False
    route_step_blocked_by_other_guard = not route_has_active_blocks
    route_step_blocked_by_trust_region = False
    route_step_blocked_by_coupled_stop = False
    optimizer_step_executed = False

    if not route_has_active_blocks:
        return {
            "route_update_count": 0,
            "route_step_blocked_by_trust_region": False,
            "route_step_blocked_by_coupled_stop": False,
            "route_step_blocked_by_other_guard": True,
            "final_route_head_grad_norm": 0.0,
            "final_route_backbone_grad_norm": 0.0,
            "final_route_logstd_grad_norm": 0.0,
            "optimizer_step_executed": False,
            "route_param_delta_norm": 0.0,
            "theta_param_delta_norm": 0.0,
            "route_params_changed": False,
            "theta_params_changed": False,
        }

    route_optimizer = agent._route_actor_optimizer()
    theta_params = _theta_param_groups(agent)
    route_params = _route_param_groups(agent)
    theta_before = _snapshot_params(theta_params)
    route_before = _snapshot_params(route_params)
    theta_log_std_before = agent.network.actor_log_std.detach()[agent._theta_action_indices()].clone()
    route_log_std_before = agent.network.actor_log_std.detach()[agent._route_action_indices()].clone()

    route_optimizer.zero_grad()
    prepared["route_only_objective"].backward()
    agent._apply_route_only_gradient_mask()
    final_route_head_grad_norm = agent._route_head_grad_norm()
    final_route_backbone_grad_norm = agent._route_backbone_grad_norm()
    final_route_logstd_grad_norm = _log_std_grad_norm(agent, agent._route_action_indices())
    torch.nn.utils.clip_grad_norm_(agent.actor_params, agent.config.max_grad_norm)
    route_optimizer.step()
    optimizer_step_executed = True

    agent._restore_parameter_list(theta_params, theta_before)
    agent._restore_actor_log_std_indices(agent._theta_action_indices(), theta_log_std_before)

    route_after = _snapshot_params(route_params)
    theta_after = _snapshot_params(theta_params)
    route_param_delta_norm = _parameter_delta_norm(route_before, route_after)
    theta_param_delta_norm = _parameter_delta_norm(theta_before, theta_after)
    route_log_std_delta_norm = float(
        (
            agent.network.actor_log_std.detach()[agent._route_action_indices()]
            - route_log_std_before
        )
        .norm()
        .item()
    )
    theta_log_std_delta_norm = float(
        (
            agent.network.actor_log_std.detach()[agent._theta_action_indices()]
            - theta_log_std_before
        )
        .norm()
        .item()
    )

    if agent._uses_factorized_trust_region_pg():
        with torch.no_grad():
            post_distribution = agent.network.policy_from_actor_input(prepared["actor_inputs"])
            post_route_log_probs = agent._true_conditional_route_policy_terms(
                prepared["actions"],
                post_distribution.mean,
            )["log_probs"]
            route_branch_approx_kl = agent._masked_mean(
                prepared["route_branch_terms"]["old_log_probs"] - post_route_log_probs,
                route_loss_mask,
            )
            if route_branch_approx_kl > agent.config.route_kl_target:
                route_step_blocked_by_trust_region = True
                route_branch_stopped = True

    if agent._uses_coupled_stop_factorized_trust_region_pg() and route_branch_stopped:
        route_step_blocked_by_coupled_stop = True

    route_optimizer.zero_grad()
    return {
        "route_update_count": 1 if optimizer_step_executed and not route_step_blocked_by_other_guard else 0,
        "route_step_blocked_by_trust_region": bool(route_step_blocked_by_trust_region),
        "route_step_blocked_by_coupled_stop": bool(route_step_blocked_by_coupled_stop),
        "route_step_blocked_by_other_guard": bool(route_step_blocked_by_other_guard),
        "final_route_head_grad_norm": final_route_head_grad_norm,
        "final_route_backbone_grad_norm": final_route_backbone_grad_norm,
        "final_route_logstd_grad_norm": final_route_logstd_grad_norm,
        "optimizer_step_executed": bool(optimizer_step_executed),
        "route_param_delta_norm": route_param_delta_norm + route_log_std_delta_norm,
        "theta_param_delta_norm": theta_param_delta_norm + theta_log_std_delta_norm,
        "route_params_changed": bool(route_param_delta_norm > 0.0 or route_log_std_delta_norm > 0.0),
        "theta_params_changed": bool(theta_param_delta_norm > 0.0 or theta_log_std_delta_norm > 0.0),
    }


def _run_mode_audit(
    mode: str,
    seed: int,
    checkpoint_path: Path,
    data: dict[str, torch.Tensor],
    batch_index: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, float | int | str]]]:
    agent, _ = _build_agent(mode, seed)
    agent.load(str(checkpoint_path), load_optimizer=False)
    prepared = _prepare_batch(agent, data, batch_index)

    route_loss_mask = prepared["route_loss_masks"] > 0.5
    if prepared["candidate_terms"] is not None:
        candidate_score_vector = prepared["candidate_terms"]["candidate_score_vector"]
        selected_score = prepared["candidate_terms"]["selected_score"]
        expected_score = prepared["candidate_terms"]["expected_score"]
        effective_advantage = prepared["candidate_terms"]["residual_credit"]
    else:
        route_adv = prepared["route_advantages_for_surrogate"]
        candidate_score_vector = torch.stack((0.5 * route_adv, -0.5 * route_adv), dim=-1)
        selected_indices = prepared["route_terms"]["selected_indices"]
        selected_score = candidate_score_vector.gather(
            dim=-1,
            index=selected_indices.unsqueeze(-1),
        ).squeeze(-1)
        expected_score = (
            prepared["route_old_policy_terms_for_credit"]["probs"] * candidate_score_vector
        ).sum(dim=-1)
        effective_advantage = route_adv

    summary = {
        "mode": mode,
        "route_active_count": int(route_loss_mask.sum().item()),
        "route_active_fraction": float(route_loss_mask.float().mean().item()),
        "selected_route_index_stats": _masked_stats(
            prepared["route_terms"]["selected_indices"].float(),
            route_loss_mask,
        ),
        "candidate_score_vector_stats": _masked_stats(
            candidate_score_vector.reshape(-1),
            route_loss_mask.unsqueeze(-1).expand_as(candidate_score_vector).reshape(-1),
        ),
        "selected_score_stats": _masked_stats(selected_score, route_loss_mask),
        "expected_score_stats": _masked_stats(expected_score, route_loss_mask),
        "route_effective_advantage_stats": _masked_stats(effective_advantage, route_loss_mask),
        "route_old_logprob_stats": _masked_stats(
            prepared["route_branch_terms"]["old_log_probs"],
            route_loss_mask,
        ),
        "route_new_logprob_stats": _masked_stats(
            prepared["route_branch_terms"]["new_log_probs"],
            route_loss_mask,
        ),
        "route_ratio_stats": _masked_stats(
            prepared["route_branch_terms"]["ratio"],
            route_loss_mask,
        ),
        "route_surrogate_stats": _masked_stats(
            prepared["route_branch_terms"]["surrogate"],
            route_loss_mask,
        ),
        "route_loss": float(prepared["route_branch_terms"]["loss"].item()),
    }

    bare_agent, _ = _build_agent(mode, seed)
    bare_agent.load(str(checkpoint_path), load_optimizer=False)
    bare_prepared = _prepare_batch(bare_agent, data, batch_index)
    bare_summary = _run_bare_backward(bare_agent, bare_prepared)

    full_agent, _ = _build_agent(mode, seed)
    full_agent.load(str(checkpoint_path), load_optimizer=False)
    full_prepared = _prepare_batch(full_agent, data, batch_index)
    full_summary = _run_full_control_step(full_agent, full_prepared)

    summary["bare_backward"] = bare_summary
    summary["full_control"] = full_summary
    raw_rows = _collect_raw_active_rows(mode, prepared)
    return summary, raw_rows


def _infer_conclusion(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_bare = candidate["bare_backward"]
    candidate_full = candidate["full_control"]

    if (
        candidate_bare["route_head_grad_norm"] == 0.0
        and candidate_bare["route_backbone_grad_norm"] == 0.0
        and candidate_bare["route_logstd_grad_norm"] == 0.0
    ):
        case = 1
        diagnosis = "裸 backward 下 route 梯度已经为 0，问题在 credit -> route loss 的计算图本身。"
    elif not candidate_full["optimizer_step_executed"] or candidate_full["route_update_count"] == 0:
        case = 2
        diagnosis = "裸 backward 有梯度，但完整控制逻辑把 route step 压掉了。"
    else:
        case = 3
        diagnosis = "裸 backward 和完整控制逻辑都正常，单 minibatch 路径已走通，更像是完整训练中的动态因素。"

    return {
        "case": case,
        "diagnosis": diagnosis,
        "baseline_route_grad_norm": baseline["bare_backward"]["route_head_grad_norm"]
        + baseline["bare_backward"]["route_backbone_grad_norm"],
        "candidate_route_grad_norm": candidate_bare["route_head_grad_norm"]
        + candidate_bare["route_backbone_grad_norm"],
        "candidate_optimizer_step_executed": bool(candidate_full["optimizer_step_executed"]),
        "candidate_route_update_count": int(candidate_full["route_update_count"]),
        "candidate_route_params_changed": bool(candidate_full["route_params_changed"]),
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    baseline = summary["baseline"]
    candidate = summary["candidate"]
    conclusion = summary["conclusion"]
    lines = [
        "# Route Credit Path Audit",
        "",
        f"- shared checkpoint: `{summary['shared_checkpoint_path']}`",
        f"- rollout seed: `{summary['rollout_seed']}`",
        f"- fixed minibatch size: `{summary['fixed_minibatch_size']}`",
        f"- fixed minibatch source: `{summary['fixed_minibatch_source']}`",
        "",
        "## Conclusion",
        "",
        f"- case: `{conclusion['case']}`",
        f"- diagnosis: {conclusion['diagnosis']}",
        "",
        "## Key Results",
        "",
        "| mode | bare route head grad | bare route backbone grad | full route update_count | optimizer.step | route params changed |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| {baseline['mode']} | {baseline['bare_backward']['route_head_grad_norm']:.6f} | "
            f"{baseline['bare_backward']['route_backbone_grad_norm']:.6f} | "
            f"{baseline['full_control']['route_update_count']} | "
            f"{int(baseline['full_control']['optimizer_step_executed'])} | "
            f"{int(baseline['full_control']['route_params_changed'])} |"
        ),
        (
            f"| {candidate['mode']} | {candidate['bare_backward']['route_head_grad_norm']:.6f} | "
            f"{candidate['bare_backward']['route_backbone_grad_norm']:.6f} | "
            f"{candidate['full_control']['route_update_count']} | "
            f"{int(candidate['full_control']['optimizer_step_executed'])} | "
            f"{int(candidate['full_control']['route_params_changed'])} |"
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else _find_latest_baseline_checkpoint()
    )
    output_root = Path(args.output_root)
    audit_dir = output_root / datetime.now().strftime("route_credit_path_audit_%Y%m%d_%H%M%S")
    audit_dir.mkdir(parents=True, exist_ok=True)

    rollout_agent, rollout_config = _build_agent(BASELINE_MODE, args.seed)
    rollout_agent.load(str(checkpoint_path), load_optimizer=False)
    data = _collect_rollout(
        rollout_agent,
        rollout_config,
        seed=args.seed,
    )

    generator = torch.Generator(device=rollout_agent.device)
    generator.manual_seed(args.seed + 17)
    permutation = torch.randperm(data["states"].size(0), generator=generator, device=rollout_agent.device)
    batch_size = min(rollout_config.ppo.mini_batch_size, int(data["states"].size(0)))
    batch_index = permutation[:batch_size]

    baseline_summary, baseline_rows = _run_mode_audit(
        BASELINE_MODE,
        args.seed,
        checkpoint_path,
        data,
        batch_index,
    )
    candidate_summary, candidate_rows = _run_mode_audit(
        CANDIDATE_MODE,
        args.seed,
        checkpoint_path,
        data,
        batch_index,
    )

    conclusion = _infer_conclusion(baseline_summary, candidate_summary)
    summary = {
        "shared_checkpoint_path": str(checkpoint_path),
        "rollout_seed": args.seed,
        "fixed_minibatch_size": int(batch_size),
        "fixed_minibatch_source": (
            "single rollout collected once from the shared baseline checkpoint, "
            "then replayed through both audit modes with identical batch indices"
        ),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "conclusion": conclusion,
    }

    raw_rows = baseline_rows + candidate_rows
    pd.DataFrame(raw_rows).to_csv(audit_dir / "route_credit_path_audit_raw.csv", index=False)
    (audit_dir / "route_credit_path_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (audit_dir / "route_credit_path_audit_baseline.json").write_text(
        json.dumps(baseline_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (audit_dir / "route_credit_path_audit_candidate_score.json").write_text(
        json.dumps(candidate_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, audit_dir / "route_credit_path_audit_summary.md")

    print(f"[route_credit_path_audit] wrote results to {audit_dir}")


if __name__ == "__main__":
    main()
