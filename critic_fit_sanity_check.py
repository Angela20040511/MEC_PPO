import argparse
import copy
import json
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
from dense_policy_ratio_mode_experiment import build_policy_ratio_mode_config
from fine_grained_epoch_early_stop_search import set_global_seeds
from rl.ppo_agent import PPOAgent
from simulator.simulator import Simulator


BASELINE_MODE = "hierarchical_actor_joint_reward_aligned_credit"
GROUP_A = "A_value_target"
GROUP_B = "B_neg_value_target"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-buffer critic-only sanity check comparing current value target "
            "against sign-flipped value target."
        )
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--output-root", type=str, default="checkpoints")
    parser.add_argument("--fit-steps", type=int, default=100)
    parser.add_argument("--checkpoint-path", type=str, default="")
    return parser.parse_args()


def _find_latest_baseline_checkpoint() -> Path:
    matches = sorted(
        Path("checkpoints").glob(f"**/policy_ratio_{BASELINE_MODE}_*/best_model.pt")
    )
    if not matches:
        raise FileNotFoundError(
            f"No baseline best_model.pt found for mode {BASELINE_MODE}"
        )
    return matches[-1]


def _build_agent(seed: int) -> tuple[PPOAgent, Any]:
    config = build_policy_ratio_mode_config(policy_ratio_mode=BASELINE_MODE, seed=seed)
    state_layout = build_state_layout(config)
    agent = PPOAgent(
        config=config.ppo,
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        critic_state_layout=state_layout,
    )
    return agent, config


def _collect_fixed_buffer(agent: PPOAgent, config: Any, seed: int) -> dict[str, torch.Tensor]:
    simulator = Simulator(config)
    state = simulator.reset(seed=seed)
    done = False

    for _ in range(config.training.time_steps):
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done, _ = simulator.step(action)
        agent.store_transition(state, action, log_prob, reward, done, value, next_state)
        state = next_state
        if done:
            break

    last_value = 0.0 if done else agent.evaluate_value(state)
    agent.finish_trajectory(last_value)
    return agent.buffer.as_tensors(agent.device)


def _simulate_running_return_stats(
    agent: PPOAgent,
    returns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    old_mean, old_std = agent._running_stats_tensors(returns)
    batch_mean = float(returns.mean().item())
    batch_var = float(returns.var(unbiased=False).item())
    batch_count = float(returns.numel())

    if agent.running_return_count == 0.0:
        target_mean = batch_mean
        target_var = max(batch_var, 1e-8)
        total_count = batch_count
    else:
        delta = batch_mean - agent.running_return_mean
        total_count = agent.running_return_count + batch_count
        mean = agent.running_return_mean + delta * batch_count / total_count
        m_a = agent.running_return_var * agent.running_return_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta**2) * agent.running_return_count * batch_count / total_count
        target_mean = mean
        target_var = max(m2 / total_count, 1e-8)

    target_mean_tensor = torch.tensor(
        target_mean,
        dtype=returns.dtype,
        device=returns.device,
    )
    target_std_tensor = torch.tensor(
        float(np.sqrt(target_var)),
        dtype=returns.dtype,
        device=returns.device,
    )
    return old_mean, old_std, target_mean_tensor, target_std_tensor


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    if left.empty or right.empty:
        return 0.0
    value = left.corr(right, method=method)
    return 0.0 if pd.isna(value) else float(value)


def _prepare_group_agent(
    checkpoint_path: Path,
    seed: int,
    returns: torch.Tensor,
) -> tuple[PPOAgent, torch.Tensor, torch.Tensor, torch.Tensor]:
    agent, _ = _build_agent(seed)
    agent.load(str(checkpoint_path), load_optimizer=False)
    old_mean, old_std, target_mean, target_std = _simulate_running_return_stats(agent, returns)
    with torch.no_grad():
        agent.network.popart_rescale(old_mean, old_std, target_mean, target_std)
    agent.running_return_mean = float(target_mean.item())
    agent.running_return_var = float(target_std.item() ** 2)
    agent.running_return_count = max(float(agent.running_return_count), float(returns.numel()))

    for parameter in agent.network.parameters():
        parameter.requires_grad = False
    for parameter in agent.network.critic_backbone.parameters():
        parameter.requires_grad = True
    for parameter in agent.network.critic_head.parameters():
        parameter.requires_grad = True

    critic_only_params = list(agent.network.critic_backbone.parameters()) + list(
        agent.network.critic_head.parameters()
    )
    agent.critic_optimizer = torch.optim.Adam(
        critic_only_params,
        lr=agent.config.critic_learning_rate,
    )
    return agent, old_mean, target_mean, target_std


def _fit_group(
    group_name: str,
    checkpoint_path: Path,
    seed: int,
    data: dict[str, torch.Tensor],
    fit_steps: int,
) -> tuple[list[dict[str, float | int | str]], dict[str, Any]]:
    states = data["states"]
    next_states = data["next_states"]
    returns = data["returns"]
    advantages = data["advantages"]
    dones = data["dones"]

    agent, _, target_mean, target_std = _prepare_group_agent(checkpoint_path, seed, returns)
    with torch.no_grad():
        critic_inputs, _, _ = agent._prepare_critic_inputs(states, update_stats=False)
        next_values_raw = agent._value_from_state_tensor(next_states).squeeze(-1)

    current_value_target = (returns - target_mean) / (target_std + 1e-8)
    if group_name == GROUP_A:
        fit_target = current_value_target.detach()
    elif group_name == GROUP_B:
        fit_target = (-current_value_target).detach()
    else:
        raise ValueError(f"Unsupported group: {group_name}")

    one_step_target = (
        data["returns"].new_tensor(
            np.asarray(agent.buffer.rewards if hasattr(agent, "buffer") else []),
            dtype=torch.float32,
        )
        if False
        else None
    )

    metrics: list[dict[str, float | int | str]] = []
    for step in range(fit_steps + 1):
        with torch.no_grad():
            normalized_pred = agent.network.normalized_value_from_critic_input(critic_inputs).squeeze(-1)
            raw_pred = normalized_pred * target_std + target_mean

        pred_series = pd.Series(normalized_pred.detach().cpu().numpy())
        raw_pred_series = pd.Series(raw_pred.detach().cpu().numpy())
        fit_target_series = pd.Series(fit_target.detach().cpu().numpy())
        returns_series = pd.Series(returns.detach().cpu().numpy())
        advantages_series = pd.Series(advantages.detach().cpu().numpy())

        loss_value = agent._compute_value_loss(normalized_pred, fit_target).item()
        metrics.append(
            {
                "group_name": group_name,
                "step": step,
                "critic_loss": float(loss_value),
                "pred_mean": float(pred_series.mean()),
                "pred_std": float(pred_series.std(ddof=0)),
                "target_mean": float(fit_target_series.mean()),
                "target_std": float(fit_target_series.std(ddof=0)),
                "raw_pred_mean": float(raw_pred_series.mean()),
                "raw_pred_std": float(raw_pred_series.std(ddof=0)),
                "pearson_pred_vs_value_target": _safe_corr(pred_series, fit_target_series, "pearson"),
                "spearman_pred_vs_value_target": _safe_corr(pred_series, fit_target_series, "spearman"),
                "pearson_pred_vs_return": _safe_corr(raw_pred_series, returns_series, "pearson"),
                "spearman_pred_vs_return": _safe_corr(raw_pred_series, returns_series, "spearman"),
                "pearson_pred_vs_advantage": _safe_corr(raw_pred_series, advantages_series, "pearson"),
                "spearman_pred_vs_advantage": _safe_corr(raw_pred_series, advantages_series, "spearman"),
                "mae_pred_vs_value_target": float(np.mean(np.abs(pred_series - fit_target_series))),
            }
        )

        if step == fit_steps:
            break

        agent.critic_optimizer.zero_grad(set_to_none=True)
        normalized_pred_train = agent.network.normalized_value_from_critic_input(critic_inputs).squeeze(-1)
        loss = agent._compute_value_loss(normalized_pred_train, fit_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(agent.network.critic_backbone.parameters()) + list(agent.network.critic_head.parameters()),
            agent.config.max_grad_norm,
        )
        agent.critic_optimizer.step()

    return metrics, {
        "group_name": group_name,
        "initial": metrics[0],
        "final": metrics[-1],
        "best_loss": float(min(row["critic_loss"] for row in metrics)),
        "fit_target_definition": (
            "current value_target"
            if group_name == GROUP_A
            else "negated current value_target"
        ),
    }


def _plot_curves(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for group_name, group_df in metrics_df.groupby("group_name"):
        plt.plot(group_df["step"], group_df["critic_loss"], marker="o", label=group_name)
    plt.xlabel("step")
    plt.ylabel("critic_loss")
    plt.title("Critic Fit Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "critic_fit_loss_curve_compare.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for group_name, group_df in metrics_df.groupby("group_name"):
        plt.plot(
            group_df["step"],
            group_df["pearson_pred_vs_value_target"],
            marker="o",
            label=f"{group_name} pred-target",
        )
    plt.xlabel("step")
    plt.ylabel("pearson correlation")
    plt.title("Critic Fit Correlation Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "critic_fit_corr_curve_compare.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else _find_latest_baseline_checkpoint()
    )

    buffer_agent, config = _build_agent(args.seed)
    buffer_agent.load(str(checkpoint_path), load_optimizer=False)
    data = _collect_fixed_buffer(buffer_agent, config, args.seed)

    all_metrics: list[dict[str, float | int | str]] = []
    group_summaries: list[dict[str, Any]] = []
    for group_name in [GROUP_A, GROUP_B]:
        metrics, summary = _fit_group(
            group_name=group_name,
            checkpoint_path=checkpoint_path,
            seed=args.seed,
            data=data,
            fit_steps=int(args.fit_steps),
        )
        all_metrics.extend(metrics)
        group_summaries.append(summary)

    metrics_df = pd.DataFrame(all_metrics)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"critic_fit_sanity_check_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "critic_fit_sanity_check_metrics.csv"
    summary_json_path = output_dir / "critic_fit_sanity_check_summary.json"
    summary_md_path = output_dir / "critic_fit_sanity_check_summary.md"
    group_a_path = output_dir / "critic_fit_group_A_value_target.csv"
    group_b_path = output_dir / "critic_fit_group_B_neg_value_target.csv"

    metrics_df.to_csv(metrics_path, index=False)
    metrics_df[metrics_df["group_name"] == GROUP_A].to_csv(group_a_path, index=False)
    metrics_df[metrics_df["group_name"] == GROUP_B].to_csv(group_b_path, index=False)
    _plot_curves(metrics_df, output_dir)

    summary = {
        "metadata": {
            "baseline_mode": BASELINE_MODE,
            "checkpoint_path": str(checkpoint_path),
            "seed": args.seed,
            "fit_steps": int(args.fit_steps),
            "buffer_steps": int(len(data["returns"])),
            "critic_only_updated_parameters": [
                "network.critic_backbone.*",
                "network.critic_head.*",
            ],
            "frozen_parameters": [
                "all actor parameters",
                "critic_block_value_head.*",
                "critic_block_path_value_head.*",
            ],
        },
        "group_summaries": group_summaries,
        "comparison": {
            "group_a_final_loss": group_summaries[0]["final"]["critic_loss"],
            "group_b_final_loss": group_summaries[1]["final"]["critic_loss"],
            "group_a_final_corr_pred_target": group_summaries[0]["final"]["pearson_pred_vs_value_target"],
            "group_b_final_corr_pred_target": group_summaries[1]["final"]["pearson_pred_vs_value_target"],
            "group_a_loss_drop": group_summaries[0]["initial"]["critic_loss"] - group_summaries[0]["final"]["critic_loss"],
            "group_b_loss_drop": group_summaries[1]["initial"]["critic_loss"] - group_summaries[1]["final"]["critic_loss"],
            "easier_group": (
                GROUP_A
                if group_summaries[0]["final"]["critic_loss"] < group_summaries[1]["final"]["critic_loss"]
                else GROUP_B
            ),
        },
    }

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    md_lines = [
        "# Critic Fit Sanity Check",
        "",
        f"- baseline_mode: `{BASELINE_MODE}`",
        f"- checkpoint_path: `{checkpoint_path}`",
        f"- fit_steps: {args.fit_steps}",
        f"- buffer_steps: {len(data['returns'])}",
        "",
        "## Group A",
        f"- initial loss: {group_summaries[0]['initial']['critic_loss']:.6f}",
        f"- final loss: {group_summaries[0]['final']['critic_loss']:.6f}",
        f"- final pred-target pearson: {group_summaries[0]['final']['pearson_pred_vs_value_target']:.4f}",
        f"- final pred-return pearson: {group_summaries[0]['final']['pearson_pred_vs_return']:.4f}",
        "",
        "## Group B",
        f"- initial loss: {group_summaries[1]['initial']['critic_loss']:.6f}",
        f"- final loss: {group_summaries[1]['final']['critic_loss']:.6f}",
        f"- final pred-target pearson: {group_summaries[1]['final']['pearson_pred_vs_value_target']:.4f}",
        f"- final pred-return pearson: {group_summaries[1]['final']['pearson_pred_vs_return']:.4f}",
        "",
        "## Comparison",
        f"- easier_group: {summary['comparison']['easier_group']}",
        f"- group_a_loss_drop: {summary['comparison']['group_a_loss_drop']:.6f}",
        f"- group_b_loss_drop: {summary['comparison']['group_b_loss_drop']:.6f}",
    ]
    with summary_md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines) + "\n")

    print(f"Saved summary JSON to {summary_json_path}")
    print(f"Saved summary MD to {summary_md_path}")
    print(f"Saved metrics CSV to {metrics_path}")


if __name__ == "__main__":
    main()
