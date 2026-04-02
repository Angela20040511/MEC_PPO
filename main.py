"""主程序入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from rl.ppo_agent import PPOAgent
from safety import validate_runtime_config
from train import build_runtime_config, rollout_agent, train_agent


def find_latest_best_checkpoint(checkpoint_dir: str) -> Path | None:
    """在 checkpoint 根目录下寻找最新一版 best_model.pt。"""
    base_dir = Path(checkpoint_dir)
    if not base_dir.exists():
        return None

    candidates = []
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            best_ckpt = subdir / "best_model.pt"
            if best_ckpt.exists():
                candidates.append(best_ckpt)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    """解析命令行参数并执行训练或回放。"""
    parser = argparse.ArgumentParser(description="MEC-PPO industrial edge computing simulator")
    parser.add_argument("--mode", choices=["train", "rollout", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=None, help="Override PPO training epochs")
    parser.add_argument("--time_steps", type=int, default=None, help="Override time steps per episode")
    parser.add_argument(
        "--topology",
        choices=["default", "dense"],
        default="default",
        help="Topology preset to use for training or rollout.",
    )
    parser.add_argument("--steps", type=int, default=50, help="Rollout steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint root directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint path for rollout",
    )
    parser.add_argument(
        "--allow-legacy-mode",
        action="store_true",
        help="Bypass runtime safety guard that blocks known legacy dense policy modes.",
    )
    args = parser.parse_args()

    config = build_runtime_config(
        num_epochs=args.epochs,
        time_steps=args.time_steps,
        topology_mode=args.topology,
    )
    validate_runtime_config(
        config,
        topology_mode=args.topology,
        allow_legacy_mode=args.allow_legacy_mode,
    )

    if args.mode == "train":
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_checkpoint_dir = Path(args.checkpoint_dir) / run_tag
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"run_tag: {run_tag}")

        _, logs = train_agent(config, checkpoint_dir=str(run_checkpoint_dir))
        for log in logs:
            print(log)

        print(f"saved best model to: {run_checkpoint_dir / 'best_model.pt'}")
        print(f"saved last model to: {run_checkpoint_dir / 'last_model.pt'}")
        print(f"saved train logs to: {run_checkpoint_dir / 'train_logs.json'}")
        print(f"saved train logs to: {run_checkpoint_dir / 'train_logs.csv'}")
        print(f"saved run artifacts to: {run_checkpoint_dir}")
        return

    if args.mode == "rollout":
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = find_latest_best_checkpoint(args.checkpoint_dir)

        agent = PPOAgent(config=config.ppo, state_dim=config.state_dim, action_dim=config.action_dim)

        if checkpoint_path is not None and checkpoint_path.exists():
            agent.load(str(checkpoint_path), load_optimizer=False)
            print(f"loaded checkpoint: {checkpoint_path}")
        else:
            print("checkpoint not found")
            print("please provide --checkpoint or train a model first")
            return

        for item in rollout_agent(agent, config=config, steps=args.steps):
            print(item)
        return

    # full mode
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_checkpoint_dir = Path(args.checkpoint_dir) / run_tag
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_tag: {run_tag}")

    _, logs = train_agent(config, checkpoint_dir=str(run_checkpoint_dir))
    for log in logs:
        print(log)

    best_checkpoint = run_checkpoint_dir / "best_model.pt"
    rollout_agent_model = PPOAgent(config=config.ppo, state_dim=config.state_dim, action_dim=config.action_dim)

    if best_checkpoint.exists():
        rollout_agent_model.load(str(best_checkpoint), load_optimizer=False)
        print(f"loaded best checkpoint for rollout: {best_checkpoint}")
    else:
        print("best checkpoint not found in current run directory")
        return

    for item in rollout_agent(rollout_agent_model, config=config, steps=args.steps):
        print(item)

    print(f"saved best model to: {run_checkpoint_dir / 'best_model.pt'}")
    print(f"saved last model to: {run_checkpoint_dir / 'last_model.pt'}")
    print(f"saved train logs to: {run_checkpoint_dir / 'train_logs.json'}")
    print(f"saved train logs to: {run_checkpoint_dir / 'train_logs.csv'}")
    print(f"saved run artifacts to: {run_checkpoint_dir}")


if __name__ == "__main__":
    main()
