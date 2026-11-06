"""
Training Script
Executes curriculum training of the Edge-Aware GNN-PPO agent (V2) or standard PPO agent.
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.training_manager import train_improved_ppo

def main():
    parser = argparse.ArgumentParser(description="Train GNN-PPO Agent for Molecular Optimization")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--rollout_steps", type=int, default=128, help="Rollout steps per PPO update")
    parser.add_argument("--epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="PPO minibatch size")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--curriculum", action="store_true", default=True, help="Use 4-stage QED curriculum")
    parser.add_argument("--output_ckpt", type=str, default="checkpoints/adaptive_v2/ppo_adaptive_v2_improved.pt")
    parser.add_argument("--output_log", type=str, default="logs/training/train_ppo_improved.csv")
    args = parser.parse_args()

    print(f"Starting PPO Training: {args.episodes} episodes | Curriculum: {args.curriculum}")
    train_improved_ppo(
        total_episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_curriculum=args.curriculum,
        output_ckpt=os.path.join(PROJECT_ROOT, args.output_ckpt),
        output_log=os.path.join(PROJECT_ROOT, args.output_log)
    )

if __name__ == "__main__":
    main()

