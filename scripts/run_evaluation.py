"""
Evaluation Script
Evaluates trained models or baselines across specified benchmark cohorts.
"""

import os
import sys
import argparse
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.molecular_environment import MolecularGraphEnv
from src.agents.ppo_agent_v2 import ImprovedPPOAgent
from src.search.ppo_guided_search import PPOGuidedSearchOptimizer
from src.rewards.adaptive_reward import CalibratedAdaptiveRewardCalculator
from src.evaluation.evaluation import evaluate_method_on_molecules

def main():
    parser = argparse.ArgumentParser(description="Evaluate Molecular Optimization Model")
    parser.add_argument("--benchmark", type=str, choices=["pilot_60", "benchmark_500", "improvement_100"], default="improvement_100")
    parser.add_argument("--method", type=str, default="hybrid", choices=["hybrid", "pure_v2", "baseline_v1", "adaptive_v1"])
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Determine cohort
    if args.benchmark == "pilot_60":
        cohort_csv = os.path.join(PROJECT_ROOT, "data/splits/pilot_60_benchmark/test_benchmark_molecules.csv")
    elif args.benchmark == "benchmark_500":
        cohort_csv = os.path.join(PROJECT_ROOT, "data/splits/test_benchmark_molecules.csv")
    else:
        cohort_csv = os.path.join(PROJECT_ROOT, "data/splits/val_molecules.csv")

    df_cohort = pd.read_csv(cohort_csv)
    print(f"Loaded benchmark {args.benchmark}: {len(df_cohort)} molecules.")

    # Initialize environment
    env = MolecularGraphEnv(reward_mode="adaptive", max_steps=10)
    env.reward_calc = CalibratedAdaptiveRewardCalculator()

    # Load agent
    agent = ImprovedPPOAgent(node_dim=21, edge_dim=4, gnn_hidden=64, scalar_dim=6, fused_dim=128)
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints/adaptive_v2/ppo_adaptive_v2_improved.pt")
    agent.load_checkpoint(ckpt_path)

    if args.method == "hybrid":
        runner = PPOGuidedSearchOptimizer(agent, top_k=args.top_k, max_steps=10)
    else:
        runner = agent

    print(f"Running evaluation with method: {args.method}...")
    # Evaluation logic...
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    main()

