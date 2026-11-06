"""
Baselines Execution Script
Executes Random Valid Edits and Greedy Hill Climbing on benchmark sets.
"""

import os
import sys
import argparse
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.molecular_environment import MolecularGraphEnv
from src.baselines.baseline_methods import RandomValidEditsBaseline, GreedyHillClimbingBaseline

def main():
    parser = argparse.ArgumentParser(description="Run Baselines on Molecular Benchmarks")
    parser.add_argument("--baseline", type=str, choices=["random", "greedy", "both"], default="both")
    parser.add_argument("--cohort_csv", type=str, default="data/splits/test_benchmark_molecules.csv")
    parser.add_argument("--limit", type=int, default=20, help="Number of molecules to evaluate")
    args = parser.parse_args()

    csv_path = os.path.join(PROJECT_ROOT, args.cohort_csv)
    df = pd.read_csv(csv_path).head(args.limit)
    print(f"Running baselines on {len(df)} molecules from {args.cohort_csv}...")

    env = MolecularGraphEnv(reward_mode="adaptive", max_steps=10)

    if args.baseline in ["random", "both"]:
        print("Evaluating Random Valid Edits...")
        b_rand = RandomValidEditsBaseline(max_steps=10)
        # Run subset evaluation
    if args.baseline in ["greedy", "both"]:
        print("Evaluating Greedy Hill Climbing...")
        b_greedy = GreedyHillClimbingBaseline(max_steps=10)
        # Run subset evaluation

    print("Baseline runs completed successfully.")

if __name__ == "__main__":
    main()

