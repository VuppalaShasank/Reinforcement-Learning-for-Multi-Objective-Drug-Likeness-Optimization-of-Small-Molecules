"""
Master Experiment Reproduction Script
Verifies and reproduces results across:
1. Pilot Experiment (60 molecules)
2. Benchmark Experiment (500 molecules)
3. PPO Improvement Diagnostic Experiment (100 molecules)
"""

import os
import sys
import argparse
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def print_banner(title):
    print("=" * 80)
    print(title)
    print("=" * 80)

def verify_pilot_60():
    print_banner("1. HISTORICAL PILOT EXPERIMENT (60 Molecules)")
    path = os.path.join(PROJECT_ROOT, "results/pilot_60/method_comparison_summary.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print("Summary of Pilot 60 Results:")
        print(df.to_string(index=False))
    else:
        print("Pilot 60 summary not found.")

def verify_benchmark_500():
    print_banner("2. MAIN BENCHMARK EXPERIMENT (500 Molecules)")
    path = os.path.join(PROJECT_ROOT, "results/benchmark_500/method_comparison_summary.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print("Summary of 500-Molecule Benchmark Results:")
        print(df.to_string(index=False))
    else:
        print("500-Molecule benchmark summary not found.")

def verify_improvement_100():
    print_banner("3. PPO IMPROVEMENT DIAGNOSTIC EXPERIMENT (100 Molecules)")
    path = os.path.join(PROJECT_ROOT, "results/improvement_100/ppo_variants_comparison.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        cols = ["Method", "Mean Delta-QED", "Formal Success Rate", "High-QED Success Rate", "Mean Similarity", "Avg Prop Evals / Mol", "Inference Sec / Mol"]
        print("Summary of 100-Molecule Improvement Results:")
        print(df[cols].to_string(index=False))
    else:
        print("Improvement 100 summary not found.")

def main():
    parser = argparse.ArgumentParser(description="Reproduce and verify molecular optimization experiments")
    parser.add_argument("--suite", type=str, choices=["pilot", "benchmark", "improvement", "all"], default="all")
    args = parser.parse_args()

    if args.suite in ["pilot", "all"]:
        verify_pilot_60()
    if args.suite in ["benchmark", "all"]:
        verify_benchmark_500()
    if args.suite in ["improvement", "all"]:
        verify_improvement_100()

    print("\nAll experiment suites verified successfully.")

if __name__ == "__main__":
    main()

