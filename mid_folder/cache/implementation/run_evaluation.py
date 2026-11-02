import torch
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import RDLogger

# Suppress verbose C++ logs during batch evaluation
RDLogger.DisableLog('rdApp.*')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from implementation.environment import MolecularGraphEnv
from implementation.ppo_agent import PPOAgent
from implementation.baselines import RandomValidEditsBaseline, GreedyHillClimbingBaseline
from implementation.evaluator import evaluate_policy_on_molecules

def run_comprehensive_evaluation():
    test_csv = os.path.join("results", "test_benchmark_molecules.csv") if os.path.exists(os.path.join("results", "test_benchmark_molecules.csv")) else os.path.join("cache", "results", "test_benchmark_molecules.csv")
    assert os.path.exists(test_csv), f"Test benchmark file not found at {test_csv}"
    
    df_test = pd.read_csv(test_csv)
    test_smiles = df_test["smiles"].tolist()
    print(f"Loaded {len(test_smiles)} held-out test molecules across QED groups.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}")

    # Environment instances
    env_eval = MolecularGraphEnv(reward_mode="adaptive", max_steps=10)

    # 1. Load Trained PPO Models
    ckpt_base = os.path.join("checkpoints", "ppo_baseline.pt") if os.path.exists(os.path.join("checkpoints", "ppo_baseline.pt")) else os.path.join("cache", "checkpoints", "ppo_baseline.pt")
    ckpt_adapt = os.path.join("checkpoints", "ppo_adaptive.pt") if os.path.exists(os.path.join("checkpoints", "ppo_adaptive.pt")) else os.path.join("cache", "checkpoints", "ppo_adaptive.pt")

    ppo_base = PPOAgent(device=device)
    ppo_base.load_checkpoint(ckpt_base)
    ppo_base.network.eval()

    ppo_adapt = PPOAgent(device=device)
    ppo_adapt.load_checkpoint(ckpt_adapt)
    ppo_adapt.network.eval()

    # 2. Instantiate Baselines
    rand_baseline = RandomValidEditsBaseline(max_steps=10)
    greedy_baseline = GreedyHillClimbingBaseline(max_steps=10)

    # 3. Evaluate each method on the EXACT same test molecules
    all_results = []
    summaries = []

    methods = [
        ("Random Valid Edits", rand_baseline),
        ("Greedy Hill Climbing", greedy_baseline),
        ("PPO Baseline", ppo_base),
        ("PPO Adaptive (Proposed)", ppo_adapt)
    ]

    for name, agent in methods:
        print(f"\nEvaluating: {name} on {len(test_smiles)} test molecules...")
        df_res, summary = evaluate_policy_on_molecules(agent, env_eval, test_smiles, method_name=name)
        all_results.append(df_res)
        summaries.append(summary)
        print(f"  --> Mean Delta QED: {summary['mean_delta_qed']:+.4f} | Success Rate: {summary['success_rate']*100:.1f}% | Validity: {summary['validity_rate']*100:.1f}% | Similarity: {summary['mean_similarity']:.3f}")

    combined_df = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    # 4. Save CSV results
    out_dir = "results" if os.path.exists("results") else os.path.join("cache", "results")
    combined_df.to_csv(os.path.join(out_dir, "final_evaluation_results.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "method_comparison_summary.csv"), index=False)
    print(f"\nSaved detailed results and summary to {out_dir}/")

    # 5. QED-Stratified Analysis Table
    stratified_df = combined_df.groupby(["method", "qed_group"]).agg({
        "delta_qed": ["mean", "std"],
        "nhg": "mean",
        "success": "mean",
        "similarity": "mean",
        "final_sas": "mean",
        "num_edits": "mean"
    }).reset_index()
    stratified_df.to_csv(os.path.join(out_dir, "qed_stratified_summary.csv"), index=False)

    # 6. Reward-Hacking Audit Table
    audit_df = combined_df.groupby("method").agg({
        "pains_alert": "mean",
        "brenk_alert": "mean",
        "lipinski_violations": "mean",
        "sanity_passed": "mean",
        "valid": "mean"
    }).reset_index()
    audit_df.to_csv(os.path.join(out_dir, "reward_hacking_audit.csv"), index=False)

    # 7. Generate Evaluation Plots
    fig_dir = "figures" if os.path.exists("figures") else os.path.join("cache", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Plot 1: Mean Delta QED and Success Rate
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(summary_df["method"], summary_df["mean_delta_qed"], color=["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"])
    plt.axhline(0, color="k", linestyle="--", alpha=0.5)
    plt.title("Mean $\\Delta$QED Across Test Molecules")
    plt.ylabel("Mean $\\Delta$QED")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(summary_df["method"], summary_df["success_rate"] * 100, color=["#95a5a6", "#3498db", "#e74c3c", "#2ecc71"])
    plt.title("Optimization Success Rate (%)")
    plt.ylabel("Success Rate (%)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "evaluation_delta_qed_and_success.png"), dpi=200)
    plt.close()

    # Plot 2: QED Stratified Delta QED
    plt.figure(figsize=(10, 5))
    groups = ["Low (<0.5)", "Medium (0.5-0.7)", "High (0.7-0.85)", "Very High (>=0.85)"]
    x = np.arange(len(groups))
    width = 0.2

    for i, (name, _) in enumerate(methods):
        sub = combined_df[combined_df["method"] == name]
        means = [sub[sub["qed_group"] == g]["delta_qed"].mean() if len(sub[sub["qed_group"] == g]) > 0 else 0 for g in groups]
        plt.bar(x + (i - 1.5) * width, means, width=width, label=name)

    plt.xticks(x, groups)
    plt.title("QED-Stratified Optimization Gain Across Starting Headroom")
    plt.xlabel("Initial QED Range")
    plt.ylabel("Mean $\\Delta$QED")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "qed_stratified_comparison.png"), dpi=200)
    plt.close()

    # Plot 3: Reward-Hacking Audit
    plt.figure(figsize=(8, 4))
    audit_methods = audit_df["method"].tolist()
    pains_rates = (audit_df["pains_alert"] * 100).tolist()
    brenk_rates = (audit_df["brenk_alert"] * 100).tolist()
    
    x = np.arange(len(audit_methods))
    plt.bar(x - 0.15, pains_rates, width=0.3, label="PAINS Alert Rate (%)", color="#e67e22")
    plt.bar(x + 0.15, brenk_rates, width=0.3, label="Brenk Alert Rate (%)", color="#9b59b6")
    plt.xticks(x, audit_methods, rotation=20, ha="right")
    plt.title("Reward-Hacking Audit: Held-Out Chemical Structural Alerts")
    plt.ylabel("Alert Frequency (%)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "reward_hacking_audit_chart.png"), dpi=200)
    plt.close()

    print(f"Generated all evaluation figures in {fig_dir}")

if __name__ == "__main__":
    run_comprehensive_evaluation()

