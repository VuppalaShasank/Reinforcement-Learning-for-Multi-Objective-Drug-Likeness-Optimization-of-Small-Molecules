import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED
from rdkit.Chem.rdchem import BondType

# Set font and style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def generate_dataset_distribution_plot():
    print("Generating ZINC-250K Property Distributions Plot...")
    zinc_path = os.path.join(PROJECT_ROOT, "data", "raw", "zinc_250k.csv")
    test_path = os.path.join(PROJECT_ROOT, "data", "splits", "test_benchmark_molecules.csv")
    
    # Load sample of ZINC
    df_zinc = pd.read_csv(zinc_path, nrows=25000)
    df_test = pd.read_csv(test_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    
    # Palette
    c_zinc = '#34495e'
    c_test = '#2980b9'
    c_accent = '#e74c3c'
    
    # 1. QED Distribution
    ax = axes[0, 0]
    ax.hist(df_zinc['qed'], bins=40, density=True, alpha=0.55, color=c_zinc, label='ZINC-250K (Sample, N=25k)')
    ax.hist(df_test['qed'], bins=25, density=True, alpha=0.65, color=c_test, label='Test Benchmark (N=500)')
    # Stratification cutoffs
    for cutoff, label in zip([0.50, 0.70, 0.85], ['Low/Med (0.50)', 'Med/High (0.70)', 'High/V.High (0.85)']):
        ax.axvline(cutoff, color=c_accent, linestyle='--', linewidth=1.2, label=f'Cutoff: {label}')
    ax.set_title('(A) Quantitative Estimate of Drug-Likeness (QED)')
    ax.set_xlabel('QED Score [0, 1]')
    ax.set_ylabel('Density')
    ax.legend(loc='upper left', fontsize=7.5, frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # 2. SAS Distribution
    ax = axes[0, 1]
    ax.hist(df_zinc['SAS'], bins=40, density=True, alpha=0.55, color=c_zinc, label='ZINC-250K')
    ax.hist(df_test['sas'], bins=25, density=True, alpha=0.65, color=c_test, label='Test Benchmark')
    ax.axvline(4.0, color=c_accent, linestyle='--', linewidth=1.5, label='Synthetic Feasibility Threshold (SAS=4.0)')
    ax.set_title('(B) Synthetic Accessibility Score (SAS)')
    ax.set_xlabel('SAS Score [1 = Easy, 10 = Difficult]')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=8, frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # 3. LogP Distribution
    ax = axes[1, 0]
    ax.hist(df_zinc['logP'], bins=40, density=True, alpha=0.55, color=c_zinc, label='ZINC-250K')
    ax.axvline(5.0, color=c_accent, linestyle='--', linewidth=1.5, label='Lipinski LogP Limit (<= 5.0)')
    ax.set_title('(C) Calculated Octanol-Water Partition Coefficient (logP)')
    ax.set_xlabel('Wildman-Crippen logP')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', fontsize=8, frameon=True)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # 4. Stratified Test Cohort Breakdown
    ax = axes[1, 1]
    groups = ['Low (<0.50)', 'Medium (0.50-0.70)', 'High (0.70-0.85)', 'Very High (>=0.85)']
    counts = [125, 125, 125, 125]
    colors = ['#1abc9c', '#3498db', '#9b59b6', '#e67e22']
    bars = ax.bar(groups, counts, color=colors, edgecolor='#2c3e50', width=0.55, alpha=0.85)
    ax.set_title('(D) Stratified Benchmark Test Cohort (N = 500)')
    ax.set_ylabel('Molecule Count')
    ax.set_ylim(0, 150)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 3, f'N={int(h)}\n(25%)', ha='center', va='bottom', fontsize=8.5)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "figures", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "zinc250k_property_distributions.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def generate_benchmark_500_method_comparison():
    print("Generating Benchmark 500 Method Comparison Plot...")
    csv_path = os.path.join(PROJECT_ROOT, "results", "benchmark_500", "method_comparison_summary.csv")
    df = pd.read_csv(csv_path)
    
    methods = df['method'].tolist()
    # Clean names for plotting
    display_methods = ['Random Valid Edits', 'PPO Baseline\n(Unshaped)', 'PPO Proposed\n(Adaptive)', 'Greedy Hill\nClimbing']
    
    colors = ['#95a5a6', '#e74c3c', '#2980b9', '#27ae60']
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    
    # (A) Mean Delta-QED with error bars
    ax = axes[0, 0]
    bars = ax.bar(display_methods, df['mean_delta_qed'], yerr=df['std_delta_qed'], capsize=4, color=colors, edgecolor='#2c3e50', alpha=0.85, width=0.55)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('(A) Mean Delta-QED (Optimization Impact)')
    ax.set_ylabel('Mean Delta-QED')
    for bar, val in zip(bars, df['mean_delta_qed']):
        ypos = val - 0.08 if val < 0 else val + 0.08
        ax.text(bar.get_x() + bar.get_width()/2., ypos, f"{val:+.3f}", ha='center', va='center', fontweight='bold', fontsize=8.5)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_ylim(-0.7, 0.3)
    
    # (B) Formal Success vs High QED Success
    ax = axes[0, 1]
    x = np.arange(len(display_methods))
    w = 0.35
    b1 = ax.bar(x - w/2, df['success_rate'] * 100, width=w, label='Formal Success (Delta>=0.1, Sim>=0.4)', color='#34495e', alpha=0.85)
    b2 = ax.bar(x + w/2, df['high_qed_success_rate'] * 100, width=w, label='High-QED Success (QED>=0.80)', color='#16a085', alpha=0.85)
    ax.set_title('(B) Success Rate Comparison (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(display_methods)
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(0, 80)
    for bar in b1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8)
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8)
    ax.legend(loc='upper left', fontsize=8, frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    # (C) Structural Similarity & Synthetic Accessibility Impact
    ax = axes[1, 0]
    b1 = ax.bar(x - w/2, df['mean_similarity'], width=w, label='Mean Tanimoto Similarity', color='#2980b9', alpha=0.85)
    b2 = ax.bar(x + w/2, df['mean_delta_sas'], width=w, label='Mean Delta-SAS (Synthesis Cost)', color='#d35400', alpha=0.85)
    ax.set_title('(C) Structural Similarity vs Synthetic Accessibility')
    ax.set_xticks(x)
    ax.set_xticklabels(display_methods)
    ax.set_ylabel('Score / Metric Value')
    ax.set_ylim(0, 2.4)
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.05, f"{h:.2f}", ha='center', va='bottom', fontsize=8)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.05, f"{h:.2f}", ha='center', va='bottom', fontsize=8)
    ax.legend(loc='upper right', fontsize=8, frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    # (D) Chemical Validity & Toxicity Alerts
    ax = axes[1, 1]
    b1 = ax.bar(x - w/2, df['pains_alert_rate'] * 100, width=w, label='PAINS Alert Rate (%)', color='#c0392b', alpha=0.85)
    b2 = ax.bar(x + w/2, df['brenk_alert_rate'] * 100, width=w, label='Brenk Alert Rate (%)', color='#8e44ad', alpha=0.85)
    ax.set_title('(D) Chemical Quality & Structural Alerts (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(display_methods)
    ax.set_ylabel('Alert Rate (%)')
    ax.set_ylim(0, 115)
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8)
    ax.legend(loc='upper left', fontsize=8, frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "figures", "comparisons")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_500_method_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def generate_benchmark_500_qed_stratified():
    print("Generating Benchmark 500 QED-Stratified Comparison Plot...")
    csv_path = os.path.join(PROJECT_ROOT, "results", "benchmark_500", "qed_stratified_summary.csv")
    df = pd.read_csv(csv_path)
    
    methods = ['Random Valid Edits', 'PPO Baseline (Unshaped)', 'PPO Proposed (Adaptive)', 'Greedy Hill Climbing']
    method_labels = ['Random', 'PPO Baseline', 'PPO Proposed', 'Greedy HC']
    colors = ['#95a5a6', '#e74c3c', '#2980b9', '#27ae60']
    groups = ['Low (<0.50)', 'Medium (0.50-0.70)', 'High (0.70-0.85)', 'Very High (>=0.85)']
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    x = np.arange(len(groups))
    w = 0.18
    
    # 1. Delta-QED by Tier
    ax = axes[0, 0]
    for idx, (m, lbl, col) in enumerate(zip(methods, method_labels, colors)):
        sub = df[df['method'] == m].set_index('qed_group').reindex(groups)
        ax.bar(x + (idx - 1.5) * w, sub['mean_delta_qed'], width=w, label=lbl, color=col, alpha=0.85)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('(A) Mean Delta-QED Across Starting-QED Tiers')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=10)
    ax.set_ylabel('Mean Delta-QED')
    ax.legend(loc='lower left', fontsize=8, frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    # 2. Final QED by Tier
    ax = axes[0, 1]
    for idx, (m, lbl, col) in enumerate(zip(methods, method_labels, colors)):
        sub = df[df['method'] == m].set_index('qed_group').reindex(groups)
        ax.bar(x + (idx - 1.5) * w, sub['mean_final_qed'], width=w, label=lbl, color=col, alpha=0.85)
    ax.axhline(0.80, color='#e74c3c', linestyle='--', linewidth=1.2, label='Drug-Like Target (QED=0.80)')
    ax.set_title('(B) Absolute Final QED Achieved by Tier')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=10)
    ax.set_ylabel('Mean Final QED')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper left', fontsize=8, frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    # 3. Tanimoto Similarity by Tier
    ax = axes[1, 0]
    for idx, (m, lbl, col) in enumerate(zip(methods, method_labels, colors)):
        sub = df[df['method'] == m].set_index('qed_group').reindex(groups)
        ax.bar(x + (idx - 1.5) * w, sub['mean_similarity'], width=w, label=lbl, color=col, alpha=0.85)
    ax.axhline(0.40, color='#e74c3c', linestyle='--', linewidth=1.2, label='Similarity Threshold (0.40)')
    ax.set_title('(C) Morgan Tanimoto Similarity Retention')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=10)
    ax.set_ylabel('Mean Tanimoto Similarity')
    ax.set_ylim(0, 0.85)
    ax.legend(loc='upper right', fontsize=8, frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    # 4. Formal Success Rate by Tier
    ax = axes[1, 1]
    for idx, (m, lbl, col) in enumerate(zip(methods, method_labels, colors)):
        sub = df[df['method'] == m].set_index('qed_group').reindex(groups)
        ax.bar(x + (idx - 1.5) * w, sub['success_rate'] * 100, width=w, label=lbl, color=col, alpha=0.85)
    ax.set_title('(D) Formal Optimization Success Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=10)
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', fontsize=8, frameon=True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "figures", "comparisons")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_500_qed_stratified.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def generate_system_architecture_diagram():
    print("Generating Overall Molecular Optimization System Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.axis('off')
    
    # Boxes
    box_props_state = dict(boxstyle='round,pad=0.6', facecolor='#ebf5fb', edgecolor='#2980b9', lw=2)
    box_props_model = dict(boxstyle='round,pad=0.6', facecolor='#e8f8f5', edgecolor='#16a085', lw=2)
    box_props_act = dict(boxstyle='round,pad=0.6', facecolor='#fef9e7', edgecolor='#f39c12', lw=2)
    box_props_env = dict(boxstyle='round,pad=0.6', facecolor='#f4ecf7', edgecolor='#8e44ad', lw=2)
    box_props_reward = dict(boxstyle='round,pad=0.6', facecolor='#fdedec', edgecolor='#e74c3c', lw=2)
    box_props_ppo = dict(boxstyle='round,pad=0.6', facecolor='#eaecee', edgecolor='#34495e', lw=2)
    
    # 1. State Input
    ax.text(0.12, 0.78, "Molecular State Representation (s_t)\n"
                       "---------------------------------------\n"
                       "- Input SMILES string\n"
                       "- Molecular Graph G = (V, E)\n"
                       "- Atom features x_v in R^21 (element, hyb, aro, ...)\n"
                       "- Bond features e_uv in R^4 (1x, 2x, 3x, aromatic)",
            ha='center', va='center', bbox=box_props_state, fontsize=9.5)
            
    # 2. GNN Encoder
    ax.text(0.48, 0.78, "Edge-Aware GNN Encoder\n"
                       "---------------------------------------\n"
                       "- 3-Layer Edge-Aware Message Passing\n"
                       "- Relational message: m_uv = ReLU(W [h_u, h_v, e_uv])\n"
                       "- Node update: GRU / Sum-Aggregation\n"
                       "- Global Readout: [MeanPool || MaxPool] in R^128",
            ha='center', va='center', bbox=box_props_model, fontsize=9.5)

    # 3. Actor-Critic Heads
    ax.text(0.85, 0.78, "Actor-Critic Network Heads\n"
                       "---------------------------------------\n"
                       "- Actor Head pi_theta(a | s):\n"
                       "   MLP(128 -> 256 -> 1901 logits)\n"
                       "- Critic Head V_phi(s):\n"
                       "   MLP(128 -> 128 -> 1 scalar value)\n"
                       "- Categorical action sampling",
            ha='center', va='center', bbox=box_props_act, fontsize=9.5)

    # 4. Action Execution & Chemistry Environment
    ax.text(0.85, 0.28, "Molecular Environment Step\n"
                       "---------------------------------------\n"
                       "- 1,901 Discrete Chemical Mutations:\n"
                       "  * 1,406 Bond Edits (Add/Rem/Order)\n"
                       "  * 475 Fragment Attachments\n"
                       "  * 20 Atom Substitutions + STOP (a=0)\n"
                       "- RDKit Valence & Sanitization Check\n"
                       "- Produces Next State s_{t+1}",
            ha='center', va='center', bbox=box_props_env, fontsize=9.5)

    # 5. Adaptive Reward Shaping Engine
    ax.text(0.48, 0.28, "Adaptive Reward Shaping Engine\n"
                       "---------------------------------------\n"
                       "R_adaptive = gamma(s_t)*Delta_QED\n"
                       "           - rho_sim(s_t, s_{t+1})\n"
                       "           - rho_slack(s_{t+1}) + r_stop\n"
                       "- Headroom Multiplier: gamma = (1 - QED(s_t))\n"
                       "- Dynamic Sim Penalty: beta * max(0, tau_sim - Sim)\n"
                       "- Constraint Slack: SAS > 4.0 & Lipinski violations",
            ha='center', va='center', bbox=box_props_reward, fontsize=9.5)

    # 6. PPO Update Engine
    ax.text(0.12, 0.28, "PPO Optimization & Updates\n"
                       "---------------------------------------\n"
                       "- Rollout Trajectory Buffer\n"
                       "- Generalized Advantage Estimation (GAE lambda)\n"
                       "- Clipped Surrogate Objective L^{CLIP}(theta)\n"
                       "- Value Function Loss L^{VF}(phi)\n"
                       "- Entropy Bonus c_ent * H(pi_theta)",
            ha='center', va='center', bbox=box_props_ppo, fontsize=9.5)

    # Connecting Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="#2c3e50")
    # State -> GNN
    ax.annotate("", xy=(0.34, 0.78), xytext=(0.26, 0.78), arrowprops=arrow_props)
    # GNN -> Actor-Critic
    ax.annotate("", xy=(0.71, 0.78), xytext=(0.62, 0.78), arrowprops=arrow_props)
    # Actor-Critic -> Environment
    ax.annotate("", xy=(0.85, 0.45), xytext=(0.85, 0.64), arrowprops=arrow_props)
    # Environment -> Reward
    ax.annotate("", xy=(0.63, 0.28), xytext=(0.72, 0.28), arrowprops=arrow_props)
    # Reward -> PPO
    ax.annotate("", xy=(0.26, 0.28), xytext=(0.33, 0.28), arrowprops=arrow_props)
    # PPO -> GNN/Actor-Critic Update
    ax.annotate("Policy & Value Gradient Updates", xy=(0.48, 0.65), xytext=(0.12, 0.45),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#c0392b", linestyle="--"),
                fontsize=8.5, color="#c0392b", ha='center')
    # Next State feedback to State Input
    ax.annotate("Next Molecule s_{t+1}", xy=(0.12, 0.63), xytext=(0.48, 0.45),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#2980b9", linestyle=":"),
                fontsize=8.5, color="#2980b9", ha='center')

    plt.title("Figure 1: Complete Molecular Optimization Pipeline with Adaptive GNN-PPO", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "figures", "optimization")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "system_architecture_diagram.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def generate_model_architecture_diagram():
    print("Generating Model Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(11, 7.0))
    ax.axis('off')
    
    # Boxes
    b_input = dict(boxstyle='square,pad=0.5', facecolor='#eaecee', edgecolor='#34495e', lw=1.5)
    b_gnn = dict(boxstyle='round,pad=0.5', facecolor='#d4efdf', edgecolor='#27ae60', lw=1.5)
    b_pool = dict(boxstyle='round,pad=0.5', facecolor='#d6eaf8', edgecolor='#2980b9', lw=1.5)
    b_head = dict(boxstyle='round,pad=0.5', facecolor='#fcf3cf', edgecolor='#f39c12', lw=1.5)
    
    ax.text(0.12, 0.72, "Molecular Graph Input\n"
                       "G = (V, E)\n\n"
                       "Atoms: |V| nodes\n"
                       "x_v in R^21\n"
                       "(Z, degree, charge,\n"
                       "aromatic, hybridization)\n\n"
                       "Bonds: |E| edges\n"
                       "e_uv in R^4\n"
                       "(Single, Double,\n"
                       "Triple, Aromatic)",
            ha='center', va='center', bbox=b_input, fontsize=9)

    ax.text(0.42, 0.72, "3-Layer Edge-Aware GNN\n"
                       "------------------------------------\n"
                       "Layer 1: Message Passing (64-d)\n"
                       "  m_uv^(1) = ReLU(W_m [h_u, h_v, e_uv])\n"
                       "  h_v^(1) = GRU(h_v^(0), sum m_uv)\n\n"
                       "Layer 2: Message Passing (64-d)\n"
                       "  m_uv^(2) = ReLU(W_m [h_u^(1), h_v^(1), e_uv])\n"
                       "  h_v^(2) = GRU(h_v^(1), sum m_uv)\n\n"
                       "Layer 3: Message Passing (64-d)\n"
                       "  m_uv^(3) = ReLU(W_m [h_u^(2), h_v^(2), e_uv])\n"
                       "  h_v^(3) = GRU(h_v^(2), sum m_uv)",
            ha='center', va='center', bbox=b_gnn, fontsize=8.5)

    ax.text(0.72, 0.72, "Global Graph Readout\n"
                       "-----------------------\n"
                       "Mean-Pooling: in R^64\n"
                       "Max-Pooling:  in R^64\n\n"
                       "Concatenation:\n"
                       "h_G = [h_mean || h_max]\n"
                       "h_G in R^128\n"
                       "(Fixed-size vector)",
            ha='center', va='center', bbox=b_pool, fontsize=9)

    ax.text(0.48, 0.22, "Actor Head (Policy Network pi_theta)\n"
                       "----------------------------------------------------\n"
                       "h_G in R^128 -> Linear(128, 256) -> ReLU -> Dropout(0.1)\n"
                       "            -> Linear(256, 1901) -> Softmax\n"
                       "Action Space: 1,901 Discrete Chemical Mutations",
            ha='center', va='center', bbox=b_head, fontsize=9)

    ax.text(0.85, 0.22, "Critic Head (Value Network V_phi)\n"
                       "-----------------------------------------\n"
                       "h_G in R^128 -> Linear(128, 128) -> ReLU\n"
                       "            -> Linear(128, 64)  -> ReLU\n"
                       "            -> Linear(64, 1)    -> Scalar V(s)",
            ha='center', va='center', bbox=b_head, fontsize=9)

    # Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="#2c3e50")
    ax.annotate("", xy=(0.28, 0.72), xytext=(0.21, 0.72), arrowprops=arrow_props)
    ax.annotate("", xy=(0.60, 0.72), xytext=(0.55, 0.72), arrowprops=arrow_props)
    
    # Readout to heads
    ax.annotate("", xy=(0.48, 0.32), xytext=(0.68, 0.58), arrowprops=arrow_props)
    ax.annotate("", xy=(0.85, 0.32), xytext=(0.76, 0.58), arrowprops=arrow_props)

    plt.title("Figure 2: Edge-Aware GNN Encoder and Actor-Critic Architecture", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "figures", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_architecture_diagram.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def generate_adaptive_reward_diagram():
    print("Generating Adaptive Reward Formulation Diagram...")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axis('off')
    
    b_main = dict(boxstyle='round,pad=0.6', facecolor='#fcf3cf', edgecolor='#d4ac0d', lw=2)
    b_term1 = dict(boxstyle='round,pad=0.5', facecolor='#d5f5e3', edgecolor='#27ae60', lw=1.5)
    b_term2 = dict(boxstyle='round,pad=0.5', facecolor='#fadbd8', edgecolor='#e74c3c', lw=1.5)
    b_term3 = dict(boxstyle='round,pad=0.5', facecolor='#ebdef0', edgecolor='#8e44ad', lw=1.5)
    b_term4 = dict(boxstyle='round,pad=0.5', facecolor='#d6eaf8', edgecolor='#2980b9', lw=1.5)

    ax.text(0.5, 0.88, "Adaptive Reward Signal at Step t:\n"
                      "R_adaptive(s_t, a_t, s_{t+1}) = gamma(s_t) * Delta_QED - rho_sim(s_t, s_{t+1}) - rho_slack(s_{t+1}) + r_stop",
            ha='center', va='center', bbox=b_main, fontsize=10.5, fontweight='bold')

    # Component 1
    ax.text(0.22, 0.55, "(1) Headroom-Scaled QED Gain\n"
                       "---------------------------------------\n"
                       "Term: gamma(s_t) * [QED(s_{t+1}) - QED(s_t)]\n"
                       "Headroom: gamma(s_t) = (1 - QED(s_t))\n\n"
                       "Behavior:\n"
                       "* High headroom (low QED ~ 0.3): gamma = 0.70\n"
                       "  Aggressively rewards exploration\n"
                       "* Low headroom (high QED ~ 0.85): gamma = 0.15\n"
                       "  Dampens reward; protects existing quality",
            ha='center', va='center', bbox=b_term1, fontsize=8.5)

    # Component 2
    ax.text(0.78, 0.55, "(2) Dynamic Similarity Penalty\n"
                       "---------------------------------------\n"
                       "Term: - beta_sim * max(0, tau_sim - Sim(s_0, s_{t+1}))\n"
                       "Threshold: tau_sim = 0.40 (Morgan Tanimoto)\n\n"
                       "Behavior:\n"
                       "* Sim >= 0.40: Penalty = 0.0 (Unpenalized)\n"
                       "* Sim < 0.40: Penalty ramps up proportionally\n"
                       "* Prevents policy from dissolving scaffold\n"
                       "  into unconstrained high-QED fragments",
            ha='center', va='center', bbox=b_term2, fontsize=8.5)

    # Component 3
    ax.text(0.22, 0.18, "(3) Constraint Slack Penalties\n"
                       "---------------------------------------\n"
                       "Term: - rho_slack(s_{t+1}) = - (P_SAS + P_Lipinski)\n"
                       "SAS Penalty: alpha_sas * max(0, SAS(s_{t+1}) - 4.0)\n"
                       "Lipinski Penalty: alpha_lip * Violations(s_{t+1})\n\n"
                       "Behavior:\n"
                       "* Curbs synthesis difficulty (SAS <= 4.0)\n"
                       "* Penalizes molecular weight and LogP violations",
            ha='center', va='center', bbox=b_term3, fontsize=8.5)

    # Component 4
    ax.text(0.78, 0.18, "(4) Terminal Stop Bonus\n"
                       "---------------------------------------\n"
                       "Term: r_stop = + 0.50 if (a_t == STOP and Delta_QED > 0)\n"
                       "      r_stop = - 0.20 if (a_t == STOP and Delta_QED <= 0)\n\n"
                       "Behavior:\n"
                       "* Solves horizon-exhaustion failure mode\n"
                       "* Encourages agent to preserve gains and exit\n"
                       "  early instead of wandering until step T=10",
            ha='center', va='center', bbox=b_term4, fontsize=8.5)

    # Connecting arrows
    arrow_props = dict(arrowstyle="->", lw=1.5, color="#2c3e50")
    ax.annotate("", xy=(0.22, 0.70), xytext=(0.40, 0.80), arrowprops=arrow_props)
    ax.annotate("", xy=(0.78, 0.70), xytext=(0.60, 0.80), arrowprops=arrow_props)
    ax.annotate("", xy=(0.22, 0.33), xytext=(0.35, 0.78), arrowprops=arrow_props)
    ax.annotate("", xy=(0.78, 0.33), xytext=(0.65, 0.78), arrowprops=arrow_props)

    plt.title("Figure 3: Mathematical Formulation of Headroom- and Constraint-Slack Adaptive Reward", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "figures", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "adaptive_reward_formulation.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def generate_representative_molecular_edits():
    print("Generating Representative Molecular Edits Visualization...")
    # Find actual molecules from benchmark_500
    res_path = os.path.join(PROJECT_ROOT, "results", "benchmark_500", "final_evaluation_results.csv")
    df = pd.read_csv(res_path)
    
    # Pick a molecule where Adaptive PPO performed well and Greedy also performed well
    # Group Low tier
    low_cases = df[df['qed_group'] == 'Low (<0.50)']
    # Find a molecule ID where both PPO Proposed and Greedy have valid outputs
    mol_ids = low_cases['molecule_id'].unique()
    
    selected_id = 'TEST_001' # default
    for mid in mol_ids:
        sub = df[df['molecule_id'] == mid]
        p_row = sub[sub['method'] == 'PPO Proposed (Adaptive)']
        g_row = sub[sub['method'] == 'Greedy Hill Climbing']
        if len(p_row) > 0 and len(g_row) > 0:
            if p_row.iloc[0]['final_qed'] > p_row.iloc[0]['initial_qed'] and g_row.iloc[0]['final_qed'] > 0.6:
                selected_id = mid
                break
                
    sub = df[df['molecule_id'] == selected_id]
    init_smi = sub.iloc[0]['initial_smiles']
    init_qed = sub.iloc[0]['initial_qed']
    init_sas = sub.iloc[0]['initial_sas']
    
    rand_row = sub[sub['method'] == 'Random Valid Edits'].iloc[0]
    base_row = sub[sub['method'] == 'PPO Baseline (Unshaped)'].iloc[0]
    prop_row = sub[sub['method'] == 'PPO Proposed (Adaptive)'].iloc[0]
    grdy_row = sub[sub['method'] == 'Greedy Hill Climbing'].iloc[0]
    
    mols = [
        Chem.MolFromSmiles(init_smi),
        Chem.MolFromSmiles(rand_row['final_smiles']),
        Chem.MolFromSmiles(base_row['final_smiles']),
        Chem.MolFromSmiles(prop_row['final_smiles']),
        Chem.MolFromSmiles(grdy_row['final_smiles'])
    ]
    
    legends = [
        f"Initial ({selected_id})\nQED: {init_qed:.3f} | SAS: {init_sas:.2f}",
        f"Random Valid\nQED: {rand_row['final_qed']:.3f} (dQ: {rand_row['delta_qed']:+.2f})\nSim: {rand_row['similarity']:.2f} | SAS: {rand_row['final_sas']:.2f}",
        f"PPO Baseline\nQED: {base_row['final_qed']:.3f} (dQ: {base_row['delta_qed']:+.2f})\nSim: {base_row['similarity']:.2f} | SAS: {base_row['final_sas']:.2f}",
        f"PPO Proposed\nQED: {prop_row['final_qed']:.3f} (dQ: {prop_row['delta_qed']:+.2f})\nSim: {prop_row['similarity']:.2f} | SAS: {prop_row['final_sas']:.2f}",
        f"Greedy HC\nQED: {grdy_row['final_qed']:.3f} (dQ: {grdy_row['delta_qed']:+.2f})\nSim: {grdy_row['similarity']:.2f} | SAS: {grdy_row['final_sas']:.2f}"
    ]
    
    # Filter any None mols if any parsing failed
    valid_mols = []
    valid_legs = []
    for m, leg in zip(mols, legends):
        if m is not None:
            valid_mols.append(m)
            valid_legs.append(leg)
            
    img = Draw.MolsToGridImage(valid_mols, molsPerRow=len(valid_mols), subImgSize=(260, 260), legends=valid_legs, useSVG=False)
    
    out_dir = os.path.join(PROJECT_ROOT, "figures", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "representative_molecular_edits.png")
    img.save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_dataset_distribution_plot()
    generate_benchmark_500_method_comparison()
    generate_benchmark_500_qed_stratified()
    generate_system_architecture_diagram()
    generate_model_architecture_diagram()
    generate_adaptive_reward_diagram()
    generate_representative_molecular_edits()
    print("All additional figures successfully created!")

