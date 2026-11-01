# Molecular Lead Optimization via GNN-PPO with Adaptive Reward Shaping
## Post-Mid-Review Continuation Capstone Workspace

**Course:** 22AIE401 — Reinforcement Learning  
**Dataset:** ZINC-250K Small Molecule Chemical Library  
**Primary Algorithm:** Proximal Policy Optimization (PPO)  
**Molecular Architecture:** Three-Layer Message-Passing GNN + PPO Actor-Critic  
**Core Innovation:** Headroom- and Constraint-Slack Adaptive Reward Shaping  

---

## 1. Workspace Organization & File Directory

All post-mid-review development, implementation code, configuration files, trained checkpoints, training logs, empirical evaluation results, figures, and pre-executed Jupyter notebooks are strictly self-contained within this `antigravity_workspace/` directory to preserve the original Phase 1–5 project files intact.

```text
antigravity_workspace/
│
├── configs/
│   └── experiment_config.yaml         # Central YAML configuration file (PPO, GNN, Env, Reward)
│
├── data/
│   └── splits/
│       ├── train_molecules.csv        # 4,000 Bemis-Murcko scaffold-disjoint training molecules
│       ├── val_molecules.csv          # 501 validation molecules
│       ├── test_benchmark_molecules.csv # 60 held-out test benchmark molecules (stratified by QED)
│       └── split_info.json            # Scaffold verification metadata confirming zero overlap
│
├── implementation/
│   ├── gnn_encoder.py                 # 3-Layer GCN with dual pooling (mean+max) and scalar fusion
│   ├── action_policy.py               # 1,901-action Actor-Critic network with chemical valency masking
│   ├── ppo_agent.py                   # PPO Agent with GAE, clipped loss, and atomic checkpointing
│   ├── training_utils.py              # Rollout worker, GAE advantage estimation, and training loop
│   ├── baseline_methods.py            # Random Valid Edits & Greedy Hill-Climbing lookahead oracle
│   └── evaluation.py                  # Evaluation harness for 8 RL metrics, PAINS, Brenk, Lipinski
│
├── checkpoints/
│   ├── ppo_adaptive.pt                # Trained weights of proposed adaptive PPO agent
│   └── ppo_baseline.pt                # Trained weights of unshaped baseline PPO agent
│
├── logs/
│   ├── train_ppo_adaptive.csv         # Episode-level training log (return, delta-QED, similarity, edits)
│   └── train_ppo_baseline.csv         # Episode-level training log for unshaped baseline
│
├── results/
│   ├── method_comparison_summary.csv  # Benchmark evaluation table (Adaptive vs Baseline vs Random vs Greedy)
│   ├── qed_stratified_summary.csv     # Performance stratified across Low, Med, High, Very High QED tiers
│   ├── reward_hacking_audit.csv       # PAINS, Brenk, Lipinski violations, and chemical sanity audit
│   ├── ablation_study_results.csv     # Empirical ablation table across 5 model variants
│   ├── sensitivity_analysis.csv       # Sensitivity grid for epsilon, lambda, beta, and sim_min
│   ├── counterfactual_action_analysis.csv # Decision explainability at critical optimization step
│   ├── rl_specific_metrics_summary.json   # Eight formal RL-specific metrics summary
│   └── final_evaluation_results.csv   # Per-molecule trajectory outcomes for all 60 benchmark molecules
│
├── figures/
│   ├── training_reward_curves.png     # 4-panel training dynamics curves
│   ├── evaluation_delta_qed_and_success.png # Overall benchmark comparison bar charts
│   ├── qed_stratified_comparison.png  # Performance across starting-QED tiers
│   ├── reward_hacking_audit_chart.png # PAINS and Brenk alert frequency comparison
│   ├── ablation_comparison_chart.png  # Trade-offs between Delta-QED and structural similarity
│   ├── convergence_sample_efficiency.png # Episode-to-success convergence trajectory
│   └── sensitivity_analysis_chart.png # Hyperparameter response surfaces
│
├── notebooks/
│   ├── 04_gnn_encoder.ipynb           # Executed: GNN forward pass, batched graphs, gradient tests
│   ├── 05_ppo_training.ipynb          # Executed: Valency masking, loss dynamics, live trajectory rollout
│   ├── 06_baseline_comparison.ipynb   # Executed: Benchmark tables, stratified analysis, molecule edits
│   └── 07_final_evaluation.ipynb      # Executed: 8 RL metrics, ablation, sensitivity, reward-hacking audit
│
├── reports/
│   └── final_project_report.md        # Comprehensive IEEE-style final technical project paper
│
├── generate_all_experiments.py        # Master reproduction script for evaluations and figures
├── build_notebooks.py                 # Master reproduction script to compile and execute notebooks
└── README.md                          # This directory documentation
```

---

## 2. Core Methodological Contributions

### 1. Three-Layer Molecular Graph Neural Network (GNN) Encoder
- Encodes atoms as 21-dimensional vectors (element identity, coordination degree, charge, hydrogens, aromaticity) and bonds as 4-dimensional edge vectors (single, double, triple, aromatic).
- Applies three message-passing layers (`GraphConv` with edge projection) to compute atom latent features.
- Concatenates **Global Mean Pooling** and **Global Max Pooling** to capture both overall scaffold composition and localized functional motifs.
- Merges the graph embedding with a 6-dimensional continuous environment state vector (current QED, SAS, Tanimoto similarity to starting molecule $M_0$, normalized step, target distance, constraint slack) through a fusion MLP to generate a 128-dimensional continuous latent state $z$.

### 2. PPO Actor-Critic with Valency Action Masking
- **Action Space:** 1,901 discrete molecular modifications (Add atom, Substitute atom, Remove atom, Modify bond order, and STOP).
- **Valency Masking:** Before computing the softmax distribution, chemically forbidden actions (such as pentavalent carbon, ring-destroying substitutions, or non-existent atom targets) are masked to $-\infty$ logits. This guarantees that **$100.0\%$ of actions produce valid chemical molecules**.
- **PPO Clipped Objective:** Ensures bounded policy gradient steps with Generalized Advantage Estimation ($\lambda=0.95, \gamma=0.99, \epsilon=0.20$).

### 3. Headroom- and Constraint-Slack Adaptive Reward Shaping
Standard RL agents exploit unconstrained scalar objectives by generating toxic, unstable, or highly dissimilar molecules (reward hacking). Our proposed reward function dynamically regulates optimization pressure:
$$R(s_t, a_t, s_{t+1}) = w_{\text{QED}} \cdot \gamma_{\text{headroom}} \cdot \Delta \text{QED} - w_{\text{SAS}} \cdot \Delta \text{SAS} - R_{\text{slack\_barrier}} + R_{\text{step}} + R_{\text{terminal}}$$
- **Headroom Multiplier $\gamma_{\text{headroom}} = 1.0 + (1.0 - \text{QED}_0)$:** Scales property gains according to remaining headroom, pushing low-QED molecules aggressively while protecting high-QED compounds from unnecessary mutations.
- **Constraint-Slack Soft Barrier:** Imposes quadratic penalties as Tanimoto similarity approaches the $0.40$ threshold, discouraging risky modifications before a violation occurs.

---

## 3. Key Empirical Results Summary

Evaluated on 60 scaffold-disjoint test molecules from ZINC-250K across four initial drug-likeness tiers:

| Method | Mean $\Delta$QED | Mean Final Similarity | Success Rate ($\%$) | PAINS Alert Rate ($\%$) | Inference Speed (s/mol) |
|---|---|---|---|---|---|
| **Random Valid Edits** | $-0.2154$ | $0.384$ | $0.0\%$ | $13.3\%$ | $0.012$s |
| **Greedy Hill-Climbing (Oracle)** | $+0.1693$ | $0.488$ | $38.3\%$ | $6.7\%$ | $0.895$s |
| **PPO Baseline (Unshaped)** | $+0.0215$ | $0.412$ | $1.7\%$ | $16.7\%$ | **$0.042$s** |
| **PPO Proposed (Adaptive)** | **$+0.0542$** | **$0.494$** | **$5.0\%$** | **$3.3\%$** | **$0.042$s** |

### Key Findings:
1. **$2.5\times$ Delta-QED Improvement:** PPO Adaptive achieves $+0.0542$ mean $\Delta$QED vs. $+0.0215$ for unshaped baseline PPO.
2. **Scaffold Preservation:** PPO Adaptive maintains a mean final similarity of $0.494$, safely above the $0.40$ constraint.
3. **5-Fold Reduction in Toxic Alerts:** PPO Adaptive suppresses PAINS alert rates to $3.3\%$ (compared to $16.7\%$ for baseline PPO and $13.3\%$ for random edits).
4. **$21.3\times$ Inference Speedup:** Evaluates molecules in $0.042$ seconds per molecule, compared to $0.895$ seconds for greedy lookahead search.

---

## 4. How to Reproduce & Run

### 1. Requirements & Environment
The code runs on standard Python 3.10+ with PyTorch, PyTorch Geometric, RDKit, Gymnasium, and Pandas:
```bash
# Verify installation
python -c "import torch, torch_geometric, rdkit, gymnasium; print('All libraries verified.')"
```

### 2. Re-Generating All Results & Figures
To re-run the complete evaluation pipeline and refresh all tables and figures in `results/` and `figures/`:
```bash
python antigravity_workspace/generate_all_experiments.py
```

### 3. Re-Building & Executing Interactive Notebooks
To compile and execute all four notebooks (`04_gnn_encoder.ipynb`, `05_ppo_training.ipynb`, `06_baseline_comparison.ipynb`, `07_final_evaluation.ipynb`) with live inline outputs and embedded plots:
```bash
python antigravity_workspace/build_notebooks.py
```

---

## 5. Guide for Presentation / Final Review

When presenting this project to evaluators:
1. **Start with `antigravity_workspace/reports/final_project_report.md`:** It contains the full academic writeup, mathematical derivations, tables, and citations.
2. **Open the 4 Pre-Executed Notebooks in `notebooks/`:**
   - **`04_gnn_encoder.ipynb`:** Show the 3-layer GNN architecture, graph tensor shapes ($[N, 21]$, $[2, 2E]$, $[2E, 4]$), and verify the $[1, 128]$ embedding output.
   - **`05_ppo_training.ipynb`:** Show how chemical valency masking eliminates invalid actions, present the 4-panel training dynamics curves, and walk through the live optimization trajectory on an unseen molecule.
   - **`06_baseline_comparison.ipynb`:** Present the comparative benchmark table and show why PPO Adaptive outperforms Random and Baseline PPO while operating $21\times$ faster than Greedy search.
   - **`07_final_evaluation.ipynb`:** Present the 8 formal RL metrics, component ablation study, hyperparameter sensitivity curves, PAINS reward-hacking audit, and counterfactual action explainability.
3. **Do not modify or delete files outside `antigravity_workspace/`:** The original Phase 1–5 project files remain completely functional and untouched.

