# Reinforcement Learning for Drug-Likeness Optimization of Small Molecules: A GNN-PPO Framework with Headroom- and Constraint-Slack Adaptive Reward Shaping

**Course:** 22AIE401 — Reinforcement Learning  
**Student:** Vuppala Shasank [AM.SC.U4AIE23064]  
**Faculty Guide:** Dr. Geetha M  
**Department:** Department of Computer Science & Engineering (Artificial Intelligence)  
**Institution:** Amrita School of Computing, Amrita Vishwa Vidyapeetham  
**Dataset:** ZINC-250K Small Molecule Chemical Library (249,455 source molecules; 5,000 selected experimental cohort)  

---

## 1. Executive Summary

This capstone project presents an end-to-end Reinforcement Learning system for **Lead Optimization in Small-Molecule Drug Discovery**. Rather than generating random chemical graphs from scratch (*de novo* generation), our system begins with an existing lead candidate ($M_0$) and sequentially applies discrete atomic and bond modifications to maximize drug-likeness (Quantitative Estimate of Drug-Likeness, QED) while strictly respecting three biochemical constraints:
1. **Chemical Validity ($100.0\%$):** Enforced via chemical valency action masking over a 1,901-action discrete space.
2. **Scaffold Similarity ($\text{Tanimoto} \ge 0.40$):** Preserves the core pharmacophoric identity of the parent molecule.
3. **Synthetic Accessibility ($\text{SAS} \le 4.5$):** Prevents the generation of chemically intractable structures.

To overcome **reward hacking** (where unconstrained policies exploit scoring heuristics by generating toxic or reactive fragments) and **scaffold collapse** (where agents dissolve the parent structure), we introduce **Headroom- and Constraint-Slack Adaptive Reward Shaping** combined with a **Three-Layer Graph Convolutional Network (GCN)** state encoder and a **Proximal Policy Optimization (PPO)** Actor-Critic architecture.

The final evaluation was conducted on an expanded, statistically rigorous, scaffold-disjoint test benchmark of **500 unseen molecules** stratified across four starting-QED tiers (Low, Medium, High, Very High; 125 molecules each), evaluating **2,000 total trajectories** across four distinct methods.

---

## 2. Final Project Repository Organization

The repository is structured so that the **final capstone implementation** is the primary workspace, while all preliminary mid-review materials are cleanly separated in `mid_folder/`:

```text
RL_Sem7_Proj/
│
├── notebooks/                              # Final Pre-Executed Experimental Notebooks
│   ├── 04_gnn_encoder.ipynb                # GNN Message Passing, Dual Pooling & Tensor Checks
│   ├── 05_ppo_training.ipynb               # Valency Masking, Training Dynamics & Live Trajectory Rollout
│   ├── 06_baseline_comparison.ipynb        # Expanded 500-Molecule Benchmark Matrix & Stratified Analysis
│   └── 07_final_evaluation.ipynb           # Formal RL Metrics, Ablation Study, Sensitivity & PAINS Audit
│
├── implementation/                         # Modular Core Python Implementation
│   ├── gnn_encoder.py                      # 3-Layer GCN with dual pooling (mean+max) & scalar state fusion
│   ├── action_policy.py                    # 1,901-action Actor-Critic network with valency masking
│   ├── ppo_agent.py                        # PPO Agent with Generalized Advantage Estimation & clipped loss
│   ├── training_utils.py                   # Rollout collection & PPO training loop
│   ├── baseline_methods.py                 # Random Valid Edits & Greedy Hill-Climbing lookahead oracle
│   ├── evaluation.py                       # Evaluation harness for 8 RL metrics, PAINS, Brenk, Lipinski
│   ├── environment.py                      # Gymnasium molecular editing environment
│   ├── molecule_editor.py                  # RDKit RWMol atomic & bond editing operations
│   ├── properties.py                       # RDKit QED, SAS, and Tanimoto calculation functions
│   └── reward.py                           # Adaptive shaped multi-objective reward function
│
├── results/                                # Empirical Evaluation Results (N=500 Benchmark; 2,000 Trajectories)
│   ├── final_evaluation_results.csv        # Detailed per-trajectory metrics across all 2,000 runs
│   ├── method_comparison_summary.csv       # Overall benchmark performance comparison table
│   ├── qed_stratified_summary.csv          # Breakdown across Low, Med, High, Very High tiers (16 rows)
│   ├── reward_hacking_audit.csv            # PAINS, Brenk, Lipinski violations & sanity pass rates
│   ├── ablation_study_results.csv          # Empirical ablation table across 5 model variants
│   ├── sensitivity_analysis.csv            # Sensitivity grid for epsilon, lambda, beta, and sim_min
│   ├── counterfactual_action_analysis.csv  # Step-by-step decision explainability analysis
│   ├── rl_specific_metrics_summary.json    # Eight formal RL-specific metrics summary
│   └── pilot_60_benchmark/                 # Preserved historical 60-molecule pilot study
│
├── figures/                                # Publication-Quality Evaluation Visualizations
│   ├── training_reward_curves.png          # 4-panel training dynamics curves
│   ├── evaluation_delta_qed_success.png    # Overall benchmark comparison bar charts (N=500)
│   ├── qed_stratified_performance.png      # Performance across starting-QED tiers (125 mols each)
│   ├── reward_hacking_and_safety.png       # PAINS alert rates and Delta-SAS degradation
│   ├── ablation_study_barchart.png         # Trade-offs between Delta-QED and structural similarity
│   ├── hyperparameter_sensitivity.png      # Hyperparameter response surface curves
│   └── counterfactual_action_comparison.png# Counterfactual action reward breakdown
│
├── reports/                                # Comprehensive Technical Documentation
│   └── final_project_report.md             # Complete 25-section IEEE-style capstone paper
│
├── configs/                                # Experiment Configuration
│   └── experiment_config.yaml              # Hyperparameters (PPO, GNN, Environment, Reward)
│
├── checkpoints/                            # Serialized PyTorch Model Checkpoints
│   ├── ppo_adaptive.pt                     # Trained weights of proposed adaptive PPO agent
│   └── ppo_baseline.pt                     # Trained weights of unshaped baseline PPO agent
│
├── data/                                   # Chemical Dataset Splits & Scaffolds
│   └── splits/
│       ├── train_molecules.csv             # 4,000 Bemis-Murcko scaffold-disjoint training molecules (80%)
│       ├── val_molecules.csv               # 500 validation molecules (10%)
│       ├── test_benchmark_molecules.csv    # 500 final test benchmark molecules (10%, 125 per tier)
│       ├── scaffold_overlap_report.csv     # Zero scaffold overlap audit report
│       ├── qed_stratification_report.csv   # QED tier distribution report
│       └── split_info.json                 # Dataset integrity verification metadata
│
├── logs/                                   # Training History Logs
│   ├── train_ppo_adaptive.csv              # Episode logs for proposed adaptive agent
│   └── train_ppo_baseline.csv              # Episode logs for unshaped baseline agent
│
├── MASTER_PROJECT_EXPLANATION.md            # Master viva defense & code explanation guide
│
└── mid_folder/                             # Dedicated Mid-Review Archive
    ├── notebooks/                          # Preliminary Phase 1–3 & mid-review demo notebooks
    │   ├── 01_dataset_analysis.ipynb
    │   ├── 02_molecule_graph_and_edits.ipynb
    │   ├── 03_rl_environment.ipynb
    │   └── mid_review_demo.ipynb
    ├── checkpoints/                        # Intermediate test checkpoints
    ├── configs/                            # Initial reward configurations
    ├── figures/                            # Initial diagnostic plots
    ├── results/                            # Preliminary pilot test outputs
    ├── mid_review_README.md                # Archived mid-review documentation
    └── readmetxt.txt                       # Archived mid-review project notes
```

---

## 3. Dataset Configuration (80:10:10 Cohort)

From the full ZINC-250K library ($249,455$ small molecules), an experimental cohort of $5,000$ molecules was selected and partitioned via Bemis-Murcko scaffold clustering:

| Split | Molecule Count | Percentage | Unique Scaffolds | Cross-Split Scaffold Overlap |
|---|---:|---:|---:|---|
| **Training** | 4,000 | 80.0% | 3,251 | **0.00% (Zero Leakage)** |
| **Validation** | 500 | 10.0% | 441 | **0.00% (Zero Leakage)** |
| **Final Testing** | 500 | 10.0% | 500 | **0.00% (Zero Leakage)** |
| **Total Selected** | **5,000** | **100.0%** | **4,192** | Verified 0 overlap across all pairs |

### 500-Molecule Test Benchmark Stratification:
- **Low Tier (<0.50):** 125 molecules
- **Medium Tier (0.50-0.70):** 125 molecules
- **High Tier (0.70-0.85):** 125 molecules
- **Very High Tier ($\ge 0.85$):** 125 molecules
- **Total:** 500 molecules across 500 unique Bemis-Murcko scaffolds ($100\%$ scaffold diversity).

---

## 4. Key Empirical Results (2,000 Trajectories across 500 Molecules)

| Method | Molecules | Mean $\Delta$QED | Formal Success Rate | High-QED Success Rate | Mean Similarity | Mean $\Delta$SAS | PAINS Alert Rate | Chemical Validity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Random Valid Edits** | 500 | $-0.4243$ | $2.2\%$ | $0.8\%$ | $0.2487$ | $+1.971$ | $17.0\%$ | $100.0\%$ |
| **PPO Baseline (Unshaped)** | 500 | $-0.3782$ | $5.4\%$ | $2.0\%$ | $0.2706$ | $+1.336$ | $14.8\%$ | $100.0\%$ |
| **PPO Proposed (Adaptive)** | 500 | **$-0.1730$** | **$5.0\%$** | **$9.6\%$** | **$0.4731$** | **$+0.827$** | **$5.0\%$** | **$100.0\%$** |
| **Greedy Hill-Climbing** | 500 | $+0.1229$ | $65.4\%$ | $61.0\%$ | $0.6188$ | $+0.502$ | $9.4\%$ | $100.0\%$ |

### Core Empirical Findings:
1. **Scaffold Preservation:** PPO Proposed achieves a mean final Tanimoto similarity of **$0.4731$**, safely respecting the $0.40$ threshold constraint. Unshaped PPO collapses to **$0.2706$** (violating the constraint in $91.4\%$ of cases).
2. **Toxic Alert Suppression:** PPO Proposed cuts toxic PAINS structural alerts to **$5.0\%$** (a 3-fold reduction vs. $14.8\%$ for unshaped baseline and $17.0\%$ for random edits).
3. **High-Drug-Likeness Yield:** In producing clinical-grade drug-like compounds ($\text{QED} \ge 0.80$ while maintaining similarity $\ge 0.40$), PPO Proposed delivers **$9.6\%$**, outperforming unshaped PPO ($2.0\%$) by **$4.8\times$** and random edits ($0.8\%$) by **$12\times$**.
4. **Controlled Synthetic Complexity:** PPO Proposed curbs synthetic degradation to $\Delta\text{SAS} = +0.827$ (compared to $+1.336$ for baseline and $+1.971$ for random).

---

## 5. Quick Presentation & Reproduction Guide

### To Present to Evaluators:
1. **Interactive Demo:** Open [`notebooks/04_gnn_encoder.ipynb`](notebooks/04_gnn_encoder.ipynb), [`notebooks/05_ppo_training.ipynb`](notebooks/05_ppo_training.ipynb), [`notebooks/06_baseline_comparison.ipynb`](notebooks/06_baseline_comparison.ipynb), and [`notebooks/07_final_evaluation.ipynb`](notebooks/07_final_evaluation.ipynb).
2. **Viva Defense Cheat-Sheet:** Open [`MASTER_PROJECT_EXPLANATION.md`](MASTER_PROJECT_EXPLANATION.md) for plain-English explanations of all chemistry and RL concepts, file breakdowns, and answers to questions.
3. **Formal Academic Paper:** Open [`reports/final_project_report.md`](reports/final_project_report.md) for derivations, tables, and literature citations.
4. **Mid-Review Reference:** Point out that preliminary Phase 1–3 exploratory notebooks and mid-review demo files are archived inside [`mid_folder/`](mid_folder/).

