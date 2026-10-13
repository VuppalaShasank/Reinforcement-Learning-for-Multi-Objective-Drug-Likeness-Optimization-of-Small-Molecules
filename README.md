# Reinforcement Learning for Drug-Likeness Optimization of Small Molecules

**Course:** 22AIE401 – Reinforcement Learning  
**Student:** Vuppala Shasank [AM.SC.U4AIE23064]  
**Faculty Guide:** Dr. Geetha M  
**Department:** Computer Science & Engineering (Artificial Intelligence)  
**Institution:** Amrita Vishwa Vidyapeetham  

---

## 1. Project Overview

This repository develops an amortized Reinforcement Learning system for optimizing existing drug-like small molecules from the **ZINC-250K** benchmark. The goal is to maximize drug-likeness (QED) while maintaining chemical validity, structural similarity to the starting molecule, and synthetic accessibility (SAS).

### Core Proposed Novelty
**Headroom- and Constraint-Slack Adaptive Reward Shaping:**  
Instead of using static reward weights, this method dynamically adapts the balance between QED improvement and constraint penalties based on:
1. **Starting QED Headroom:** $H(q_0) = \max(1.0 - q_0, 0.05)$, normalizing drug-likeness gains across low, medium, and high initial QED molecules.
2. **Similarity Slack Barrier:** Enforces smooth quadratic barrier penalties near the threshold ($\tau_{\text{sim}} = 0.40$), escalating to strict linear penalties upon constraint breach.
3. **SAS Slack Barrier:** Enforces quadratic barrier penalties as synthetic accessibility approaches $\tau_{\text{SAS}} = 4.5$.
4. **Step Cost:** Penalizes superfluous structural edits.

---

## 2. Repository Structure

```
RL_Sem7_Proj/
│
├── notebooks/                              # Primary experimental notebooks
│   ├── 01_dataset_analysis.ipynb           # ZINC-250K inspection & QED/SAS distributions
│   ├── 02_molecule_graph_and_edits.ipynb   # Molecular graph conversion & valid chemical edits
│   ├── 03_rl_environment.ipynb             # Gymnasium environment transitions
│   ├── 04_gnn_ppo_training.ipynb           # 3-Layer GNN encoder forward pass & PPO curves
│   ├── 05_baseline_comparisons.ipynb       # Evaluation against Random and Greedy baselines
│   ├── 06_evaluation_and_reward_hacking_audit.ipynb # QED stratification & PAINS/Brenk audit
│   └── mid_review_demo.ipynb               # Dedicated presentation demo notebook
│
├── implementation/                         # Modular Python source code
│   ├── action_handler.py                   # 1,901-action discrete space with valence masking
│   ├── actor_critic.py                     # Actor-Critic network with masked distribution
│   ├── baselines.py                        # Random Valid Edits & Greedy Hill Climbing
│   ├── data_pipeline.py                    # Bemis-Murcko scaffold 80/10/10 splitting
│   ├── environment.py                      # Gymnasium MolecularGraphEnv environment
│   ├── evaluator.py                        # Metrics & PAINS/Brenk/Lipinski audit
│   ├── gnn_encoder.py                      # 3-Layer GNN state encoder module
│   ├── graph_utils.py                      # SMILES to PyG Data feature extraction
│   ├── molecule_editor.py                  # RDKit RWMol atomic & bond edit operators
│   ├── ppo_agent.py                        # PPO Agent with GAE and clipped objective
│   ├── properties.py                       # Authentic RDKit QED, SAS (sascorer), Similarity
│   ├── reward_module.py                    # Baseline and Adaptive reward calculators
│   ├── run_evaluation.py                   # Benchmark evaluation execution script
│   └── train_experiments.py                # PPO training execution script
│
├── checkpoints/                            # Trained PyTorch model weights
│   ├── ppo_baseline.pt                     # PPO policy trained with Baseline reward
│   └── ppo_adaptive.pt                     # PPO policy trained with Adaptive reward
│
├── figures/                                # Generated experimental plots
│   ├── reward_comparison_diagnostics.png   # Penalty function curves
│   ├── training_reward_curves.png          # PPO training return and Delta QED curves
│   ├── evaluation_delta_qed_and_success.png# Benchmark comparison charts
│   ├── qed_stratified_comparison.png       # Optimization gain by starting headroom
│   └── reward_hacking_audit_chart.png      # PAINS and Brenk structural alert rates
│
├── results/                                # Evaluation CSVs and split metadata
│   ├── method_comparison_summary.csv       # Aggregate performance comparison
│   ├── qed_stratified_summary.csv          # Results stratified by starting QED bin
│   ├── reward_hacking_audit.csv            # Lipinski, PAINS, Brenk filter statistics
│   ├── final_evaluation_results.csv        # Molecule-level test outputs
│   ├── split_info.json                     # Scaffold split statistics
│   └── test_benchmark_molecules.csv        # 60 stratified benchmark test molecules
│
├── configs/                                # Experiment configurations
│   └── reward_config.yaml                  # Reward function hyperparameters
│
├── data/                                   # Processed datasets
│   └── processed/
│       ├── dataset_stats.json
│       └── zinc_processed.csv
│
├── ZINC-250K/                              # Raw benchmark dataset
│   └── 250k_rndm_zinc_drugs_clean_3.csv
│
└── cache/                                  # Internal run logs, audit reports & scratch cache
```

---

## 3. Quickstart & Demonstration

### How to Run the Mid-Review Demonstration
Open Jupyter and run [`notebooks/mid_review_demo.ipynb`](file:///c:/Users/vuppa/OneDrive/Desktop/RL_Sem7_Proj/notebooks/mid_review_demo.ipynb).  
It loads a real ZINC-250K molecule, evaluates baseline QED and authentic SAS, applies a chemically valid edit, calculates new properties and Tanimoto similarity, and computes the step reward.

### How to Run Training
```powershell
python implementation/train_experiments.py
```
Trains both PPO-Baseline and PPO-Adaptive models on training scaffolds, logs episode metrics, and saves model checkpoints to `checkpoints/`.

### How to Run Benchmark Evaluation
```powershell
python implementation/run_evaluation.py
```
Evaluates Random Valid Edits, Greedy Hill Climbing, PPO Baseline, and PPO Adaptive on 60 unseen test scaffolds and updates `results/` and `figures/`.

---

## 4. Key Results

- **Chemical Validity:** **100.0%** across all generated trajectories due to pre-action chemical valence masking.
- **Scaffold Preservation:** PPO Adaptive maintains a mean structural similarity of **0.494** (preserving the starting scaffold above $\tau_{\text{sim}} \ge 0.40$), whereas PPO Baseline drifts to 0.265.
- **Reward-Hacking Audit:** PPO Adaptive reduces the PAINS structural alert rate to **3.3%** (a 5$\times$ reduction compared to PPO Baseline's 16.7%) and achieves only **0.167 Lipinski violations** per molecule.
