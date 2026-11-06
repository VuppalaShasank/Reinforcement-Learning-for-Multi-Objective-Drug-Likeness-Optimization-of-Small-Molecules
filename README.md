# Reinforcement Learning for Multi-Objective Drug-Likeness Optimization of Small Molecules

> **From Adaptive Policy Gradients to Policy-Guided Local Search**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-17%2F17%20Passing-brightgreen.svg)](#running-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** Shasank Vuppala  
**Affiliation:** Department of CSE (AI), Amrita Vishwa Vidyapeetham, Amritapuri  
**Course:** 22AIE401 — Reinforcement Learning  
**Dataset:** ZINC-250K (249,455 drug-like molecules)

---

## Overview

This project formulates **lead optimization** — a core step in computational drug discovery — as a **Markov Decision Process (MDP)** over molecular graphs. An RL agent learns to apply discrete chemical edits (add atom, remove atom, substitute element, change bond) to improve a molecule's **drug-likeness (QED)** while preserving its structural identity, synthetic feasibility, and pharmacological safety.

### Five Key Contributions

| # | Contribution | Where in Code |
|---|---|---|
| 1 | **Edge-Aware GNN Encoder** — 3-layer message-passing network with 4D bond features (Single, Double, Triple, Aromatic) | [`src/models/gnn_encoder_v2.py`](src/models/gnn_encoder_v2.py) |
| 2 | **Adaptive Reward Shaping** — Headroom-calibrated QED gain, smooth quadratic similarity barrier, SAS penalty, and STOP preservation bonus | [`src/rewards/adaptive_reward.py`](src/rewards/adaptive_reward.py) |
| 3 | **Dynamic Valence-Aware Action Masking** — Filters chemically invalid actions (1,901 total) before policy sampling | [`src/environment/action_masking.py`](src/environment/action_masking.py) |
| 4 | **PPO Actor-Critic Training** — Clipped surrogate objective with GAE rollouts and entropy regularisation | [`src/agents/ppo_agent_v2.py`](src/agents/ppo_agent_v2.py) |
| 5 | **PPO-Guided Hybrid Top-5 Search** — Policy ranks candidates, 1-step lookahead selects best; 47× fewer evaluations than greedy | [`src/search/ppo_guided_search.py`](src/search/ppo_guided_search.py) |

---

## Results

Evaluated on **100 held-out, scaffold-separated test molecules** (25 per QED tier):

| Method | Mean ΔQED | High-QED Success | Similarity | Violations | Prop Evals/Mol | Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Random Valid Edits | −0.465 | 0% | 0.222 | 95% | 9.51 | 0.19s |
| PPO Baseline (Unshaped) | −0.387 | 2% | 0.264 | 87% | 9.53 | 0.21s |
| PPO Adaptive (V1) | −0.205 | 10% | 0.461 | 46% | 9.56 | 0.23s |
| PPO Adaptive (V2 Edge-Aware) | −0.177 | 9% | 0.715 | **0%** | 2.92 | **0.10s** |
| **PPO-Guided Search (Hybrid Top-5)** | **+0.018** | **41%** | **0.879** | **0%** | **5.68** | 0.15s |
| Greedy Hill Climbing (Oracle) | +0.108 | 67% | 0.626 | 0% | 268.22 | 0.68s |

**Key takeaway:** PPO-Guided Search achieves **47× fewer property evaluations** than Greedy Hill Climbing while preserving **0.879 structural similarity** (vs 0.626 for Greedy) and the **lowest PAINS toxicophore rate** (5.0%).

---

## Repository Structure

```
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/                                # Core Python package (10 subpackages, 21 modules)
│   ├── chemistry/                      # Molecular graph features, editing, QED/SAS/PAINS
│   │   ├── molecule_editor.py          #   1,901 discrete chemical edits
│   │   ├── graph_utils.py              #   21D node + 4D edge feature extraction
│   │   ├── properties.py               #   QED, SAS, Tanimoto, PAINS, Brenk filters
│   │   └── descriptors.py              #   Scalar descriptors (MW, LogP, HBD, HBA, TPSA)
│   ├── environment/                    # Gymnasium MDP environment
│   │   ├── molecular_environment.py    #   MolecularGraphEnv with state/reward/done
│   │   └── action_masking.py           #   Valence-aware action masking (1,901 actions)
│   ├── models/                         # Neural network architectures
│   │   ├── gnn_encoder.py              #   3-Layer GNN (V1, node-only)
│   │   ├── gnn_encoder_v2.py           #   Edge-Aware 3-Layer GNN (V2, 4D bond features)
│   │   ├── action_policy.py            #   Actor-Critic policy head (V1)
│   │   └── action_policy_v2.py         #   Actor-Critic with masking support (V2)
│   ├── agents/                         # RL agents
│   │   ├── ppo_agent.py                #   PPO with GAE & clipped objective (V1)
│   │   └── ppo_agent_v2.py             #   Improved PPO with curriculum (V2)
│   ├── rewards/                        # Reward functions
│   │   ├── reward.py                   #   Potential-based ΔQED reward (V1)
│   │   └── adaptive_reward.py          #   Calibrated adaptive reward with STOP bonus (V2)
│   ├── search/                         # Hybrid search
│   │   └── ppo_guided_search.py        #   Top-5 PPO-Guided Local Search
│   ├── baselines/                      # Benchmark methods
│   │   └── baseline_methods.py         #   Random Valid Edits & Greedy Hill Climbing
│   ├── training/                       # Training infrastructure
│   │   ├── training_manager.py         #   Training loop orchestrator
│   │   └── training_utils.py           #   GAE rollouts, logging, utilities
│   └── evaluation/                     # Metrics and statistical tests
│       ├── evaluation.py               #   Benchmark evaluator & Pareto metrics
│       ├── reward_hacking_audit.py     #   PAINS/Brenk chemical safety auditor
│       └── statistical_analysis.py     #   Paired t-test & Wilcoxon signed-rank
│
├── notebooks/                          # 8 end-to-end Jupyter notebooks
│   ├── 01_dataset_analysis.ipynb       #   ZINC-250K curation & Bemis-Murcko splitting
│   ├── 02_molecule_graph_and_edits.ipynb #   Graph representation & chemical edits
│   ├── 03_rl_environment.ipynb         #   Gymnasium env, valence masking, transitions
│   ├── 04_gnn_encoder.ipynb            #   Edge-Aware GNN architecture & forward pass
│   ├── 05_ppo_training.ipynb           #   PPO training loop & convergence curves
│   ├── 06_baseline_comparison.ipynb    #   500-molecule benchmark: Random vs Greedy vs PPO
│   ├── 07_ppo_improvement_analysis.ipynb #   Failure diagnostics & Hybrid Top-5 search
│   └── 08_final_evaluation.ipynb       #   Ablation, sensitivity analysis & safety audits
│
├── code_output_figures/                # 10 publication figures
├── data/                               # Dataset & scaffold-separated splits
│   ├── raw/zinc_250k.csv               #   ZINC-250K (249,455 molecules)
│   ├── processed/                      #   Cleaned & canonicalized molecules
│   └── splits/                         #   Train (4K) / Val (500) / Test (500)
├── checkpoints/                        # Trained model weights
│   ├── baseline/                       #   PPO Baseline
│   ├── adaptive_v1/                    #   PPO Adaptive V1
│   ├── adaptive_v2/                    #   PPO Adaptive V2 (Edge-Aware)
│   └── hybrid/                         #   PPO-Guided Hybrid Top-5
├── results/                            # Benchmark CSVs across all experiments
├── logs/                               # Training logs & diagnostic audits
├── configs/                            # Experiment & evaluation YAML configs
├── scripts/                            # CLI scripts for training, eval & reproduction
├── tests/                              # 17 unit tests (full pipeline coverage)
├── reports/                            # IEEE Access LaTeX paper & project reports
└── docs/                               # Methodology, protocols & limitations
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- RDKit (with `SA_Score` contrib)
- PyTorch ≥ 2.0 & PyTorch Geometric ≥ 2.5

### Installation

```bash
git clone https://github.com/your-username/RL_Sem7_Proj.git
cd RL_Sem7_Proj
pip install -r requirements.txt
```

### Running Tests

```bash
python -m pytest                      # 17/17 tests, ~8 seconds
```

### Reproducing All Results

```bash
python scripts/reproduce_experiments.py --suite all
```

Verifies metrics across the Pilot (60-molecule), Benchmark (500-molecule), and Improvement (100-molecule) cohorts.

### Training

```bash
python scripts/run_training.py --episodes 100 --curriculum
```

### Running Baselines

```bash
python scripts/run_baselines.py --baseline both --limit 20
```

### Generating Figures

```bash
python scripts/generate_academic_figures.py
```

---

## Notebook Guide

The research pipeline follows 8 sequential notebooks:

| # | Notebook | Focus | Key Output |
|---|---|---|---|
| 1 | `01_dataset_analysis` | Data curation | Bemis-Murcko scaffold splitting, QED/SAS distributions |
| 2 | `02_molecule_graph_and_edits` | Representation | 21D node + 4D edge features, 1,901 edit verification |
| 3 | `03_rl_environment` | MDP environment | `MolecularGraphEnv`, valence action masking |
| 4 | `04_gnn_encoder` | GNN architecture | Edge-Aware message passing, dual-pooling readout |
| 5 | `05_ppo_training` | Policy training | PPO Actor-Critic, GAE rollouts, loss curves |
| 6 | `06_baseline_comparison` | Benchmark | 500-molecule evaluation: Random vs Greedy vs PPO |
| 7 | `07_ppo_improvement_analysis` | Diagnostics | Action entropy collapse, Hybrid Top-5 search |
| 8 | `08_final_evaluation` | Final evaluation | Ablation study, sensitivity analysis, safety audits |

---

## Reward Function

The adaptive reward (V2) in [`src/rewards/adaptive_reward.py`](src/rewards/adaptive_reward.py) computes:

```
r(s, a, s') = γ_headroom · ΔQED  −  P_sim(sim)  −  P_SAS(sas)  −  c_step
```

Where:
- **γ_headroom** = 1 + min(1.5, max(0, (0.90 − q₀) / 0.90)) — amplifies signal on low-QED leads
- **P_sim** — smooth quadratic penalty in [0.40, 0.55], steep linear barrier below 0.40
- **P_SAS** — analogous barrier keeping SAS ≤ max(SAS₀, 4.5)
- **STOP bonus** — +5.0 if ΔQED ≥ 0.10 and constraints met; +2.0 × ΔQED preservation bonus otherwise

---

## Model Checkpoints

| Checkpoint | Description | File |
|---|---|---|
| PPO Baseline | Potential-based ΔQED reward | `checkpoints/baseline/ppo_baseline.pt` |
| PPO Adaptive V1 | Headroom + constraint slack | `checkpoints/adaptive_v1/ppo_adaptive.pt` |
| PPO Adaptive V2 | Edge-Aware GNN + STOP bonus | `checkpoints/adaptive_v2/ppo_adaptive_v2_improved.pt` |
| Hybrid Top-5 | Policy for guided local search | `checkpoints/hybrid/ppo_guided_hybrid.pt` |

SHA256 checksums are in [`checkpoints/checkpoint_manifest.json`](checkpoints/checkpoint_manifest.json).

---

## Limitations

1. **Single-path rollout drift** — Pure reactive PPO suffers from non-monotonic reward landscapes; the Hybrid Top-5 lookahead mitigates this.
2. **Fixed K=5 lookahead** — Dynamic, entropy-based candidate sizing is left for future work.
3. **Surrogate objectives** — QED and SAS are fast heuristics, not physics-based docking scores. The framework's 5.68 evaluations/molecule could scale to expensive 3D docking oracles.

See [`docs/limitations.md`](docs/limitations.md) for a full discussion.

---

## Citation

```bibtex
@article{vuppala2026rl_drug_opt,
  title   = {Reinforcement Learning for Multi-Objective Drug-Likeness Optimization 
             of Small Molecules: From Adaptive Policy Gradients to 
             Policy-Guided Local Search},
  author  = {Vuppala, Shasank},
  journal = {Course Project Report, 22AIE401 Reinforcement Learning},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.
