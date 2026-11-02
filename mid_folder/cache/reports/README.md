# Project Quickstart & Execution Guide
**Project:** Reinforcement Learning for Drug-Likeness Optimization of Small Molecules  
**Course:** 22AIE401 – Reinforcement Learning  
**Author:** Vuppala Shasank [AM.SC.U4AIE23064]  

---

## 1. Environment Setup

The project runs using the verified Python environment with full CUDA GPU support:
- Python Environment: `C:\Users\vuppa\torch-gpu-env`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)
- CUDA: Active (`torch.cuda.is_available() == True`)

All required libraries are pre-configured:
- `torch` (2.9.1+cu128)
- `torch_geometric` (2.8.1)
- `rdkit` (2026.9.1 with authentic C++ `sascorer`)
- `gymnasium` (1.4.0)
- `pandas`, `numpy`, `scipy`, `matplotlib`, `scikit-learn`

---

## 2. Execution Commands

From the workspace root (`c:\Users\vuppa\OneDrive\Desktop\RL_Sem7_Proj`):

### Run All Automated Unit Tests:
```powershell
& "C:\Users\vuppa\torch-gpu-env\Scripts\python.exe" cache/tests/test_existing_pipeline.py
& "C:\Users\vuppa\torch-gpu-env\Scripts\python.exe" cache/tests/test_rewards.py
& "C:\Users\vuppa\torch-gpu-env\Scripts\python.exe" cache/tests/test_environment.py
& "C:\Users\vuppa\torch-gpu-env\Scripts\python.exe" cache/tests/test_gnn_ppo.py
```

### Run PPO Model Training (Baseline and Adaptive):
```powershell
& "C:\Users\vuppa\torch-gpu-env\Scripts\python.exe" cache/implementation/train_experiments.py
```
Outputs:
- Checkpoints: `cache/checkpoints/ppo_baseline.pt`, `cache/checkpoints/ppo_adaptive.pt`
- Logs: `cache/logs/train_ppo_baseline.csv`, `cache/logs/train_ppo_adaptive.csv`
- Curves: `cache/figures/training_reward_curves.png`

### Run Held-Out Test Evaluation & Comparison:
```powershell
& "C:\Users\vuppa\torch-gpu-env\Scripts\python.exe" cache/implementation/run_evaluation.py
```
Outputs:
- Results CSVs: `cache/results/final_evaluation_results.csv`, `cache/results/method_comparison_summary.csv`
- QED Stratified Table: `cache/results/qed_stratified_summary.csv`
- Reward-Hacking Audit: `cache/results/reward_hacking_audit.csv`
- Publication Figures: `cache/figures/evaluation_delta_qed_and_success.png`, `cache/figures/qed_stratified_comparison.png`, `cache/figures/reward_hacking_audit_chart.png`

---

## 3. Jupyter Notebooks

All 6 notebooks are located under `cache/notebooks/`:
1. `01_dataset_preprocessing.ipynb`: Dataset inspection, Bemis-Murcko scaffold splitting.
2. `02_molecular_graph_and_edits.ipynb`: Graph feature extraction, valence masked edits.
3. `03_environment_and_rewards.ipynb`: Gym environment and reward barrier penalties.
4. `04_gnn_ppo_training.ipynb`: GNN Actor-Critic forward pass and PPO training curves.
5. `05_baseline_comparisons.ipynb`: Comparison against Random and Greedy Hill Climbing.
6. `06_evaluation_and_reward_hacking_audit.ipynb`: QED stratification and held-out audit.

Launch via Jupyter:
```powershell
& "C:\Users\vuppa\torch-gpu-env\Scripts\jupyter.exe" notebook cache/notebooks
```

---

## 4. Completed vs. Future Tasks

### Completed Tasks:
- [x] Initial implementation audit (`cache/reports/audit_report.md`).
- [x] Genuine RDKit SAScorer integration (no fallbacks).
- [x] Molecular Graph representation (21 node features, 4 edge features, 6 scalars).
- [x] 1,901-action chemical valence pre-action masking.
- [x] Baseline potential-based reward implemented and unit-tested.
- [x] Novel Headroom- and Constraint-Slack Adaptive Reward implemented and unit-tested.
- [x] 3-Layer GNN encoder with mean/max pooling fusion.
- [x] PPO Actor-Critic agent with GAE, clipped objective, and entropy regularization.
- [x] GPU acceleration on NVIDIA RTX 5060 Laptop GPU.
- [x] Bemis-Murcko scaffold 80/10/10 split with 0 scaffold overlap.
- [x] Random Valid Edits and Greedy Hill Climbing baselines.
- [x] Evaluation across 60 held-out benchmark test molecules.
- [x] Reward-hacking audit (Lipinski, PAINS, Brenk, Structural sanity).
- [x] All 6 structured Jupyter notebooks created under `cache/notebooks/`.

### Future / Planned Work:
- Full-scale multi-million step distributed PPO training cluster.
- Incorporation of ring-closure edits beyond local single-bond modifications.
