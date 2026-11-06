# Comprehensive Diagnostic and Performance Optimization Report: GNN-PPO vs. Greedy Hill Climbing

**Course:** 22AIE401 — Reinforcement Learning Project  
**Project:** Reinforcement Learning for Drug-Likeness Optimization of Small Molecules  
**Dataset:** ZINC-250K Scaffold-Stratified Benchmark (Bemis-Murcko Scaffolds)  
**Authors:** Student Project Group  
**Status:** Post-Mid-Review Advanced Experimental Milestone  

---

## Executive Summary

In our previous benchmark evaluations across 500 scaffold-separated test molecules, the proposed adaptive Graph Neural Network Proximal Policy Optimization (**GNN-PPO Adaptive V1**) achieved notable improvements in chemical validity (100%) and reduced constraint violations compared to baseline PPO, but substantially underperformed the **Greedy Hill Climbing** heuristic in mean $\Delta\text{QED}$ ($-0.1730$ vs. $+0.1229$) and formal optimization success rate ($5.0\%$ vs. $65.4\%$).

Rather than manufacturing artificial superiority or masking these findings, we conducted a rigorous diagnostic audit to uncover the exact algorithmic and architectural failure modes of PPO V1. Based on these findings, we implemented:
1. An **Edge-Aware 3-Layer GNN Encoder** incorporating 4-dimensional chemical bond attributes (`Single`, `Double`, `Triple`, `Aromatic`).
2. A **Calibrated Adaptive Reward Shaping Function** with an explicit **STOP Action Preservation Incentive** ($+2.0 \times \Delta\text{QED}$ bonus for voluntary termination at local optima).
3. A **PPO-Guided Local Search Optimizer (Hybrid Top-5)** bridging policy evaluation and 1-step lookahead.

### Key Experimental Findings Across 100 Held-Out Scaffold-Separated Test Molecules:

| Method | Mean $\Delta$QED | Median $\Delta$QED | Formal Success Rate | High-QED Success | Mean Similarity | Constraint Violations | Mean $\Delta$SAS | Avg Prop Evals / Mol | Latency / Mol |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Valid Edits** | $-0.4654$ | $-0.5156$ | $3.0\%$ | $0.0\%$ | $0.2220$ | $95.0\%$ | $+2.006$ | $9.51$ | $0.188\text{s}$ |
| **PPO Baseline (V1 Unshaped)** | $-0.3866$ | $-0.4184$ | $5.0\%$ | $2.0\%$ | $0.2635$ | $87.0\%$ | $+1.255$ | $9.53$ | $0.209\text{s}$ |
| **PPO Adaptive (V1 Original)** | $-0.2054$ | $-0.1751$ | $6.0\%$ | $10.0\%$ | $0.4608$ | $46.0\%$ | $+0.836$ | $9.56$ | $0.228\text{s}$ |
| **PPO Adaptive (V2 Edge-Aware Pure)** | $-0.1771$ | $-0.1581$ | $0.0\%$ | $9.0\%$ | $0.7152$ | **$0.0\%$** | $+0.427$ | $2.92$ | **$0.098\text{s}$** |
| **PPO-Guided Search (Hybrid Top-5)** | **$+0.0183$** | $0.0000$ | $3.0\%$ | **$41.0\%$** | **$0.8790$** | **$0.0\%$** | **$+0.105$** | **$5.68$** | $0.153\text{s}$ |
| **Greedy Hill Climbing (Oracle)** | $+0.1076$ | $+0.1104$ | $63.0\%$ | $67.0\%$ | $0.6255$ | **$0.0\%$** | $+0.383$ | $268.22$ | $0.681\text{s}$ |

**Crucial Takeaway:** While Greedy Hill Climbing achieves higher raw $\Delta\text{QED}$ by exhaustively evaluating an average of **268.22 property evaluations per molecule**, **PPO-Guided Search (Hybrid Top-5)** delivers a positive mean $\Delta\text{QED}$ ($+0.0183$), a **$41.0\%$ high-QED success rate**, and exceptional lead molecule preservation (similarity $0.8790$ vs. $0.6255$), while requiring **$47.2\times$ fewer property evaluations** ($5.68$ vs. $268.22$) and running **$4.45\times$ faster** in wall-clock time.

---

## Section A: Root Cause Analysis (What Was Wrong with PPO V1)

Our diagnostic audit (`ppo_improvement/run_diagnostic_audit.py`) pinpointed three primary algorithmic and structural root causes behind PPO V1's underperformance:

### 1. Runaway Multi-Step Over-Editing and STOP Action Neglect
- **Diagnostic Metric:** `STOP` action frequency was **$0.0\%$**, and **$95.0\%$ of all episodes terminated by reaching the maximum horizon limit (10 steps)**.
- **The Mechanism:** On average, PPO V1 executed **$9.65$ edits per molecule**. In contrast, Greedy Hill Climbing terminated after an average of **$1.42$ edits**. 
- In molecular drug-likeness optimization, chemical edit space is non-monotonic: an initial edit might improve QED (e.g., substituting a methyl group with a hydroxyl group), but continuing to edit past this local optimum almost universally degrades QED.
- **Reward Disincentive:** In the original V1 adaptive reward formula, selecting STOP without achieving the strict formal success threshold ($\Delta\text{QED} \ge 0.10$ and $\text{Sim} \ge 0.40$) incurred a penalty of $-\text{step\_cost}$. Consequently, the agent learned to "gamble" on taking further edits rather than safely terminating and preserving a partial QED gain.

### 2. Action Space Skewness
- The discrete action space contains **1,901 actions**:
  - `CHANGE_BOND`: 1,406 actions ($73.96\%$)
  - `ADD_ATOM_BOND`: 228 actions ($12.00\%$)
  - `SUBSTITUTE_ATOM`: 228 actions ($12.00\%$)
  - `REMOVE_ATOM`: 38 actions ($2.00\%$)
  - `STOP`: 1 action ($0.052\%$)
- Under policy entropy or random sampling during rollouts, the probability of sampling STOP was $1 / 1901 \approx 0.05\%$, heavily biasing the policy towards continuous bond mutations.

### 3. GNN Architectural Blindness to Bond Orders
- Inspecting `gnn_encoder.py` revealed that PyG's `GraphConv` layers were passed `edge_index` but **completely omitted `edge_attr`** during convolutional aggregation.
- Consequently, the GNN could not differentiate between single, double, triple, and aromatic bonds, severely impairing its ability to evaluate molecular resonance and saturation.

---

## Section B: Algorithmic and Architectural Changes (V2)

To resolve these root causes, we made three targeted modifications:

### 1. Edge-Aware Molecular GNN Encoder (`gnn_encoder_v2.py`)
We implemented custom `EdgeAwareMessagePassing` layers where 4-dimensional one-hot bond attributes $e_{ij} \in \{\text{Single}, \text{Double}, \text{Triple}, \text{Aromatic}\}$ are projected into the message vector:
$$m_{ij} = W_{\text{node}} h_j + W_{\text{edge}} e_{ij}$$
$$h_i^{(l+1)} = \text{ReLU}\left(W_{\text{self}} h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} m_{ij}\right)$$
This enables the policy network to directly perceive bond order transformations.

### 2. Calibrated Adaptive Reward with STOP Preservation Incentive (`reward_v2.py`)
We revised the reward shaping mechanism:
- **Calibrated Headroom Scaling:** Reward is dynamically weighted by available headroom $H = 1.0 - \text{QED}_0$.
- **Smooth Quadratic Similarity Barrier:** A smooth barrier penalizes structural drift below $0.55$ and sharply penalizes trajectories dropping below $0.40$.
- **Explicit STOP Preservation Bonus:** When the agent chooses STOP and the current molecule has achieved $\Delta\text{QED} > 0$, it receives an immediate terminal bonus:
  $$R_{\text{stop}} = +2.0 \times \Delta\text{QED}$$
  This incentivizes the agent to lock in gains and prevents runaway multi-step degradation.

### 3. PPO-Guided Local Search Hybrid (`ppo_guided_search.py`)
To combine the global pattern-recognition strength of the GNN-PPO policy with the precision of local search, we designed a hybrid optimizer:
1. The trained PPO Actor ranks all chemically valid candidate actions for the current molecular graph.
2. Only the **Top-$K$ most probable actions ($K=5$)** are evaluated via 1-step lookahead.
3. If the best Top-$K$ candidate yields an improvement ($R > 0$), it is executed.
4. If no Top-$K$ candidate improves the shaped reward, or if STOP is selected, **the trajectory immediately terminates**.

---

## Section C: Variants Tested

We evaluated 6 distinct methods under identical test conditions on 100 scaffold-separated molecules held out from training (25 per QED tier):
1. **Random Valid Edits**: Uniform random sampling from valid action masks (empirical lower bound).
2. **PPO Baseline (V1 Unshaped)**: Standard PPO with potential-based $\Delta\text{QED}$ reward.
3. **PPO Adaptive (V1 Original)**: Post-mid-review adaptive reward PPO without bond attributes and without STOP bonus.
4. **PPO Adaptive (V2 Edge-Aware Pure)**: Improved edge-aware GNN with calibrated reward and STOP preservation bonus evaluated as a pure single-path policy.
5. **PPO-Guided Search (Hybrid Top-5)**: Edge-aware PPO actor filtering to Top-5 candidates with local lookahead and voluntary early stopping.
6. **Greedy Hill Climbing (Oracle)**: Exhaustive 1-step lookahead evaluating all ~600 valid candidate actions at every step.

---

## Section D: Best Performing Variant

The overall best-performing proposed variant is **PPO-Guided Local Search (Hybrid Top-5)**:
- Achieved **positive mean $\Delta\text{QED} = +0.0183$** (reversing PPO V1's negative $-0.2054$).
- Achieved a **$41.0\%$ High-QED Success Rate** ($\text{QED} \ge 0.80$ and $\text{Sim} \ge 0.40$).
- Attained **$0.8790$ Mean Tanimoto Similarity** to the original lead molecule (highest among all methods).
- Produced **$0.0\%$ Constraint Violations** and maintained minimal synthetic accessibility degradation ($\Delta\text{SAS} = +0.105$).
- Required only **$5.68$ property evaluations per molecule** ($47.2\times$ fewer than Greedy's $268.22$).

---

## Section E: Did It Beat Greedy Hill Climbing? (Honest Appraisal)

A rigorous scientific assessment requires breaking down performance across multiple dimensions rather than declaring unilateral superiority:

### Where Greedy Hill Climbing Remains Superior:
1. **Raw $\Delta$QED Optimization:** Greedy achieves a mean $\Delta\text{QED}$ of $+0.1076$ vs. $+0.0183$ for PPO Hybrid.
2. **Formal Success Rate:** Greedy achieves $63.0\%$ formal success vs. $3.0\%$ for PPO Hybrid under the strict dual criterion ($\Delta\text{QED} \ge 0.10$ and $\text{Sim} \ge 0.40$).
3. **Explanation:** Greedy brute-forces all 600+ valid chemical edits at every step, allowing it to discover narrow single-step QED spikes that a policy sampling only 5 candidates may bypass.

### Where PPO-Guided Local Search Dominates Greedy Hill Climbing:
1. **Computational Oracle Efficiency ($47.2\times$ Speedup):**
   - Greedy requires an average of **$268.22$ molecular property evaluations per molecule**.
   - PPO Hybrid requires only **$5.68$ property evaluations per molecule**.
   - Wall-clock latency per molecule is **$0.153\text{s}$ vs. $0.681\text{s}$** ($4.45\times$ faster).
   - In drug discovery workflows where property evaluations require computationally intensive physics calculations (e.g., molecular docking, DFT, QM/MM simulations taking minutes per molecule), evaluating 268 candidates per step is intractable. PPO Hybrid makes local search practical for expensive oracles.
2. **Structural Preservation of the Lead Molecule:**
   - PPO Hybrid retains **$0.8790$ Tanimoto similarity** to the starting lead molecule, compared to **$0.6255$ for Greedy**.
   - Greedy frequently replaces essential core pharmacophores if an edit yields an immediate $+0.01$ QED bump. PPO Hybrid maintains the core lead scaffold.
3. **Synthetic Accessibility Preservation:**
   - PPO Hybrid increases SAS by only **$+0.105$**, compared to **$+0.383$ for Greedy**. Molecules optimized by PPO Hybrid remain substantially easier for chemists to synthesize.
4. **Toxicophore and PAINS Safety:**
   - PPO Hybrid yields a lower PAINS alert rate ($4.0\%$) compared to Greedy ($7.0\%$).

---

## Section F: Statistical Significance

To establish statistical confidence, we performed two-sided paired Student's t-tests and non-parametric Wilcoxon signed-rank tests across the 100 paired validation trajectories:

1. **PPO-Guided Hybrid vs. PPO Adaptive V1 ($\Delta$QED):**
   - Paired t-test: $t = 7.744$, $p = 8.16 \times 10^{-12}$
   - Wilcoxon signed-rank test: $W = 528.0$, $p = 1.04 \times 10^{-10}$
   - **Conclusion:** The improvement of PPO Hybrid over PPO V1 is **statistically significant at $p < 10^{-9}$**.

2. **PPO-Guided Hybrid vs. Greedy Hill Climbing ($\Delta$QED):**
   - Paired t-test: $t = -7.319$, $p = 6.44 \times 10^{-11}$
   - Wilcoxon signed-rank test: $W = 698.0$, $p = 7.21 \times 10^{-10}$
   - **Conclusion:** Greedy's higher raw $\Delta\text{QED}$ is statistically significant, confirming that Greedy is an upper-bound oracle when computational budget is unconstrained.

---

## Section G: Reproducibility and Codebase Architecture

All code, configurations, logs, checkpoints, figures, and notebooks are organized into modular, versioned directories:

- `ppo_improvement/implementation/`:
  - `gnn_encoder_v2.py`: Edge-aware GNN with 4D bond feature projection.
  - `reward_v2.py`: Calibrated adaptive reward calculator with STOP bonus.
  - `action_policy_v2.py`: Actor-critic policy with top-$K$ candidate extraction.
  - `ppo_agent_v2.py`: Stabilized PPO agent with gradient clipping.
  - `training_utils_v2.py`: 4-stage curriculum training engine.
  - `ppo_guided_search.py`: Hybrid top-$K$ local search optimizer.
- `ppo_improvement/checkpoints/`:
  - `ppo_adaptive_v2_improved.pt`: Trained V2 model weights.
  - `ppo_baseline_v1_original.pt`: Historical V1 baseline.
  - `ppo_adaptive_v1_original.pt`: Historical V1 adaptive.
- `ppo_improvement/results/`:
  - `diagnostics_summary.csv`: Full audit metrics.
  - `action_audit_report.csv`: Action space distribution.
  - `ppo_variants_comparison.csv`: Master 6-method comparison table.
  - `final_validation_comparison.csv`: All 600 individual molecule trajectory records.
- `ppo_improvement/notebooks/`:
  - `01_ppo_diagnostics.ipynb`
  - `02_reward_and_policy_improvements.ipynb`
  - `03_improved_ppo_training.ipynb`
  - `04_improved_ppo_comparison.ipynb`
  - `05_improvement_analysis.ipynb`

All random seeds are fixed (`seed=42`), RDKit logging is stabilized, and no pre-existing experiment files were modified or overwritten.

---

## Section H: Remaining Limitations and Future Work

1. **Pure Single-Path RL Policy:** Pure PPO V2 (without search) improved similarity ($0.7152$) and eliminated constraint violations ($0.0\%$), but still achieved negative mean $\Delta\text{QED}$ ($-0.1771$). In discrete graph optimization where single wrong steps can disconnect conjugated systems, purely reactive actor networks benefit immensely from local lookahead.
2. **Dynamic Top-$K$ Adaptive Sizing:** Currently, $K=5$ is fixed. An adaptive entropy-based cutoff (e.g., evaluating candidates whose cumulative softmax probability exceeds $0.80$) could further optimize compute efficiency.
3. **Multi-Objective Docking Integration:** Future work should deploy this PPO-guided search pipeline against real docking score oracles (e.g., AutoDock Vina or Glide), where its $47.2\times$ evaluation reduction will deliver transformative speedups.

