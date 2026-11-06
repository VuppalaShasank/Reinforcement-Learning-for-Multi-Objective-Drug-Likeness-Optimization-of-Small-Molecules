# Master Experiment Summary: Reinforcement Learning for Drug-Likeness Optimization

**Course:** 22AIE401 — Reinforcement Learning Project  
**Repository:** Final Reorganized Academic Submission  

---

## 1. Experimental Overview

This project investigates reinforcement learning methods for optimizing the drug-likeness (QED) of drug-like small molecules from the ZINC-250K dataset under chemical validity, structural similarity ($\ge 0.40$), and synthetic accessibility (SAS) constraints.

Across three distinct benchmark iterations, we evaluated:
1. **Pilot Experiment (60 molecules):** Initial proof-of-concept testing 4 methods.
2. **Main Benchmark (500 molecules):** Comprehensive evaluation across 500 Bemis-Murcko scaffold-separated test molecules stratified across 4 starting-QED tiers.
3. **PPO Improvement Diagnostic Experiment (100 molecules):** Root-cause diagnosis of PPO underperformance, development of an Edge-Aware 3-Layer GNN (incorporating 4D bond attributes), Calibrated Adaptive Reward Shaping with STOP Action Preservation ($+2.0 \times \Delta\text{QED}$ bonus), and PPO-Guided Local Search (Top-5 Hybrid).

---

## 2. Benchmark Results Comparison

### Benchmark A: Historical Pilot Experiment (60 Molecules)

| Method | Mean $\Delta$QED | Success Rate | Mean Similarity | Mean $\Delta$SAS | Mean Edits | PAINS Alert Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Valid Edits** | $-0.4797$ | $0.0\%$ | $0.2046$ | $+2.073$ | $9.92$ | $13.3\%$ |
| **PPO Baseline (Unshaped)** | $-0.4376$ | $5.0\%$ | $0.2646$ | $+1.406$ | $9.60$ | $16.7\%$ |
| **PPO Adaptive (V1)** | $-0.1477$ | $5.0\%$ | $0.4941$ | $+0.723$ | $9.62$ | $3.3\%$ |
| **Greedy Hill Climbing** | $+0.1095$ | $56.7\%$ | $0.5989$ | $+0.400$ | $1.77$ | $6.7\%$ |

---

### Benchmark B: Main Test Benchmark (500 Molecules)

| Method | Mean $\Delta$QED | Median $\Delta$QED | Formal Success Rate | High-QED Success | Mean Similarity | Constraint Violations | Mean $\Delta$SAS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Valid Edits** | $-0.4243$ | $-0.4399$ | $2.2\%$ | $0.8\%$ | $0.2487$ | $93.6\%$ | $+1.971$ |
| **PPO Baseline (Unshaped)** | $-0.3782$ | $-0.3982$ | $5.4\%$ | $2.0\%$ | $0.2706$ | $91.4\%$ | $+1.336$ |
| **PPO Proposed (Adaptive V1)** | $-0.1730$ | $-0.1530$ | $5.0\%$ | $9.6\%$ | $0.4731$ | $40.6\%$ | $+0.827$ |
| **Greedy Hill Climbing** | $+0.1229$ | $+0.1167$ | $65.4\%$ | $61.0\%$ | $0.6188$ | **$0.0\%$** | $+0.502$ |

---

### Benchmark C: PPO Improvement Diagnostic Experiment (100 Molecules)

| Method | Mean $\Delta$QED | Formal Success | High-QED Success | Mean Similarity | Constraint Violations | Mean $\Delta$SAS | Avg Prop Evals / Mol | Latency / Mol |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Valid Edits** | $-0.4654$ | $3.0\%$ | $0.0\%$ | $0.2220$ | $95.0\%$ | $+2.006$ | $9.51$ | $0.188\text{s}$ |
| **PPO Baseline (V1 Unshaped)** | $-0.3866$ | $5.0\%$ | $2.0\%$ | $0.2635$ | $87.0\%$ | $+1.255$ | $9.53$ | $0.209\text{s}$ |
| **PPO Adaptive (V1 Original)** | $-0.2054$ | $6.0\%$ | $10.0\%$ | $0.4608$ | $46.0\%$ | $+0.836$ | $9.56$ | $0.228\text{s}$ |
| **PPO Adaptive (V2 Edge-Aware Pure)** | $-0.1771$ | $0.0\%$ | $9.0\%$ | $0.7152$ | **$0.0\%$** | $+0.427$ | $2.92$ | **$0.098\text{s}$** |
| **PPO-Guided Search (Hybrid Top-5)** | **$+0.0183$** | $3.0\%$ | **$41.0\%$** | **$0.8790$** | **$0.0\%$** | **$+0.105$** | **$5.68$** | $0.153\text{s}$ |
| **Greedy Hill Climbing (Oracle)** | $+0.1076$ | $63.0\%$ | $67.0\%$ | $0.6255$ | **$0.0\%$** | $+0.383$ | $268.22$ | $0.681\text{s}$ |

---

## 3. Scientific Conclusions

1. **Pure Single-Path RL vs. Local Search:** Pure reactive policies (PPO V1 and V2) struggle in combinatorial chemistry because a single suboptimal edit can degrade molecular resonance and QED.
2. **The Power of PPO-Guided Local Search:** Filtering candidate edits using the PPO Actor and evaluating only the Top-5 candidates via lookahead achieves positive mean $\Delta\text{QED}$ ($+0.0183$) and $41\%$ high-QED success.
3. **The Pareto Frontier (Quality vs. Compute):**
   - Greedy Hill Climbing maximizes QED ($+0.1076$) but brute-forces **268.22 property evaluations per molecule**.
   - PPO-Guided Search achieves **$47.2\times$ fewer property evaluations** ($5.68$ per molecule) and **$4.45\times$ faster wall-clock speed**, while preserving initial lead identity ($0.8790$ vs. $0.6255$) and synthetic accessibility ($\Delta\text{SAS} = +0.105$ vs. $+0.383$).

