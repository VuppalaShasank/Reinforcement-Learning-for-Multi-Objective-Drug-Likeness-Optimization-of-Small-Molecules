# Experimental Benchmark Results Index

This directory archives the empirical evaluations conducted for the project:  
**Reinforcement Learning for Drug-Likeness Optimization of Small Molecules (Course: 22AIE401)**.

To maintain strict scientific provenance, three distinct experimental benchmarks are recorded separately:

---

## 1. Pilot Experiment (`results/pilot_60/`)
- **Cohort Size:** 60 molecules (scaffold-separated).
- **Stratification:** 15 molecules per QED tier (Low, Medium, High, Very High).
- **Methods Evaluated:**
  1. Random Valid Edits
  2. Greedy Hill Climbing
  3. PPO Baseline (unshaped)
  4. PPO Proposed (Adaptive V1)
- **Status:** Historical pilot milestone. Preserved as early exploratory proof-of-concept.
- **Key Artifacts:**
  - `final_evaluation_results.csv`: 240 evaluation trajectories.
  - `method_comparison_summary.csv`: Aggregated performance metrics.
  - `ablation_study_results.csv`: Component ablation records.
  - `sensitivity_analysis.csv`: Headroom weight sensitivity.

---

## 2. Main Benchmark Experiment (`results/benchmark_500/`)
- **Cohort Size:** 500 molecules (Bemis-Murcko scaffold-separated, 0% overlap with 4,000 train and 500 val molecules).
- **Stratification:** 125 molecules per QED tier:
  - Low: $\text{QED} < 0.50$ (125)
  - Medium: $0.50 \le \text{QED} < 0.70$ (125)
  - High: $0.70 \le \text{QED} < 0.85$ (125)
  - Very High: $\text{QED} \ge 0.85$ (125)
- **Methods Evaluated:**
  1. Random Valid Edits (2.2% formal success, Mean $\Delta$QED $-0.4243$)
  2. PPO Baseline Unshaped (5.4% formal success, Mean $\Delta$QED $-0.3782$)
  3. PPO Proposed Adaptive V1 (5.0% formal success, Mean $\Delta$QED $-0.1730$, 40.6% violations vs. 91.4%)
  4. Greedy Hill Climbing (65.4% formal success, Mean $\Delta$QED $+0.1229$)
- **Status:** Primary 500-molecule benchmark milestone.
- **Key Artifacts:**
  - `final_evaluation_results.csv`: 2,000 evaluation trajectories.
  - `method_comparison_summary.csv`: Primary benchmark table.
  - `qed_stratified_summary.csv`: Stratified breakdown across the 4 starting-QED tiers.
  - `counterfactual_action_analysis.csv`: Action-level step analysis.
  - `reward_hacking_audit.csv`: Verification of chemical sanity.

---

## 3. PPO Improvement Diagnostic Experiment (`results/improvement_100/`)
- **Cohort Size:** 100 held-out validation molecules (scaffold-separated, 25 per QED tier).
- **Objective:** Diagnose PPO underperformance, integrate Edge-Aware GNN message passing, calibrate reward shaping with a STOP action preservation bonus ($+2.0 \times \Delta\text{QED}$), and benchmark the PPO-Guided Local Search Hybrid.
- **Methods Evaluated (6 Variants):**
  1. Random Valid Edits
  2. PPO Baseline (V1 Unshaped)
  3. PPO Adaptive (V1 Original)
  4. PPO Adaptive (V2 Edge-Aware Pure)
  5. PPO-Guided Search (Hybrid Top-5)
  6. Greedy Hill Climbing (Oracle)
- **Status:** Advanced optimization milestone demonstrating computational oracle efficiency ($47.2\times$ fewer property evaluations than Greedy) and positive mean $\Delta\text{QED}$ ($+0.0183$).
- **Key Artifacts:**
  - `final_validation_comparison.csv`: 600 individual trajectory records.
  - `ppo_variants_comparison.csv`: Master 6-method comparison table.
  - `action_audit_report.csv`: Action space composition audit.
  - `diagnostics_summary.csv`: Empirical diagnostics of termination and rollout behavior.

---

## Benchmark Cross-Reference Rules
- Do NOT merge rows from different sample sizes into an unlabelled table.
- When presenting results, always state whether the metrics originate from the Pilot (60), Benchmark (500), or Diagnostic Improvement (100) cohort.

