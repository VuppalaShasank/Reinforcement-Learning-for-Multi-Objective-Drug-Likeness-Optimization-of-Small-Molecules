# Experimental Evaluation Protocol

## 1. Scaffold-Based Data Separation
To prevent data contamination, all dataset splits are constructed using Bemis-Murcko scaffolds:
- **Training Set:** 4,000 molecules.
- **Validation Set:** 500 molecules (0% scaffold overlap with training).
- **Test Benchmark Set:** 500 molecules (0% scaffold overlap with training and validation).
- Cross-split scaffold overlap is audited and verified via `data/splits/scaffold_overlap_report.csv`.

## 2. QED-Stratification Protocol
Test benchmarks are stratified into 4 balanced starting-QED tiers:
1. **Low:** $\text{QED}_0 < 0.50$ (High headroom for optimization).
2. **Medium:** $0.50 \le \text{QED}_0 < 0.70$ (Moderate headroom).
3. **High:** $0.70 \le \text{QED}_0 < 0.85$ (Tight headroom).
4. **Very High:** $\text{QED}_0 \ge 0.85$ (Preservation-dominant regime).

## 3. Evaluation Metrics
- **Mean $\Delta\text{QED}$:** Final QED minus starting QED ($\text{QED}_T - \text{QED}_0$).
- **Formal Success Rate:** Proportion of trajectories achieving $\Delta\text{QED} \ge +0.10$, $\text{Similarity} \ge 0.40$, and 100% chemical validity.
- **High-QED Success Rate:** Proportion achieving $\text{QED}_T \ge 0.80$, $\text{Similarity} \ge 0.40$, and validity.
- **Mean Tanimoto Similarity:** Morgan circular fingerprint similarity (radius=2, 2048 bits) to starting lead $M_0$.
- **Constraint Violation Rate:** Percentage of final molecules with $\text{Similarity} < 0.40$.
- **Mean $\Delta\text{SAS}$:** Change in Synthetic Accessibility Score ($\text{SAS}_T - \text{SAS}_0$). Lower is better.
- **PAINS Alert Rate:** Frequency of Pan-Assay Interference Compounds alerts detected by RDKit filter catalogs.
- **Property Evaluations per Molecule:** Average number of candidate molecule property evaluations required to complete an optimization trajectory.

