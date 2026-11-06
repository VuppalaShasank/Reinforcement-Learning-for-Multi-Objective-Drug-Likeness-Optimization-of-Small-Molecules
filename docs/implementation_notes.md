# Implementation Notes and Engineering Details

## 1. Chemical Environment & RDKit Sanitization
- **Molecular Graph Conversion:** Atoms are mapped to 21-dimensional one-hot vectors (element identity: C, N, O, F, S, Cl; degree, formal charge, hybridization, aromaticity). Bonds are mapped to 4-dimensional one-hot vectors (`Single`, `Double`, `Triple`, `Aromatic`).
- **Valence Guard:** Before applying any edit, the `MoleculeEditor` checks maximum valence constraints ($C \le 4, N \le 3, O \le 2, F \le 1, S \le 6, Cl \le 1$). Edits producing radical centers or disconnected components are rejected.
- **Sanitization:** Edits must successfully pass `Chem.SanitizeMol()`. Molecules failing sanitization return `None` and are assigned an invalid-step penalty.

## 2. PPO Algorithmic Stability
- **Gradient Norm Clipping:** Global gradient norm is clipped to $0.50$ via `nn.utils.clip_grad_norm_()` to prevent policy collapse during edge convolutions.
- **Adam Optimizer Stability:** Epsilon is set to $\epsilon = 1 \times 10^{-5}$ to avoid division by zero when gradients are sparse across the 1,901 action logits.
- **Advantage Normalization:** Advantages calculated via Generalized Advantage Estimation (GAE, $\lambda = 0.95, \gamma = 0.99$) are normalized across minibatches ($\mu = 0, \sigma = 1$).

## 3. Top-K Candidate Extraction
- The actor network computes masked logits over all 1,901 actions.
- The `get_top_k_actions(obs, k=5)` method extracts the indices of the $K$ largest logits that correspond to chemically valid actions (`mask == True`).
- This guarantees that property oracles are queried only for chemically viable candidates.

