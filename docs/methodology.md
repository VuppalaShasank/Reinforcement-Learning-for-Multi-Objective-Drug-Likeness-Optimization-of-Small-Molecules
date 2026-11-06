# Methodology: Reinforcement Learning for Drug-Likeness Optimization

## 1. Problem Formulation
Lead optimization in medicinal chemistry requires enhancing a small molecule's drug-likeness (quantified via the Quantitative Estimate of Drug-likeness, QED) while maintaining structural similarity to the starting lead and preserving synthetic accessibility (SAS).

We formulate this task as a Finite-Horizon Markov Decision Process (MDP) $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:
- **State Space $\mathcal{S}$**: Molecular graphs $G = (V, E, X, E_{\text{attr}})$ accompanied by scalar state features $s_t = [t/T, \text{QED}_t, \text{SAS}_t, \text{Sim}(M_0, M_t), \text{QED}_0, \text{SAS}_0] \in \mathbb{R}^6$.
- **Action Space $\mathcal{A}$**: 1,901 discrete molecular editing actions:
  - `STOP` (Action 0): Voluntarily terminate optimization.
  - `CHANGE_BOND` (Actions 1–1,406): Alter bond orders between valid atom pairs.
  - `ADD_ATOM_BOND` (Actions 1,407–1,634): Append atom bonded to an existing atom.
  - `SUBSTITUTE_ATOM` (Actions 1,635–1,862): Mutate an atom element.
  - `REMOVE_ATOM` (Actions 1,863–1,900): Delete an atom without creating disconnected fragments.
- **Transition Dynamics $\mathcal{P}$**: Deterministic chemical editing via RDKit with explicit valence checking.
- **Reward Function $\mathcal{R}$**: Adaptive multi-objective reward shaping.

---

## 2. Neural Architecture

### 2.1 Edge-Aware Graph Neural Network (GNN V2)
Standard message passing often ignores chemical bond orders. Our improved architecture utilizes `EdgeAwareMessagePassing`:
$$m_{ij} = W_{\text{node}} h_j + W_{\text{edge}} e_{ij}, \quad e_{ij} \in \{\text{Single}, \text{Double}, \text{Triple}, \text{Aromatic}\}$$
$$h_i^{(l+1)} = \text{ReLU}\left(W_{\text{self}} h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} m_{ij}\right)$$

Readout concatenates global mean pooling and global max pooling:
$$h_G = [\text{MeanPool}(\{h_i\}), \text{MaxPool}(\{h_i\})] \in \mathbb{R}^{128}$$

The graph embedding is fused with the scalar descriptor vector:
$$z = \text{ReLU}(W_{\text{fuse}} [h_G, s_t]) \in \mathbb{R}^{128}$$

### 2.2 Actor-Critic Policy
- **Actor Head**: Fully connected layer producing unnormalized logits for all 1,901 actions.
  - Valid action masking applies $-\infty$ to invalid actions before Softmax:
    $$\pi(a | s) = \text{Softmax}(\text{logits} + \log M(s))$$
- **Critic Head**: Linear projection estimating state value $V(s) \in \mathbb{R}$.

---

## 3. Calibrated Adaptive Reward Shaping
Reward shaping dynamically adjusts weights based on initial property headroom:
$$H_0 = 1.0 - \text{QED}_0$$
$$R(s_t, a_t, s_{t+1}) = w_{\text{qed}}(H_0) \Delta\text{QED} - w_{\text{sim}} \text{Barrier}(\text{Sim}) - w_{\text{sas}} \Delta\text{SAS} - \text{StepCost}$$

### STOP Action Preservation Incentive
If the agent voluntarily chooses `STOP` while retaining positive improvement:
$$R_{\text{stop}} = +2.0 \times \Delta\text{QED}$$
This eliminates runaway over-editing and prevents late-stage molecular degradation.

---

## 4. PPO-Guided Local Search Hybrid
To bridge policy evaluation and local search:
1. The trained actor ranks all valid actions.
2. The Top-$K$ ($K=5$) most probable actions are evaluated via 1-step lookahead.
3. If $\max(R) > 0$, the best action is executed.
4. If no candidate improves the shaped score, the trajectory immediately terminates.

