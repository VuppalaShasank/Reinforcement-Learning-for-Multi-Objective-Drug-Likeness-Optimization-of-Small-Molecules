# Complete Project Explainer & Viva Preparation Guide
## Reinforcement Learning for Multi-Objective Drug-Likeness Optimization of Small Molecules: From Adaptive Policy Gradients to Policy-Guided Local Search

**Course:** 22AIE401 – Reinforcement Learning  
**Author:** Shasank Vuppala (Roll No.: AM.SC.U4AIE23064)  
**Institution:** Amrita Vishwa Vidyapeetham, Department of CSE (Artificial Intelligence)

---

## Executive Summary: What Is This Project About?

Imagine you are a medicinal chemist at a pharmaceutical company. You ran an experiment and found a molecule (a "hit") that kills a disease target, but it cannot be used as a medicine yet because:
1. It is **not drug-like enough** (poor solubility, poor oral absorption, low bioavailability).
2. If you change it too much, it **loses its biological activity** (scaffold collapse).
3. If you make it too complex, **chemists cannot synthesize it** in the laboratory.
4. If you add bad chemical groups, it becomes **toxic** or interferes with biological assays (PAINS alerts).

**The Task:** We want an AI that takes this existing starting molecule and makes small, chemically valid edits to improve its oral drug-likeness (QED), while strictly preserving its core structure (Tanimoto similarity $\ge 0.40$), keeping it easy to synthesize (SAS $\le 4.5$), and avoiding toxic chemical groups.

**How We Solved It:**
1. We formulated molecular editing as a **Reinforcement Learning Markov Decision Process (MDP)** with 1,901 discrete chemical actions and valency masking (guaranteeing 100% chemical validity).
2. We designed an **Edge-Aware Graph Neural Network (GNN)** encoder that explicitly distinguishes bond types (single, double, triple, aromatic).
3. We invented **Calibrated Adaptive Reward Shaping** (dynamic headroom scaling, quadratic similarity barrier, slack penalties, and a STOP bonus).
4. We conducted a massive benchmark on **500 unseen molecules (2,000 trajectories)** from ZINC-250K.
5. We discovered that pure reactive PPO suffered from **horizon exhaustion** and negative drift because it took too many unguided edits ($9.58$ steps vs. $1.42$ for greedy).
6. To solve this, we invented the **PPO-Guided Local Search Hybrid (Top-5 Lookahead)**: the trained PPO network acts as an intelligent candidate filter, ranking valid edits down to only the Top-5 candidates, which are evaluated with 1-step lookahead.
7. **The Result:** The Hybrid reversed negative drift to **$+0.0183$ $\Delta\text{QED}$**, achieved **$41.0\%$ high-QED success**, attained industry-leading scaffold retention (**$0.8790$ similarity**), and cut computational oracle evaluations by **$47.2\times$** ($5.68$ vs. $268.22$ per molecule), enabling practical use with real 3D physical docking!

---

## 1. Problem Formulation: The MDP

We formulate lead optimization as a **Finite-Horizon Markov Decision Process (MDP)**:
$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, T)$$

### 1.1 State Space ($\mathcal{S}$)
At each step $t$, the state is a tuple $s_t = (G_t, \mathbf{u}_t)$:
1. **Molecular Graph $G_t = (V_t, E_t, \mathbf{X}_t, \mathbf{E}_{\text{attr}, t})$**:
   - Vertices $V_t$: Heavy atoms, bounded by $|V_t| \le 38$.
   - Node Feature Matrix $\mathbf{X}_t \in \mathbb{R}^{|V_t| \times 21}$: One-hot encoding of element (C, N, O, F, S, Cl), atomic degree (0–5), formal charge ($-1, 0, +1$), hybridization ($sp, sp^2, sp^3$), aromaticity (boolean), and implicit hydrogens.
   - Bond Attribute Matrix $\mathbf{E}_{\text{attr}, t} \in \mathbb{R}^{|E_t| \times 4}$: One-hot encoding of bond order:
     $$\mathbf{e}_{ij} \in \{\texttt{Single}, \texttt{Double}, \texttt{Triple}, \texttt{Aromatic}\}$$
2. **Scalar Trajectory Context Vector $\mathbf{u}_t \in \mathbb{R}^6$**:
   $$\mathbf{u}_t = \left[ \frac{t}{T}, \, \operatorname{QED}(M_t), \, \operatorname{SAS}(M_t), \, \operatorname{Sim}(M_0, M_t), \, \operatorname{QED}(M_0), \, \operatorname{SAS}(M_0) \right]^\top$$
   This tells the agent how far along it is in the trajectory ($t/10$), where it started, and how close it is to the parent molecule.

### 1.2 Action Space ($\mathcal{A}$)
Total discrete actions: **$|\mathcal{A}| = 1,901$**, divided into 5 chemical categories:
- **Action 0: `STOP`** (1 action, $0.05\%$): Voluntarily stops editing, locking in the current molecule as the final candidate.
- **Actions 1–228: `ADD_ATOM_BOND`** (228 actions, $12.00\%$): Adds an atom of element $e \in \{\text{C}, \text{N}, \text{O}, \text{F}, \text{S}, \text{Cl}\}$ to existing atom $i \le 37$ via a single bond ($38 \times 6 = 228$).
- **Actions 229–266: `REMOVE_ATOM`** (38 actions, $2.00\%$): Deletes atom $i$ and incident bonds (only if graph remains connected).
- **Actions 267–494: `SUBSTITUTE_ATOM`** (228 actions, $12.00\%$): Mutates the elemental species of atom $i$ ($38 \times 6 = 228$).
- **Actions 495–1,900: `CHANGE_BOND`** (1,406 actions, $73.96\%$): Modifies the bond order across atom pairs $(i, j)$ with $i < j \le 37$. There are $\binom{38}{2} = 703$ pairs $\times 2$ directions (increment/decrement) $= 1,406$.

### 1.3 Dynamic Action Masking (100% Chemical Validity)
Before the policy chooses an action, an action mask $\mathbf{m}(s_t) \in \{0, 1\}^{1901}$ is evaluated:
- An action is valid ($m_a = 1$) only if:
  1. No atom exceeds maximum valence ($\text{C}\le4, \text{N}\le3, \text{O}\le2, \text{F}\le1, \text{S}\le6, \text{Cl}\le1$).
  2. Heavy atom count does not exceed 38.
  3. Graph remains a single connected piece.
  4. Passes RDKit's chemical sanitization (`Chem.SanitizeMol`).
- Action 0 (`STOP`) is always valid ($m_0 = 1$).
- Invalid actions get set to $-\infty$ logits:
  $$\tilde{z}_a(s_t) = \begin{cases} z_a(s_t), & \text{if } m_a(s_t) = 1 \\ -\infty, & \text{if } m_a(s_t) = 0 \end{cases}$$
  This mathematically guarantees that invalid actions have $0.0\%$ probability, ensuring **100% chemically valid molecules at all times**!

### 1.4 Horizon & Transition
- $T = 10$ editing steps maximum.
- State transitions are deterministic: $s_{t+1} = \mathcal{P}(s_t, a_t)$.

---

## 2. Important Chemistry Descriptors & Equations

### 2.1 Quantitative Estimate of Drug-likeness (QED)
Bickerton et al. (2012) modeled oral drug-likeness by fitting desirability functions $d_i(x_i)$ to 8 properties (MW, $\log P$, HBD, HBA, PSA, ROTB, AROM, ALERTS):
$$\operatorname{QED}(M) = \exp\left( \frac{\sum_{i=1}^8 w_i \ln d_i(x_i)}{\sum_{i=1}^8 w_i} \right) \in [0, 1]$$
- $\text{QED} < 0.50$: Low / poor drug-likeness.
- $\text{QED} \in [0.50, 0.70]$: Moderate drug-likeness.
- $\text{QED} \in [0.70, 0.80]$: Strong drug-likeness profile.
- $\text{QED} \ge 0.80$: High chemical elegance (commercial oral pharmaceutical standard).

### 2.2 Morgan Circular Fingerprints & Tanimoto Similarity
Measures structural retention between starting lead $M_0$ and edited lead $M_t$:
$$\operatorname{Sim}(M_0, M_t) = \frac{|\mathbf{f}_0 \cap \mathbf{f}_t|}{|\mathbf{f}_0 \cup \mathbf{f}_t|} = \frac{\sum_k f_{0,k} f_{t,k}}{\sum_k f_{0,k} + \sum_k f_{t,k} - \sum_k f_{0,k} f_{t,k}}$$
- Standard medicinal chemistry constraint: $\operatorname{Sim}(M_0, M_t) \ge 0.40$ (to preserve the target-binding pharmacophore).

### 2.3 Synthetic Accessibility Score (SAS)
Ertl & Schuffenhauer (2009):
$$\operatorname{SAS}(M) = \operatorname{FragmentScore}(M) - \operatorname{ComplexityPenalty}(M) \in [1.0, 10.0]$$
- $1.0$: Extremely easy to synthesize.
- $\le 4.0$: Feasible via standard organic synthesis.
- $> 5.0$: Highly complex / total synthesis required.

### 2.4 Formal Lead Optimization Success Criteria
A trajectory is formally successful if and only if:
$$\Delta\operatorname{QED} \ge +0.10 \quad \text{AND} \quad \operatorname{Sim}(M_0, M_T) \ge 0.40 \quad \text{AND} \quad \operatorname{SAS}(M_T) \le \max(\operatorname{SAS}(M_0), 4.5)$$

---

## 3. Reward Function: Unshaped vs. Calibrated Adaptive

### 3.1 The Unshaped Baseline (Why it failed)
$$R_{\text{unshaped}} = \Delta\operatorname{QED} - 0.5 \times \Delta\operatorname{SAS}$$
**Why it failed:**
- It had **no similarity penalty**. The agent discovered a shortcut ("reward hacking"): if it chopped off rings and unraveled the molecule into flexible aliphatic chains, QED went up nominally, but Tanimoto similarity crashed to $0.27$, and $91.4\%$ of trajectories violated the $\ge 0.40$ constraint!
- It had **no headroom awareness**. Editing a molecule that already had $\text{QED} = 0.90$ was treated the same as one with $0.30$.
- It had **no STOP bonus**, so the agent never learned to voluntarily terminate.

### 3.2 Our Calibrated Adaptive Reward Shaping Function
$$\mathcal{R}(s_t, a_t, s_{t+1}) = \gamma_{\text{headroom}} \cdot \Delta\operatorname{QED} - \rho_{\text{sim}} - \rho_{\text{slack}} + r_{\text{stop}}$$

#### Term 1: Dynamic Initial-Headroom Scaling ($\gamma_{\text{headroom}}$)
$$\gamma_{\text{headroom}}(q_0) = 1.0 + \min\left(1.5, \, \max\left(0.0, \, \frac{0.90 - q_0}{0.90}\right)\right)$$
- If starting molecule has **low QED** ($q_0 \approx 0.30$): Headroom is high, so $\gamma \approx 1.67$. The agent gets a $67\%$ reward bonus to explore aggressive improvements.
- If starting molecule has **high QED** ($q_0 \ge 0.85$): Headroom is tiny ($0.90 - 0.85 = 0.05$), so $\gamma \approx 1.05$. The agent is discouraged from risking destruction of an already good drug.

#### Term 2: Smooth Quadratic Similarity Barrier ($\rho_{\text{sim}}$)
$$\rho_{\text{sim}} = \begin{cases}
0, & \text{if } \operatorname{Sim} \ge 0.55 \quad (\text{Safe Zone}) \\
1.0 \left( \frac{0.55 - \operatorname{Sim}}{0.15} \right)^2, & \text{if } 0.40 \le \operatorname{Sim} < 0.55 \quad (\text{Warning Zone}) \\
2.0 + 5.0(0.40 - \operatorname{Sim}), & \text{if } \operatorname{Sim} < 0.40 \quad (\text{Violation Zone})
\end{cases}$$
- Instead of a harsh cliff (which destroys neural network gradients), it uses a smooth quadratic curve that gently pushes the agent back as it approaches $0.40$, followed by a steep penalty if it breaches $0.40$.

#### Term 3: Constraint Slack Penalties ($\rho_{\text{slack}}$)
$$\rho_{\text{slack}} = 0.5 \times \max(0, \operatorname{SAS}_{t+1} - \max(\operatorname{SAS}_0, 4.5)) + 0.25 \times \text{Violations}_{\text{Lipinski}}$$
- Penalizes synthetically difficult molecules and Lipinski Rule of Five violations.

#### Term 4: Voluntary STOP-Action Terminal Bonus ($r_{\text{stop}}$)
When the agent executes Action 0 (`STOP`):
$$r_{\text{stop}} = \begin{cases}
+5.0, & \text{if Formal Success Criteria are met} \\
+2.0 \times \Delta\operatorname{QED}, & \text{if } \Delta\operatorname{QED} > 0 \text{ and } \operatorname{Sim} \ge 0.40 \\
0.0, & \text{otherwise}
\end{cases}$$
- This gives the agent an explicit reward for stopping early when it has reached a local peak, instead of mindlessly editing until step 10!

---

## 4. Model Architecture & PPO Training

### 4.1 Edge-Aware Graph Neural Network (GNN) Encoder
Standard GCNs treat all bonds the same. But in chemistry, an **aromatic bond** in a benzene ring is completely different from a single or double bond.
Our `EdgeAwareMessagePassing` layer explicitly incorporates the 4D bond vector $\mathbf{e}_{ij}$:
$$\mathbf{m}_{ij}^{(l)} = \operatorname{ReLU}\left( \mathbf{W}_{\text{source}}^{(l)} \mathbf{h}_j^{(l)} + \mathbf{W}_{\text{edge}}^{(l)} \mathbf{e}_{ij} \right)$$
$$\mathbf{h}_i^{(l+1)} = \operatorname{ReLU}\left( \operatorname{BatchNorm}\left( \mathbf{W}_{\text{root}}^{(l)} \mathbf{h}_i^{(l)} + \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij}^{(l)} \right) \right)$$
- We stack **3 layers**: $21 \to 64 \to 64 \to 64$.
- **Dual Global Pooling:** Combines mean pooling (global molecular properties) and max pooling (prominent functional groups):
  $$\mathbf{h}_{\text{graph}} = \left[ \operatorname{mean}(\mathbf{h}_i^{(3)}), \, \max(\mathbf{h}_i^{(3)}) \right] \in \mathbb{R}^{128}$$
- **State Fusion:** Concatenates $\mathbf{h}_{\text{graph}}$ with the 6-dim context vector $\mathbf{u}_t$ and passes through a fusion layer $\to \mathbf{z}_t \in \mathbb{R}^{128}$.

### 4.2 Actor--Critic Heads
1. **Actor ($\pi_\theta$):** MLP ($128 \to 128 \to 1,901$) with masked softmax $\to$ outputs probability distribution over valid actions.
2. **Critic ($V_\phi$):** MLP ($128 \to 64 \to 1$) $\to$ predicts expected cumulative return.

### 4.3 PPO Optimization
- **Generalized Advantage Estimation (GAE-$\lambda$):** $\gamma = 0.99, \lambda = 0.95$.
- **Clipped Surrogate Objective:**
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta) \hat{A}_t, \, \operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
  with clipping parameter $\epsilon = 0.20$.
- **Total Loss:** $L^{\text{PPO}} = -L^{\text{CLIP}} + 0.5 L^{\text{VF}} - 0.01 \mathcal{S}[\pi_\theta]$.

---

## 5. Phase 1: Benchmark 500 Results & The Problem We Discovered

We evaluated 4 methods across **500 unseen molecules** (2,000 trajectories) from ZINC-250K:

| Metric | Random Valid Edits | PPO Baseline (Unshaped) | PPO Proposed (Adaptive V1) | Greedy Hill Climbing |
| :--- | :---: | :---: | :---: | :---: |
| **Mean $\Delta$QED** | $-0.4243$ | $-0.3782$ | **$-0.1730$** | $\mathbf{+0.1229}$ |
| **Mean Similarity** | $0.2487$ | $0.2706$ | **$0.4731$** | $0.6188$ |
| **Constraint Violations ($\text{Sim} < 0.40$)** | $93.6\%$ | $91.4\%$ | **$40.6\%$** | $\mathbf{0.0\%}$ |
| **PAINS Toxicophore Alert Rate** | $17.0\%$ | $14.8\%$ | $\mathbf{5.0\%}$ | $9.4\%$ |
| **Mean Lipinski Violations** | $0.704$ | $0.704$ | **$0.288$** | $\mathbf{0.016}$ |
| **Formal Success Rate** | $2.2\%$ | $5.4\%$ | $5.0\%$ | $\mathbf{65.4\%}$ |
| **Mean Steps per Trajectory** | $9.39$ | $9.53$ | $9.58$ | $\mathbf{1.42}$ |
| **Voluntary STOP Rate** | $0.0\%$ | $0.0\%$ | $0.0\%$ | $\mathbf{100.0\%}$ |

### What Did We Conclude from Phase 1?
1. **Adaptive Reward Worked for Safety & Constraints:** It improved $\Delta\text{QED}$ by $+0.205$ over unshaped, cut similarity violations from $91.4\%$ to $40.6\%$, halved Lipinski violations ($0.288$ vs. $0.704$), and achieved the lowest PAINS alert rate ($5.0\%$, even better than Greedy's $9.4\%$).
2. **THE BIG PROBLEM:** Both PPO models still had **negative mean $\Delta\text{QED}$** ($-0.1730$) and low formal success ($5.0\%$), whereas Greedy achieved $+0.1229$ and $65.4\%$ success!

---

## 6. Failure Diagnostics: Why Did Pure Reactive PPO Drift?

We ran an in-depth empirical diagnostic on 100 validation molecules:
1. **Action Space Imbalance (73.96% Bond Edits):** Out of 1,901 actions, 1,406 are bond edits. When PPO explores with entropy regularization, it samples bond changes. But breaking bonds in aromatic rings or forming strained rings almost always destroys drug-likeness.
2. **Horizon Exhaustion (95% $T=10$, 0% STOP):** Pure PPO took $9.58$ edits per molecule. It never chose STOP!
3. **Step-Wise Quality Decay:** When we plotted $\Delta\text{QED}$ step-by-step, molecules reached their **peak quality at step 1 or 2** ($\Delta\text{QED} \approx +0.03$). Because reactive PPO had no lookahead, it blindly kept editing for 8 more steps, decaying all the way down to $-0.1730$.
4. **Huge Branching Factor:** Even after valency masking, there are $\sim88$ valid actions per step. In an 88-way tree, purely sampling actions without lookahead has a tiny probability of making 10 perfect consecutive edits.

---

## 7. Phase 2: What Did We Do Next? The Hybrid Breakthrough!

Greedy Hill Climbing works because it checks actions before executing them. But Greedy has a fatal flaw: **it evaluates every single valid candidate at every step** ($268.22$ candidate evaluations per molecule). If each evaluation takes 1 minute in a 3D physical docking simulation, Greedy takes **4.5 hours per molecule**!

### The Solution: PPO-Guided Local Search Hybrid (Top-5 Lookahead)
Instead of checking all 88 candidates, we use the trained PPO Actor as an **intelligent candidate filter**:
1. Run the molecular graph through the Edge-Aware GNN $\to$ get policy probabilities $\pi_\theta(a|s_t)$.
2. Sort valid actions by probability and select only the **Top-$K$ candidates ($K=5$)** plus the `STOP` action.
3. Evaluate the shaped reward for only those 5 candidate molecules using 1-step lookahead.
4. Pick the candidate with the highest reward: $a^* = \arg\max_{a \in \text{Top-5}} r_a$.
5. **Early Stopping Rule:** If $a^* = 0$ (`STOP`) or if the best reward is non-positive ($r_{a^*} \le 0$), **terminate immediately**!

---

## 8. Final Results: The Hybrid vs. Greedy Comparison

Evaluated on the 100-molecule scaffold-separated diagnostic benchmark:

| Method | Mean $\Delta$QED | High-QED Success ($\text{QED}\ge0.80, \text{Sim}\ge0.40$) | Mean Similarity | Constraint Violations | Mean $\Delta$SAS | Oracle Calls / Mol | Wall-Clock Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Valid Edits** | $-0.4654$ | $0.0\%$ | $0.2220$ | $95.0\%$ | $+2.006$ | $9.51$ | $0.188\text{ s}$ |
| **PPO Baseline (Unshaped)** | $-0.3866$ | $2.0\%$ | $0.2635$ | $87.0\%$ | $+1.255$ | $9.53$ | $0.209\text{ s}$ |
| **PPO Proposed (Adaptive V1)**| $-0.2054$ | $10.0\%$ | $0.4608$ | $46.0\%$ | $+0.836$ | $9.56$ | $0.228\text{ s}$ |
| **PPO Adaptive V2 (Edge-Aware)**| $-0.1771$ | $9.0\%$ | $0.7152$ | $\mathbf{0.0\%}$ | $+0.427$ | $2.92$ | $\mathbf{0.098\text{ s}}$ |
| **PPO-Guided Hybrid (Top-5)** | $\mathbf{+0.0183}$ | $\mathbf{41.0\%}$ | $\mathbf{0.8790}$ | $\mathbf{0.0\%}$ | $\mathbf{+0.105}$ | $\mathbf{5.68}$ | $0.153\text{ s}$ |
| **Greedy Hill Climbing** | $+0.1076$ | $67.0\%$ | $0.6255$ | $\mathbf{0.0\%}$ | $+0.383$ | $268.22$ | $0.681\text{ s}$ |

### Key Conclusions:
1. **Reversed Negative Drift:** The PPO Hybrid turned a negative drift ($-0.205$) into a **positive mean $\Delta\text{QED} = +0.0183$** and achieved a **$41.0\%$ High-QED Success Rate**.
2. **Industry-Leading Scaffold Preservation:** The Hybrid achieved **$0.8790$ Tanimoto similarity**, vastly outperforming Greedy ($0.6255$). Greedy aggressively alters core scaffolds to chase immediate QED, whereas our Hybrid preserves the drug's core pharmacophore.
3. **Minimal Synthesis Cost:** SAS increased by only $+0.105$ (nearly identical to the starting molecule).
4. **The $47.2\times$ Computational Speedup:**
   $$\text{Speedup Factor} = \frac{268.22}{5.68} \approx \mathbf{47.2\times}$$
   - The Hybrid requires only **$5.68$ oracle evaluations per molecule** compared to **$268.22$ for Greedy**.
   - With real 3D docking ($60\text{ s}$ per evaluation):
     - Greedy: $268.22 \times 60\text{ s} \approx \mathbf{4.47\text{ hours}}$ per molecule.
     - PPO Hybrid: $5.68 \times 60\text{ s} \approx \mathbf{5.68\text{ minutes}}$ per molecule!
   - This makes our hybrid practically deployable in actual pharmaceutical drug discovery pipelines.

---

## 9. Quick Q&A for Presentation and Viva

**Q1: Why didn't you use SMILES strings with an RNN or VAE?**  
*Answer:* String-based models frequently produce chemically invalid strings (broken parentheses, invalid valencies) and disconnected fragments. By using molecular graphs with dynamic action masking, our agent is mathematically guaranteed to generate **100% chemically valid molecules** at every step.

**Q2: What is the difference between your GNN and a standard GCN?**  
*Answer:* Standard GCNs only look at neighboring atoms and treat all bonds identically. Our Edge-Aware GNN explicitly encodes the 4D bond attribute vector ($\mathbf{e}_{ij} \in \{\text{Single}, \text{Double}, \text{Triple}, \text{Aromatic}\}$) into each message-passing step, allowing the network to distinguish conjugated aromatic rings from aliphatic chains.

**Q3: Why did pure PPO show negative $\Delta\text{QED}$ during Phase 1?**  
*Answer:* Our failure diagnostics proved that pure PPO suffered from horizon exhaustion ($9.58$ steps on average with 0% voluntary STOP). The agent reached peak quality at step 1 or 2, but without lookahead or an explicit stop bonus, it continued taking random-like edits that degraded the molecule. Furthermore, 74% of the action space consists of bond modifications, which act as an exploratory entropy sink.

**Q4: How does your PPO-Guided Hybrid work?**  
*Answer:* It bridges policy learning with constrained local search. The trained PPO actor ranks all valid actions and filters them down to the Top-5 candidates plus STOP. We evaluate only those 5 candidate edits using 1-step lookahead and pick the best one. If no candidate improves the reward, the agent terminates immediately.

**Q5: If Greedy achieved $+0.12$ $\Delta\text{QED}$ and the Hybrid achieved $+0.02$, why is the Hybrid better?**  
*Answer:* (1) **Scaffold Retention:** The Hybrid preserves the starting molecule far better ($0.8790$ similarity vs. $0.6255$ for Greedy). Greedy mutates the core structure to chase QED. (2) **Computational Feasibility:** Greedy requires $268.22$ property evaluations per molecule, while the Hybrid requires only $5.68$---a **$47.2\times$ reduction**. In real drug discovery where property evaluation involves 3D physical docking (taking 1 minute per molecule), Greedy takes 4.5 hours per molecule, which is impossible for screening libraries, while our Hybrid takes under 6 minutes.
