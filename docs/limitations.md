# Experimental Limitations and Future Directions

## 1. Limitations of Pure Single-Path RL Policy
In a discrete graph action space of 1,901 actions, pure PPO rollouts (without local search) suffer from non-monotonic reward landscapes. Although Edge-Aware GNN and calibrated reward shaping reduced edits to $1.96$ and eliminated constraint violations ($0.0\%$), single-path policy sampling without lookahead still yielded negative mean $\Delta\text{QED}$ ($-0.1771$). Combining the policy with Top-$K$ lookahead is necessary to guarantee monotonic improvement.

## 2. Fixed Top-K Lookahead Horizon
In the current implementation of `PPOGuidedSearchOptimizer`, $K=5$ is fixed across all steps and molecules. When policy entropy is very low (high certainty), evaluating fewer candidates ($K=2$) might suffice; when entropy is high, expanding $K$ dynamically could discover superior edits.

## 3. Proxy Scoring vs. High-Fidelity Physics Oracles
QED and SAS are fast proxy heuristics computed via RDKit. In drug discovery campaigns, optimization is directed towards 3D binding affinity (docking scores or free energy perturbation). While Greedy Hill Climbing would be computationally prohibitive with docking oracles (~268 docking runs per molecule), PPO-Guided Local Search (~5 runs per molecule) offers a scalable framework for future physical docking integration.

## 4. Local Structural Modifications
The action space operates on single-atom and single-bond edits (maximum atom limit 38). Fragment-based bioisosteric replacements (e.g., swapping entire phenyl rings for pyridine or pyrazole rings in one step) are not directly represented in the current action space and must be assembled through sequential edits.

