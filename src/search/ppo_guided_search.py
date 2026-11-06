import time
import numpy as np
from rdkit import Chem
from src.environment.action_masking import apply_action
from src.chemistry.properties import calculate_properties


class PPOGuidedSearchOptimizer:
    """
    PPO-Guided Local Search Optimizer (Hybrid Extension).
    
    Bridging Policy Learning and Local Lookahead:
    1. Rather than blindly executing a single action or exhaustively simulating all ~150 candidates,
       the trained Edge-Aware PPO Actor ranks all valid candidate edits.
    2. Only the Top-K most promising candidate actions (e.g. K=5) are evaluated via 1-step lookahead.
    3. If the best Top-K edit improves the shaped reward (reward > 0), it is executed.
    4. If no Top-K edit improves the score or if STOP is selected, the trajectory IMMEDIATELY TERMINATES.
    
    Advantages:
    - Eliminates runaway multi-step degradation (stops at local peak).
    - Achieves >10x property-evaluation efficiency compared to exhaustive Greedy Hill-Climbing.
    """
    def __init__(self, ppo_agent, top_k=5, max_steps=10):
        self.ppo_agent = ppo_agent
        self.top_k = top_k
        self.max_steps = max_steps
        self.name = f"PPO-Guided Search (Top-{top_k})"

    def optimize(self, starting_smiles, env):
        t0 = time.time()
        obs, info = env.reset(options={"starting_smiles": starting_smiles})
        trajectory = [info["smiles"]]
        total_reward = 0.0
        prop_eval_count = 0

        for step in range(self.max_steps):
            # 1. Query PPO Actor for top-k candidate actions
            top_k_indices = self.ppo_agent.get_top_k_actions(obs, k=self.top_k)
            
            # Ensure STOP (0) is always considered
            candidate_pool = list(set([0] + top_k_indices))
            
            best_action = 0
            best_reward = 0.0
            best_cand_mol = None
            best_cand_props = None
            current_mol = env.current_mol

            # 2. Evaluate only the filtered Top-K candidates
            for act in candidate_pool:
                if act == 0:
                    continue
                cand_mol = apply_action(current_mol, act, env.editor)
                if cand_mol is None:
                    continue
                cand_props = calculate_properties(cand_mol, env.original_mol)
                prop_eval_count += 1
                
                rew, _ = env.reward_calc.calculate_reward(
                    env.current_props, cand_props, env.original_props
                )
                if rew > best_reward:
                    best_reward = rew
                    best_action = act
                    best_cand_mol = cand_mol
                    best_cand_props = cand_props

            # 3. Stopping Decision: If no candidate improves shaped reward, STOP
            if best_action == 0 or best_reward <= 0:
                env.step(0) # Voluntary STOP
                break

            # 4. Execute the best candidate action
            obs, reward, terminated, truncated, step_info = env.step(best_action)
            total_reward += reward

            if step_info.get("valid", False) and "smiles" in step_info:
                trajectory.append(step_info["smiles"])

            if terminated or truncated:
                break

        elapsed = time.time() - t0
        final_props = env.current_props
        return {
            "initial_smiles": starting_smiles,
            "final_smiles": Chem.MolToSmiles(env.current_mol),
            "initial_qed": env.original_props["qed"],
            "final_qed": final_props["qed"],
            "delta_qed": final_props["qed"] - env.original_props["qed"],
            "initial_sas": env.original_props["sas"],
            "final_sas": final_props["sas"],
            "delta_sas": final_props["sas"] - env.original_props["sas"],
            "similarity": final_props["similarity"],
            "valid": final_props["valid"],
            "success": env.reward_calc.check_success(final_props, env.original_props),
            "num_edits": env.current_step,
            "total_reward": total_reward,
            "trajectory": trajectory,
            "prop_eval_count": prop_eval_count,
            "latency_sec": elapsed
        }

