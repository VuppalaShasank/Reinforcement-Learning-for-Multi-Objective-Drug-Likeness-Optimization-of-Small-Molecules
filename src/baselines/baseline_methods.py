import os
import sys
import numpy as np
from rdkit import Chem

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.environment.action_masking import apply_action
from src.chemistry.properties import calculate_properties


class RandomValidEditsBaseline:
    """
    Baseline 1: Random Valid Edits (Lower-Bound Benchmark).
    At each step, samples uniformly at random from chemically valid candidate actions
    identified by the action mask. Terminates if STOP is selected or step limit is reached.
    """
    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.name = "Random Valid Edits"

    def optimize(self, starting_smiles, env):
        obs, info = env.reset(options={"starting_smiles": starting_smiles})
        trajectory = [info["smiles"]]
        total_reward = 0.0

        for step in range(self.max_steps):
            mask = obs["action_mask"]
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                break

            action = int(np.random.choice(valid_indices))
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward

            if step_info.get("valid", False) and "smiles" in step_info:
                trajectory.append(step_info["smiles"])

            if terminated or truncated:
                break

        final_mol = env.current_mol
        final_props = env.current_props
        return {
            "initial_smiles": starting_smiles,
            "final_smiles": Chem.MolToSmiles(final_mol),
            "initial_qed": env.original_props["qed"],
            "final_qed": final_props["qed"],
            "delta_qed": final_props["qed"] - env.original_props["qed"],
            "initial_sas": env.original_props["sas"],
            "final_sas": final_props["sas"],
            "similarity": final_props["similarity"],
            "valid": final_props["valid"],
            "success": env.reward_calc.check_success(final_props, env.original_props),
            "num_edits": env.current_step,
            "total_reward": total_reward,
            "trajectory": trajectory
        }


class GreedyHillClimbingBaseline:
    """
    Baseline 2: Greedy Hill Climbing (Immediate-Step Oracle).
    At each step, exhaustively simulates all valid candidate actions and greedily executes
    the edit yielding the highest immediate reward. Stops if no edit improves the score.
    """
    def __init__(self, max_steps=10):
        self.max_steps = max_steps
        self.name = "Greedy Hill Climbing"

    def optimize(self, starting_smiles, env):
        obs, info = env.reset(options={"starting_smiles": starting_smiles})
        trajectory = [info["smiles"]]
        total_reward = 0.0

        for step in range(self.max_steps):
            mask = obs["action_mask"]
            valid_indices = np.where(mask)[0]

            if len(valid_indices) <= 1:
                env.step(0)  # STOP
                break

            best_action = 0  # Default STOP
            best_reward = -float("inf")
            current_mol = env.current_mol

            for act in valid_indices:
                if act == 0:
                    continue
                cand_mol = apply_action(current_mol, act, env.editor)
                if cand_mol is None:
                    continue
                cand_props = calculate_properties(cand_mol, env.original_mol)
                rew, _ = env.reward_calc.calculate_reward(env.current_props, cand_props, env.original_props)
                if rew > best_reward:
                    best_reward = rew
                    best_action = act

            if best_reward <= 0 or best_action == 0:
                obs, reward, _, _, _ = env.step(0)
                total_reward += reward
                break

            obs, reward, terminated, truncated, step_info = env.step(best_action)
            total_reward += reward

            if step_info.get("valid", False) and "smiles" in step_info:
                trajectory.append(step_info["smiles"])

            if terminated or truncated:
                break

        final_mol = env.current_mol
        final_props = env.current_props
        return {
            "initial_smiles": starting_smiles,
            "final_smiles": Chem.MolToSmiles(final_mol),
            "initial_qed": env.original_props["qed"],
            "final_qed": final_props["qed"],
            "delta_qed": final_props["qed"] - env.original_props["qed"],
            "initial_sas": env.original_props["sas"],
            "final_sas": final_props["sas"],
            "similarity": final_props["similarity"],
            "valid": final_props["valid"],
            "success": env.reward_calc.check_success(final_props, env.original_props),
            "num_edits": env.current_step,
            "total_reward": total_reward,
            "trajectory": trajectory
        }

