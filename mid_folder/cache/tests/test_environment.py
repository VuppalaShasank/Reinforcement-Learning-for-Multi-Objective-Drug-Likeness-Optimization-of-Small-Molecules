import os
import sys
import torch
import numpy as np
from rdkit import Chem

# Insert project root into sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from implementation.environment import MolecularGraphEnv
from implementation.action_handler import action_to_details, get_action_mask

def test_molecular_graph_env():
    print("=== Testing MolecularGraphEnv API & Transitions ===")
    
    # 1. Test Baseline Mode
    env_base = MolecularGraphEnv(reward_mode="baseline", max_steps=10)
    obs, info = env_base.reset(options={"starting_smiles": "c1ccccc1"}) # Benzene
    assert "pyg_data" in obs and "scalars" in obs and "action_mask" in obs
    assert obs["action_mask"][0] == True, "STOP action must always be valid"
    assert obs["pyg_data"].x.shape[1] == 21, f"Expected 21 node features, got {obs['pyg_data'].x.shape[1]}"
    print(f"Reset Benzene: {obs['pyg_data'].x.shape[0]} atoms, {obs['action_mask'].sum()} valid candidate actions")

    # Pick a valid non-stop action from mask
    valid_indices = np.where(obs["action_mask"])[0]
    non_stop = [idx for idx in valid_indices if idx != 0]
    assert len(non_stop) > 0, "Benzene must have valid non-stop actions"
    chosen_act = non_stop[0]
    act_desc = action_to_details(chosen_act)
    print(f"Stepping with valid action: {act_desc}")

    next_obs, rew, term, trunc, step_info = env_base.step(chosen_act)
    assert step_info["valid"] == True
    print(f"Step Result: reward={rew:.4f}, new smiles={step_info.get('smiles')}, term={term}, trunc={trunc}")

    # Step with STOP
    stop_obs, stop_rew, stop_term, stop_trunc, stop_info = env_base.step(0)
    assert stop_term == True, "STOP action must terminate"
    print(f"STOP action terminated episode: PASSED (reward={stop_rew:.4f})")

    # 2. Test Adaptive Mode
    print("\n=== Testing Adaptive Reward Mode in Env ===")
    env_adapt = MolecularGraphEnv(reward_mode="adaptive", max_steps=5)
    obs_a, info_a = env_adapt.reset(options={"starting_smiles": "CC(=O)Oc1ccccc1C(=O)O"}) # Aspirin
    print(f"Reset Aspirin (Adaptive): initial QED={info_a['qed']:.4f}, SAS={info_a['sas']:.4f}")

    # Run 5 steps or until terminated
    for s in range(5):
        mask = obs_a["action_mask"]
        valid_acts = np.where(mask)[0]
        # Pick valid non-stop action if available, else STOP
        act = valid_acts[1] if len(valid_acts) > 1 else 0
        obs_a, rew, term, trunc, step_info = env_adapt.step(act)
        print(f"  Step {s+1}: act={step_info['action']['type']}, rew={rew:.4f}, QED={step_info.get('qed'):.4f}, term={term}, trunc={trunc}")
        if term or trunc:
            break

    print("\nALL ENVIRONMENT API & TRANSITION TESTS PASSED!")

if __name__ == "__main__":
    test_molecular_graph_env()
