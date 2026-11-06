import os
import sys
import time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, DataStructs, AllChem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.chemistry.properties import calculate_properties

def get_morgan_fingerprint(mol, radius=2, nBits=2048):
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    except Exception:
        return None
from src.environment.action_masking import apply_action, action_to_details

# PAINS & Brenk Filter Catalogs
pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog(pains_params)

brenk_params = FilterCatalogParams()
brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
BRENK_CATALOG = FilterCatalog(brenk_params)


def check_lipinski_violations(mol):
    """
    Computes Lipinski Rule of 5 violations:
    1. MW <= 500
    2. LogP <= 5.0
    3. H-bond Donors <= 5
    4. H-bond Acceptors <= 10
    """
    if mol is None:
        return 4
    violations = 0
    try:
        if Descriptors.MolWt(mol) > 500: violations += 1
        if Descriptors.MolLogP(mol) > 5.0: violations += 1
        if Lipinski.NumHDonors(mol) > 5: violations += 1
        if Lipinski.NumHAcceptors(mol) > 10: violations += 1
    except Exception:
        violations = 4
    return violations


def check_pains_alert(mol):
    if mol is None:
        return False
    try:
        return PAINS_CATALOG.HasMatch(mol)
    except Exception:
        return False


def check_brenk_alert(mol):
    if mol is None:
        return False
    try:
        return BRENK_CATALOG.HasMatch(mol)
    except Exception:
        return False


def check_structural_sanity(mol):
    """
    Sanity checks for unreasonable chemical structures:
    - Ring size <= 8
    - Contains at least one carbon atom for organic molecules
    - Exactly 1 connected molecular component (no fragmentation)
    """
    if mol is None:
        return False, "Null molecule"
    try:
        ssr = Chem.GetSymmSSSR(mol)
        for ring in ssr:
            if len(ring) > 8:
                return False, f"Large ring of size {len(ring)}"
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        if 'C' not in atoms and len(atoms) > 3:
            return False, "No carbon atoms"
        if len(Chem.GetMolFrags(mol)) > 1:
            return False, "Disconnected graph fragments"
        return True, "Passed"
    except Exception as e:
        return False, str(e)


def compute_normalized_headroom_gain(q0, q_final, eps=0.05):
    """
    Normalized Headroom Gain (NHG):
    NHG = (QED_final - q0) / max(1.0 - q0, eps)
    """
    headroom = max(1.0 - q0, eps)
    return (q_final - q0) / headroom


def compute_diversity(smiles_list):
    """
    Computes average pairwise Tanimoto distance across generated molecules.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    if len(mols) < 2:
        return 0.0
    fps = [get_morgan_fingerprint(m) for m in mols]
    n = len(fps)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return float(1.0 - np.mean(sims)) if len(sims) > 0 else 0.0


def audit_molecule(mol):
    """
    Performs full reward-hacking audit on a molecule.
    """
    if mol is None:
        return {
            "valid": False,
            "lipinski_violations": 4,
            "pains_alert": False,
            "brenk_alert": False,
            "sanity_passed": False
        }
    sanity_pass, _ = check_structural_sanity(mol)
    return {
        "valid": True,
        "lipinski_violations": check_lipinski_violations(mol),
        "pains_alert": check_pains_alert(mol),
        "brenk_alert": check_brenk_alert(mol),
        "sanity_passed": sanity_pass
    }


def evaluate_policy_on_molecules(agent, molecules_smiles, env, method_name="PPO"):
    """
    Evaluates an agent (PPO or baseline) across a list of test molecules.
    Returns:
    - Detailed DataFrame of results
    - Summary dictionary of aggregate performance metrics
    """
    records = []
    latencies = []

    for idx, smiles in enumerate(molecules_smiles):
        t0 = time.time()
        if hasattr(agent, "optimize"):
            # Baseline object
            res = agent.optimize(smiles, env)
        else:
            # PPO Agent
            obs, info = env.reset(options={"starting_smiles": smiles})
            total_reward = 0.0
            trajectory = [info["smiles"]]

            for step in range(env.max_steps):
                action, _, _ = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward
                if step_info.get("valid", False) and "smiles" in step_info:
                    trajectory.append(step_info["smiles"])
                if terminated or truncated:
                    break

            final_mol = env.current_mol
            final_props = env.current_props
            res = {
                "initial_smiles": smiles,
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

        elapsed = time.time() - t0
        latencies.append(elapsed)

        # Normalized Headroom Gain
        res["nhg"] = compute_normalized_headroom_gain(res["initial_qed"], res["final_qed"])

        # Reward-hacking audit
        final_mol = Chem.MolFromSmiles(res["final_smiles"])
        audit = audit_molecule(final_mol)
        res.update(audit)
        res["method"] = method_name
        res["latency_sec"] = elapsed

        # QED stratification group
        q0 = res["initial_qed"]
        if q0 < 0.5:
            res["qed_group"] = "Low (<0.5)"
        elif q0 < 0.7:
            res["qed_group"] = "Medium (0.5-0.7)"
        elif q0 < 0.85:
            res["qed_group"] = "High (0.7-0.85)"
        else:
            res["qed_group"] = "Very High (>=0.85)"

        records.append(res)

    df = pd.DataFrame(records)
    diversity = compute_diversity(df["final_smiles"].tolist())

    summary = {
        "method": method_name,
        "count": len(df),
        "mean_delta_qed": float(df["delta_qed"].mean()),
        "std_delta_qed": float(df["delta_qed"].std() if len(df) > 1 else 0.0),
        "mean_nhg": float(df["nhg"].mean()),
        "success_rate": float(df["success"].mean()),
        "validity_rate": float(df["valid"].mean()),
        "mean_similarity": float(df["similarity"].mean()),
        "mean_delta_sas": float((df["final_sas"] - df["initial_sas"]).mean()),
        "mean_edits": float(df["num_edits"].mean()),
        "mean_latency_sec": float(np.mean(latencies)),
        "diversity": diversity,
        "pains_alert_rate": float(df["pains_alert"].mean()),
        "brenk_alert_rate": float(df["brenk_alert"].mean()),
        "mean_lipinski_violations": float(df["lipinski_violations"].mean()),
        "sanity_pass_rate": float(df["sanity_passed"].mean())
    }
    return df, summary


def counterfactual_action_analysis(starting_smiles, env, ppo_agent, num_alternatives=3):
    """
    Action-Level Counterfactual Explainability Module.
    For a given molecular state:
    1. Evaluates PPO-selected action.
    2. Applies alternative valid candidate actions independently from the exact same state.
    3. Computes property changes (Delta QED, Similarity, SAS change, immediate reward).
    4. Returns a clean comparison table explaining why the agent chose the action.
    """
    obs, info = env.reset(options={"starting_smiles": starting_smiles})
    mask = obs["action_mask"]
    valid_indices = np.where(mask)[0]

    # Selected PPO action
    ppo_action, _, _ = ppo_agent.select_action(obs, deterministic=True)
    ppo_desc = action_to_details(ppo_action)

    # Step copy for PPO action
    current_mol = env.current_mol
    orig_mol = env.original_mol
    current_props = env.current_props

    def eval_action(act):
        desc = action_to_details(act)
        if desc["type"] == "STOP":
            rew, _ = env.reward_calc.calculate_reward(current_props, current_props, env.original_props)
            return {
                "action_type": "STOP",
                "action_desc": "STOP",
                "delta_qed": 0.0,
                "similarity": current_props["similarity"],
                "delta_sas": 0.0,
                "reward": rew,
                "is_ppo_choice": (act == ppo_action)
            }
        new_mol = apply_action(current_mol, act, env.editor)
        if new_mol is None:
            return None
        new_props = calculate_properties(new_mol, orig_mol)
        rew, _ = env.reward_calc.calculate_reward(current_props, new_props, env.original_props)
        return {
            "action_type": desc["type"],
            "action_desc": str(desc),
            "delta_qed": new_props["qed"] - current_props["qed"],
            "similarity": new_props["similarity"],
            "delta_sas": new_props["sas"] - current_props["sas"],
            "reward": rew,
            "is_ppo_choice": (act == ppo_action)
        }

    outcomes = []
    ppo_res = eval_action(ppo_action)
    if ppo_res:
        outcomes.append(ppo_res)

    # Sample alternatives
    other_actions = [a for a in valid_indices if a != ppo_action]
    np.random.shuffle(other_actions)
    for act in other_actions[:num_alternatives]:
        res = eval_action(act)
        if res:
            outcomes.append(res)

    return pd.DataFrame(outcomes)
