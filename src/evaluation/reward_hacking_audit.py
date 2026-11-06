import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Initialize PAINS and Brenk filter catalogs
pains_params = FilterCatalogParams()
pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog(pains_params)

brenk_params = FilterCatalogParams()
brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
BRENK_CATALOG = FilterCatalog(brenk_params)

def check_lipinski_violations(mol):
    """
    Computes number of Lipinski Rule of 5 violations:
    1. MW <= 500
    2. LogP <= 5
    3. H-bond Donors <= 5
    4. H-bond Acceptors <= 10
    """
    if mol is None:
        return 4
    violations = 0
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if hbd > 5: violations += 1
        if hba > 10: violations += 1
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
    - Maximum ring size > 8
    - Extreme heteroatom to carbon ratio
    - Disconnected fragments
    """
    if mol is None:
        return False, "Null molecule"
    try:
        # Check rings
        ssr = Chem.GetSymmSSSR(mol)
        for ring in ssr:
            if len(ring) > 8:
                return False, f"Large ring of size {len(ring)}"
                
        # Check heteroatom ratio
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        c_count = atoms.count('C')
        hetero_count = len(atoms) - c_count
        if c_count == 0 and len(atoms) > 3:
            return False, "No carbon atoms"
            
        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:
            return False, "Disconnected graph fragments"
            
        return True, "Passed sanity checks"
    except Exception as e:
        return False, str(e)

def compute_headroom_gain(q0, q_final, eps=0.05):
    """
    Normalized Headroom Gain:
    NHG = (QED_final - q0) / max(1.0 - q0, eps)
    """
    headroom = max(1.0 - q0, eps)
    return (q_final - q0) / headroom

def audit_molecule(mol):
    """
    Performs full reward-hacking audit on a generated molecule:
    - Chemical validity
    - Lipinski violations (0-4)
    - PAINS alert (bool)
    - Brenk alert (bool)
    - Structural sanity (bool)
    """
    if mol is None:
        return {
            "valid": False,
            "lipinski_violations": 4,
            "pains_alert": False,
            "brenk_alert": False,
            "sanity_passed": False,
            "sanity_reason": "Invalid molecule"
        }
    passed_sanity, reason = check_structural_sanity(mol)
    return {
        "valid": True,
        "lipinski_violations": check_lipinski_violations(mol),
        "pains_alert": check_pains_alert(mol),
        "brenk_alert": check_brenk_alert(mol),
        "sanity_passed": passed_sanity,
        "sanity_reason": reason
    }

def evaluate_policy_on_molecules(agent, env, smiles_list, method_name="PPO"):
    """
    Evaluates an agent (PPO or Baseline) across a list of starting SMILES.
    Returns:
    - results_df: DataFrame with per-molecule metrics and audit checks.
    - summary_stats: dict of aggregate metrics (mean, std, success rate, etc.).
    """
    records = []
    
    for idx, s_smiles in enumerate(smiles_list):
        obs, info = env.reset(options={"starting_smiles": s_smiles})
        orig_qed = info["qed"]
        orig_sas = info["sas"]
        
        # Run episode
        while True:
            # Check if agent has select_action or is a baseline object
            if hasattr(agent, "select_action"):
                action, _, _ = agent.select_action(obs, deterministic=True)
            elif hasattr(agent, "optimize"):
                # Baseline object handles full trajectory
                res = agent.optimize(s_smiles, env)
                break
            else:
                action = 0 # STOP fallback
                
            obs, reward, terminated, truncated, step_info = env.step(action)
            if terminated or truncated:
                res = {
                    "initial_smiles": s_smiles,
                    "final_smiles": Chem.MolToSmiles(env.current_mol) if env.current_mol else s_smiles,
                    "initial_qed": orig_qed,
                    "final_qed": env.current_props["qed"],
                    "delta_qed": env.current_props["qed"] - orig_qed,
                    "initial_sas": orig_sas,
                    "final_sas": env.current_props["sas"],
                    "similarity": env.current_props["similarity"],
                    "valid": env.current_props["valid"],
                    "success": env.reward_calc.check_success(env.current_props, env.original_props),
                    "num_edits": env.current_step,
                    "total_reward": reward
                }
                break

        # Compute headroom gain
        nhg = compute_headroom_gain(res["initial_qed"], res["final_qed"])
        res["nhg"] = nhg
        
        # Reward-hacking audit on final molecule
        final_mol = Chem.MolFromSmiles(res["final_smiles"])
        audit = audit_molecule(final_mol)
        res.update(audit)
        res["method"] = method_name
        
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
        "pains_alert_rate": float(df["pains_alert"].mean()),
        "brenk_alert_rate": float(df["brenk_alert"].mean()),
        "mean_lipinski_violations": float(df["lipinski_violations"].mean()),
        "sanity_pass_rate": float(df["sanity_passed"].mean())
    }
    return df, summary

