import os
import sys
import json
import torch
import pandas as pd
from rdkit import Chem

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from implementation.properties import calculate_qed, calculate_sas, calculate_similarity, calculate_properties
from implementation.molecule_editor import MoleculeEditor
from cache.internal_scripts.legacy_phase1_5.environment import MolecularEnvironment
from implementation.reward import RewardCalculator

def test_pipeline():
    print("=== Testing Hardware & Libraries ===")
    print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")
        
    print("\n=== Testing Phase 1: Dataset & RDKit Properties ===")
    dataset_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "ZINC-250K", "250k_rndm_zinc_drugs_clean_3.csv")
    assert os.path.exists(dataset_path), f"Dataset not found at {dataset_path}"
    df = pd.read_csv(dataset_path, nrows=50)
    sample_smiles = df.iloc[0]["smiles"].strip()
    mol0 = Chem.MolFromSmiles(sample_smiles)
    assert mol0 is not None, "Failed to parse SMILES from ZINC-250K"
    
    qed0 = calculate_qed(mol0)
    sas0 = calculate_sas(mol0)
    print(f"Starting molecule: {sample_smiles}")
    print(f"Calculated QED: {qed0:.4f}, SAS: {sas0:.4f}")
    assert 0.0 < qed0 < 1.0, f"Invalid QED: {qed0}"
    assert 1.0 < sas0 < 10.0, f"Invalid SAS: {sas0}"
    
    print("\n=== Testing Phase 2: Molecule Editor & Chemical Edits ===")
    editor = MoleculeEditor()
    # Test valid edit: Add F to carbon 0
    mol_add = editor.add_atom(mol0, atom_idx=0, new_element="F")
    assert mol_add is not None, "Valid Add Atom failed"
    print("Valid Add Atom (F to atom 0): PASSED")
    
    # Test invalid edit: Add to invalid index
    mol_inv = editor.add_atom(mol0, atom_idx=999, new_element="F")
    assert mol_inv is None, "Expected invalid atom_idx to return None"
    print("Invalid Action Rejection: PASSED")
    
    # Test substitution: replace atom 0 with N if valid
    mol_sub = editor.substitute_atom(mol0, atom_idx=0, new_element="N")
    # Might be valid or invalid depending on valency, but shouldn't crash
    print(f"Substitution test: {'Accepted' if mol_sub else 'Safely Rejected'}")
    
    print("\n=== Testing Phase 3: Environment Transitions & STOP ===")
    env = MolecularEnvironment()
    obs, info = env.reset(options={"starting_smiles": sample_smiles})
    assert "qed" in obs and "sas" in obs and "similarity" in obs
    print(f"Initial env reset obs: QED={obs['qed'][0]:.4f}, SAS={obs['sas'][0]:.4f}, Sim={obs['similarity'][0]:.4f}")
    
    # Step with STOP action (action_type 4)
    stop_action = {"type": 4, "atom1": 0, "atom2": 0, "element": 0, "bond_delta": 1}
    next_obs, reward, terminated, truncated, step_info = env.step(stop_action)
    assert terminated is True, "STOP action must terminate episode"
    print(f"STOP Action terminated episode: PASSED (reward={reward:.4f})")
    
    print("\nALL EXISTING PHASE 1-5 PIPELINE TESTS PASSED!")

if __name__ == "__main__":
    test_pipeline()
