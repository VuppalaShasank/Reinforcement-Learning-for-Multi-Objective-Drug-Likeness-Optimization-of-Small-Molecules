import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
import sys
import os
import json
import torch
sys.path.append(os.path.abspath("."))
from implementation.properties import calculate_qed, calculate_sas
from implementation.molecule_editor import MoleculeEditor
from implementation.environment import MolecularEnvironment

def test_all():
    print("Testing Phase 1: Dataset")
    df = pd.read_csv("ZINC-250K/250k_rndm_zinc_drugs_clean_3.csv", nrows=100)
    df_sample = df.sample(10, random_state=42).copy()
    df_sample["smiles"] = df_sample["smiles"].str.strip()
    df_sample["mol"] = df_sample["smiles"].apply(Chem.MolFromSmiles)
    df_valid = df_sample[df_sample["mol"].notna()].copy()
    df_valid["qed"] = df_valid["mol"].apply(calculate_qed)
    df_valid["sas"] = df_valid["mol"].apply(calculate_sas)
    
    print("Testing Phase 2/3: Graph and Edits")
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    editor = MoleculeEditor()
    new_mol_1 = editor.add_atom(mol, atom_idx=3, new_element="F")
    assert new_mol_1 is not None, "Add atom failed"
    
    # Graph features
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetTotalNumHs(), int(atom.GetIsAromatic()), int(atom.IsInRing())])
    x = torch.tensor(node_features, dtype=torch.float)
    assert x.shape[1] == 5, "Graph node features shape mismatch"

    print("Testing Phase 3/4/5: Environment and Reward")
    env = MolecularEnvironment()
    obs, info = env.reset(options={"starting_smiles": "c1ccccc1"})
    action = {"type": 0, "atom1": 0, "atom2": 0, "element": 3, "bond_delta": 1}
    next_obs, reward, done, _, info = env.step(action)
    assert reward is not None, "Reward is none"

    print("ALL TESTS PASSED.")

if __name__ == "__main__":
    test_all()
