import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data

ALLOWED_ELEMENTS = ['C', 'N', 'O', 'F', 'S', 'Cl']
MAX_ATOMS = 38

def one_hot_encode(val, possible_values):
    vec = [0] * (len(possible_values) + 1)
    if val in possible_values:
        vec[possible_values.index(val)] = 1
    else:
        vec[-1] = 1 # 'Other'
    return vec

def get_node_features(mol):
    """
    Extracts 21-dimensional node features for all atoms in mol:
    - Element (one-hot across C, N, O, F, S, Cl, other): 7 dims
    - Formal charge (-2, -1, 0, 1, 2, other): 6 dims
    - Hydrogen count (0, 1, 2, 3, 4, other): 6 dims
    - Is aromatic: 1 dim
    - Is in ring: 1 dim
    Total: 7 + 6 + 6 + 1 + 1 = 21 dims
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return torch.zeros((1, 21), dtype=torch.float32)

    features = []
    for atom in mol.GetAtoms():
        elem_symbol = atom.GetSymbol()
        elem_feat = one_hot_encode(elem_symbol, ALLOWED_ELEMENTS)
        
        charge_feat = one_hot_encode(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        h_feat = one_hot_encode(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        aromatic_feat = [1.0 if atom.GetIsAromatic() else 0.0]
        ring_feat = [1.0 if atom.IsInRing() else 0.0]
        
        atom_vec = elem_feat + charge_feat + h_feat + aromatic_feat + ring_feat
        features.append(atom_vec)
        
    return torch.tensor(features, dtype=torch.float32)

def get_edge_features(mol):
    """
    Extracts directed edge indices and 4-dimensional edge features:
    - Bond type: Single, Double, Triple, Aromatic (4 dims)
    """
    if mol is None or mol.GetNumBonds() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float32)
        return edge_index, edge_attr

    bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_type = bond.GetBondType()
        b_vec = [1.0 if b_type == bt else 0.0 for bt in bond_types]

        # Undirected edges in PyG require both directions
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([b_vec, b_vec])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    return edge_index, edge_attr

def mol_to_pyg_data(mol, current_step=0, max_steps=20, qed=0.0, sas=0.0, sim=1.0, q0=0.0, sas_0=0.0):
    """
    Converts RDKit Mol and scalar state descriptors into a PyTorch Geometric Data object.
    Scalars: [t/T, qed, sas, sim, q0, sas_0] (6 dims)
    """
    x = get_node_features(mol)
    edge_index, edge_attr = get_edge_features(mol)
    
    step_norm = float(current_step) / float(max_steps) if max_steps > 0 else 0.0
    scalars = torch.tensor([step_norm, qed, sas, sim, q0, sas_0], dtype=torch.float32)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.scalars = scalars.unsqueeze(0) # [1, 6]
    data.num_atoms = mol.GetNumAtoms() if mol else 0
    return data

