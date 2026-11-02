import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from implementation.molecule_editor import MoleculeEditor

ALLOWED_ELEMENTS = ['C', 'N', 'O', 'F', 'S', 'Cl']
MAX_ATOMS = 38
NUM_ELEMENTS = len(ALLOWED_ELEMENTS)

# Action ranges:
# 0: STOP (1)
# 1 .. 228: ADD (38 * 6 = 228)
# 229 .. 266: REMOVE (38)
# 267 .. 494: SUBSTITUTE (38 * 6 = 228)
# 495 .. 1900: CHANGE_BOND (38 * 37 // 2 * 2 = 703 * 2 = 1406)
TOTAL_ACTIONS = 1 + (MAX_ATOMS * NUM_ELEMENTS) + MAX_ATOMS + (MAX_ATOMS * NUM_ELEMENTS) + (MAX_ATOMS * (MAX_ATOMS - 1) // 2 * 2)

# Precompute pair to index mapping for bonds
PAIR_TO_IDX = {}
IDX_TO_PAIR = []
idx = 0
for i in range(MAX_ATOMS):
    for j in range(i + 1, MAX_ATOMS):
        PAIR_TO_IDX[(i, j)] = idx
        IDX_TO_PAIR.append((i, j))
        idx += 1

def action_to_details(action_idx):
    """
    Decodes an integer action_idx into an operation dictionary:
    - {'type': 'STOP'}
    - {'type': 'ADD', 'atom': i, 'element': elem}
    - {'type': 'REMOVE', 'atom': i}
    - {'type': 'SUBSTITUTE', 'atom': i, 'element': elem}
    - {'type': 'CHANGE_BOND', 'atom1': i, 'atom2': j, 'delta': delta}
    """
    action_idx = int(action_idx)
    if action_idx == 0:
        return {'type': 'STOP'}
    
    curr = 1
    # ADD
    if action_idx < curr + MAX_ATOMS * NUM_ELEMENTS:
        offset = action_idx - curr
        atom_idx = int(offset // NUM_ELEMENTS)
        elem_idx = int(offset % NUM_ELEMENTS)
        return {'type': 'ADD', 'atom': atom_idx, 'element': ALLOWED_ELEMENTS[elem_idx]}
    curr += MAX_ATOMS * NUM_ELEMENTS
    
    # REMOVE
    if action_idx < curr + MAX_ATOMS:
        atom_idx = int(action_idx - curr)
        return {'type': 'REMOVE', 'atom': atom_idx}
    curr += MAX_ATOMS
    
    # SUBSTITUTE
    if action_idx < curr + MAX_ATOMS * NUM_ELEMENTS:
        offset = action_idx - curr
        atom_idx = int(offset // NUM_ELEMENTS)
        elem_idx = int(offset % NUM_ELEMENTS)
        return {'type': 'SUBSTITUTE', 'atom': atom_idx, 'element': ALLOWED_ELEMENTS[elem_idx]}
    curr += MAX_ATOMS * NUM_ELEMENTS
    
    # CHANGE_BOND
    offset = action_idx - curr
    pair_idx = int(offset // 2)
    delta_idx = int(offset % 2)
    delta = -1 if delta_idx == 0 else 1
    atom1, atom2 = IDX_TO_PAIR[pair_idx]
    return {'type': 'CHANGE_BOND', 'atom1': int(atom1), 'atom2': int(atom2), 'delta': int(delta)}

def get_action_mask(mol, editor=None):
    """
    Computes a boolean action mask of shape [1901] for the current molecule.
    Action 0 (STOP) is always valid.
    Only chemically valid local modifications have mask = True.
    """
    if editor is None:
        editor = MoleculeEditor(allowed_elements=ALLOWED_ELEMENTS)
        
    mask = np.zeros(TOTAL_ACTIONS, dtype=bool)
    mask[0] = True # STOP is always valid

    if mol is None:
        return mask

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return mask

    # 1. ADD ATOM checks
    # Cannot add if already at max atoms limit
    if num_atoms < MAX_ATOMS:
        curr = 1
        for a_idx in range(num_atoms):
            # Check maximum valency of target atom
            target_atom = mol.GetAtomWithIdx(a_idx)
            val = target_atom.GetExplicitValence() + target_atom.GetImplicitValence()
            # Fast check: elements usually don't accept additional bonds beyond standard valencies
            for e_idx, elem in enumerate(ALLOWED_ELEMENTS):
                act_idx = curr + a_idx * NUM_ELEMENTS + e_idx
                # Test edit
                res_mol = editor.add_atom(mol, a_idx, elem)
                if res_mol is not None and res_mol.GetNumAtoms() <= MAX_ATOMS:
                    mask[act_idx] = True

    # 2. REMOVE ATOM checks
    curr_remove = 1 + MAX_ATOMS * NUM_ELEMENTS
    if num_atoms > 1:
        for a_idx in range(num_atoms):
            act_idx = curr_remove + a_idx
            res_mol = editor.remove_atom(mol, a_idx)
            if res_mol is not None:
                mask[act_idx] = True

    # 3. SUBSTITUTE ATOM checks
    curr_sub = curr_remove + MAX_ATOMS
    for a_idx in range(num_atoms):
        curr_elem = mol.GetAtomWithIdx(a_idx).GetSymbol()
        for e_idx, elem in enumerate(ALLOWED_ELEMENTS):
            act_idx = curr_sub + a_idx * NUM_ELEMENTS + e_idx
            if elem == curr_elem:
                continue # Skip identity substitution
            res_mol = editor.substitute_atom(mol, a_idx, elem)
            if res_mol is not None:
                mask[act_idx] = True

    # 4. CHANGE BOND checks
    curr_bond = curr_sub + MAX_ATOMS * NUM_ELEMENTS
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i > j:
            i, j = j, i
        if (i, j) in PAIR_TO_IDX:
            pair_idx = PAIR_TO_IDX[(i, j)]
            # delta = -1
            act_dec = curr_bond + pair_idx * 2 + 0
            res_dec = editor.change_bond_order(mol, i, j, -1)
            if res_dec is not None:
                mask[act_dec] = True
            # delta = +1
            act_inc = curr_bond + pair_idx * 2 + 1
            res_inc = editor.change_bond_order(mol, i, j, 1)
            if res_inc is not None:
                mask[act_inc] = True

    return mask

def apply_action(mol, action_idx, editor=None):
    """
    Applies the decoded action to mol and returns new_mol.
    Returns None if the action is chemically invalid or cannot be applied.
    """
    if editor is None:
        editor = MoleculeEditor(allowed_elements=ALLOWED_ELEMENTS)
        
    details = action_to_details(action_idx)
    op = details['type']

    if op == 'STOP':
        return mol # Terminal, molecule unchanged
    elif op == 'ADD':
        return editor.add_atom(mol, details['atom'], details['element'])
    elif op == 'REMOVE':
        return editor.remove_atom(mol, details['atom'])
    elif op == 'SUBSTITUTE':
        return editor.substitute_atom(mol, details['atom'], details['element'])
    elif op == 'CHANGE_BOND':
        return editor.change_bond_order(mol, details['atom1'], details['atom2'], details['delta'])
    return None

