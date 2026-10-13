import random
from rdkit import Chem

class MoleculeEditor:
    def __init__(self, allowed_elements=None):
        if allowed_elements is None:
            self.allowed_elements = ['C', 'N', 'O', 'F', 'S', 'Cl']
        else:
            self.allowed_elements = allowed_elements
            
    def _sanitize(self, rw_mol):
        try:
            Chem.SanitizeMol(rw_mol)
            return rw_mol.GetMol()
        except Exception:
            return None

    def add_atom(self, mol, atom_idx, new_element):
        """Adds a new atom of new_element bonded to atom_idx with a single bond."""
        if mol is None or atom_idx >= mol.GetNumAtoms():
            return None
        rw_mol = Chem.RWMol(mol)
        new_idx = rw_mol.AddAtom(Chem.Atom(new_element))
        rw_mol.AddBond(atom_idx, new_idx, Chem.BondType.SINGLE)
        return self._sanitize(rw_mol)

    def remove_atom(self, mol, atom_idx):
        """Removes an atom from the molecule if it doesn't break it into fragments."""
        if mol is None or atom_idx >= mol.GetNumAtoms():
            return None
        # Don't allow removing the last atom
        if mol.GetNumAtoms() <= 1:
            return None
            
        rw_mol = Chem.RWMol(mol)
        rw_mol.RemoveAtom(atom_idx)
        
        # Check if disconnected (has multiple fragments)
        frags = Chem.GetMolFrags(rw_mol.GetMol())
        if len(frags) > 1:
            return None
            
        return self._sanitize(rw_mol)

    def substitute_atom(self, mol, atom_idx, new_element):
        """Changes the element of an existing atom."""
        if mol is None or atom_idx >= mol.GetNumAtoms():
            return None
        rw_mol = Chem.RWMol(mol)
        rw_mol.GetAtomWithIdx(atom_idx).SetAtomicNum(Chem.Atom(new_element).GetAtomicNum())
        return self._sanitize(rw_mol)

    def change_bond_order(self, mol, atom1_idx, atom2_idx, bond_order_delta):
        """Changes the bond order between atom1 and atom2 by bond_order_delta."""
        if mol is None:
            return None
        if atom1_idx >= mol.GetNumAtoms() or atom2_idx >= mol.GetNumAtoms() or atom1_idx == atom2_idx:
            return None
            
        rw_mol = Chem.RWMol(mol)
        bond = rw_mol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        
        if bond is None:
            # We don't add new bonds between arbitrary distant atoms in this simple function 
            # (unless extending to ring closures)
            return None
            
        current_type = bond.GetBondType()
        order_map = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}
        reverse_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
        
        if current_type not in order_map:
            return None
            
        new_order = order_map[current_type] + bond_order_delta
        
        if new_order <= 0:
            # Bond removal - check if it disconnects graph
            rw_mol.RemoveBond(atom1_idx, atom2_idx)
            frags = Chem.GetMolFrags(rw_mol.GetMol())
            if len(frags) > 1:
                return None
        elif new_order > 3:
            return None # Can't exceed triple bond
        else:
            bond.SetBondType(reverse_map[new_order])
            
        return self._sanitize(rw_mol)
