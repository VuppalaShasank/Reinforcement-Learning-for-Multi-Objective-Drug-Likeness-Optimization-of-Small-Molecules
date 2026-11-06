"""
Chemistry Descriptors Module
Calculates standard physicochemical descriptors, Lipinski Rule of 5 violations,
and synthetic accessibility score (SAS).
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
from src.chemistry.properties import calculate_properties, calculate_sas
from src.evaluation.reward_hacking_audit import check_pains_alert, check_lipinski_violations

def get_molecular_descriptors(mol):
    """
    Computes a comprehensive dictionary of molecular descriptors.
    """
    if mol is None:
        return {}
    
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    aromatic_rings = Lipinski.NumAromaticRings(mol)
    
    # Lipinski Rule of Five Violations
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    return {
        "molecular_weight": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "rotatable_bonds": rotatable_bonds,
        "aromatic_rings": aromatic_rings,
        "lipinski_violations": check_lipinski_violations(mol),
        "pains_alert": check_pains_alert(mol)
    }
