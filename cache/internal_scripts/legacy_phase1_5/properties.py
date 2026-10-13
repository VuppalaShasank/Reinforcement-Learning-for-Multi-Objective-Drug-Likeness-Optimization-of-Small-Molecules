import os
import sys
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import RDConfig

# Add sascorer to path if possible
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    SASCORER_AVAILABLE = True
except ImportError:
    SASCORER_AVAILABLE = False
    raise ImportError("RDKit SA_Score (sascorer) is required but not available. Please ensure RDKit contrib modules are installed.")

def calculate_qed(mol):
    if mol is None:
        return 0.0
    try:
        return QED.qed(mol)
    except Exception:
        return 0.0

def calculate_sas(mol):
    if mol is None:
        return 10.0 # High penalty for invalid
    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return 10.0

def calculate_similarity(mol1, mol2):
    if mol1 is None or mol2 is None:
        return 0.0
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0

def calculate_properties(mol, original_mol=None):
    """Calculates multiple properties for a given molecule."""
    props = {
        'valid': mol is not None,
        'qed': calculate_qed(mol),
        'sas': calculate_sas(mol),
        'similarity': calculate_similarity(original_mol, mol) if original_mol else 1.0
    }
    return props

def is_valid(mol):
    """Check if molecule is chemically valid according to RDKit."""
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False
