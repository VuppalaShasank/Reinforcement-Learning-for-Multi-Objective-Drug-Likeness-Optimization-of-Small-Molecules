"""
Script to create the expanded 500-molecule final test benchmark from ZINC-250K,
adjust the validation split from 501 to 500, verify strict zero scaffold overlap,
and output detailed verification reports and split metadata.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

# Project imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from implementation.properties import calculate_sas, calculate_qed

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(WORKSPACE_DIR, "data", "splits")
ZINC_CSV = os.path.join(ROOT_DIR, "ZINC-250K", "250k_rndm_zinc_drugs_clean_3.csv")


def main():
    print("================================================================================")
    print("GENERATING EXPANDED FINAL TEST BENCHMARK (500 MOLECULES) FROM ZINC-250K")
    print("================================================================================")

    # 1. Load train molecules
    train_path = os.path.join(SPLITS_DIR, "train_molecules.csv")
    df_train = pd.read_csv(train_path)
    print(f"Loaded train molecules: {len(df_train)}")

    # 2. Adjust validation split to exactly 500 molecules
    val_path = os.path.join(SPLITS_DIR, "val_molecules.csv")
    df_val_orig = pd.read_csv(val_path)
    df_val = df_val_orig.head(500).copy()
    df_val.to_csv(val_path, index=False)
    print(f"Adjusted validation molecules from {len(df_val_orig)} to {len(df_val)}")

    # 3. Canonicalize train and val smiles and scaffolds
    train_smiles_set = set(df_train["smiles"].str.strip())
    val_smiles_set = set(df_val["smiles"].str.strip())
    
    train_scaffolds = set(df_train["scaffold"].dropna())
    val_scaffolds = set(df_val["scaffold"].dropna())
    forbidden_scaffolds = set(train_scaffolds | val_scaffolds)
    forbidden_smiles = set(train_smiles_set | val_smiles_set)

    print(f"Unique train scaffolds: {len(train_scaffolds)}")
    print(f"Unique val scaffolds: {len(val_scaffolds)}")
    print(f"Total forbidden scaffolds (Train U Val): {len(forbidden_scaffolds)}")

    # 4. Target counts for stratified test set
    target_counts = {
        "Low (<0.50)": 125,
        "Medium (0.50-0.70)": 125,
        "High (0.70-0.85)": 125,
        "Very High (>=0.85)": 125
    }
    
    selected_by_group = {k: [] for k in target_counts}
    test_scaffolds_set = set()
    
    np.random.seed(42)
    chunksize = 25000

    print("\nStreaming through ZINC-250K to select 125 molecules per tier...")
    for chunk in pd.read_csv(ZINC_CSV, chunksize=chunksize):
        # Shuffle chunk deterministically
        chunk = chunk.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        for _, row in chunk.iterrows():
            # Check if all targets met
            if all(len(selected_by_group[k]) >= target_counts[k] for k in target_counts):
                break
                
            raw_smiles = str(row["smiles"]).strip()
            if raw_smiles in forbidden_smiles:
                continue
                
            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is None:
                continue
                
            canon_smiles = Chem.MolToSmiles(mol)
            if canon_smiles in forbidden_smiles:
                continue
                
            # Compute scaffold
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            except Exception:
                continue
                
            if not scaffold or scaffold in forbidden_scaffolds or scaffold in test_scaffolds_set:
                # Require non-empty scaffold, and avoid duplicate scaffolds in test to ensure diversity!
                continue
                
            # Recompute properties using project standard functions
            qed_val = calculate_qed(mol)
            sas_val = calculate_sas(mol)
            mw_val = float(Descriptors.MolWt(mol))
            heavy_atoms = int(mol.GetNumHeavyAtoms())
            
            # Determine group
            if qed_val < 0.50:
                grp = "Low (<0.50)"
            elif qed_val < 0.70:
                grp = "Medium (0.50-0.70)"
            elif qed_val < 0.85:
                grp = "High (0.70-0.85)"
            else:
                grp = "Very High (>=0.85)"
                
            if len(selected_by_group[grp]) < target_counts[grp]:
                selected_by_group[grp].append({
                    "original_smiles": raw_smiles,
                    "canonical_smiles": canon_smiles,
                    "scaffold_smiles": scaffold,
                    "qed": qed_val,
                    "sas": sas_val,
                    "molecular_weight": mw_val,
                    "heavy_atom_count": heavy_atoms,
                    "qed_group": grp
                })
                test_scaffolds_set.add(scaffold)
                forbidden_smiles.add(canon_smiles)
                
        if all(len(selected_by_group[k]) >= target_counts[k] for k in target_counts):
            break

    # Combine all groups
    all_test = []
    mol_idx = 1
    for grp in ["Low (<0.50)", "Medium (0.50-0.70)", "High (0.70-0.85)", "Very High (>=0.85)"]:
        items = selected_by_group[grp]
        print(f"Group {grp:22s}: selected {len(items)} molecules")
        for item in items:
            item["molecule_id"] = f"TEST_{mol_idx:03d}"
            item["smiles"] = item["canonical_smiles"] # standard smiles column
            mol_idx += 1
            all_test.append(item)

    df_test_500 = pd.DataFrame(all_test)
    cols_order = [
        "molecule_id", "smiles", "canonical_smiles", "original_smiles",
        "scaffold_smiles", "qed", "sas", "molecular_weight", "heavy_atom_count", "qed_group"
    ]
    df_test_500 = df_test_500[cols_order]
    
    test_500_path = os.path.join(SPLITS_DIR, "test_benchmark_molecules.csv")
    df_test_500.to_csv(test_500_path, index=False)
    print(f"\nSaved 500-molecule test benchmark to: {test_500_path}")

    # 5. Verification Checks & Reports
    train_scaff_set = set(df_train["scaffold"].dropna())
    val_scaff_set = set(df_val["scaffold"].dropna())
    test_scaff_set = set(df_test_500["scaffold_smiles"].dropna())

    train_smiles_set = set(df_train["smiles"].str.strip())
    val_smiles_set = set(df_val["smiles"].str.strip())
    test_smiles_set = set(df_test_500["canonical_smiles"].str.strip())

    overlap_train_val = len(train_scaff_set & val_scaff_set)
    overlap_train_test = len(train_scaff_set & test_scaff_set)
    overlap_val_test = len(val_scaff_set & test_scaff_set)

    smiles_overlap_train_val = len(train_smiles_set & val_smiles_set)
    smiles_overlap_train_test = len(train_smiles_set & test_smiles_set)
    smiles_overlap_val_test = len(val_smiles_set & test_smiles_set)

    print("\n--- ZERO-LEAKAGE VERIFICATION REPORT ---")
    print(f"1. Train vs. Val scaffold overlap:        {overlap_train_val}")
    print(f"2. Train vs. Test scaffold overlap:       {overlap_train_test}")
    print(f"3. Val vs. Test scaffold overlap:         {overlap_val_test}")
    print(f"4. Canonical SMILES overlap (all pairs):   {smiles_overlap_train_val + smiles_overlap_train_test + smiles_overlap_val_test}")
    print(f"5. Total unique scaffolds in 500-test:    {len(test_scaff_set)} (100% scaffold diversity)")

    assert overlap_train_val == 0, "Error: Train-Val scaffold leakage!"
    assert overlap_train_test == 0, "Error: Train-Test scaffold leakage!"
    assert overlap_val_test == 0, "Error: Val-Test scaffold leakage!"
    assert smiles_overlap_train_test == 0, "Error: Train-Test SMILES leakage!"
    assert len(df_test_500) == 500, f"Error: Expected 500 test molecules, got {len(df_test_500)}"

    # Save scaffold overlap report
    overlap_df = pd.DataFrame([
        {"comparison": "Train vs Validation", "scaffold_overlap_count": overlap_train_val, "smiles_overlap_count": smiles_overlap_train_val, "status": "PASSED (Zero Leakage)"},
        {"comparison": "Train vs Final Test", "scaffold_overlap_count": overlap_train_test, "smiles_overlap_count": smiles_overlap_train_test, "status": "PASSED (Zero Leakage)"},
        {"comparison": "Validation vs Final Test", "scaffold_overlap_count": overlap_val_test, "smiles_overlap_count": smiles_overlap_val_test, "status": "PASSED (Zero Leakage)"}
    ])
    overlap_report_path = os.path.join(SPLITS_DIR, "scaffold_overlap_report.csv")
    overlap_df.to_csv(overlap_report_path, index=False)
    print(f"Saved scaffold overlap report to: {overlap_report_path}")

    # Save QED stratification report
    strat_records = []
    for grp, grp_df in df_test_500.groupby("qed_group"):
        strat_records.append({
            "qed_group": grp,
            "target_count": 125,
            "actual_count": len(grp_df),
            "mean_qed": float(grp_df["qed"].mean()),
            "std_qed": float(grp_df["qed"].std()),
            "min_qed": float(grp_df["qed"].min()),
            "max_qed": float(grp_df["qed"].max()),
            "mean_sas": float(grp_df["sas"].mean()),
            "mean_molecular_weight": float(grp_df["molecular_weight"].mean()),
            "mean_heavy_atom_count": float(grp_df["heavy_atom_count"].mean()),
            "unique_scaffolds": int(grp_df["scaffold_smiles"].nunique())
        })
    strat_report_df = pd.DataFrame(strat_records)
    strat_report_path = os.path.join(SPLITS_DIR, "qed_stratification_report.csv")
    strat_report_df.to_csv(strat_report_path, index=False)
    print(f"Saved QED stratification report to: {strat_report_path}")

    # Save split info JSON
    split_info = {
        "dataset_name": "ZINC-250K",
        "total_source_molecules": 249455,
        "experimental_subset_size": 5000,
        "split_ratio": "80:10:10 (Train:Val:Test)",
        "train_molecules_count": len(df_train),
        "val_molecules_count": len(df_val),
        "test_molecules_count": len(df_test_500),
        "pilot_test_molecules_count": 60,
        "unique_scaffolds_train": len(train_scaff_set),
        "unique_scaffolds_val": len(val_scaff_set),
        "unique_scaffolds_test": len(test_scaff_set),
        "cross_split_scaffold_overlap": {
            "train_val": overlap_train_val,
            "train_test": overlap_train_test,
            "val_test": overlap_val_test
        },
        "qed_group_counts": {k: len(selected_by_group[k]) for k in target_counts},
        "selection_random_seed": 42,
        "scaffold_method": "Bemis-Murcko Scaffold",
        "verification_status": "VERIFIED_ZERO_LEAKAGE"
    }
    split_info_path = os.path.join(SPLITS_DIR, "split_info.json")
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved split info metadata to: {split_info_path}")
    print("\nDataset split generation completed successfully!")


if __name__ == "__main__":
    main()

