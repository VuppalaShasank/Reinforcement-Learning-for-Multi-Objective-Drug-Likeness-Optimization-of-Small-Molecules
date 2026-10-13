import os
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def prepare_scaffold_splits(dataset_path, output_dir, sample_size=5000, random_seed=42):
    """
    Loads ZINC-250K, cleans SMILES, computes Murcko scaffolds,
    and produces an 80/10/10 train/val/test split with zero scaffold overlap.
    Also extracts a balanced test benchmark stratified by QED.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading ZINC-250K dataset from: {dataset_path}")
    
    # Read sample or full (sample_size for responsive, reliable execution)
    df = pd.read_csv(dataset_path, nrows=sample_size)
    df["smiles"] = df["smiles"].str.strip()
    
    print(f"Loaded {len(df)} molecules. Generating Bemis-Murcko scaffolds...")
    scaffolds = {}
    valid_records = []
    
    for idx, row in df.iterrows():
        s = row["smiles"]
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaffold = ""
            
        row_dict = {
            "smiles": s,
            "qed": float(row["qed"]),
            "sas": float(row["SAS"]),
            "scaffold": scaffold
        }
        valid_records.append(row_dict)
        scaffolds.setdefault(scaffold, []).append(len(valid_records) - 1)

    clean_df = pd.DataFrame(valid_records)
    print(f"Valid molecules: {len(clean_df)}, Unique scaffolds: {len(scaffolds)}")

    # Sort scaffolds by size for balanced splitting
    scaffold_list = list(scaffolds.keys())
    rng = np.random.RandomState(random_seed)
    rng.shuffle(scaffold_list)

    total_mols = len(clean_df)
    train_cutoff = int(0.80 * total_mols)
    val_cutoff = int(0.90 * total_mols)

    train_indices, val_indices, test_indices = [], [], []
    curr_count = 0
    for scaf in scaffold_list:
        m_idxs = scaffolds[scaf]
        if curr_count < train_cutoff:
            train_indices.extend(m_idxs)
        elif curr_count < val_cutoff:
            val_indices.extend(m_idxs)
        else:
            test_indices.extend(m_idxs)
        curr_count += len(m_idxs)

    train_df = clean_df.iloc[train_indices].reset_index(drop=True)
    val_df = clean_df.iloc[val_indices].reset_index(drop=True)
    test_df = clean_df.iloc[test_indices].reset_index(drop=True)

    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Verify zero scaffold overlap between train and test
    train_scafs = set(train_df["scaffold"].unique())
    test_scafs = set(test_df["scaffold"].unique())
    overlap = len(train_scafs.intersection(test_scafs))
    print(f"Train/Test Scaffold Overlap: {overlap} (Must be 0)")
    assert overlap == 0 or (overlap == 1 and "" in train_scafs and "" in test_scafs)

    # Save split info
    split_info = {
        "total_valid": len(clean_df),
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "unique_scaffolds": len(scaffolds),
        "train_scaffolds": len(train_scafs),
        "test_scaffolds": len(test_scafs)
    }
    
    with open(os.path.join(output_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # Save benchmark test set (stratified by QED)
    low_test = test_df[test_df["qed"] < 0.5]
    med_test = test_df[(test_df["qed"] >= 0.5) & (test_df["qed"] < 0.7)]
    high_test = test_df[(test_df["qed"] >= 0.7) & (test_df["qed"] < 0.85)]
    vhigh_test = test_df[test_df["qed"] >= 0.85]

    print(f"Test QED Distribution: Low={len(low_test)}, Med={len(med_test)}, High={len(high_test)}, Very High={len(vhigh_test)}")

    # Sample test benchmark: 15 molecules per QED bin for thorough, reproducible evaluation
    benchmark_mols = []
    for grp, g_df in [("Low", low_test), ("Medium", med_test), ("High", high_test), ("Very High", vhigh_test)]:
        sample_n = min(len(g_df), 15)
        if sample_n > 0:
            samp = g_df.sample(sample_n, random_state=random_seed).copy()
            samp["qed_group"] = grp
            benchmark_mols.append(samp)

    benchmark_df = pd.concat(benchmark_mols, ignore_index=True)
    benchmark_df.to_csv(os.path.join(output_dir, "test_benchmark_molecules.csv"), index=False)
    train_df.to_csv(os.path.join(output_dir, "train_molecules.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_molecules.csv"), index=False)
    
    print(f"Benchmark test set saved with {len(benchmark_df)} molecules across QED groups.")
    return train_df, val_df, benchmark_df

if __name__ == "__main__":
    ds_path = os.path.join("ZINC-250K", "250k_rndm_zinc_drugs_clean_3.csv")
    out_dir = os.path.join("cache", "results")
    prepare_scaffold_splits(ds_path, out_dir, sample_size=5000)

