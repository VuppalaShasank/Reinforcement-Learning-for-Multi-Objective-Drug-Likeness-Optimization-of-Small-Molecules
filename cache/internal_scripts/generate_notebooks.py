import nbformat as nbf
import os

def create_notebook(filename, cells_data):
    nb = nbf.v4.new_notebook()
    cells = []
    for ctype, content in cells_data:
        if ctype == 'md':
            cells.append(nbf.v4.new_markdown_cell(content))
        elif ctype == 'code':
            cells.append(nbf.v4.new_code_cell(content))
    nb['cells'] = cells
    
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

os.makedirs('notebooks', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# 1. Dataset Analysis Notebook
ds_cells = [
    ('md', '# Phase 1: ZINC-250K Dataset Analysis\n\n### Objective\nLoad the dataset, strip newlines, calculate basic statistics, bin molecules by QED, and save the processed dataset.'),
    ('code', 'import pandas as pd\nimport matplotlib.pyplot as plt\nfrom rdkit import Chem\nimport sys\nimport os\nimport json\nsys.path.append(os.path.abspath(".."))\nfrom implementation.properties import calculate_qed, calculate_sas'),
    ('code', '# Load ZINC dataset\ndf = pd.read_csv("../ZINC-250K/250k_rndm_zinc_drugs_clean_3.csv")\nprint(f"Total original rows: {len(df)}")'),
    ('md', '### Clean and Validate SMILES\nWe will take a sample for speed (use full dataset for final training).'),
    ('code', 'df_sample = df.sample(1000, random_state=42).copy()\ndf_sample["smiles"] = df_sample["smiles"].str.strip()\n\n# Convert to RDKit Mols\ndf_sample["mol"] = df_sample["smiles"].apply(Chem.MolFromSmiles)\nvalid_count = df_sample["mol"].notna().sum()\nprint(f"Valid molecules in sample: {valid_count} / {len(df_sample)}")\n\n# Filter out invalid\ndf_valid = df_sample[df_sample["mol"].notna()].copy()'),
    ('md', '### Calculate QED and SAS'),
    ('code', 'df_valid["qed"] = df_valid["mol"].apply(calculate_qed)\ndf_valid["sas"] = df_valid["mol"].apply(calculate_sas)\n\nplt.hist(df_valid["qed"], bins=30, color="skyblue", edgecolor="black")\nplt.title("QED Distribution")\nplt.xlabel("QED")\nplt.ylabel("Frequency")\nplt.show()'),
    ('code', 'low_qed = len(df_valid[df_valid["qed"] < 0.5])\nmed_qed = len(df_valid[(df_valid["qed"] >= 0.5) & (df_valid["qed"] < 0.7)])\nhigh_qed = len(df_valid[df_valid["qed"] >= 0.7])\nprint(f"Low QED: {low_qed}\\nMedium QED: {med_qed}\\nHigh QED: {high_qed}")'),
    ('md', '### Save Processed Dataset and Statistics'),
    ('code', 'stats = {\n    "total_rows_processed": len(df_valid),\n    "low_qed_count": low_qed,\n    "med_qed_count": med_qed,\n    "high_qed_count": high_qed\n}\nwith open("../data/processed/dataset_stats.json", "w") as f:\n    json.dump(stats, f)\n\n# Save without the mol object\ndf_out = df_valid.drop(columns=["mol"])\ndf_out.to_csv("../data/processed/zinc_processed.csv", index=False)\nprint("Processed dataset and statistics saved to data/processed/")')
]
create_notebook('notebooks/01_dataset_analysis.ipynb', ds_cells)

# 2. Molecule Graph and Edits Notebook
graph_cells = [
    ('md', '# Phase 2 & 3: Molecular Graph and Edits\n\n### Objective\nDemonstrate how a SMILES is converted to a graph with PyTorch Geometric-compatible features, and apply edits.\n\n### Note on Edit Vocabulary\nThe mid-review edit vocabulary restricts added and substituted atoms to `C, N, O, F, S, Cl` to keep the action space manageable. There is no support for `Br` or `I` edits currently.'),
    ('code', 'from rdkit import Chem\nfrom rdkit.Chem import Draw\nimport torch\nimport sys\nimport os\nsys.path.append(os.path.abspath(".."))\nfrom implementation.molecule_editor import MoleculeEditor'),
    ('md', '### Convert RDKit Mol to Graph representation\nWe extract node features (element, formal charge, H count, aromaticity, ring membership) and edge features (bond type).'),
    ('code', 'def mol_to_graph_features(mol):\n    if mol is None:\n        return None\n    \n    # Node features\n    node_features = []\n    for atom in mol.GetAtoms():\n        feat = [\n            atom.GetAtomicNum(),\n            atom.GetFormalCharge(),\n            atom.GetTotalNumHs(),\n            int(atom.GetIsAromatic()),\n            int(atom.IsInRing())\n        ]\n        node_features.append(feat)\n        \n    # Edge features\n    edge_indices = []\n    edge_features = []\n    for bond in mol.GetBonds():\n        i = bond.GetBeginAtomIdx()\n        j = bond.GetEndAtomIdx()\n        bond_type = bond.GetBondTypeAsDouble()\n        \n        # Undirected graph (add both directions)\n        edge_indices.extend([[i, j], [j, i]])\n        edge_features.extend([[bond_type], [bond_type]])\n        \n    return {\n        "x": torch.tensor(node_features, dtype=torch.float),\n        "edge_index": torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long),\n        "edge_attr": torch.tensor(edge_features, dtype=torch.float)\n    }'),
    ('code', 'smiles = "CC(=O)Oc1ccccc1C(=O)O"\nmol = Chem.MolFromSmiles(smiles)\ngraph_data = mol_to_graph_features(mol)\n\nprint("Graph Dimensions:")\nprint(f"Nodes (x): {graph_data[\'x\'].shape} -> [num_atoms, num_node_features]")\nprint(f"Edges (edge_index): {graph_data[\'edge_index\'].shape} -> [2, num_directed_edges]")\nprint(f"Edge Attributes (edge_attr): {graph_data[\'edge_attr\'].shape} -> [num_directed_edges, num_edge_features]")\n\ndisplay(Draw.MolToImage(mol))'),
    ('md', '### Apply Molecular Edits\nLet\'s test the `MoleculeEditor`.'),
    ('code', 'editor = MoleculeEditor()\nprint(f"Allowed Elements for Editing: {editor.allowed_elements}")\n\n# 1. Add Atom (Fluorine to Carbon 3)\nnew_mol_1 = editor.add_atom(mol, atom_idx=3, new_element="F")\nif new_mol_1:\n    print("Added Fluorine:")\n    display(Draw.MolToImage(new_mol_1))'),
    ('code', '# 2. Remove Atom (Remove the newly added Fluorine)\nif new_mol_1:\n    f_idx = new_mol_1.GetNumAtoms() - 1\n    new_mol_2 = editor.remove_atom(new_mol_1, f_idx)\n    if new_mol_2:\n        print("Removed Fluorine:")\n        display(Draw.MolToImage(new_mol_2))')
]
create_notebook('notebooks/02_molecule_graph_and_edits.ipynb', graph_cells)

# 3. RL Environment Notebook
env_cells = [
    ('md', '# Phase 3: RL Environment\n\n### Objective\nDemonstrate the Gymnasium environment.\n\n### Note on Pending Integrations (Phase 6+)\nThis environment prototype correctly manages state transitions, step tracking, valid RDKit edits, and rewards. However, the final integration of the PyTorch Geometric / GNN-based graph state along with scalar properties directly into the observation space is pending and belongs to the next stage (Phase 6: GNN Encoder). Currently, the observation space only surfaces the scalar properties (QED, SAS, step, similarity) to test the transition loop.'),
    ('code', 'import sys\nimport os\nsys.path.append(os.path.abspath(".."))\nfrom implementation.environment import MolecularEnvironment\nfrom rdkit.Chem import Draw'),
    ('code', 'env = MolecularEnvironment()\nobs, info = env.reset(options={"starting_smiles": "c1ccccc1"})\nprint("Initial state (Scalars only pending Phase 6):", obs)\nDraw.MolToImage(env.current_mol)'),
    ('code', '# Take a step (Add atom Carbon to atom index 0)\naction = {"type": 0, "atom1": 0, "atom2": 0, "element": 0, "bond_delta": 1}\nobs, reward, done, _, info = env.step(action)\nprint("Next state (Scalars):", obs)\nprint("Reward:", reward)\nDraw.MolToImage(env.current_mol)')
]
create_notebook('notebooks/03_rl_environment.ipynb', env_cells)

# 4. Mid Review Demo Notebook
demo_cells = [
    ('md', '# MID-REVIEW DEMONSTRATION\n\n### Project: Reinforcement Learning for Drug-Likeness Optimization of Small Molecules\n\nThis notebook demonstrates the end-to-end pipeline (Phases 1-5). It focuses on the chemical logic and multi-objective reward calculations.\n\n**Note:** Added and substituted atoms are restricted to `C, N, O, F, S, Cl`. Support for `Br` and `I` is not currently included.'),
    ('code', 'import sys\nimport os\nimport pandas as pd\nsys.path.append(os.path.abspath(".."))\nfrom rdkit import Chem\nfrom rdkit.Chem import Draw\nfrom implementation.environment import MolecularEnvironment\nfrom implementation.properties import calculate_properties'),
    ('md', '### 1. Select a Molecule from ZINC'),
    ('code', 'try:\n    df = pd.read_csv("../data/processed/zinc_processed.csv", nrows=100)\nexcept FileNotFoundError:\n    # Fallback to original dataset if processed not yet saved\n    df = pd.read_csv("../ZINC-250K/250k_rndm_zinc_drugs_clean_3.csv", nrows=100)\n    \nstart_smiles = df.iloc[42]["smiles"].strip()\nstart_mol = Chem.MolFromSmiles(start_smiles)\nprint("Original SMILES:", start_smiles)\ndisplay(Draw.MolToImage(start_mol))'),
    ('md', '### 2. Calculate Initial Properties'),
    ('code', 'initial_props = calculate_properties(start_mol)\nprint(f"Original QED: {initial_props[\'qed\']:.4f}")\nprint(f"Original SAS: {initial_props[\'sas\']:.4f}")'),
    ('md', '### 3. Initialize Environment & Perform Edit\nWe will attempt to add a Fluorine atom to the first atom (index 0).'),
    ('code', 'env = MolecularEnvironment()\nobs, _ = env.reset(options={"starting_smiles": start_smiles})\n\n# Let\'s add a Fluorine (element index 3 in [C, N, O, F, S, Cl]) to atom 0\naction = {"type": 0, "atom1": 0, "atom2": 0, "element": 3, "bond_delta": 1}\nnext_obs, reward, done, _, info = env.step(action)\n\nprint("Valid Edit Attempted.")'),
    ('md', '### 4. Display Resulting Molecule & Properties'),
    ('code', 'new_mol = env.current_mol\nmodified_smiles = Chem.MolToSmiles(new_mol)\n\nprint("Modified SMILES:", modified_smiles)\nprint(f"New QED: {next_obs[\'qed\'][0]:.4f}")\nprint(f"New SAS: {next_obs[\'sas\'][0]:.4f}")\nprint(f"Tanimoto Similarity: {next_obs[\'similarity\'][0]:.4f}")\nprint(f"Chemically Valid: {info.get(\'invalid_penalty\') is None}")\nprint(f"Reward Received: {reward:.4f}")\n\ndisplay(Draw.MolToImage(new_mol))')
]
create_notebook('notebooks/mid_review_demo.ipynb', demo_cells)

print("Notebooks generated successfully.")
