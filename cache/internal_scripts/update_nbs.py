import json

p1 = 'c:/Users/vuppa/OneDrive/Desktop/RL_Sem7_Proj/notebooks/01_dataset_analysis.ipynb'
with open(p1, 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

for cell in nb1['cells']:
    if cell['cell_type'] == 'markdown':
        for i, line in enumerate(cell['source']):
            if 'We will take a sample for speed (use full dataset for final training).' in line:
                cell['source'][i] = line.replace(
                    'We will take a sample for speed (use full dataset for final training).',
                    'We will take a reproducible 1,000-molecule sample for fast mid-review analysis, while the complete 249,455-molecule dataset will be used for final preprocessing/training.'
                )
    elif cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if 'high_qed = len(df_valid[df_valid["qed"] >= 0.7])' in line:
                cell['source'][i] = line.replace(
                    'high_qed = len(df_valid[df_valid["qed"] >= 0.7])',
                    'high_qed = len(df_valid[(df_valid["qed"] >= 0.7) & (df_valid["qed"] < 0.85)])'
                )

with open(p1, 'w', encoding='utf-8') as f:
    json.dump(nb1, f, indent=1)

p2 = 'c:/Users/vuppa/OneDrive/Desktop/RL_Sem7_Proj/notebooks/02_molecule_graph_and_edits.ipynb'
with open(p2, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

for cell in nb2['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if '# 1. Add Atom (Fluorine to Carbon 3)' in line:
                cell['source'][i] = line.replace(
                    '# 1. Add Atom (Fluorine to Carbon 3)',
                    '# 1. Add Atom (Fluorine to Oxygen 3 - Invalid Action)'
                )
        if len(cell['source']) > 0 and '# 2. Remove Atom (Remove the newly added Fluorine)' in cell['source'][0]:
            cell['source'] = [
                '# 2. Add Atom (Fluorine to Carbon 0 - Valid Action)\n',
                'valid_mol = editor.add_atom(mol, atom_idx=0, new_element="F")\n',
                'if valid_mol:\n',
                '    print("Successfully Added Fluorine to Carbon 0:")\n',
                '    display(Draw.MolToImage(valid_mol))\n',
                'else:\n',
                '    print("Action rejected: Invalid molecule")\n'
            ]

new_cell = {
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
    '# 3. Remove Atom (Remove the newly added Fluorine)\n',
    'if valid_mol:\n',
    '    f_idx = valid_mol.GetNumAtoms() - 1\n',
    '    removed_mol = editor.remove_atom(valid_mol, f_idx)\n',
    '    if removed_mol:\n',
    '        print("Successfully Removed Fluorine:")\n',
    '        display(Draw.MolToImage(removed_mol))\n'
   ]
}
nb2['cells'].append(new_cell)

with open(p2, 'w', encoding='utf-8') as f:
    json.dump(nb2, f, indent=1)
