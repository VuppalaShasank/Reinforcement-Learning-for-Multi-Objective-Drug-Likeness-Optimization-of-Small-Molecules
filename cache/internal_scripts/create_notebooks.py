import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def make_notebook(filepath, cells_spec):
    nb = nbf.v4.new_notebook()
    cells = []
    for cell_type, content in cells_spec:
        if cell_type == "md":
            cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            cells.append(nbf.v4.new_code_cell(content))
    nb["cells"] = cells
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created {filepath}")
    return filepath

def execute_notebook_file(filepath):
    print(f"Executing {filepath}...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    with open(filepath, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(filepath)}})
        with open(filepath, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Successfully executed and updated {filepath}")
    except Exception as e:
        print(f"Error executing {filepath}: {e}")

def generate_all_notebooks():
    nb_dir = os.path.join("cache", "notebooks")
    os.makedirs(nb_dir, exist_ok=True)

    # 1. Dataset Preprocessing Notebook
    nb1_cells = [
        ("md", "# Notebook 1: ZINC-250K Preprocessing & Scaffold Splitting\n"
               "**Project:** Reinforcement Learning for Drug-Likeness Optimization of Small Molecules\n"
               "**Course:** 22AIE401 – Reinforcement Learning\n\n"
               "### Objective\n"
               "1. Inspect the ZINC-250K dataset.\n"
               "2. Recompute genuine QED and SAS scores using RDKit.\n"
               "3. Partition the dataset using Bemis-Murcko scaffolds into 80% Train, 10% Validation, and 10% Test splits (zero scaffold overlap).\n"
               "4. Stratify test molecules into 4 QED headroom groups: Low, Medium, High, and Very High."),
        ("code", "import os, sys, json\n"
                 "import pandas as pd\n"
                 "import matplotlib.pyplot as plt\n"
                 "sys.path.insert(0, os.path.abspath('../..'))\n"
                 "from cache.implementation.data_pipeline import prepare_scaffold_splits\n"
                 "ds_path = os.path.join('../..', 'ZINC-250K', '250k_rndm_zinc_drugs_clean_3.csv')\n"
                 "out_dir = os.path.join('../..', 'cache', 'results')\n"
                 "train_df, val_df, bench_df = prepare_scaffold_splits(ds_path, out_dir, sample_size=1000)\n"
                 "print('Benchmark molecules sample:')\n"
                 "bench_df[['smiles', 'qed', 'sas', 'qed_group']].head()"),
        ("code", "plt.figure(figsize=(8, 4))\n"
                 "plt.hist(train_df['qed'], bins=25, alpha=0.7, color='steelblue', label='Train QED')\n"
                 "plt.hist(bench_df['qed'], bins=25, alpha=0.7, color='coral', label='Test Benchmark QED')\n"
                 "plt.title('QED Distribution across Train & Test Benchmark')\n"
                 "plt.xlabel('QED')\n"
                 "plt.ylabel('Count')\n"
                 "plt.legend()\n"
                 "plt.grid(True, alpha=0.3)\n"
                 "plt.show()")
    ]
    make_notebook(os.path.join(nb_dir, "01_dataset_preprocessing.ipynb"), nb1_cells)

    # 2. Molecular Graph and Edits Notebook
    nb2_cells = [
        ("md", "# Notebook 2: Molecular Graph Representation & Valence Masked Edits\n\n"
               "### Objective\n"
               "1. Convert SMILES to PyTorch Geometric `Data` graphs with 21 node features and 4 edge features.\n"
               "2. Demonstrate valid atomic additions, removals, substitutions, and bond changes.\n"
               "3. Illustrate chemical valence pre-action masking across the 1901 discrete action vocabulary."),
        ("code", "import os, sys, torch\n"
                 "from rdkit import Chem\n"
                 "from rdkit.Chem import Draw\n"
                 "sys.path.insert(0, os.path.abspath('../..'))\n"
                 "from cache.implementation.graph_utils import mol_to_pyg_data\n"
                 "from cache.implementation.action_handler import get_action_mask, action_to_details, apply_action\n"
                 "from cache.implementation.action_handler import TOTAL_ACTIONS\n\n"
                 "smiles = 'CC(=O)Oc1ccccc1C(=O)O' # Aspirin\n"
                 "mol = Chem.MolFromSmiles(smiles)\n"
                 "data = mol_to_pyg_data(mol, current_step=0, max_steps=10, qed=0.55, sas=1.6)\n"
                 "print(f'PyG Data: Nodes x={data.x.shape}, Edge Index={data.edge_index.shape}, Edge Attr={data.edge_attr.shape}')\n"
                 "mask = get_action_mask(mol)\n"
                 "print(f'Total Actions: {TOTAL_ACTIONS}, Valid Actions: {mask.sum()}')\n"
                 "Draw.MolToImage(mol)")
    ]
    make_notebook(os.path.join(nb_dir, "02_molecular_graph_and_edits.ipynb"), nb2_cells)

    # 3. Environment and Rewards Notebook
    nb3_cells = [
        ("md", "# Notebook 3: Gymnasium Molecular Environment & Adaptive Reward Shaping\n\n"
               "### Objective\n"
               "1. Test `MolecularGraphEnv` reset and step transitions.\n"
               "2. Compare the Baseline Potential-Based Reward against the Proposed Novel Headroom- & Constraint-Slack Adaptive Reward.\n"
               "3. Visualize barrier penalty behavior near similarity ($\tau_{sim}=0.4$) and SAS ($\tau_{SAS}=4.5$) boundaries."),
        ("code", "import os, sys\n"
                 "import matplotlib.pyplot as plt\n"
                 "from PIL import Image\n"
                 "sys.path.insert(0, os.path.abspath('../..'))\n"
                 "from cache.implementation.environment import MolecularGraphEnv\n\n"
                 "env = MolecularGraphEnv(reward_mode='adaptive', max_steps=5)\n"
                 "obs, info = env.reset(options={'starting_smiles': 'c1ccccc1'})\n"
                 "print(f'Reset Benzene: QED={info[\"qed\"]:.4f}, SAS={info[\"sas\"]:.4f}')\n"
                 "obs, rew, term, trunc, sinfo = env.step(1) # Add C\n"
                 "print(f'Step 1: Reward={rew:.4f}, New SMILES={sinfo.get(\"smiles\")}')\n\n"
                 "diag_img = os.path.join('../..', 'cache', 'figures', 'reward_comparison_diagnostics.png')\n"
                 "if os.path.exists(diag_img):\n"
                 "    img = Image.open(diag_img)\n"
                 "    plt.figure(figsize=(10, 4))\n"
                 "    plt.imshow(img)\n"
                 "    plt.axis('off')\n"
                 "    plt.show()")
    ]
    make_notebook(os.path.join(nb_dir, "03_environment_and_rewards.ipynb"), nb3_cells)

    # 4. GNN and PPO Training Notebook
    nb4_cells = [
        ("md", "# Notebook 4: 3-Layer GNN Encoder & PPO Actor-Critic Policy Training\n\n"
               "### Objective\n"
               "1. Demonstrate forward pass of the 3-Layer GNN encoder and Actor-Critic heads.\n"
               "2. Inspect real PPO training logs and comparative learning curves for Baseline vs Adaptive reward modes.\n"
               "3. Verify checkpoint loading from `cache/checkpoints/`."),
        ("code", "import os, sys, torch\n"
                 "import pandas as pd\n"
                 "import matplotlib.pyplot as plt\n"
                 "from PIL import Image\n"
                 "sys.path.insert(0, os.path.abspath('../..'))\n"
                 "from cache.implementation.ppo_agent import PPOAgent\n\n"
                 "ckpt_path = os.path.join('../..', 'cache', 'checkpoints', 'ppo_adaptive.pt')\n"
                 "agent = PPOAgent()\n"
                 "meta = agent.load_checkpoint(ckpt_path)\n"
                 "print(f'Successfully loaded PPO checkpoint: {meta}')\n\n"
                 "curve_img = os.path.join('../..', 'cache', 'figures', 'training_reward_curves.png')\n"
                 "if os.path.exists(curve_img):\n"
                 "    img = Image.open(curve_img)\n"
                 "    plt.figure(figsize=(12, 5))\n"
                 "    plt.imshow(img)\n"
                 "    plt.axis('off')\n"
                 "    plt.show()")
    ]
    make_notebook(os.path.join(nb_dir, "04_gnn_ppo_training.ipynb"), nb4_cells)

    # 5. Baseline Comparisons Notebook
    nb5_cells = [
        ("md", "# Notebook 5: Baseline Comparisons on Held-Out Scaffold Benchmark\n\n"
               "### Objective\n"
               "1. Compare PPO Baseline and PPO Adaptive against Random Valid Edits and Greedy Hill Climbing on the exact same 60 test molecules.\n"
               "2. Inspect performance metrics: $\Delta$QED, Normalized Headroom Gain (NHG), Success Rate, Structural Similarity, and SAS change."),
        ("code", "import os, sys\n"
                 "import pandas as pd\n"
                 "import matplotlib.pyplot as plt\n"
                 "from PIL import Image\n\n"
                 "summary_path = os.path.join('../..', 'cache', 'results', 'method_comparison_summary.csv')\n"
                 "df_sum = pd.read_csv(summary_path)\n"
                 "display(df_sum[['method', 'count', 'mean_delta_qed', 'mean_nhg', 'success_rate', 'mean_similarity', 'mean_delta_sas']])\n\n"
                 "fig_path = os.path.join('../..', 'cache', 'figures', 'evaluation_delta_qed_and_success.png')\n"
                 "if os.path.exists(fig_path):\n"
                 "    img = Image.open(fig_path)\n"
                 "    plt.figure(figsize=(12, 5))\n"
                 "    plt.imshow(img)\n"
                 "    plt.axis('off')\n"
                 "    plt.show()")
    ]
    make_notebook(os.path.join(nb_dir, "05_baseline_comparisons.ipynb"), nb5_cells)

    # 6. Evaluation and Reward Hacking Audit Notebook
    nb6_cells = [
        ("md", "# Notebook 6: QED Stratification & Held-Out Reward-Hacking Audit\n\n"
               "### Objective\n"
               "1. Examine QED-stratified performance across starting headroom (Low, Medium, High, Very High).\n"
               "2. Perform reward-hacking audit using held-out signals: Lipinski Rule of 5 violations, PAINS alerts, Brenk alerts, and structural sanity checks."),
        ("code", "import os, sys\n"
                 "import pandas as pd\n"
                 "import matplotlib.pyplot as plt\n"
                 "from PIL import Image\n\n"
                 "strat_path = os.path.join('../..', 'cache', 'results', 'qed_stratified_summary.csv')\n"
                 "audit_path = os.path.join('../..', 'cache', 'results', 'reward_hacking_audit.csv')\n"
                 "print('=== Reward-Hacking Audit Summary ===')\n"
                 "display(pd.read_csv(audit_path))\n\n"
                 "fig1 = os.path.join('../..', 'cache', 'figures', 'qed_stratified_comparison.png')\n"
                 "fig2 = os.path.join('../..', 'cache', 'figures', 'reward_hacking_audit_chart.png')\n"
                 "for p in [fig1, fig2]:\n"
                 "    if os.path.exists(p):\n"
                 "        plt.figure(figsize=(10, 4))\n"
                 "        plt.imshow(Image.open(p))\n"
                 "        plt.axis('off')\n"
                 "        plt.show()")
    ]
    make_notebook(os.path.join(nb_dir, "06_evaluation_and_reward_hacking_audit.ipynb"), nb6_cells)

    print("All 6 notebooks created successfully under cache/notebooks/")

if __name__ == "__main__":
    generate_all_notebooks()
