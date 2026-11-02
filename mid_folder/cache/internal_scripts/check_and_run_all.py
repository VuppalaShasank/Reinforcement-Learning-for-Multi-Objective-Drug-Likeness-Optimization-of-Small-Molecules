import os
import sys
import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def patch_notebook_imports(nb_path):
    print(f"Patching imports for {os.path.basename(nb_path)}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    for cell in nb.cells:
        if cell.cell_type == "code":
            # Check if import torch is present as first import
            code = cell.source
            if "import torch" not in code:
                # Add import torch at the very beginning
                cell.source = "import torch\n" + code
            break
            
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

def update_notebook_04():
    p = os.path.join(WORKSPACE, "notebooks", "04_gnn_ppo_training.ipynb")
    nb = nbformat.v4.new_notebook()
    
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# Notebook 4: 3-Layer GNN Encoder & PPO Actor-Critic Policy Training\n\n"
            "### Objectives\n"
            "1. Inspect the 3-Layer Graph Neural Network encoder architecture and feature dimensions.\n"
            "2. Perform forward pass through the Molecular Actor-Critic network with action masking.\n"
            "3. Load trained PPO policy weights (`checkpoints/ppo_adaptive.pt` and `ppo_baseline.pt`).\n"
            "4. Analyze real training return curves and Delta-QED progression across training episodes."
        ),
        nbformat.v4.new_code_cell(
            "import torch\n"
            "import os\n"
            "import sys\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from PIL import Image\n\n"
            "# Ensure project root is in sys.path\n"
            "root_dir = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.path.abspath('.')\n"
            "if root_dir not in sys.path:\n"
            "    sys.path.insert(0, root_dir)\n\n"
            "from rdkit import Chem\n"
            "from implementation.graph_utils import mol_to_pyg_data\n"
            "from implementation.gnn_encoder import MolecularGNNEncoder\n"
            "from implementation.actor_critic import MolecularActorCritic\n"
            "from implementation.ppo_agent import PPOAgent\n"
            "from implementation.environment import MolecularGraphEnv\n\n"
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
            "print(f'PyTorch Version: {torch.__version__}, Active Device: {device}')"
        ),
        nbformat.v4.new_markdown_cell(
            "### 1. 3-Layer GNN Encoder Architecture & Forward Pass\n"
            "The GNN processes the molecular graph with 21 node features and 4 bond features."
        ),
        nbformat.v4.new_code_cell(
            "mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O') # Aspirin\n"
            "data = mol_to_pyg_data(mol, current_step=0, max_steps=10, qed=0.55, sas=1.6, sim=1.0, q0=0.55, sas_0=1.6).to(device)\n"
            "scalars = data.scalars.to(device)\n\n"
            "encoder = MolecularGNNEncoder().to(device)\n"
            "embedding = encoder(data, scalars)\n"
            "print(f'Number of atoms: {data.x.shape[0]}, Node feature dim: {data.x.shape[1]}')\n"
            "print(f'Edge index shape: {data.edge_index.shape}, Edge feature dim: {data.edge_attr.shape[1]}')\n"
            "print(f'Global graph representation shape: {embedding.shape} (Expected: [1, 128])')"
        ),
        nbformat.v4.new_markdown_cell(
            "### 2. Molecular Actor-Critic Policy & Value Heads\n"
            "Evaluates action distribution over the 1,901-action discrete space and predicts state value."
        ),
        nbformat.v4.new_code_cell(
            "actor_critic = MolecularActorCritic().to(device)\n"
            "logits, value = actor_critic(data, scalars)\n"
            "print(f'Actor output logits shape: {logits.shape} (Total actions: 1,901)')\n"
            "print(f'Critic estimated state value V(s): {value.item():.4f}')"
        ),
        nbformat.v4.new_markdown_cell(
            "### 3. Load Trained Model Weights & Inspection\n"
            "Loads the trained checkpoint from `checkpoints/`."
        ),
        nbformat.v4.new_code_cell(
            "ckpt_path = os.path.join(root_dir, 'checkpoints', 'ppo_adaptive.pt')\n"
            "if not os.path.exists(ckpt_path):\n"
            "    ckpt_path = os.path.join(root_dir, 'cache', 'checkpoints', 'ppo_adaptive.pt')\n\n"
            "agent = PPOAgent(device=device)\n"
            "meta = agent.load_checkpoint(ckpt_path)\n"
            "print(f'Successfully loaded PPO Adaptive checkpoint! Metadata: {meta}')"
        ),
        nbformat.v4.new_markdown_cell(
            "### 4. Training Learning Curves\n"
            "Inspect episode returns and Delta-QED progression comparing Baseline vs Adaptive reward shaping."
        ),
        nbformat.v4.new_code_cell(
            "fig_path = os.path.join(root_dir, 'figures', 'training_reward_curves.png')\n"
            "if os.path.exists(fig_path):\n"
            "    img = Image.open(fig_path)\n"
            "    plt.figure(figsize=(12, 5))\n"
            "    plt.imshow(img)\n"
            "    plt.axis('off')\n"
            "    plt.title('PPO Training Return & Delta-QED Curves', fontsize=14)\n"
            "    plt.show()"
        )
    ]
    with open(p, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook 04 updated successfully.")

def update_notebook_05():
    p = os.path.join(WORKSPACE, "notebooks", "05_baseline_comparisons.ipynb")
    nb = nbformat.v4.new_notebook()
    
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# Notebook 5: Baseline Comparisons on Held-Out Scaffold Benchmark\n\n"
            "### Objectives\n"
            "1. Evaluate PPO Baseline and PPO Adaptive against Random Valid Edits and Greedy Hill Climbing on the exact same 60 test molecules.\n"
            "2. Compare performance across key metrics: Delta-QED, Normalized Headroom Gain (NHG), Success Rate, Structural Similarity, and SAS change."
        ),
        nbformat.v4.new_code_cell(
            "import torch\n"
            "import os\n"
            "import sys\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from PIL import Image\n\n"
            "root_dir = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.path.abspath('.')\n"
            "if root_dir not in sys.path:\n"
            "    sys.path.insert(0, root_dir)\n\n"
            "summary_path = os.path.join(root_dir, 'results', 'method_comparison_summary.csv')\n"
            "df_sum = pd.read_csv(summary_path)\n"
            "print('=== Benchmark Method Comparison (60 Held-Out Scaffolds) ===')\n"
            "display(df_sum)"
        ),
        nbformat.v4.new_markdown_cell(
            "### 1. Benchmark Comparison Plots\n"
            "Visualizing Delta-QED, Success Rate, and Tanimoto Structural Similarity preservation."
        ),
        nbformat.v4.new_code_cell(
            "fig_path = os.path.join(root_dir, 'figures', 'evaluation_delta_qed_and_success.png')\n"
            "if os.path.exists(fig_path):\n"
            "    img = Image.open(fig_path)\n"
            "    plt.figure(figsize=(12, 5))\n"
            "    plt.imshow(img)\n"
            "    plt.axis('off')\n"
            "    plt.title('Optimization Delta-QED & Success Rate by Method', fontsize=14)\n"
            "    plt.show()"
        ),
        nbformat.v4.new_markdown_cell(
            "### 2. Sample Optimized Molecules\n"
            "Inspect sample test molecules optimized by PPO Adaptive."
        ),
        nbformat.v4.new_code_cell(
            "eval_csv = os.path.join(root_dir, 'results', 'final_evaluation_results.csv')\n"
            "df_eval = pd.read_csv(eval_csv)\n"
            "sample_df = df_eval[df_eval['method'] == 'PPO Adaptive (Proposed)'].head(5)\n"
            "display(sample_df[['initial_smiles', 'final_smiles', 'initial_qed', 'final_qed', 'delta_qed', 'similarity', 'success']])"
        )
    ]
    with open(p, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook 05 updated successfully.")

def update_notebook_06():
    p = os.path.join(WORKSPACE, "notebooks", "06_evaluation_and_reward_hacking_audit.ipynb")
    nb = nbformat.v4.new_notebook()
    
    nb.cells = [
        nbformat.v4.new_markdown_cell(
            "# Notebook 6: QED Stratification & Held-Out Reward-Hacking Audit\n\n"
            "### Objectives\n"
            "1. Examine QED-stratified performance across initial headroom bins (Low, Medium, High, Very High).\n"
            "2. Audit against reward-hacking using held-out filters: Lipinski Rule of 5, PAINS alerts, Brenk alerts, and structural sanity."
        ),
        nbformat.v4.new_code_cell(
            "import torch\n"
            "import os\n"
            "import sys\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from PIL import Image\n\n"
            "root_dir = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.path.abspath('.')\n"
            "if root_dir not in sys.path:\n"
            "    sys.path.insert(0, root_dir)\n\n"
            "strat_path = os.path.join(root_dir, 'results', 'qed_stratified_summary.csv')\n"
            "df_strat = pd.read_csv(strat_path)\n"
            "print('=== QED-Stratified Results by Starting Headroom ===')\n"
            "display(df_strat)"
        ),
        nbformat.v4.new_markdown_cell(
            "### 1. Starting Headroom Comparison Chart\n"
            "Shows how Adaptive reward shaping avoids saturation on high-QED molecules."
        ),
        nbformat.v4.new_code_cell(
            "fig_strat = os.path.join(root_dir, 'figures', 'qed_stratified_comparison.png')\n"
            "if os.path.exists(fig_strat):\n"
            "    img = Image.open(fig_strat)\n"
            "    plt.figure(figsize=(10, 4))\n"
            "    plt.imshow(img)\n"
            "    plt.axis('off')\n"
            "    plt.title('Delta QED Gain Stratified by Initial QED', fontsize=14)\n"
            "    plt.show()"
        ),
        nbformat.v4.new_markdown_cell(
            "### 2. Reward-Hacking Audit Table & Chart\n"
            "Evaluates whether policies generate structural alerts or un-druglike motifs."
        ),
        nbformat.v4.new_code_cell(
            "audit_path = os.path.join(root_dir, 'results', 'reward_hacking_audit.csv')\n"
            "df_audit = pd.read_csv(audit_path)\n"
            "print('=== Structural Alert & Druglikeness Audit ===')\n"
            "display(df_audit)\n\n"
            "fig_audit = os.path.join(root_dir, 'figures', 'reward_hacking_audit_chart.png')\n"
            "if os.path.exists(fig_audit):\n"
            "    img = Image.open(fig_audit)\n"
            "    plt.figure(figsize=(10, 4))\n"
            "    plt.imshow(img)\n"
            "    plt.axis('off')\n"
            "    plt.title('PAINS & Brenk Alert Rates Across Methods', fontsize=14)\n"
            "    plt.show()"
        )
    ]
    with open(p, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print("Notebook 06 updated successfully.")

def execute_notebook(nb_path):
    print(f"\n==========================================")
    print(f"Executing: {os.path.basename(nb_path)}")
    print(f"==========================================")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    notebook_dir = os.path.dirname(nb_path)
    
    try:
        ep.preprocess(nb, {"metadata": {"path": notebook_dir}})
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"SUCCESS: {os.path.basename(nb_path)} executed completely with 0 errors!")
        return True
    except Exception as e:
        print(f"ERROR in {os.path.basename(nb_path)}: {e}")
        return False

if __name__ == "__main__":
    # 1. Update notebooks 04, 05, 06
    update_notebook_04()
    update_notebook_05()
    update_notebook_06()
    
    # 2. Patch imports for 01, 02, 03, mid_review_demo
    for nb_name in ["01_dataset_analysis.ipynb", "02_molecule_graph_and_edits.ipynb", "03_rl_environment.ipynb", "mid_review_demo.ipynb"]:
        patch_notebook_imports(os.path.join(WORKSPACE, "notebooks", nb_name))
        
    # 3. Execute all notebooks sequentially
    all_nbs = [
        "01_dataset_analysis.ipynb",
        "02_molecule_graph_and_edits.ipynb",
        "03_rl_environment.ipynb",
        "04_gnn_ppo_training.ipynb",
        "05_baseline_comparisons.ipynb",
        "06_evaluation_and_reward_hacking_audit.ipynb",
        "mid_review_demo.ipynb"
    ]
    
    results = {}
    for nb in all_nbs:
        p = os.path.join(WORKSPACE, "notebooks", nb)
        results[nb] = execute_notebook(p)
        
    print("\n\n==========================================")
    print("SUMMARY OF NOTEBOOK EXECUTIONS:")
    print("==========================================")
    all_passed = True
    for nb, status in results.items():
        print(f"  {nb}: {'PASSED' if status else 'FAILED'}")
        if not status:
            all_passed = False
            
    if all_passed:
        print("\nALL 7 NOTEBOOKS EXECUTED CLEANLY AND SUCCESSFULLY!")
    else:
        print("\nSOME NOTEBOOKS FAILED. CHECK LOGS ABOVE.")
