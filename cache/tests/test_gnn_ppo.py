import os
import sys
import torch
import numpy as np
from rdkit import Chem

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from implementation.graph_utils import mol_to_pyg_data
from implementation.gnn_encoder import MolecularGNNEncoder
from implementation.actor_critic import MolecularActorCritic
from implementation.ppo_agent import PPOAgent
from implementation.evaluator import audit_molecule, check_lipinski_violations, check_pains_alert

def test_gnn_and_ppo():
    print("=== Testing 3-Layer GNN Encoder ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O") # Aspirin
    data = mol_to_pyg_data(mol, current_step=2, max_steps=10, qed=0.55, sas=1.6, sim=0.9, q0=0.55, sas_0=1.6).to(device)
    scalars = data.scalars.to(device)

    encoder = MolecularGNNEncoder().to(device)
    embedding = encoder(data, scalars)
    print(f"GNN Embedding shape: {embedding.shape} (Expected [1, 128])")
    assert embedding.shape == (1, 128), "GNN output dimension mismatch"

    print("\n=== Testing Molecular Actor-Critic ===")
    ac = MolecularActorCritic().to(device)
    mask = torch.ones((1, 1901), dtype=torch.bool, device=device)
    # Mask out some arbitrary actions
    mask[0, 5:50] = False
    
    action, log_prob, entropy, value = ac.get_action_and_value(data, scalars, mask)
    print(f"Sampled Action: {action.item()}, LogProb: {log_prob.item():.4f}, Value: {value.item():.4f}, Entropy: {entropy.item():.4f}")
    assert action.item() not in range(5, 50), "Masked actions must never be sampled!"

    print("\n=== Testing PPO Agent Update & Checkpointing ===")
    agent = PPOAgent(device=device)
    
    # Create a small synthetic rollout
    rollout = {
        "pyg_data_list": [mol_to_pyg_data(mol, current_step=i, max_steps=10) for i in range(4)],
        "scalars": torch.randn((4, 6)),
        "action_masks": torch.ones((4, 1901), dtype=torch.bool),
        "actions": torch.tensor([0, 1, 2, 0], dtype=torch.long),
        "old_log_probs": torch.tensor([-0.5, -1.2, -0.8, -0.1]),
        "returns": torch.tensor([1.0, 0.5, -0.2, 0.0]),
        "advantages": torch.tensor([0.2, -0.1, 0.4, -0.5]),
        "values": torch.tensor([0.8, 0.6, -0.6, 0.5])
    }
    
    loss_metrics = agent.update(rollout, ppo_epochs=2, batch_size=2)
    print("PPO Mini-Batch Update Metrics:", loss_metrics)
    assert "policy_loss" in loss_metrics and "value_loss" in loss_metrics
    assert np.isfinite(loss_metrics["policy_loss"]) and np.isfinite(loss_metrics["value_loss"])

    # Test checkpoint save and load
    ckpt_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "checkpoints", "test_smoke_ckpt.pt")
    agent.save_checkpoint(ckpt_path, metadata={"test": True, "step": 10})
    assert os.path.exists(ckpt_path), "Checkpoint file was not saved"
    meta = agent.load_checkpoint(ckpt_path)
    print(f"Checkpoint loaded successfully with metadata: {meta}")

    print("\n=== Testing Reward-Hacking Audit Functions ===")
    audit_res = audit_molecule(mol)
    print(f"Aspirin Audit: Lipinski={audit_res['lipinski_violations']}, PAINS={audit_res['pains_alert']}, Brenk={audit_res['brenk_alert']}, Sanity={audit_res['sanity_passed']}")
    assert audit_res["valid"] is True
    assert audit_res["lipinski_violations"] == 0, "Aspirin should have 0 Lipinski violations"

    print("\nALL GNN, PPO, AND AUDIT UNIT TESTS PASSED!")

if __name__ == "__main__":
    test_gnn_and_ppo()
