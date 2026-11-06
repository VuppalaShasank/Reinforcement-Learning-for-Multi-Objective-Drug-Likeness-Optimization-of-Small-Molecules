import unittest
import torch
from rdkit import Chem
from src.environment.molecular_environment import MolecularGraphEnv
from src.agents.ppo_agent_v2 import ImprovedPPOAgent

class TestPPOAgent(unittest.TestCase):
    def setUp(self):
        self.env = MolecularGraphEnv(reward_mode="adaptive", max_steps=5)
        self.agent = ImprovedPPOAgent(
            node_dim=21, edge_dim=4, gnn_hidden=64, scalar_dim=6, fused_dim=128
        )
        self.obs, _ = self.env.reset(options={"starting_smiles": "CC(=O)Oc1ccccc1C(=O)O"})

    def test_select_action(self):
        action, log_prob, val_est = self.agent.select_action(self.obs, deterministic=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 1901)
        self.assertTrue(self.obs["action_mask"][action])

    def test_top_k_actions(self):
        top_k = self.agent.get_top_k_actions(self.obs, k=5)
        self.assertLessEqual(len(top_k), 5)
        for act in top_k:
            self.assertTrue(self.obs["action_mask"][act])

if __name__ == '__main__':
    unittest.main()

