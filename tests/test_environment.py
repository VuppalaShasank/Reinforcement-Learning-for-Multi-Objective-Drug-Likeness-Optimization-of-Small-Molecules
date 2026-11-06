import unittest
from rdkit import Chem
from src.environment.molecular_environment import MolecularGraphEnv

class TestMolecularEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = MolecularGraphEnv(reward_mode="adaptive", max_steps=5)
        self.smiles = "CC(=O)Oc1ccccc1C(=O)O"

    def test_env_reset(self):
        obs, info = self.env.reset(options={"starting_smiles": self.smiles})
        self.assertIn("pyg_data", obs)
        self.assertIn("scalars", obs)
        self.assertIn("action_mask", obs)
        self.assertEqual(len(obs["action_mask"]), 1901)
        self.assertEqual(info["smiles"], self.smiles)

    def test_env_step_stop(self):
        self.env.reset(options={"starting_smiles": self.smiles})
        obs, reward, terminated, truncated, info = self.env.step(0) # STOP
        self.assertTrue(terminated)
        self.assertFalse(truncated)

if __name__ == '__main__':
    unittest.main()

