import unittest
from rdkit import Chem
from src.chemistry.molecule_editor import MoleculeEditor
from src.environment.action_masking import apply_action, TOTAL_ACTIONS

class TestMoleculeEditor(unittest.TestCase):
    def setUp(self):
        self.editor = MoleculeEditor()
        self.smiles = "c1ccccc1" # Benzene
        self.mol = Chem.MolFromSmiles(self.smiles)

    def test_total_actions(self):
        self.assertEqual(TOTAL_ACTIONS, 1901)
        self.assertEqual(len(self.editor.allowed_elements), 6)

    def test_stop_action(self):
        # Action 0 is STOP
        res = apply_action(self.mol, 0, self.editor)
        self.assertIsNotNone(res)
        self.assertEqual(Chem.MolToSmiles(res), Chem.MolToSmiles(self.mol))

if __name__ == '__main__':
    unittest.main()
