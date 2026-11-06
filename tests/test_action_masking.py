import unittest
from rdkit import Chem
from src.environment.action_masking import get_action_mask, TOTAL_ACTIONS
from src.chemistry.molecule_editor import MoleculeEditor

class TestActionMasking(unittest.TestCase):
    def setUp(self):
        self.editor = MoleculeEditor()
        self.mol = Chem.MolFromSmiles("c1ccccc1")

    def test_mask_shape(self):
        mask = get_action_mask(self.mol, self.editor)
        self.assertEqual(len(mask), TOTAL_ACTIONS)
        self.assertEqual(len(mask), 1901)

    def test_stop_is_always_valid(self):
        mask = get_action_mask(self.mol, self.editor)
        self.assertTrue(mask[0]) # STOP must always be valid

    def test_has_valid_actions(self):
        mask = get_action_mask(self.mol, self.editor)
        self.assertGreater(mask.sum(), 1)

if __name__ == '__main__':
    unittest.main()

