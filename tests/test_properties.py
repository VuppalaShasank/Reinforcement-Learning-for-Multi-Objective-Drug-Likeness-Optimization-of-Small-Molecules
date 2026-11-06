import unittest
from rdkit import Chem
from src.chemistry.properties import calculate_properties, calculate_qed, calculate_sas, calculate_similarity

class TestChemistryProperties(unittest.TestCase):
    def setUp(self):
        self.aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
        self.aspirin = Chem.MolFromSmiles(self.aspirin_smiles)
        
    def test_qed_calculation(self):
        qed_val = calculate_qed(self.aspirin)
        self.assertGreater(qed_val, 0.0)
        self.assertLessEqual(qed_val, 1.0)
        
    def test_sas_calculation(self):
        sas_val = calculate_sas(self.aspirin)
        self.assertGreaterEqual(sas_val, 1.0)
        self.assertLessEqual(sas_val, 10.0)
        
    def test_similarity_identical(self):
        sim = calculate_similarity(self.aspirin, self.aspirin)
        self.assertAlmostEqual(sim, 1.0, places=4)
        
    def test_calculate_properties_dict(self):
        props = calculate_properties(self.aspirin, self.aspirin)
        self.assertTrue(props['valid'])
        self.assertIn('qed', props)
        self.assertIn('sas', props)
        self.assertIn('similarity', props)

if __name__ == '__main__':
    unittest.main()

