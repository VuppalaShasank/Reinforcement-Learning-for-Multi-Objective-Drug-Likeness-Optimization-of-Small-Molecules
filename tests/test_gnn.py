import unittest
import torch
from rdkit import Chem
from src.chemistry.graph_utils import mol_to_pyg_data
from src.models.gnn_encoder import MolecularGNNEncoder
from src.models.gnn_encoder_v2 import EdgeAwareMolecularGNNEncoder

class TestGNNEncoders(unittest.TestCase):
    def setUp(self):
        self.mol = Chem.MolFromSmiles("c1ccccc1O")
        self.pyg = mol_to_pyg_data(self.mol)
        self.scalars = torch.tensor([[0.1, 0.5, 3.0, 1.0, 0.5, 3.0]], dtype=torch.float32)

    def test_legacy_gnn_forward(self):
        encoder = MolecularGNNEncoder(node_dim=21, edge_dim=4, gnn_hidden=64, scalar_dim=6, fused_dim=128)
        out = encoder(self.pyg, self.scalars)
        self.assertEqual(out.shape, (1, 128))

    def test_edge_aware_gnn_forward(self):
        encoder = EdgeAwareMolecularGNNEncoder(node_dim=21, edge_dim=4, gnn_hidden=64, scalar_dim=6, fused_dim=128)
        out = encoder(self.pyg, self.scalars)
        self.assertEqual(out.shape, (1, 128))

if __name__ == '__main__':
    unittest.main()
