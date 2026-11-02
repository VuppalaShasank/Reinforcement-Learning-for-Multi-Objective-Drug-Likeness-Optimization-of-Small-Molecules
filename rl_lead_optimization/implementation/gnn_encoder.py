import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool

# Import base molecular graph extraction from implementation
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from implementation.graph_utils import get_node_features, get_edge_features, mol_to_pyg_data


class MolecularGNNEncoder(nn.Module):
    """
    Three-Layer Graph Neural Network State Encoder for Molecular Graphs.
    
    Architecture:
    - Layer 1: GraphConv(21 -> 64) + BatchNorm1d(64) + ReLU
    - Layer 2: GraphConv(64 -> 64) + BatchNorm1d(64) + ReLU
    - Layer 3: GraphConv(64 -> 64) + BatchNorm1d(64) + ReLU
    - Readout: Concat(global_mean_pool, global_max_pool) -> Linear(128 -> 64) + ReLU
    - Fusion MLP: Linear(64 + 6 -> 128) -> ReLU -> Linear(128 -> 128) -> ReLU
    
    Inputs:
    - pyg_data: PyG Data or Batch object containing:
        x: [N, 21] Node features (element, formal charge, Hs, aromaticity, ring)
        edge_index: [2, E] Graph connectivity
        edge_attr: [E, 4] Bond types (Single, Double, Triple, Aromatic)
    - scalars: [B, 6] Scalar descriptors [t/T, QED_t, SAS_t, Sim(M_t, M_0), QED_0, SAS_0]
    
    Output:
    - Fixed-size fused state representation [B, 128]
    """
    def __init__(self, node_dim=21, edge_dim=4, gnn_hidden=64, scalar_dim=6, fused_dim=128):
        super(MolecularGNNEncoder, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.gnn_hidden = gnn_hidden
        self.scalar_dim = scalar_dim
        self.fused_dim = fused_dim
        
        # 3 Graph Convolution Message-Passing Layers
        self.conv1 = GraphConv(node_dim, gnn_hidden)
        self.conv2 = GraphConv(gnn_hidden, gnn_hidden)
        self.conv3 = GraphConv(gnn_hidden, gnn_hidden)
        
        # Batch Normalization for stability across molecular sizes
        self.bn1 = nn.BatchNorm1d(gnn_hidden)
        self.bn2 = nn.BatchNorm1d(gnn_hidden)
        self.bn3 = nn.BatchNorm1d(gnn_hidden)
        
        # Graph Readout Projection: Mean-pool (64) + Max-pool (64) = 128
        self.graph_proj = nn.Linear(gnn_hidden * 2, gnn_hidden)
        
        # Fusion MLP combining graph-level topological context with scalar state features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(gnn_hidden + scalar_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU()
        )

    def forward(self, pyg_data, scalars):
        """
        Forward pass for single molecular graph or mini-batch.
        """
        x, edge_index = pyg_data.x, pyg_data.edge_index
        batch = pyg_data.batch if hasattr(pyg_data, 'batch') and pyg_data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Message passing layers
        h1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index)))
        
        # Dual-pooling readout (mean + max)
        pool_mean = global_mean_pool(h3, batch)
        pool_max = global_max_pool(h3, batch)
        h_graph = torch.cat([pool_mean, pool_max], dim=-1)
        h_graph_proj = F.relu(self.graph_proj(h_graph))
        
        # Match dimensions for scalar descriptors
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)
            
        # Concatenate graph embedding and scalar properties
        combined = torch.cat([h_graph_proj, scalars], dim=-1)
        state_embed = self.fusion_mlp(combined)
        return state_embed


def molecule_to_graph_data(mol, current_step=0, max_steps=10, qed=0.0, sas=0.0, sim=1.0, q0=0.0, sas_0=0.0):
    """
    Converts an RDKit Mol object into a PyTorch Geometric Data object with 21 node features,
    4 edge features, and 6 scalar state descriptors.
    """
    return mol_to_pyg_data(
        mol=mol,
        current_step=current_step,
        max_steps=max_steps,
        qed=qed,
        sas=sas,
        sim=sim,
        q0=q0,
        sas_0=sas_0
    )
