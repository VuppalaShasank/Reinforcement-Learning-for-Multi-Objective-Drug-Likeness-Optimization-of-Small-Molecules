import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.chemistry.graph_utils import get_node_features, get_edge_features, mol_to_pyg_data


class EdgeAwareMessagePassing(MessagePassing):
    """
    Message-Passing Layer incorporating multi-dimensional bond attributes:
    m_{j -> i} = ReLU(W_source * h_j + W_edge * e_{ji})
    h_i' = ReLU(BN(W_root * h_i + sum_{j in N(i)} m_{j -> i}))
    """
    def __init__(self, in_dim, out_dim, edge_dim=4):
        super(EdgeAwareMessagePassing, self).__init__(aggr='add')
        self.lin_source = nn.Linear(in_dim, out_dim)
        self.lin_edge = nn.Linear(edge_dim, out_dim)
        self.lin_root = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        # Transform source nodes and edge attributes
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        root = self.lin_root(x)
        return self.relu(self.bn(root + out))

    def message(self, x_j, edge_attr):
        # x_j: [E, in_dim], edge_attr: [E, edge_dim]
        m = self.lin_source(x_j) + self.lin_edge(edge_attr)
        return F.relu(m)


class EdgeAwareMolecularGNNEncoder(nn.Module):
    """
    Improved 3-Layer Edge-Aware Graph Neural Network State Encoder.
    
    Key Improvements over V1:
    1. Genuinely incorporates 4-dimensional chemical bond attributes (Single, Double, Triple, Aromatic).
    2. MessagePassing with edge projection allows differentiating aromatic vs aliphatic ring bonds.
    3. Retains permutation-invariance and dual-pooling (mean + max readout).
    4. Enhanced Fusion MLP combining graph embedding [128] with 6-dim scalar environment descriptors.
    """
    def __init__(self, node_dim=21, edge_dim=4, gnn_hidden=64, scalar_dim=6, fused_dim=128):
        super(EdgeAwareMolecularGNNEncoder, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.gnn_hidden = gnn_hidden
        self.scalar_dim = scalar_dim
        self.fused_dim = fused_dim

        # 3 Edge-Aware Message-Passing Layers
        self.conv1 = EdgeAwareMessagePassing(node_dim, gnn_hidden, edge_dim)
        self.conv2 = EdgeAwareMessagePassing(gnn_hidden, gnn_hidden, edge_dim)
        self.conv3 = EdgeAwareMessagePassing(gnn_hidden, gnn_hidden, edge_dim)

        # Graph Readout Projection: Mean-pool (64) + Max-pool (64) = 128
        self.graph_proj = nn.Linear(gnn_hidden * 2, gnn_hidden)

        # Fusion MLP combining graph-level context with continuous environment scalars
        self.fusion_mlp = nn.Sequential(
            nn.Linear(gnn_hidden + scalar_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU()
        )

    def forward(self, pyg_data, scalars):
        x, edge_index = pyg_data.x, pyg_data.edge_index
        edge_attr = pyg_data.edge_attr
        
        # Handle empty edges (single isolated atom edge cases)
        if edge_attr is None or edge_attr.size(0) == 0:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), dtype=torch.float32, device=x.device)
            
        batch = pyg_data.batch if hasattr(pyg_data, 'batch') and pyg_data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Edge-aware message passing
        h1 = self.conv1(x, edge_index, edge_attr)
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)

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

