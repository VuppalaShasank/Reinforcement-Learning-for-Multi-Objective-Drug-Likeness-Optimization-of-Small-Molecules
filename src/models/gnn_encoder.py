import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool

class MolecularGNNEncoder(nn.Module):
    """
    3-Layer Graph Neural Network state encoder for molecular graphs.
    Input:
    - Node features: x [N, 21]
    - Edge index: edge_index [2, E]
    - Edge attributes: edge_attr [E, 4]
    - Scalar state features: scalars [B, 6] (step/T, QED, SAS, Sim, q0, sas_0)
    
    Output:
    - Fixed-size fused state representation [B, hidden_dim]
    """
    def __init__(self, node_dim=21, edge_dim=4, gnn_hidden=64, scalar_dim=6, fused_dim=128):
        super(MolecularGNNEncoder, self).__init__()
        
        # 3 Graph Convolution Layers
        self.conv1 = GraphConv(node_dim, gnn_hidden)
        self.conv2 = GraphConv(gnn_hidden, gnn_hidden)
        self.conv3 = GraphConv(gnn_hidden, gnn_hidden)
        
        # Batch normalization for graph layers
        self.bn1 = nn.BatchNorm1d(gnn_hidden)
        self.bn2 = nn.BatchNorm1d(gnn_hidden)
        self.bn3 = nn.BatchNorm1d(gnn_hidden)
        
        # Graph readout projection: combines mean pool and max pool (2 * gnn_hidden)
        self.graph_proj = nn.Linear(gnn_hidden * 2, gnn_hidden)
        
        # Fusion MLP combining graph embedding and scalar descriptors
        self.fusion_mlp = nn.Sequential(
            nn.Linear(gnn_hidden + scalar_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU()
        )

    def forward(self, pyg_data, scalars):
        """
        pyg_data: PyTorch Geometric Data or Batch object
        scalars: Tensor of shape [B, 6]
        """
        x, edge_index = pyg_data.x, pyg_data.edge_index
        batch = pyg_data.batch if hasattr(pyg_data, 'batch') and pyg_data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Layer 1
        h1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        # Layer 2
        h2 = F.relu(self.bn2(self.conv2(h1, edge_index)))
        # Layer 3
        h3 = F.relu(self.bn3(self.conv3(h2, edge_index)))
        
        # Global Readout: Mean and Max pooling
        pool_mean = global_mean_pool(h3, batch)
        pool_max = global_max_pool(h3, batch)
        h_graph = torch.cat([pool_mean, pool_max], dim=-1)
        h_graph_proj = F.relu(self.graph_proj(h_graph))
        
        # Match batch size if scalars has multiple entries
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)
            
        # Concatenate graph embedding with scalar descriptors
        combined = torch.cat([h_graph_proj, scalars], dim=-1)
        state_embed = self.fusion_mlp(combined)
        return state_embed

