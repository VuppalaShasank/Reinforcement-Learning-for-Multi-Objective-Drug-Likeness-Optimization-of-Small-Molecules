import os
import sys
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.gnn_encoder_v2 import EdgeAwareMolecularGNNEncoder
from src.environment.action_masking import TOTAL_ACTIONS


class EdgeAwareMolecularActorCritic(nn.Module):
    """
    Improved Actor-Critic Network with Edge-Aware Message Passing GNN backbone.
    
    Backbone:
    - 3-Layer Edge-Aware GNN Encoder incorporating bond attributes (Single, Double, Triple, Aromatic).
    - Fused Latent Embedding Dim = 128.
    
    Heads:
    - Actor: MLP(128 -> 128 -> 1901) producing action logits with chemical valency masking.
    - Critic: MLP(128 -> 64 -> 1) estimating the state value V(s).
    """
    def __init__(self, node_dim=21, edge_dim=4, gnn_hidden=64, scalar_dim=6, fused_dim=128, num_actions=TOTAL_ACTIONS):
        super(EdgeAwareMolecularActorCritic, self).__init__()
        self.num_actions = num_actions
        self.encoder = EdgeAwareMolecularGNNEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            gnn_hidden=gnn_hidden,
            scalar_dim=scalar_dim,
            fused_dim=fused_dim
        )
        
        # Critic head: V(s)
        self.critic = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Actor head: Action logits
        self.actor = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, pyg_data, scalars):
        state_embed = self.encoder(pyg_data, scalars)
        logits = self.actor(state_embed)
        value = self.critic(state_embed)
        return logits, value

    def get_value(self, pyg_data, scalars):
        state_embed = self.encoder(pyg_data, scalars)
        return self.critic(state_embed).squeeze(-1)

    def get_action_and_value(self, pyg_data, scalars, action_mask, action=None, deterministic=False):
        state_embed = self.encoder(pyg_data, scalars)
        logits = self.actor(state_embed)
        value = self.critic(state_embed).squeeze(-1)
        
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
            
        mask_bool = action_mask.bool()
        mask_bool[:, 0] = True # STOP action (idx 0) is always chemically valid
        
        # Mask out chemically forbidden actions
        masked_logits = logits.masked_fill(~mask_bool, -1e8)
        dist = Categorical(logits=masked_logits)
        
        if action is None:
            if deterministic:
                action = torch.argmax(masked_logits, dim=-1)
            else:
                action = dist.sample()
                
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_top_k_candidates(self, pyg_data, scalars, action_mask, k=5):
        """
        Returns the top-k chemically valid candidate actions ranked by the Actor's policy logits.
        """
        with torch.no_grad():
            state_embed = self.encoder(pyg_data, scalars)
            logits = self.actor(state_embed)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            mask_bool = action_mask.bool()
            mask_bool[:, 0] = True
            
            masked_logits = logits.masked_fill(~mask_bool, -1e8)
            num_valid = mask_bool.sum().item()
            effective_k = min(k, max(1, num_valid))
            topk_vals, topk_indices = torch.topk(masked_logits[0], k=effective_k)
            
        return topk_indices.cpu().numpy().tolist()

