import os
import sys
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rl_lead_optimization.implementation.gnn_encoder import MolecularGNNEncoder
from implementation.action_handler import TOTAL_ACTIONS, get_action_mask, action_to_details, apply_action


class MolecularActorCritic(nn.Module):
    """
    Actor-Critic Network for Molecular Optimization with chemical valency action masking.
    
    Backbone:
    - 3-Layer Molecular GNN Encoder (fused embedding dim = 128)
    
    Heads:
    - Actor: MLP(128 -> 128 -> 1901) producing action logits.
             Invalid chemical actions are masked out before computing the categorical policy.
    - Critic: MLP(128 -> 64 -> 1) estimating the state value V(s).
    """
    def __init__(self, node_dim=21, edge_dim=4, gnn_hidden=64, scalar_dim=6, fused_dim=128, num_actions=TOTAL_ACTIONS):
        super(MolecularActorCritic, self).__init__()
        self.num_actions = num_actions
        self.encoder = MolecularGNNEncoder(
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
        """
        Samples an action from the masked categorical distribution,
        and returns action, log_prob, entropy, and estimated value.
        """
        state_embed = self.encoder(pyg_data, scalars)
        logits = self.actor(state_embed)
        value = self.critic(state_embed).squeeze(-1)
        
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
            
        mask_bool = action_mask.bool()
        # STOP action (idx 0) is always guaranteed valid
        mask_bool[:, 0] = True
        
        # Mask out invalid actions
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
