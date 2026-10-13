import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from implementation.gnn_encoder import MolecularGNNEncoder

class MolecularActorCritic(nn.Module):
    """
    Combined Actor-Critic network with 3-Layer GNN encoder backbone.
    Supports valid-action masking for chemically compliant policy learning.
    """
    def __init__(self, node_dim=21, edge_dim=4, gnn_hidden=64, scalar_dim=6, fused_dim=128, num_actions=1901):
        super(MolecularActorCritic, self).__init__()
        self.num_actions = num_actions
        self.encoder = MolecularGNNEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            gnn_hidden=gnn_hidden,
            scalar_dim=scalar_dim,
            fused_dim=fused_dim
        )
        
        # Critic head: outputs scalar state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Actor head: outputs unmasked action logits
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
        Computes action distribution masked by chemical valency, samples action,
        and computes log-probability, entropy, and state-value.
        """
        state_embed = self.encoder(pyg_data, scalars)
        logits = self.actor(state_embed) # [B, num_actions]
        value = self.critic(state_embed).squeeze(-1) # [B]
        
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
            
        # Mask out invalid actions by setting logits to large negative value
        mask_bool = action_mask.bool()
        # Ensure at least one action is valid per batch item (action 0 STOP is always valid)
        mask_bool[:, 0] = True
        
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

