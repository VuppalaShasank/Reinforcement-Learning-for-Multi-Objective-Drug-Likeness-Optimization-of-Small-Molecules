import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch_geometric.data import Batch

from implementation.actor_critic import MolecularActorCritic
from implementation.environment import MolecularGraphEnv

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for molecular graph optimization.
    Features:
    - GNN Actor-Critic with chemical action masking
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective
    - Entropy regularization
    - Value function clipping
    """
    def __init__(
        self,
        node_dim=21,
        edge_dim=4,
        gnn_hidden=64,
        scalar_dim=6,
        fused_dim=128,
        num_actions=1901,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.network = MolecularActorCritic(
            node_dim=node_dim,
            edge_dim=edge_dim,
            gnn_hidden=gnn_hidden,
            scalar_dim=scalar_dim,
            fused_dim=fused_dim,
            num_actions=num_actions
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def select_action(self, obs, deterministic=False):
        """
        Takes observation dict from MolecularGraphEnv and returns action, log_prob, value.
        """
        pyg_data = obs["pyg_data"].to(self.device)
        scalars = torch.tensor(obs["scalars"], dtype=torch.float32, device=self.device).unsqueeze(0)
        action_mask = torch.tensor(obs["action_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(
                pyg_data=pyg_data,
                scalars=scalars,
                action_mask=action_mask,
                deterministic=deterministic
            )
            
        return action.item(), log_prob.item(), value.item()

    def update(self, rollouts, ppo_epochs=4, batch_size=32):
        """
        Performs PPO policy and value updates over collected transition rollouts.
        rollouts: dict containing:
        - pyg_data_list: list of PyG Data objects
        - scalars: tensor [N, 6]
        - action_masks: tensor [N, 1901]
        - actions: tensor [N]
        - old_log_probs: tensor [N]
        - returns: tensor [N]
        - advantages: tensor [N]
        - values: tensor [N]
        """
        num_samples = len(rollouts["actions"])
        if num_samples == 0:
            return {}

        b_scalars = rollouts["scalars"].to(self.device)
        b_masks = rollouts["action_masks"].to(self.device)
        b_actions = rollouts["actions"].to(self.device)
        b_old_log_probs = rollouts["old_log_probs"].to(self.device)
        b_returns = rollouts["returns"].to(self.device)
        b_advantages = rollouts["advantages"].to(self.device)
        pyg_list = rollouts["pyg_data_list"]

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(ppo_epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                mb_pyg = Batch.from_data_list([pyg_list[i] for i in mb_idx]).to(self.device)
                mb_scalars = b_scalars[mb_idx]
                mb_masks = b_masks[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_log_probs = b_old_log_probs[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_adv = b_advantages[mb_idx]

                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    pyg_data=mb_pyg,
                    scalars=mb_scalars,
                    action_mask=mb_masks,
                    action=mb_actions
                )

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                # Policy Loss with PPO clipping
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Function Loss
                vf_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                # Entropy Regularization
                entropy_loss = entropy.mean()

                loss = pg_loss + (self.vf_coef * vf_loss) - (self.ent_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy_loss.item()
                num_updates += 1

        return {
            "policy_loss": total_pg_loss / max(num_updates, 1),
            "value_loss": total_vf_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1)
        }

    def save_checkpoint(self, filepath, metadata=None):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": metadata or {}
        }, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("metadata", {})

