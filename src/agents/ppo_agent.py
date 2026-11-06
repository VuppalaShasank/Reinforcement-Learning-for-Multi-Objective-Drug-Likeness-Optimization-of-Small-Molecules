import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import Batch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.action_policy import MolecularActorCritic


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for Molecular Graph Optimization.
    
    Key Features:
    - Actor-Critic network with 3-Layer GNN encoder backbone.
    - Dynamic chemical valency action masking over 1,901 discrete chemical edits.
    - Generalized Advantage Estimation (GAE, lambda=0.95).
    - Clipped surrogate policy objective (clip_coef=0.20).
    - Value function optimization with MSE loss.
    - Entropy regularization to encourage chemical exploration.
    - Atomic checkpoint saving and seamless resumption.
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
        clip_coef=0.20,
        ent_coef=0.01,
        vf_coef=0.50,
        max_grad_norm=0.50,
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
        Takes observation dict from environment and returns sampled action, log_prob, and value.
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
        Executes PPO mini-batch updates over collected rollout transitions.
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

        # Advantage normalization
        if len(b_advantages) > 1:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
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

                # Approximate KL divergence for monitoring
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()

                # Clipped Surrogate Policy Loss
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
                total_approx_kl += approx_kl.item()
                num_updates += 1

        denom = max(num_updates, 1)
        return {
            "policy_loss": total_pg_loss / denom,
            "value_loss": total_vf_loss / denom,
            "entropy": total_entropy / denom,
            "approx_kl": total_approx_kl / denom
        }

    def save_checkpoint(self, filepath, metadata=None):
        """
        Atomic checkpoint saving: writes to .tmp first to prevent corruption on interruption.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tmp_path = filepath + ".tmp"
        payload = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": metadata or {}
        }
        torch.save(payload, tmp_path)
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(tmp_path, filepath)

    def load_checkpoint(self, filepath):
        """
        Loads network weights and optimizer state from checkpoint.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("metadata", {})

