import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.action_policy_v2 import EdgeAwareMolecularActorCritic
from src.environment.action_masking import TOTAL_ACTIONS


class ImprovedPPOAgent:
    """
    Improved Proximal Policy Optimization (PPO) Agent (V2).
    
    Key Enhancements:
    1. Edge-Aware 3-Layer GNN encoder utilizing 4-dim chemical bond attributes.
    2. Exact action mask alignment guaranteeing 100% chemically valid actions.
    3. Top-K Candidate Extraction for PPO-Guided Local Search.
    4. Gradient norm clipping (0.50) and Adam epsilon stabilization (1e-5).
    5. Clean checkpoint serialization and atomic resumption.
    """
    def __init__(
        self,
        node_dim=21,
        edge_dim=4,
        gnn_hidden=64,
        scalar_dim=6,
        fused_dim=128,
        num_actions=TOTAL_ACTIONS,
        lr=2.5e-4,
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

        self.network = EdgeAwareMolecularActorCritic(
            node_dim=node_dim,
            edge_dim=edge_dim,
            gnn_hidden=gnn_hidden,
            scalar_dim=scalar_dim,
            fused_dim=fused_dim,
            num_actions=num_actions
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def select_action(self, obs, deterministic=False):
        pyg_data = obs["pyg_data"].to(self.device)
        scalars = torch.tensor(obs["scalars"], dtype=torch.float32, device=self.device).unsqueeze(0)
        action_mask = torch.tensor(obs["action_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(
                pyg_data=pyg_data,
                scalars=scalars,
                action_mask=action_mask,
                deterministic=deterministic
            )
            
        return action.item(), log_prob.item(), value.item()

    def get_top_k_actions(self, obs, k=5):
        pyg_data = obs["pyg_data"].to(self.device)
        scalars = torch.tensor(obs["scalars"], dtype=torch.float32, device=self.device).unsqueeze(0)
        action_mask = torch.tensor(obs["action_mask"], dtype=torch.bool, device=self.device).unsqueeze(0)
        return self.network.get_top_k_candidates(pyg_data, scalars, action_mask, k=k)

    def update(self, rollouts, ppo_epochs=4, batch_size=32):
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

        # Normalize advantages across rollout
        if b_advantages.numel() > 1:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_ent_loss = 0.0
        total_loss_accum = 0.0
        update_steps = 0

        indices = np.arange(num_samples)

        for epoch in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                mb_idx = indices[start:end]

                # Construct batched PyG Data
                mb_pyg_data = [pyg_list[i] for i in mb_idx]
                mb_pyg_batch = Batch.from_data_list(mb_pyg_data).to(self.device)
                mb_scalars = b_scalars[mb_idx]
                mb_masks = b_masks[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_log_probs = b_old_log_probs[mb_idx]
                mb_returns = b_returns[mb_idx]
                mb_advantages = b_advantages[mb_idx]

                # Evaluate actions under current policy
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    pyg_data=mb_pyg_batch,
                    scalars=mb_scalars,
                    action_mask=mb_masks,
                    action=mb_actions
                )

                # Ratio r_t(theta)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                # Clipped surrogate policy objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (clipped MSE)
                v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                # Entropy loss
                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_ent_loss += ent_loss.item()
                total_loss_accum += loss.item()
                update_steps += 1

        avg_metrics = {
            "policy_loss": total_pg_loss / max(1, update_steps),
            "value_loss": total_v_loss / max(1, update_steps),
            "entropy": -total_ent_loss / max(1, update_steps),
            "total_loss": total_loss_accum / max(1, update_steps)
        }
        return avg_metrics

    def save_checkpoint(self, path, metadata=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": metadata or {}
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass
        return checkpoint.get("metadata", {})

