import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment.molecular_environment import MolecularGraphEnv
from src.agents.ppo_agent_v2 import ImprovedPPOAgent
from src.rewards.adaptive_reward import CalibratedAdaptiveRewardCalculator


class CurriculumSampler:
    """
    Curriculum-based Molecule Sampler:
    Stage A (Episodes 1-30): Low QED (<0.50)
    Stage B (Episodes 31-60): Medium QED (0.50-0.70)
    Stage C (Episodes 61-90): High QED (0.70-0.85)
    Stage D (Episodes 91+): Full Mixed Spectrum
    """
    def __init__(self, df_train):
        self.df_train = df_train
        self.pool_low = df_train[df_train["qed"] < 0.50]["smiles"].tolist()
        self.pool_med = df_train[(df_train["qed"] >= 0.50) & (df_train["qed"] < 0.70)]["smiles"].tolist()
        self.pool_high = df_train[(df_train["qed"] >= 0.70) & (df_train["qed"] < 0.85)]["smiles"].tolist()
        self.pool_all = df_train["smiles"].tolist()

    def sample_smiles(self, episode):
        if episode <= 25 and len(self.pool_low) > 0:
            return np.random.choice(self.pool_low)
        elif episode <= 50 and len(self.pool_med) > 0:
            return np.random.choice(self.pool_med)
        elif episode <= 75 and len(self.pool_high) > 0:
            return np.random.choice(self.pool_high)
        else:
            return np.random.choice(self.pool_all)


def train_improved_ppo(
    total_episodes=100,
    rollout_steps=128,
    ppo_epochs=4,
    batch_size=32,
    lr=2.5e-4,
    use_curriculum=True,
    output_ckpt="checkpoints/ppo_adaptive_v2_improved.pt",
    output_log="logs/train_ppo_improved.csv"
):
    print("=" * 80)
    print("TRAINING IMPROVED GNN-PPO AGENT (V2: EDGE-AWARE GNN + CALIBRATED REWARD)")
    print("=" * 80)

    # 1. Load Training Data
    train_csv = os.path.join(PROJECT_ROOT, "data", "splits", "train_molecules.csv")
    df_train = pd.read_csv(train_csv)
    print(f"Loaded {len(df_train)} training molecules.")
    sampler = CurriculumSampler(df_train)

    # 2. Setup Environment with Calibrated Reward V2
    env = MolecularGraphEnv(reward_mode="adaptive", max_steps=10)
    env.reward_calc = CalibratedAdaptiveRewardCalculator()

    # 3. Initialize Improved PPO Agent
    agent = ImprovedPPOAgent(
        node_dim=21, edge_dim=4, gnn_hidden=64, scalar_dim=6, fused_dim=128,
        lr=lr, gamma=0.99, gae_lambda=0.95, clip_coef=0.20, ent_coef=0.01
    )

    history = []
    step_buffer = {
        "pyg_data_list": [],
        "scalars": [],
        "action_masks": [],
        "actions": [],
        "old_log_probs": [],
        "rewards": [],
        "dones": [],
        "values": []
    }

    t0_train = time.time()
    
    for episode in range(1, total_episodes + 1):
        if use_curriculum:
            starting_smiles = sampler.sample_smiles(episode)
        else:
            starting_smiles = np.random.choice(sampler.pool_all)

        obs, info = env.reset(options={"starting_smiles": starting_smiles})
        q0 = env.original_props["qed"]
        sas0 = env.original_props["sas"]

        ep_return = 0.0
        ep_steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, log_prob, val_est = agent.select_action(obs, deterministic=False)
            
            # Record step transition
            step_buffer["pyg_data_list"].append(obs["pyg_data"])
            step_buffer["scalars"].append(torch.tensor(obs["scalars"], dtype=torch.float32))
            step_buffer["action_masks"].append(torch.tensor(obs["action_mask"], dtype=torch.bool))
            step_buffer["actions"].append(action)
            step_buffer["old_log_probs"].append(log_prob)
            step_buffer["values"].append(val_est)

            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_return += reward
            ep_steps += 1

            step_buffer["rewards"].append(reward)
            step_buffer["dones"].append(float(terminated or truncated))

            # Trigger PPO Update when rollout buffer reaches capacity
            if len(step_buffer["actions"]) >= rollout_steps:
                # GAE Advantage Estimation
                rewards = np.array(step_buffer["rewards"])
                values = np.array(step_buffer["values"])
                dones = np.array(step_buffer["dones"])
                N = len(rewards)
                advantages = np.zeros(N, dtype=np.float32)
                last_gae = 0.0

                # Compute next value for bootstrap
                with torch.no_grad():
                    next_val = agent.network.get_value(
                        obs["pyg_data"].to(agent.device),
                        torch.tensor(obs["scalars"], dtype=torch.float32, device=agent.device).unsqueeze(0)
                    ).item()

                for t in reversed(range(N)):
                    if t == N - 1:
                        next_v = next_val
                    else:
                        next_v = values[t + 1]
                    delta = rewards[t] + 0.99 * next_v * (1.0 - dones[t]) - values[t]
                    last_gae = delta + 0.99 * 0.95 * (1.0 - dones[t]) * last_gae
                    advantages[t] = last_gae

                returns = advantages + values

                rollout_data = {
                    "pyg_data_list": step_buffer["pyg_data_list"],
                    "scalars": torch.stack(step_buffer["scalars"]),
                    "action_masks": torch.stack(step_buffer["action_masks"]),
                    "actions": torch.tensor(step_buffer["actions"], dtype=torch.long),
                    "old_log_probs": torch.tensor(step_buffer["old_log_probs"], dtype=torch.float32),
                    "returns": torch.tensor(returns, dtype=torch.float32),
                    "advantages": torch.tensor(advantages, dtype=torch.float32)
                }

                # Update Actor-Critic
                loss_info = agent.update(rollout_data, ppo_epochs=ppo_epochs, batch_size=batch_size)

                # Reset buffer
                step_buffer = {
                    "pyg_data_list": [], "scalars": [], "action_masks": [],
                    "actions": [], "old_log_probs": [], "rewards": [],
                    "dones": [], "values": []
                }

        # Episode logging
        final_qed = env.current_props["qed"]
        delta_qed = final_qed - q0
        sim = env.current_props["similarity"]
        final_sas = env.current_props["sas"]
        success = env.reward_calc.check_success(env.current_props, env.original_props)

        record = {
            "episode": episode,
            "q0": q0,
            "final_qed": final_qed,
            "delta_qed": delta_qed,
            "similarity": sim,
            "final_sas": final_sas,
            "delta_sas": final_sas - sas0,
            "edits": ep_steps,
            "return": ep_return,
            "success": success
        }
        history.append(record)

        if episode % 10 == 0 or episode == total_episodes:
            df_hist = pd.DataFrame(history)
            df_hist.to_csv(output_log, index=False)
            recent_delta = df_hist["delta_qed"].tail(10).mean()
            recent_sim = df_hist["similarity"].tail(10).mean()
            recent_succ = df_hist["success"].tail(10).mean() * 100
            print(f"Episode {episode:3d}/{total_episodes} | Return: {ep_return:+.3f} | Delta-QED (10-ep): {recent_delta:+.4f} | Sim: {recent_sim:.3f} | Succ: {recent_succ:.1f}%")

    # Save Final Checkpoint
    agent.save_checkpoint(output_ckpt, metadata={"episodes": total_episodes, "final_delta_qed": float(np.mean([h['delta_qed'] for h in history[-20:]]))})
    print(f"\nTraining completed in {time.time() - t0_train:.1f}s.")
    print(f"Saved trained checkpoint to: {output_ckpt}")
    print(f"Saved training log to: {output_log}")
    return agent, pd.DataFrame(history)


if __name__ == "__main__":
    train_improved_ppo()

