import torch
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from implementation.environment import MolecularGraphEnv
from implementation.ppo_agent import PPOAgent

def train_ppo_model(
    reward_mode="adaptive",
    train_csv=None,
    num_episodes=60,
    max_steps_per_episode=10,
    update_every_episodes=5,
    ppo_epochs=4,
    batch_size=16,
    lr=3e-4,
    checkpoint_path=None,
    log_csv_path=None,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" Starting PPO Training Run: Mode = {reward_mode.upper()} ")
    print(f" Device: {device}, Episodes: {num_episodes}, Horizon: {max_steps_per_episode} ")
    print(f"=======================================================")

    # Load training starting molecules
    if train_csv and os.path.exists(train_csv):
        df_train = pd.read_csv(train_csv)
        train_smiles_list = df_train["smiles"].tolist()
    else:
        train_smiles_list = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]

    env = MolecularGraphEnv(reward_mode=reward_mode, max_steps=max_steps_per_episode)
    agent = PPOAgent(lr=lr, device=device)

    # Rollout buffer
    rollouts = {
        "pyg_data_list": [],
        "scalars": [],
        "action_masks": [],
        "actions": [],
        "old_log_probs": [],
        "rewards": [],
        "dones": [],
        "values": []
    }

    logs = []
    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        # Sample random starting molecule from training set
        start_smi = train_smiles_list[(ep - 1) % len(train_smiles_list)]
        obs, info = env.reset(options={"starting_smiles": start_smi})
        
        ep_reward = 0.0
        ep_edits = 0
        q0 = info["qed"]
        
        while True:
            action, log_prob, value = agent.select_action(obs, deterministic=False)
            
            # Store transition
            rollouts["pyg_data_list"].append(obs["pyg_data"].cpu())
            rollouts["scalars"].append(torch.tensor(obs["scalars"], dtype=torch.float32))
            rollouts["action_masks"].append(torch.tensor(obs["action_mask"], dtype=torch.bool))
            rollouts["actions"].append(action)
            rollouts["old_log_probs"].append(log_prob)
            rollouts["values"].append(value)
            
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            done = terminated or truncated
            rollouts["rewards"].append(reward)
            rollouts["dones"].append(done)
            
            ep_reward += reward
            ep_edits += 1
            obs = next_obs
            
            if done:
                break

        final_qed = env.current_props["qed"]
        final_sas = env.current_props["sas"]
        final_sim = env.current_props["similarity"]
        delta_qed = final_qed - q0
        is_success = env.reward_calc.check_success(env.current_props, env.original_props)

        ep_stat = {
            "episode": ep,
            "reward_mode": reward_mode,
            "return": ep_reward,
            "delta_qed": delta_qed,
            "initial_qed": q0,
            "final_qed": final_qed,
            "similarity": final_sim,
            "final_sas": final_sas,
            "edits": ep_edits,
            "success": is_success,
            "valid": env.current_props["valid"]
        }
        logs.append(ep_stat)

        if ep % 5 == 0 or ep == num_episodes:
            elapsed = time.time() - start_time
            print(f"[{reward_mode.capitalize()} Ep {ep:3d}/{num_episodes}] Return: {ep_reward:7.3f} | Delta QED: {delta_qed:+6.3f} | Sim: {final_sim:5.3f} | Success: {is_success} | Time: {elapsed:.1f}s")

        # PPO Update
        if ep % update_every_episodes == 0 and len(rollouts["actions"]) > 0:
            # Compute GAE advantages and returns
            rewards = rollouts["rewards"]
            dones = rollouts["dones"]
            values = rollouts["values"] + [0.0] # Terminal value approx
            
            advantages = []
            gae = 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + agent.gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
                gae = delta + agent.gamma * agent.gae_lambda * (1.0 - float(dones[t])) * gae
                advantages.insert(0, gae)
                
            returns = [adv + val for adv, val in zip(advantages, values[:-1])]

            update_data = {
                "pyg_data_list": rollouts["pyg_data_list"],
                "scalars": torch.stack(rollouts["scalars"]),
                "action_masks": torch.stack(rollouts["action_masks"]),
                "actions": torch.tensor(rollouts["actions"], dtype=torch.long),
                "old_log_probs": torch.tensor(rollouts["old_log_probs"], dtype=torch.float32),
                "returns": torch.tensor(returns, dtype=torch.float32),
                "advantages": torch.tensor(advantages, dtype=torch.float32)
            }

            loss_info = agent.update(update_data, ppo_epochs=ppo_epochs, batch_size=batch_size)
            print(f"   --> PPO Update: Policy Loss = {loss_info.get('policy_loss', 0):.4f}, Value Loss = {loss_info.get('value_loss', 0):.4f}")

            # Clear rollout buffer
            for k in rollouts:
                rollouts[k] = []

    # Save checkpoint and logs
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        agent.save_checkpoint(checkpoint_path, metadata={"reward_mode": reward_mode, "num_episodes": num_episodes})
        print(f"Checkpoint saved to: {checkpoint_path}")

    df_logs = pd.DataFrame(logs)
    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        df_logs.to_csv(log_csv_path, index=False)
        print(f"Training log saved to: {log_csv_path}")

    return agent, df_logs

def run_both_training_experiments():
    train_csv = os.path.join("results", "train_molecules.csv") if os.path.exists(os.path.join("results", "train_molecules.csv")) else os.path.join("cache", "results", "train_molecules.csv")
    
    # 1. Train PPO Baseline
    ckpt_base = os.path.join("checkpoints", "ppo_baseline.pt")
    log_base = os.path.join("results", "train_ppo_baseline.csv") if os.path.exists("results") else os.path.join("cache", "logs", "train_ppo_baseline.csv")
    agent_base, logs_base = train_ppo_model(
        reward_mode="baseline",
        train_csv=train_csv,
        num_episodes=60,
        max_steps_per_episode=8,
        update_every_episodes=5,
        checkpoint_path=ckpt_base,
        log_csv_path=log_base,
        seed=101
    )

    # 2. Train PPO Adaptive (Proposed Contribution)
    ckpt_adapt = os.path.join("checkpoints", "ppo_adaptive.pt")
    log_adapt = os.path.join("results", "train_ppo_adaptive.csv") if os.path.exists("results") else os.path.join("cache", "logs", "train_ppo_adaptive.csv")
    agent_adapt, logs_adapt = train_ppo_model(
        reward_mode="adaptive",
        train_csv=train_csv,
        num_episodes=60,
        max_steps_per_episode=8,
        update_every_episodes=5,
        checkpoint_path=ckpt_adapt,
        log_csv_path=log_adapt,
        seed=202
    )

    # 3. Generate Comparative Training Curve Plot
    fig_dir = "figures" if os.path.exists("figures") else os.path.join("cache", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_path = os.path.join(fig_dir, "training_reward_curves.png")

    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Episode Returns (Rolling mean)
    plt.subplot(1, 2, 1)
    base_smooth = logs_base["return"].rolling(window=5, min_periods=1).mean()
    adapt_smooth = logs_adapt["return"].rolling(window=5, min_periods=1).mean()
    plt.plot(logs_base["episode"], base_smooth, 'r-', label="PPO Baseline", linewidth=2)
    plt.plot(logs_adapt["episode"], adapt_smooth, 'b-', label="PPO Adaptive (Proposed)", linewidth=2)
    plt.title("PPO Episode Return Trajectory")
    plt.xlabel("Episode")
    plt.ylabel("Return (5-ep moving average)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Delta QED
    plt.subplot(1, 2, 2)
    base_qed = logs_base["delta_qed"].rolling(window=5, min_periods=1).mean()
    adapt_qed = logs_adapt["delta_qed"].rolling(window=5, min_periods=1).mean()
    plt.plot(logs_base["episode"], base_qed, 'r--', label="PPO Baseline $\\Delta$QED", linewidth=2)
    plt.plot(logs_adapt["episode"], adapt_qed, 'b-', label="PPO Adaptive $\\Delta$QED", linewidth=2)
    plt.axhline(0.0, color='k', linestyle=':', alpha=0.5)
    plt.title("Optimization $\\Delta$QED Over Training")
    plt.xlabel("Episode")
    plt.ylabel("$\\Delta$QED (5-ep moving average)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nTraining curves plot saved to: {plot_path}")

if __name__ == "__main__":
    run_both_training_experiments()

