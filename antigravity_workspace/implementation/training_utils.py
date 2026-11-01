import os
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from implementation.environment import MolecularGraphEnv
from antigravity_workspace.implementation.ppo_agent import PPOAgent


def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """
    Computes Generalized Advantage Estimation (GAE) and returns targets.
    """
    advantages = []
    gae = 0.0
    num_steps = len(rewards)
    
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
            
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages.insert(0, gae)
        
    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def run_training_experiment(
    reward_mode="adaptive",
    train_csv=None,
    val_csv=None,
    num_episodes=60,
    max_steps_per_episode=10,
    update_every_episodes=5,
    ppo_epochs=4,
    batch_size=16,
    lr=3e-4,
    checkpoint_dir=None,
    log_csv_path=None,
    seed=42
):
    """
    Executes a complete, fault-tolerant PPO training experiment across starting molecules.
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting PPO Training [{reward_mode.upper()}] on {device} (Seed: {seed}) ===")

    # Load starting molecules
    if train_csv is None or not os.path.exists(train_csv):
        train_csv = os.path.join(ROOT_DIR, "antigravity_workspace", "data", "splits", "train_molecules.csv")
        if not os.path.exists(train_csv):
            train_csv = os.path.join(ROOT_DIR, "results", "train_molecules.csv")

    df_train = pd.read_csv(train_csv)
    train_smiles_list = df_train["smiles"].tolist()
    print(f"Loaded {len(train_smiles_list)} training starting molecules.")

    # Create environment and PPO agent
    config_path = os.path.join(ROOT_DIR, "antigravity_workspace", "configs", "experiment_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(ROOT_DIR, "configs", "reward_config.yaml")

    env = MolecularGraphEnv(reward_mode=reward_mode, max_steps=max_steps_per_episode, config_path=config_path)
    agent = PPOAgent(lr=lr, device=device)

    # Rollout buffer
    buf_pyg = []
    buf_scalars = []
    buf_masks = []
    buf_actions = []
    buf_log_probs = []
    buf_rewards = []
    buf_values = []
    buf_dones = []

    episode_logs = []
    best_eval_return = -float("inf")
    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        # Sample starting molecule
        start_smiles = np.random.choice(train_smiles_list)
        try:
            obs, info = env.reset(options={"starting_smiles": start_smiles})
        except Exception as e:
            # Fault-tolerance: skip unparseable molecule gracefully
            continue

        ep_reward = 0.0
        ep_steps = 0
        ep_success = False

        for step in range(max_steps_per_episode):
            action, log_prob, value = agent.select_action(obs)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            # Store transition
            buf_pyg.append(obs["pyg_data"])
            buf_scalars.append(torch.tensor(obs["scalars"], dtype=torch.float32))
            buf_masks.append(torch.tensor(obs["action_mask"], dtype=torch.bool))
            buf_actions.append(action)
            buf_log_probs.append(log_prob)
            buf_rewards.append(reward)
            buf_values.append(value)
            buf_dones.append(done)

            ep_reward += reward
            ep_steps += 1
            if step_info.get("success", False):
                ep_success = True

            obs = next_obs
            if done:
                break

        # Compute episode properties
        final_props = env.current_props
        initial_props = env.original_props
        delta_qed = final_props["qed"] - initial_props["qed"]
        final_sim = final_props["similarity"]

        # PPO Update condition
        loss_info = {}
        if ep % update_every_episodes == 0 and len(buf_rewards) > 0:
            last_val = 0.0 if buf_dones[-1] else agent.select_action(obs)[2]
            advantages, returns = compute_gae(
                rewards=buf_rewards,
                values=buf_values,
                dones=buf_dones,
                next_value=last_val,
                gamma=agent.gamma,
                gae_lambda=agent.gae_lambda
            )

            rollouts = {
                "pyg_data_list": buf_pyg,
                "scalars": torch.stack(buf_scalars),
                "action_masks": torch.stack(buf_masks),
                "actions": torch.tensor(buf_actions, dtype=torch.long),
                "old_log_probs": torch.tensor(buf_log_probs, dtype=torch.float32),
                "returns": returns,
                "advantages": advantages
            }

            loss_info = agent.update(rollouts, ppo_epochs=ppo_epochs, batch_size=batch_size)

            # Clear buffer
            buf_pyg.clear()
            buf_scalars.clear()
            buf_masks.clear()
            buf_actions.clear()
            buf_log_probs.clear()
            buf_rewards.clear()
            buf_values.clear()
            buf_dones.clear()

        # Log episode
        log_entry = {
            "episode": ep,
            "seed": seed,
            "reward_mode": reward_mode,
            "reward": ep_reward,
            "delta_qed": delta_qed,
            "final_qed": final_props["qed"],
            "initial_qed": initial_props["qed"],
            "similarity": final_sim,
            "final_sas": final_props["sas"],
            "steps": ep_steps,
            "success": ep_success,
            "policy_loss": loss_info.get("policy_loss", np.nan),
            "value_loss": loss_info.get("value_loss", np.nan),
            "entropy": loss_info.get("entropy", np.nan),
            "elapsed_sec": time.time() - start_time
        }
        episode_logs.append(log_entry)

        if ep % 10 == 0 or ep == num_episodes:
            print(f"  Ep {ep}/{num_episodes} | Return: {ep_reward:.3f} | Delta QED: {delta_qed:+.3f} | Sim: {final_sim:.3f} | Success: {ep_success}")

    df_logs = pd.DataFrame(episode_logs)
    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        df_logs.to_csv(log_csv_path, index=False)
        print(f"Saved training log to: {log_csv_path}")

    # Checkpoint saving
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        latest_ckpt = os.path.join(checkpoint_dir, f"ppo_{reward_mode}_latest.pt")
        final_ckpt = os.path.join(checkpoint_dir, f"ppo_{reward_mode}.pt")
        meta = {
            "episodes": num_episodes,
            "reward_mode": reward_mode,
            "seed": seed,
            "mean_return": float(df_logs["reward"].tail(10).mean())
        }
        agent.save_checkpoint(latest_ckpt, metadata=meta)
        agent.save_checkpoint(final_ckpt, metadata=meta)
        print(f"Saved checkpoints to: {final_ckpt}")

    return agent, df_logs

