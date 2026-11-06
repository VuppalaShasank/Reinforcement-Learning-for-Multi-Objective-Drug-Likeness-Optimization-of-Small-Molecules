import os
import yaml
import numpy as np


class CalibratedAdaptiveRewardCalculator:
    """
    Calibrated Headroom- and Constraint-Slack Adaptive Reward Function (V2).
    
    Key Mathematical Innovations:
    1. Calibrated Headroom Gain:
       gamma_headroom = 1.0 + min(1.5, max(0.0, (0.90 - q0) / 0.90))
       Provides higher incentive on low-QED leads without extreme destabilization.
    2. Continuous Smooth Constraint Barrier:
       Soft quadratic penalty in [0.40, 0.55]; steep linear barrier below 0.40.
    3. Step Cost:
       -0.01 per edit to discourage unproductive wandering.
    4. STOP Action Preservation Incentive (Crucial V2 Correction):
       When the agent takes the STOP action:
       - If Delta-QED >= 0.10 and Sim >= 0.40 and SAS <= allowed: awards +5.0 terminal bonus.
       - If Delta-QED > 0 and Sim >= 0.40: awards preservation bonus of +2.0 * Delta-QED.
       - This eliminates the penalty on STOP and incentivizes locking in gains at local peaks.
    """
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.step_cost = float(config.get("step_cost", 0.01))
        self.sim_threshold = float(config.get("sim_threshold", 0.40))
        self.sim_margin_buffer = float(config.get("sim_margin_buffer", 0.15))
        self.sim_soft_weight = float(config.get("sim_soft_penalty_weight", 2.0))
        self.sim_hard_weight = float(config.get("sim_hard_penalty_weight", 12.0))
        self.sim_violation_base = float(config.get("sim_violation_base", 3.0))

        self.max_sas_base = float(config.get("max_sas_base", 4.5))
        self.sas_margin_buffer = float(config.get("sas_margin_buffer", 0.50))
        self.sas_soft_weight = float(config.get("sas_soft_penalty_weight", 1.0))
        self.sas_hard_weight = float(config.get("sas_hard_penalty_weight", 5.0))
        self.sas_violation_base = float(config.get("sas_violation_base", 1.0))

        self.invalid_penalty = float(config.get("invalid_penalty", -1.0))
        self.success_bonus = float(config.get("terminal_bonus", 5.0))
        self.preservation_weight = float(config.get("preservation_weight", 2.0))

    def _compute_sim_penalty(self, sim):
        slack = sim - self.sim_threshold
        if slack < 0:
            return self.sim_violation_base + self.sim_hard_weight * abs(slack)
        elif slack < self.sim_margin_buffer:
            ratio = 1.0 - (slack / self.sim_margin_buffer)
            return self.sim_soft_weight * (ratio ** 2)
        else:
            return 0.0

    def _compute_sas_penalty(self, sas, sas_0):
        allowed_sas = max(sas_0, self.max_sas_base)
        slack = allowed_sas - sas
        if slack < 0:
            return self.sas_violation_base + self.sas_hard_weight * abs(slack)
        elif slack < self.sas_margin_buffer:
            ratio = 1.0 - (slack / self.sas_margin_buffer)
            return self.sas_soft_weight * (ratio ** 2)
        else:
            return 0.0

    def calculate_reward(self, prev_props, new_props, original_props, is_stop=False):
        if not new_props.get("valid", False):
            return self.invalid_penalty, {
                "reward_mode": "adaptive_v2",
                "valid": False,
                "total_reward": self.invalid_penalty,
                "success": False
            }

        q0 = original_props["qed"]
        sas0 = original_props["sas"]
        
        # Headroom multiplier
        headroom = max(0.10, 0.90 - q0) / 0.90
        gamma_headroom = 1.0 + min(1.5, max(0.0, headroom))

        # 1. STOP Action Handling
        if is_stop or prev_props == new_props:
            delta_q_total = new_props["qed"] - q0
            sim = new_props["similarity"]
            allowed_sas = max(sas0, self.max_sas_base)
            is_success = (delta_q_total >= 0.10 and sim >= self.sim_threshold and new_props["sas"] <= allowed_sas)
            
            if is_success:
                stop_reward = self.success_bonus
            elif delta_q_total > 0 and sim >= self.sim_threshold:
                # Preservation bonus for locking in partial improvements
                stop_reward = self.preservation_weight * delta_q_total
            else:
                stop_reward = 0.0 # No penalty for stopping if no gain
                
            return stop_reward, {
                "reward_mode": "adaptive_v2",
                "valid": True,
                "is_stop": True,
                "stop_reward": stop_reward,
                "total_reward": stop_reward,
                "success": is_success
            }

        # 2. Step Modification Reward
        raw_delta_qed = new_props["qed"] - prev_props["qed"]
        shaped_qed_gain = gamma_headroom * raw_delta_qed

        p_sim_prev = self._compute_sim_penalty(prev_props["similarity"])
        p_sim_new = self._compute_sim_penalty(new_props["similarity"])
        delta_p_sim = p_sim_new - p_sim_prev

        p_sas_prev = self._compute_sas_penalty(prev_props["sas"], sas0)
        p_sas_new = self._compute_sas_penalty(new_props["sas"], sas0)
        delta_p_sas = p_sas_new - p_sas_prev

        step_reward = shaped_qed_gain - delta_p_sim - delta_p_sas - self.step_cost
        
        # Formal success condition check
        is_success = self.check_success(new_props, original_props)
        bonus = self.success_bonus if is_success else 0.0
        total_reward = step_reward + bonus

        diagnostics = {
            "reward_mode": "adaptive_v2",
            "valid": True,
            "q0": q0,
            "gamma_headroom": gamma_headroom,
            "raw_delta_qed": raw_delta_qed,
            "shaped_qed_gain": shaped_qed_gain,
            "delta_p_sim": delta_p_sim,
            "delta_p_sas": delta_p_sas,
            "step_cost": self.step_cost,
            "bonus": bonus,
            "total_reward": total_reward,
            "success": is_success
        }
        return total_reward, diagnostics

    def check_success(self, current_props, original_props):
        if not current_props.get("valid", False):
            return False
        delta_qed = current_props["qed"] - original_props["qed"]
        sim = current_props["similarity"]
        sas = current_props["sas"]
        allowed_sas = max(original_props["sas"], self.max_sas_base)
        return (delta_qed >= 0.10 and sim >= self.sim_threshold and sas <= allowed_sas)


def get_reward_calculator(mode="adaptive", config_path=None):
    """
    Factory function to instantiate either Baseline or Calibrated Adaptive reward calculators.
    """
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            full_cfg = yaml.safe_load(f)
            config = full_cfg.get(f"{mode}_reward", {})
            
    if mode == "baseline":
        from src.rewards.reward import RewardCalculator
        return RewardCalculator(
            step_cost=config.get("step_cost", 0.01),
            sim_threshold=config.get("sim_threshold", 0.40),
            max_sas=config.get("max_sas", 4.5)
        )
    elif mode == "adaptive":
        return CalibratedAdaptiveRewardCalculator(config)
    else:
        raise ValueError(f"Unknown reward mode: {mode}. Must be 'baseline' or 'adaptive'.")


