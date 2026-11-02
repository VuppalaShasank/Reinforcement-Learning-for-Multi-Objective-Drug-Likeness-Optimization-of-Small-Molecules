import os
import yaml
import numpy as np

class BaselineRewardCalculator:
    """
    Implements the baseline potential-based reward function from the approved project proposal:
    Phi(M) = QED(M) - P_sim(M) - P_SAS(M)
    r_t = Phi(M_{t+1}) - Phi(M_t) - c
    """
    def __init__(self, config=None):
        if config is None:
            config = {
                "step_cost": 0.01,
                "sim_threshold": 0.4,
                "sim_penalty_weight": 10.0,
                "max_sas_base": 4.5,
                "sas_penalty_weight": 1.0,
                "invalid_penalty": -1.0,
                "terminal_bonus": 10.0,
                "success_delta_qed": 0.1,
                "success_min_sim": 0.4
            }
        self.step_cost = float(config.get("step_cost", 0.01))
        self.sim_threshold = float(config.get("sim_threshold", 0.4))
        self.sim_penalty_weight = float(config.get("sim_penalty_weight", 10.0))
        self.max_sas_base = float(config.get("max_sas_base", 4.5))
        self.sas_penalty_weight = float(config.get("sas_penalty_weight", 1.0))
        self.invalid_penalty = float(config.get("invalid_penalty", -1.0))
        self.terminal_bonus = float(config.get("terminal_bonus", 10.0))
        self.success_delta_qed = float(config.get("success_delta_qed", 0.1))
        self.success_min_sim = float(config.get("success_min_sim", 0.4))

    def _phi(self, qed, sas, sim, sas_0):
        # Similarity penalty
        p_sim = 0.0
        if sim < self.sim_threshold:
            p_sim = (self.sim_threshold - sim) * self.sim_penalty_weight

        # SAS penalty
        p_sas = 0.0
        allowed_sas = max(sas_0, self.max_sas_base)
        if sas > allowed_sas:
            p_sas = (sas - allowed_sas) * self.sas_penalty_weight

        return qed - p_sim - p_sas

    def calculate_reward(self, prev_props, new_props, original_props):
        """
        Calculates the baseline potential-difference step reward.
        prev_props, new_props, original_props: dicts containing 'qed', 'sas', 'similarity', 'valid'
        """
        if not new_props.get("valid", False):
            return self.invalid_penalty, {
                "reward_mode": "baseline",
                "valid": False,
                "step_reward": self.invalid_penalty,
                "invalid_penalty": self.invalid_penalty,
                "delta_phi": 0.0,
                "step_cost": self.step_cost,
                "success": False
            }

        sas_0 = original_props["sas"]
        phi_t = self._phi(prev_props["qed"], prev_props["sas"], prev_props["similarity"], sas_0)
        phi_t_next = self._phi(new_props["qed"], new_props["sas"], new_props["similarity"], sas_0)

        delta_phi = phi_t_next - phi_t
        step_reward = delta_phi - self.step_cost

        # Success check against original starting molecule M0
        is_success = self.check_success(new_props, original_props)
        bonus = self.terminal_bonus if is_success else 0.0
        total_reward = step_reward + bonus

        diagnostics = {
            "reward_mode": "baseline",
            "valid": True,
            "phi_t": phi_t,
            "phi_t_next": phi_t_next,
            "delta_phi": delta_phi,
            "step_cost": self.step_cost,
            "bonus": bonus,
            "step_reward": step_reward,
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
        return (delta_qed >= self.success_delta_qed and 
                sim >= self.success_min_sim and 
                sas <= allowed_sas)


class AdaptiveRewardCalculator:
    """
    Proposed Novel Research Contribution:
    Headroom- and Constraint-Slack Adaptive Reward Shaping.
    
    Dynamically balances QED optimization and molecular constraint adherence based on:
    1. QED Headroom: H(q0) = max(1.0 - q0, eps). Normalizes Delta QED by available headroom.
    2. Similarity Slack: Slack_sim = Sim(M, M0) - tau_sim. Applies quadratic barrier penalties
       as similarity approaches tau_sim, and hard linear violation penalties below it.
    3. SAS Slack: Slack_sas = tau_sas - SAS(M). Applies quadratic barrier penalty near tau_sas,
       and hard linear violation penalties above it.
    4. Step cost to penalize superfluous edits.
    """
    def __init__(self, config=None):
        if config is None:
            config = {
                "step_cost": 0.01,
                "sim_threshold": 0.4,
                "sim_margin_buffer": 0.15,
                "sim_soft_penalty_weight": 2.0,
                "sim_hard_penalty_weight": 10.0,
                "sim_violation_base": 2.0,
                "max_sas_base": 4.5,
                "sas_margin_buffer": 0.50,
                "sas_soft_penalty_weight": 1.0,
                "sas_hard_penalty_weight": 5.0,
                "sas_violation_base": 1.0,
                "headroom_min_epsilon": 0.05,
                "headroom_weight": 1.0,
                "invalid_penalty": -1.0,
                "terminal_bonus": 10.0,
                "success_delta_qed": 0.1,
                "success_min_sim": 0.4
            }
        self.step_cost = float(config.get("step_cost", 0.01))
        self.sim_threshold = float(config.get("sim_threshold", 0.4))
        self.sim_margin_buffer = float(config.get("sim_margin_buffer", 0.15))
        self.sim_soft_weight = float(config.get("sim_soft_penalty_weight", 2.0))
        self.sim_hard_weight = float(config.get("sim_hard_penalty_weight", 10.0))
        self.sim_violation_base = float(config.get("sim_violation_base", 1.0))
        
        self.max_sas_base = float(config.get("max_sas_base", 4.5))
        self.sas_margin_buffer = float(config.get("sas_margin_buffer", 0.50))
        self.sas_soft_weight = float(config.get("sas_soft_penalty_weight", 1.0))
        self.sas_hard_weight = float(config.get("sas_hard_penalty_weight", 5.0))
        self.sas_violation_base = float(config.get("sas_violation_base", 0.5))

        self.headroom_eps = float(config.get("headroom_min_epsilon", 0.05))
        self.headroom_weight = float(config.get("headroom_weight", 1.0))
        self.invalid_penalty = float(config.get("invalid_penalty", -1.0))
        self.terminal_bonus = float(config.get("terminal_bonus", 10.0))
        self.success_delta_qed = float(config.get("success_delta_qed", 0.1))
        self.success_min_sim = float(config.get("success_min_sim", 0.4))

    def _compute_sim_penalty(self, sim):
        slack = sim - self.sim_threshold
        if slack < 0:
            # Violation regime: base violation penalty + linear slope
            return self.sim_violation_base + self.sim_hard_weight * abs(slack)
        elif slack < self.sim_margin_buffer:
            # Buffer warning regime: smooth quadratic penalty
            ratio = 1.0 - (slack / self.sim_margin_buffer)
            return self.sim_soft_weight * (ratio ** 2)
        else:
            return 0.0

    def _compute_sas_penalty(self, sas, sas_0):
        allowed_sas = max(sas_0, self.max_sas_base)
        slack = allowed_sas - sas
        if slack < 0:
            # Violation regime: SAS exceeded
            return self.sas_violation_base + self.sas_hard_weight * abs(slack)
        elif slack < self.sas_margin_buffer:
            # Buffer warning regime: approaching threshold
            ratio = 1.0 - (slack / self.sas_margin_buffer)
            return self.sas_soft_weight * (ratio ** 2)
        else:
            return 0.0

    def calculate_reward(self, prev_props, new_props, original_props):
        if not new_props.get("valid", False):
            return self.invalid_penalty, {
                "reward_mode": "adaptive",
                "valid": False,
                "step_reward": self.invalid_penalty,
                "invalid_penalty": self.invalid_penalty,
                "headroom_gain": 0.0,
                "sim_penalty_change": 0.0,
                "sas_penalty_change": 0.0,
                "step_cost": self.step_cost,
                "success": False
            }

        q0 = original_props["qed"]
        sas_0 = original_props["sas"]
        
        # 1. Headroom adaptive gain
        headroom = max(1.0 - q0, self.headroom_eps)
        raw_delta_qed = new_props["qed"] - prev_props["qed"]
        headroom_gain = self.headroom_weight * (raw_delta_qed / headroom)

        # 2. Penalty changes
        p_sim_prev = self._compute_sim_penalty(prev_props["similarity"])
        p_sim_new = self._compute_sim_penalty(new_props["similarity"])
        delta_p_sim = p_sim_new - p_sim_prev

        p_sas_prev = self._compute_sas_penalty(prev_props["sas"], sas_0)
        p_sas_new = self._compute_sas_penalty(new_props["sas"], sas_0)
        delta_p_sas = p_sas_new - p_sas_prev

        step_reward = headroom_gain - delta_p_sim - delta_p_sas - self.step_cost

        # 3. Success bonus
        is_success = self.check_success(new_props, original_props)
        bonus = self.terminal_bonus if is_success else 0.0
        total_reward = step_reward + bonus

        diagnostics = {
            "reward_mode": "adaptive",
            "valid": True,
            "q0": q0,
            "headroom": headroom,
            "raw_delta_qed": raw_delta_qed,
            "headroom_gain": headroom_gain,
            "p_sim_new": p_sim_new,
            "delta_p_sim": delta_p_sim,
            "p_sas_new": p_sas_new,
            "delta_p_sas": delta_p_sas,
            "step_cost": self.step_cost,
            "bonus": bonus,
            "step_reward": step_reward,
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
        return (delta_qed >= self.success_delta_qed and 
                sim >= self.success_min_sim and 
                sas <= allowed_sas)


def get_reward_calculator(mode="baseline", config_path=None):
    """
    Factory function to instantiate either Baseline or Adaptive reward calculators.
    """
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            full_cfg = yaml.safe_load(f)
            config = full_cfg.get(f"{mode}_reward", {})
            
    if mode == "baseline":
        return BaselineRewardCalculator(config)
    elif mode == "adaptive":
        return AdaptiveRewardCalculator(config)
    else:
        raise ValueError(f"Unknown reward mode: {mode}. Must be 'baseline' or 'adaptive'.")

