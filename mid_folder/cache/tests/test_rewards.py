import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Set path to import implementation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from implementation.reward_module import BaselineRewardCalculator, AdaptiveRewardCalculator, get_reward_calculator

def run_reward_tests():
    print("=== Testing Reward Formulations: Baseline vs Adaptive ===")
    config_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "configs", "reward_config.yaml")
    
    r_base = get_reward_calculator("baseline", config_path)
    r_adapt = get_reward_calculator("adaptive", config_path)

    orig = {"qed": 0.50, "sas": 3.0, "similarity": 1.0, "valid": True}
    prev = {"qed": 0.50, "sas": 3.0, "similarity": 1.0, "valid": True}

    # Test 1: Positive QED edit without violating constraints
    new_good = {"qed": 0.58, "sas": 3.1, "similarity": 0.85, "valid": True}
    rew_b1, diag_b1 = r_base.calculate_reward(prev, new_good, orig)
    rew_a1, diag_a1 = r_adapt.calculate_reward(prev, new_good, orig)
    print(f"Test 1 (Positive QED): Baseline={rew_b1:.4f}, Adaptive={rew_a1:.4f}")
    assert rew_b1 > 0 and rew_a1 > 0, "Positive edit must yield positive reward"

    # Test 2: Edit improving QED but breaking similarity constraint (< 0.4)
    new_viol_sim = {"qed": 0.75, "sas": 3.1, "similarity": 0.30, "valid": True}
    rew_b2, diag_b2 = r_base.calculate_reward(prev, new_viol_sim, orig)
    rew_a2, diag_a2 = r_adapt.calculate_reward(prev, new_viol_sim, orig)
    print(f"Test 2 (Sim Violation sim=0.30): Baseline={rew_b2:.4f}, Adaptive={rew_a2:.4f}")
    assert rew_b2 < 0 and rew_a2 < 0, "Similarity violation must be penalized"

    # Test 3: Edit worsening SAS (SAS > 4.5)
    new_viol_sas = {"qed": 0.55, "sas": 5.5, "similarity": 0.80, "valid": True}
    rew_b3, diag_b3 = r_base.calculate_reward(prev, new_viol_sas, orig)
    rew_a3, diag_a3 = r_adapt.calculate_reward(prev, new_viol_sas, orig)
    print(f"Test 3 (SAS Violation sas=5.5): Baseline={rew_b3:.4f}, Adaptive={rew_a3:.4f}")
    assert rew_b3 < rew_b1 and rew_a3 < rew_a1, "SAS penalty must degrade reward"

    # Test 4: Invalid action
    new_invalid = {"valid": False}
    rew_b4, diag_b4 = r_base.calculate_reward(prev, new_invalid, orig)
    rew_a4, diag_a4 = r_adapt.calculate_reward(prev, new_invalid, orig)
    print(f"Test 4 (Invalid Action): Baseline={rew_b4:.4f}, Adaptive={rew_a4:.4f}")
    assert rew_b4 == -1.0 and rew_a4 == -1.0, "Invalid action must receive -1.0 penalty"

    # Test 5: Unnecessary edit (identical molecule)
    rew_b5, diag_b5 = r_base.calculate_reward(prev, prev, orig)
    rew_a5, diag_a5 = r_adapt.calculate_reward(prev, prev, orig)
    print(f"Test 5 (Identical Edit / Step Cost): Baseline={rew_b5:.4f}, Adaptive={rew_a5:.4f}")
    assert rew_b5 == -0.01 and rew_a5 == -0.01, "Zero-progress edit must cost exactly step_cost"

    # Test 6: Low headroom molecule (q0 = 0.94)
    orig_high = {"qed": 0.94, "sas": 2.5, "similarity": 1.0, "valid": True}
    prev_high = {"qed": 0.94, "sas": 2.5, "similarity": 1.0, "valid": True}
    new_high = {"qed": 0.96, "sas": 2.5, "similarity": 0.95, "valid": True}
    rew_b6, diag_b6 = r_base.calculate_reward(prev_high, new_high, orig_high)
    rew_a6, diag_a6 = r_adapt.calculate_reward(prev_high, new_high, orig_high)
    print(f"Test 6 (Low Headroom q0=0.94, +0.02 delta): Baseline={rew_b6:.4f}, Adaptive={rew_a6:.4f}")
    assert np.isfinite(rew_a6), "Adaptive reward must handle near-zero headroom safely"
    assert rew_a6 > rew_b6, "Adaptive reward must scale smaller delta QED when headroom is restricted"

    # Test 7: Similarity near threshold (0.42 vs 0.38)
    near_sim_safe = {"qed": 0.60, "sas": 3.0, "similarity": 0.42, "valid": True}
    near_sim_violation = {"qed": 0.60, "sas": 3.0, "similarity": 0.38, "valid": True}
    rew_a_safe, _ = r_adapt.calculate_reward(prev, near_sim_safe, orig)
    rew_a_viol, _ = r_adapt.calculate_reward(prev, near_sim_violation, orig)
    print(f"Test 7 (Sim Border: 0.42 vs 0.38): Safe={rew_a_safe:.4f}, Violation={rew_a_viol:.4f}")
    assert rew_a_safe > rew_a_viol, "Safe border must yield higher reward than violation"

    # Test 8: Success Condition bonus
    success_state = {"qed": 0.62, "sas": 3.0, "similarity": 0.70, "valid": True}
    rew_b_succ, diag_b_succ = r_base.calculate_reward(prev, success_state, orig)
    rew_a_succ, diag_a_succ = r_adapt.calculate_reward(prev, success_state, orig)
    print(f"Test 8 (Success Terminal Bonus): Baseline={rew_b_succ:.4f}, Adaptive={rew_a_succ:.4f}")
    assert diag_b_succ["success"] is True and diag_a_succ["success"] is True
    assert rew_b_succ >= 10.0 and rew_a_succ >= 10.0

    print("\nALL REWARD UNIT TESTS PASSED SUCCESSFULLY!")

    # Generate diagnostic plot comparing penalties
    sims = np.linspace(0.2, 1.0, 100)
    p_sim_base = [(r_base.sim_threshold - s) * r_base.sim_penalty_weight if s < r_base.sim_threshold else 0 for s in sims]
    p_sim_adapt = [r_adapt._compute_sim_penalty(s) for s in sims]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(sims, p_sim_base, 'r--', label='Baseline Penalty $P_{sim}$')
    plt.plot(sims, p_sim_adapt, 'b-', label='Adaptive Barrier Penalty')
    plt.axvline(0.4, color='k', linestyle=':', label='Constraint $\\tau_{sim}=0.4$')
    plt.title('Similarity Penalty Comparison')
    plt.xlabel('Tanimoto Similarity to $M_0$')
    plt.ylabel('Penalty Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    sass = np.linspace(2.0, 7.0, 100)
    p_sas_base = [(s - 4.5) * r_base.sas_penalty_weight if s > 4.5 else 0 for s in sass]
    p_sas_adapt = [r_adapt._compute_sas_penalty(s, 3.0) for s in sass]

    plt.subplot(1, 2, 2)
    plt.plot(sass, p_sas_base, 'r--', label='Baseline Penalty $P_{SAS}$')
    plt.plot(sass, p_sas_adapt, 'b-', label='Adaptive Barrier Penalty')
    plt.axvline(4.5, color='k', linestyle=':', label='Max Allowed SAS = 4.5')
    plt.title('SAS Penalty Comparison')
    plt.xlabel('Synthetic Accessibility Score (SAS)')
    plt.ylabel('Penalty Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "reward_comparison_diagnostics.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Diagnostic plot saved to: {out_path}")

if __name__ == "__main__":
    run_reward_tests()
