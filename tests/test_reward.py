import unittest
from src.rewards.adaptive_reward import CalibratedAdaptiveRewardCalculator

class TestRewardFunctions(unittest.TestCase):
    def setUp(self):
        self.calc = CalibratedAdaptiveRewardCalculator()

    def test_positive_delta_qed(self):
        props_prev = {"qed": 0.50, "sas": 3.0, "similarity": 1.0, "valid": True}
        props_curr = {"qed": 0.60, "sas": 3.0, "similarity": 0.85, "valid": True}
        props_init = {"qed": 0.50, "sas": 3.0, "similarity": 1.0, "valid": True}
        
        rew, info = self.calc.calculate_reward(props_prev, props_curr, props_init, is_stop=False)
        self.assertGreater(rew, 0.0)

    def test_stop_action_preservation_bonus(self):
        props_curr = {"qed": 0.65, "sas": 3.0, "similarity": 0.80, "valid": True}
        props_init = {"qed": 0.50, "sas": 3.0, "similarity": 1.0, "valid": True}
        
        rew_stop, info_stop = self.calc.calculate_reward(props_curr, props_curr, props_init, is_stop=True)
        # Should include stop preservation reward
        self.assertGreater(rew_stop, 0.0)
        self.assertIn("stop_reward", info_stop)

if __name__ == '__main__':
    unittest.main()
