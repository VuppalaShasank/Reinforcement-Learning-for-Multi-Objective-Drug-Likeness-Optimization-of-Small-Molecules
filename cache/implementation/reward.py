class RewardCalculator:
    def __init__(self, step_cost=0.01, sim_threshold=0.4, max_sas=4.5):
        self.step_cost = step_cost
        self.sim_threshold = sim_threshold
        self.max_sas_base = max_sas
        
    def _phi(self, qed, sas, sim, sas_0):
        # Calculate P_sim
        p_sim = 0.0
        if sim < self.sim_threshold:
            p_sim = (self.sim_threshold - sim) * 10.0 # penalty multiplier
            
        # Calculate P_SAS
        p_sas = 0.0
        allowed_sas = max(sas_0, self.max_sas_base)
        if sas > allowed_sas:
            p_sas = (sas - allowed_sas) * 1.0 # penalty multiplier
            
        return qed - p_sim - p_sas

    def calculate_reward(self, prev_props, new_props, sas_0):
        """
        Calculates the step reward based on the change in the potential function Phi.
        prev_props: dict containing 'qed', 'sas', 'similarity', 'valid'
        new_props: dict containing 'qed', 'sas', 'similarity', 'valid'
        """
        # Invalid actions get a negative reward
        if not new_props['valid']:
            return -1.0, {"invalid_penalty": -1.0}
            
        phi_t = self._phi(prev_props['qed'], prev_props['sas'], prev_props['similarity'], sas_0)
        phi_t_next = self._phi(new_props['qed'], new_props['sas'], new_props['similarity'], sas_0)
        
        reward = phi_t_next - phi_t - self.step_cost
        
        # Check success condition for terminal bonus
        success = False
        delta_qed = new_props['qed'] - prev_props['qed'] # actually from M0 to M_t? Wait, Submission 2 says: Delta QED >= 0.1
        # To be precise, delta QED is typically from M0. Let's assume prev_props contains original QED.
        # But this function only sees prev_props. We will pass delta_qed explicitly if needed, but let's just return the step reward here.
        
        components = {
            "phi_t": phi_t,
            "phi_t_next": phi_t_next,
            "step_reward": reward
        }
        return reward, components
        
    def check_success(self, current_props, original_props):
        delta_qed = current_props['qed'] - original_props['qed']
        sim = current_props['similarity']
        sas = current_props['sas']
        allowed_sas = max(original_props['sas'], self.max_sas_base)
        
        return (delta_qed >= 0.1) and (sim >= self.sim_threshold) and (sas <= allowed_sas) and current_props['valid']
