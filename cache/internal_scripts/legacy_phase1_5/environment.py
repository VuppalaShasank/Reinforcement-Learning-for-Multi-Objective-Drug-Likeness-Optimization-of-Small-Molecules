import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rdkit import Chem

from implementation.molecule_editor import MoleculeEditor
from implementation.reward import RewardCalculator
from implementation.properties import calculate_properties

class MolecularEnvironment(gym.Env):
    def __init__(self, max_steps=20):
        super(MolecularEnvironment, self).__init__()
        self.max_steps = max_steps
        self.editor = MoleculeEditor()
        self.reward_calc = RewardCalculator()
        
        # Action space: 
        # 0: Add Atom, 1: Remove Atom, 2: Substitute Atom, 3: Change Bond Order, 4: STOP
        # Since this requires specifying WHICH atom, it's actually a parameterized action space.
        # For simplicity in this mid-review basic env, we can represent actions as a dict or tuple.
        # We will keep it simple: just passing an action dictionary for direct interaction.
        self.action_space = spaces.Dict({
            'type': spaces.Discrete(5),
            'atom1': spaces.Discrete(38), # Max atoms 38
            'atom2': spaces.Discrete(38), 
            'element': spaces.Discrete(len(self.editor.allowed_elements)),
            'bond_delta': spaces.Discrete(3) # e.g. -1, +1
        })
        
        self.observation_space = spaces.Dict({
            'qed': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'sas': spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
            'similarity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'step': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            # In a full GNN this would include the graph data
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'starting_smiles' in options:
            self.original_mol = Chem.MolFromSmiles(options['starting_smiles'])
        else:
            # Default fallback for testing
            self.original_mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O') # Aspirin
            
        self.current_mol = self.original_mol
        self.original_props = calculate_properties(self.original_mol)
        self.current_props = calculate_properties(self.current_mol, self.original_mol)
        
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            'qed': np.array([self.current_props['qed']], dtype=np.float32),
            'sas': np.array([self.current_props['sas']], dtype=np.float32),
            'similarity': np.array([self.current_props['similarity']], dtype=np.float32),
            'step': np.array([self.current_step / self.max_steps], dtype=np.float32)
        }

    def step(self, action):
        action_type = action['type']
        
        new_mol = None
        if action_type == 4: # STOP
            done = True
            new_mol = self.current_mol
        else:
            done = False
            # Apply action
            if action_type == 0: # Add
                elem = self.editor.allowed_elements[action['element']]
                new_mol = self.editor.add_atom(self.current_mol, action['atom1'], elem)
            elif action_type == 1: # Remove
                new_mol = self.editor.remove_atom(self.current_mol, action['atom1'])
            elif action_type == 2: # Substitute
                elem = self.editor.allowed_elements[action['element']]
                new_mol = self.editor.substitute_atom(self.current_mol, action['atom1'], elem)
            elif action_type == 3: # Change Bond Order
                delta = action['bond_delta'] - 1 # Map 0, 1, 2 to -1, 0, +1
                new_mol = self.editor.change_bond_order(self.current_mol, action['atom1'], action['atom2'], delta)

        if new_mol is None:
            # Invalid action
            reward, info = self.reward_calc.calculate_reward(self.current_props, {'valid': False}, self.original_props['sas'])
            # We don't advance the state or step count for invalid actions, but they penalize the agent.
            # In some frameworks, episode terminates. We'll just return the penalty and same state.
            return self._get_obs(), reward, False, False, info
            
        new_props = calculate_properties(new_mol, self.original_mol)
        reward, info = self.reward_calc.calculate_reward(self.current_props, new_props, self.original_props['sas'])
        
        self.current_mol = new_mol
        self.current_props = new_props
        self.current_step += 1
        
        if self.current_step >= self.max_steps:
            done = True
            
        if self.reward_calc.check_success(self.current_props, self.original_props):
            reward += 10.0 # Terminal bonus
            done = True
            info['success'] = True
            
        return self._get_obs(), reward, done, False, info
