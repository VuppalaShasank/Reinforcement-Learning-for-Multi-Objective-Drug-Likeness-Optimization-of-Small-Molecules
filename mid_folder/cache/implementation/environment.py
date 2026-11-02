import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from rdkit import Chem

from implementation.molecule_editor import MoleculeEditor
from implementation.reward_module import get_reward_calculator
from implementation.graph_utils import mol_to_pyg_data
from implementation.action_handler import TOTAL_ACTIONS, get_action_mask, apply_action, action_to_details
from implementation.properties import calculate_properties

class MolecularGraphEnv(gym.Env):
    """
    Standard Gymnasium Environment for Molecular Optimization.
    Observations include:
    - Graph representation (PyTorch Geometric Data object)
    - Scalar features: [t/T, QED, SAS, Similarity, QED_0, SAS_0]
    - Action mask: boolean array of shape [1901] ensuring valid action selection
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, reward_mode="adaptive", max_steps=20, config_path=None):
        super(MolecularGraphEnv, self).__init__()
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.editor = MoleculeEditor()
        self.reward_calc = get_reward_calculator(mode=reward_mode, config_path=config_path)

        # Action space: 1901 discrete actions (STOP + ADD + REMOVE + SUBSTITUTE + BOND)
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)
        
        # Observation space placeholder
        self.observation_space = spaces.Dict({
            "scalars": spaces.Box(low=0.0, high=10.0, shape=(6,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(TOTAL_ACTIONS,), dtype=np.int8)
        })

        self.original_mol = None
        self.current_mol = None
        self.original_props = None
        self.current_props = None
        self.current_step = 0
        self.trajectory_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "starting_smiles" in options:
            smiles = options["starting_smiles"]
            self.original_mol = Chem.MolFromSmiles(smiles)
        else:
            # Default fallback molecule: Aspirin
            self.original_mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")

        self.current_mol = Chem.Mol(self.original_mol)
        self.original_props = calculate_properties(self.original_mol)
        self.current_props = calculate_properties(self.current_mol, self.original_mol)
        self.current_step = 0
        self.trajectory_history = [Chem.MolToSmiles(self.current_mol)]

        obs = self._get_obs()
        info = {
            "smiles": Chem.MolToSmiles(self.current_mol),
            "qed": self.current_props["qed"],
            "sas": self.current_props["sas"],
            "similarity": 1.0,
            "success": False
        }
        return obs, info

    def _get_obs(self):
        pyg_data = mol_to_pyg_data(
            mol=self.current_mol,
            current_step=self.current_step,
            max_steps=self.max_steps,
            qed=self.current_props["qed"],
            sas=self.current_props["sas"],
            sim=self.current_props["similarity"],
            q0=self.original_props["qed"],
            sas_0=self.original_props["sas"]
        )
        mask = get_action_mask(self.current_mol, self.editor)
        scalars = np.array([
            self.current_step / self.max_steps,
            self.current_props["qed"],
            self.current_props["sas"],
            self.current_props["similarity"],
            self.original_props["qed"],
            self.original_props["sas"]
        ], dtype=np.float32)

        return {
            "pyg_data": pyg_data,
            "scalars": scalars,
            "action_mask": mask
        }

    def step(self, action_idx):
        action_idx = int(action_idx)
        details = action_to_details(action_idx)
        terminated = False
        truncated = False

        # 1. STOP action
        if details["type"] == "STOP":
            terminated = True
            # Zero-change potential transition with step cost
            reward, diag = self.reward_calc.calculate_reward(self.current_props, self.current_props, self.original_props)
            info = {
                "action": details,
                "valid": True,
                "reward_diagnostics": diag,
                "success": diag.get("success", False),
                "terminal_stop": True
            }
            obs = self._get_obs()
            return obs, reward, terminated, truncated, info

        # 2. Apply modification
        new_mol = apply_action(self.current_mol, action_idx, self.editor)
        
        # 3. Invalid action handling
        if new_mol is None:
            reward, diag = self.reward_calc.calculate_reward(self.current_props, {"valid": False}, self.original_props)
            info = {
                "action": details,
                "valid": False,
                "reward_diagnostics": diag,
                "success": False
            }
            obs = self._get_obs()
            # In molecular RL, invalid action receives penalty; episode does not terminate unless max steps reached
            self.current_step += 1
            if self.current_step >= self.max_steps:
                truncated = True
            return obs, reward, terminated, truncated, info

        # 4. Valid action handling
        new_props = calculate_properties(new_mol, self.original_mol)
        reward, diag = self.reward_calc.calculate_reward(self.current_props, new_props, self.original_props)

        self.current_mol = new_mol
        self.current_props = new_props
        self.current_step += 1
        self.trajectory_history.append(Chem.MolToSmiles(self.current_mol))

        is_success = diag.get("success", False)
        if is_success:
            terminated = True # Reached success termination condition
        elif self.current_step >= self.max_steps:
            truncated = True

        info = {
            "action": details,
            "valid": True,
            "reward_diagnostics": diag,
            "success": is_success,
            "smiles": Chem.MolToSmiles(self.current_mol),
            "qed": self.current_props["qed"],
            "sas": self.current_props["sas"],
            "similarity": self.current_props["similarity"]
        }
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

from implementation.reward import RewardCalculator

class MolecularEnvironment(gym.Env):
    """
    Phase 1-5 Prototype Environment with dictionary-based parameterized actions.
    Maintained for mid-review interactive demonstration and property tracking.
    """
    def __init__(self, max_steps=20):
        super(MolecularEnvironment, self).__init__()
        self.max_steps = max_steps
        self.editor = MoleculeEditor()
        self.reward_calc = RewardCalculator()

        self.action_space = spaces.Dict({
            'type': spaces.Discrete(5),
            'atom1': spaces.Discrete(38),
            'atom2': spaces.Discrete(38),
            'element': spaces.Discrete(len(self.editor.allowed_elements)),
            'bond_delta': spaces.Discrete(3)
        })

        self.observation_space = spaces.Dict({
            'qed': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'sas': spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
            'similarity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'step': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'starting_smiles' in options:
            self.original_mol = Chem.MolFromSmiles(options['starting_smiles'])
        else:
            self.original_mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')

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
        if isinstance(action, (int, np.integer)):
            # If called with integer action, delegate or adapt
            action = {"type": 4} if action == 0 else {"type": 0, "atom1": 0, "element": 0, "atom2": 0, "bond_delta": 1}

        action_type = action['type']
        new_mol = None
        if action_type == 4: # STOP
            done = True
            new_mol = self.current_mol
        else:
            done = False
            if action_type == 0: # Add
                elem = self.editor.allowed_elements[action['element']]
                new_mol = self.editor.add_atom(self.current_mol, action['atom1'], elem)
            elif action_type == 1: # Remove
                new_mol = self.editor.remove_atom(self.current_mol, action['atom1'])
            elif action_type == 2: # Substitute
                elem = self.editor.allowed_elements[action['element']]
                new_mol = self.editor.substitute_atom(self.current_mol, action['atom1'], elem)
            elif action_type == 3: # Change Bond Order
                delta = action['bond_delta'] - 1
                new_mol = self.editor.change_bond_order(self.current_mol, action['atom1'], action['atom2'], delta)

        if new_mol is None:
            reward, info = self.reward_calc.calculate_reward(self.current_props, {'valid': False}, self.original_props['sas'])
            return self._get_obs(), reward, False, False, info

        new_props = calculate_properties(new_mol, self.original_mol)
        reward, info = self.reward_calc.calculate_reward(self.current_props, new_props, self.original_props['sas'])

        self.current_mol = new_mol
        self.current_props = new_props
        self.current_step += 1

        if self.current_step >= self.max_steps:
            done = True

        if self.reward_calc.check_success(self.current_props, self.original_props):
            reward += 10.0
            done = True
            info['success'] = True

        return self._get_obs(), reward, done, False, info


