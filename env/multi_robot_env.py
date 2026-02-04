import numpy as np
from gymnasium.spaces import Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from .sim.marl_sim_env import MARL_SIM

class MultiRobotEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.self_obs_dim = config.get("self_obs_dim", 4)
        self.neighbor_obs_dim = config.get("neighbor_obs_dim", 4)
        self.num_use_neighbor_obs = config.get("num_use_neighbor_obs", 6)
        self.H = config.get("H", 80)
        self.num_robots = config.get("num_robots", 4)
        self.env_yaml_path = config.get("env_yaml_path", 'config/random_env.yaml')
        self.disable_plotting = config.get("disable_plotting", False)
        
        # action dim
        self.action_dim = config.get("action_dim", 2)
        
        # observation dim
        self.neigh_flat_dim = self.num_use_neighbor_obs * self.neighbor_obs_dim
        self.sdf_flat_dim = self.H * self.H
        self.obs_dim = self.self_obs_dim + self.neigh_flat_dim + 1 + self.sdf_flat_dim + self.action_dim # +2 for prior direction, same as action dim

        # robots
        self.agents = self.possible_agents = [f'robot_{i}' for i in range(self.num_robots)]
        
        # sim
        self.sim = MARL_SIM(self.env_yaml_path, disable_plotting=self.disable_plotting)
    
    def get_observation_space(self, agent_id):
        if agent_id.startswith("robot_"):
            return Box(low=-np.inf,
                       high=np.inf,
                       shape=(self.obs_dim,),
                       dtype=np.float32
                    )
        else:
            raise ValueError(f"bad agent id: {agent_id}!")
    
    # action是α和theta, 范围[-1, 1]
    def get_action_space(self, agent_id):
        if agent_id.startswith("robot_"):
            return Box(low=-1.0,
                       high=1.0,
                       shape=(self.action_dim,),
                       dtype=np.float32
                    )
        else:
            raise ValueError(f"bad agent id: {agent_id}!")

    # ------- RLlib 接口：reset -------
    def reset(self, *, seed=None, options=None):
        obs_dict, info_dict = self.sim.reset()
        return obs_dict, info_dict

    # ------- RLlib 接口：step -------
    def step(self, action_dict):
        obs, rewards, terminateds, truncateds, infos = self.sim.step(action_dict)
        return obs, rewards, terminateds, truncateds, infos