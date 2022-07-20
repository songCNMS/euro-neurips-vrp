import random
from collections import namedtuple
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from efficient_env.efficient_env_opt import EfficientReplenishmentEnv, Utils


class Inventory_Environment(gym.Env):
    environment_name = "Inventory Management Environment"

    def __init__(self, n_agents, max_capacity, sampler_seq_len, joint_training=False):
        self.n_agents = n_agents
        self.max_capacity = max_capacity
        self.sampler_seq_len = sampler_seq_len
        self.mode = 'train'
        self.joint_training = joint_training
        self.episode_duration_training = Utils.get_env_config()["episod_duration"]
        self.stock_trajectory = np.zeros((self.n_agents, 3, self.episode_duration_training+1))
        self.joint_reset()
        self.reset()
        self.action_space = self.joint_env.action_space
        self.observation_space = self.joint_env.observation_space
        self._max_episode_steps = self.sampler_seq_len
        self.trials = 10
        self.reward_threshold = np.float("inf")
        self.id = "Inventory"
        self.num_states = self.joint_env.state_dim
        self.reward_norm = self.joint_env.max_price * np.median(self.joint_env.demand_mean)

    def reset(self, relative_start_day=None, local_agent_idx=None):
        if local_agent_idx is not None: self.local_agent_idx = local_agent_idx
        else: self.local_agent_idx = np.random.randint(1, self.n_agents+1)
        self.C_trajectory = np.zeros((3, self.sampler_seq_len + 1))
        demand_mean = self.joint_env.demand_mean[self.local_agent_idx-1]
        sampled_capacity = int((1.0 + np.random.random() * 20.0) * demand_mean)
        if self.joint_training:
            self.local_env = EfficientReplenishmentEnv(
                n_agents=self.n_agents, max_capacity=self.max_capacity, hist_len=7, 
                C_trajectory=self.C_trajectory, local_SKU=self.local_agent_idx,
                relative_start_day=relative_start_day,
                sampler_seq_len=self.sampler_seq_len, mode=self.mode
            )
            relative_start_day = self.local_env.tracker_offset
            for i in range(self.sampler_seq_len):
                for j in range(3):
                    self.C_trajectory[j, i] = max(demand_mean,
                                                    np.sum(self.stock_trajectory[:, j, relative_start_day+i]) 
                                                        - self.stock_trajectory[self.local_agent_idx-1, j, relative_start_day+i])
            self.local_env.set_C_trajectory(self.C_trajectory)
        else:
            self.local_env = EfficientReplenishmentEnv(
                n_agents=self.n_agents, max_capacity=sampled_capacity, hist_len=7, 
                C_trajectory=self.C_trajectory, local_SKU=self.local_agent_idx,
                relative_start_day=relative_start_day,
                sampler_seq_len=self.sampler_seq_len, mode=self.mode
            )

        self.sku_name = self.local_env.sku_names[0]
        self.state = self.local_env.reset()[0]
        self.next_state = None
        self.reward = 0.0
        self.done = False
        self.episode_steps = 0
        return self.state

    def joint_reset(self, relative_start_day=None, sampler_seq_len=None):
        _sampler_seq_len = (sampler_seq_len if sampler_seq_len else self.sampler_seq_len)
        self.joint_env = EfficientReplenishmentEnv(
        n_agents=self.n_agents, max_capacity=self.max_capacity, 
        hist_len=7, C_trajectory=None, local_SKU=None,
        relative_start_day=relative_start_day,
        sampler_seq_len=_sampler_seq_len)
        states = self.joint_env.reset()
        states = np.clip(states, 0.0, 20.0)
        return states
        
    def joint_step(self, actions):
        actions = [(a+1.0)/2.0 for a in actions]
        states, rewards, dones, infos = self.joint_env.step(actions)
        states = np.clip(states, 0.0, 20.0)
        rewards = [reward / self.reward_norm for reward in rewards]
        return states, rewards, dones, infos

    def step(self, action):
        if np.isscalar(action):
            action = [action]
        action = [(a+1.0)/2.0 for a in action]
        states, rewards, dones, _ = self.local_env.step(action)
        self.state = np.clip(states[0], 0.0, 20.0)
        self.reward = rewards[0] / self.reward_norm
        self.done = dones[0]
        assert (not np.isnan(self.state).any()), f"nan state {self.state}"
        return self.state, self.reward, self.done, {}

    def switch_mode(self, mode):
        self.mode = mode
        self.local_env.switch_mode(mode)
        self.joint_env.switch_mode(mode)



