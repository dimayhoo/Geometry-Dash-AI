from stable_baselines3 import PPO, Environment, DQN, A2C, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch as th


# one jump is approximately 140.2 axmol units. 
# Should be tested multiple times to be absolutely sure. 
X_JUMP_LENGTH = 140


'''ONE_CUBE_LENGTH
ONE_BLOCK_LENGTH
BLOCKS_PER_STATE
STATE_WIDTH
STATE_HEIGHT'''

''' My way of handling steps
External: Running agent and storing states, acitons, rewards.
In the env, extract states in some order (maybe, purely random) from 
    a some source and using as steps.
'''

class GameEnv(gym.Env):
    def __init__(self, lvl_id):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space

    def step(self, action):
        pass


    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        return super().close()
    
    def estimate_reward(self):
        pass

    def get_level_state(self):
        pass

    def get_reward(self):
        # NOTE: if we use negative reward to jumps, then we should
        # include a condition "is there a jump in X blocks before current state?"
        # because otherwise agent will learn to avoid jumping in the moment 
        # hoping of a future action. This isn't preferable. Moreover, I need 
        # the opposite: to make a jump in front of an obstacle as early as possible. 
        pass




