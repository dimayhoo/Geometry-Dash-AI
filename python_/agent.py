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

class GameEnv(gym.Env):
    def __init__(self, lvl_name, state_width, state_height, start_pos):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space
        self.state
        self.state_width
        self.state_height

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


def get_action(env, model, random=False):
    if random:
        return env.action_space.sample()
    
    return model.predict(env, env.state)



