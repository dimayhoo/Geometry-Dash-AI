from stable_baselines3 import PPO, DQN, A2C, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import gymnasium as gym
from gymnasium import spaces
#import torch as th

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
    # TODO: should I include dtypes?
    def __init__(self, env_data):
        super().__init__()
        #state_shape, state_dtype, other_params_shape
        self.action_space = spaces.Discrete(2)
        # low and high values are taken from objects encodings. They are only approximated.
        self.observation_space = spaces.Dict({
            "lvl_frame": spaces.Box(
                # TODO: if you load the model, load observation space from there (because min and max values differ) or just use global max and global min amon all levels.
                low=env_data["min_max"][0], 
                high=env_data["min_max"][1], 
                shape=env_data["state_shape"], 
                dtype=env_data["state_dtype"]),
            "other_params": spaces.MultiBinary(env_data["other_params_len"])
        })
        self.observations_data = [] # Will be dynamically initialised
        self.state = None

        self.reset()

    def step(self, action):
        #self.state = self.states_data.pop(0)...
        
        # NOTE: there is no done in my env and that's okay, because 
        # data isn't generated in real time
        """
        Executes an action and returns the resulting observation, reward, done, and info.
        
        Args:
            action (int): The action taken by the agent.
        
        Returns:
            tuple:
                - observation (dict): The new observation after the action.
                - reward (float): The reward obtained from taking the action.
                - done (bool): Whether the episode has ended.
                - info (dict): Additional information.
        """
        if not self.observations_data:
            # If no more data, end the episode
            done = True
            observation = self.observation_space.sample()  # Replace with actual observation
            reward = 0.0
            info = {}
        
        else:
            # Implement action effects and environment transitions here
            # Example placeholders:
            observation, action, done, death_dict = self.observations_data.pop()
            reward = self.calculate_reward(action, done, death_dict)
            info = {}
            print(reward, done)
        
        trancated = False
        
        return observation, reward, done, trancated,  info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.observations_data = []
        initial_observation = self.observation_space.sample()
        info = {}
        return initial_observation, info

    def render(self):
        pass

    def close(self):
        return super().close()

    def calculate_reward(self, action, done, death_dict):
        # NOTE: if we use negative reward to jumps, then we should
        # include a condition "is there a jump in X blocks before current state?"
        # because otherwise agent will learn to avoid jumping in the moment 
        # hoping of a future action. This isn't preferable. Moreover, I need 
        # the opposite: to make a jump in front of an obstacle as early as possible. 
        reward = 0

        if done:
            reward -= death_dict / 10
            if action:
                reward -= 5

        elif action:
            reward -= 0.05
        
        return reward




