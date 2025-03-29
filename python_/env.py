from stable_baselines3 import PPO, DQN, A2C, SAC, TD3, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import gymnasium as gym
from gymnasium import spaces
from constants import COMMUNITY_LEVEL_MAX_HEIGHT_PIXELS, LEVEL_MIN_HEIGHT, BASIC_LEVEL_MAX_HEIGHT_PIXELS
#import torch as th
import numpy as np
from helpers import save_step

# one jump is approximately 140.2 axmol units. 
# Should be tested multiple times to be absolutely sure. 
X_JUMP_LENGTH = 140

''' My way of handling steps
External: Running agent and storing states, acitons, rewards.
In the env, extract states in some order (maybe, purely random) from 
    a some source and using as steps.

Current situation:
The model doesn't learn. It just repeat its actions without any progress.  
The data inputs are examined and different combinations of learning rate, step
    and batch sizes are tried, but none of this succeed. 
The conclusion is expertise-level understanding is required to make this reinforcement
    learning working. 
Another alternative is to use supervised learning over the stored states. This option wasn't tested yet.
'''


class GameEnv(gym.Env):
    # TODO: should I include dtypes?
    def __init__(self, env_data):
        super().__init__()
        #state_shape, state_dtype, other_params_shape
        self.action_space = spaces.Discrete(2)
        
        # NOTE: Gym Env uses low and high values just as a verifiction that the current
        # state is valid and appropriate, and there wasn't an error along the way.
        # They don't affect learning.
        # NOTE: although you must align every observation with the observation
        # space, otherwise model will throw an error.
        
        '''# TODO: if you load the model, load observation space from there 
        (because min and max values differ) or just use global max and global min amon all levels.'''

        box_space = spaces.Box(low=LEVEL_MIN_HEIGHT, # I suspect it's ground level's 105, but I am not sure.
                high=BASIC_LEVEL_MAX_HEIGHT_PIXELS, 
                shape=env_data["state_shape"], 
                dtype=env_data["state_dtype"], seed=42)

        self.observation_space = spaces.Dict({
            "lvl_frame": box_space,
            "other_params": spaces.MultiBinary(env_data["other_params_len"]),
            "layer_speed": spaces.Discrete(20)
        })

        self.observation_data = [] # Will be dynamically initialised
        self.state = None
        self.count_steps = 0
        self.count_sampling = 0

        self.is_save_step = True

        self.reset()
        print("Environment has been initialised and reset. ")

    def step(self, action):
        #self.state = self.states_data.pop(0)...
        
        # NOTE: there is no done in my env and that's okay, because 
        # data isn't generated in real time
        """
        Executes an action and returns the resulting observation, reward, done, and info.
        
        Args:
            action (int): The action taken by the agent in the moment. 
                Agent thinks it takes the current action, but in reality I use the recorded one.
                For this reason I have to swap the reward if his action differs with the recorded.
        
        Returns:
            tuple:
                - observation (dict): The new observation after the action.
                - reward (float): The reward obtained from taking the action.
                - done (bool): Whether the episode has ended.
                - info (dict): Additional information.
        """
        if not self.observation_data:
            # NOTE: condition is required because my number of current steps
            # can be different from model's n_steps. So it has to roll out mine
            # with sample observations. Although reward is poorly set.
            # If no more data, end the episode
            done = True
            observation = self.observation_space.sample()  # Replace with actual observation
            reward = 0
            info = {}

            self.count_sampling += 1
            #print("Sampling observation space. ")
        
        else:
            # Implement action effects and environment transitions here
            # Example placeholders:
            observation, recorded_action, done, state_death_dict = self.observation_data.pop()
            
            reward = self.calculate_reward(recorded_action, done, state_death_dict)

            if recorded_action != action: 
                print("RECORDED ACTION ISN'T THE MODEL'S ACTION. SWAPING THE REWARD. ")
                reward = -reward

            info = {
                "obs": observation,
                "act": action,
                "rec_act": recorded_action, 
                "reward": reward,
                "done": done,
                "state_death_dict": state_death_dict
            }

            if self.is_save_step:
                save_step(info, path="observations/env/", max_batch_count=9) # two obs

            #print("Getting obs from obs data.")
            #print(observation, reward)

            '''if 20 < self.count_steps < 100:
                print(info)'''
            
        
        trancated = False
        self.count_steps += 1

        print(f"Step report. Number steps: {self.count_steps}. Number not sampled: {self.count_steps - self.count_sampling}. Reward: {reward}. ")
        
        return observation, reward, done, trancated,  info


    def reset(self, seed=None, options=None):
        #super().reset(seed=seed) # - it will reset self.oservation_data
        initial_observation = self.observation_space.sample()
        info = {}
        return initial_observation, info

    def render(self):
        pass

    def close(self):
        return super().close()

    def calculate_reward(self, action, done, state_death_dict):
        # NOTE: if we use negative reward to jumps, then we should
        # include a condition "is there a jump in X blocks before current state?"
        # because otherwise agent will learn to avoid jumping in the moment 
        # hoping of a future action. This isn't preferable. Moreover, I need 
        # the opposite: to make a jump in front of an obstacle as early as possible. 

        #return (-1000 if action else 1000) * -1
        reward = 1 #10

        if done:
            reward += -1 * state_death_dict * 4 # the closer the more punishment
            if action:
                reward += -5

        elif action:
            # I incentivise those jumps which very successful.
            # So it's trying to jump in the current state -> as early as possible.
            # However it makes an agent to jump as much as possible...
            # Another problem is early jumping on an incomplete, unseen lvl frame.
            reward += -1
        
        # It seems like the model tries hard to get the least possible reward. 
        # It minimazes instead of maximazing.
        return reward #* (-1)




