# filepath: /c:/Users/Dmitry/Documents/Coding/VSCFiles/IndependentProjects/GeometryDashAI/python_/agent.py

''' NOTE: Agent functionality
The purpose of the Agent class is to handle state creation,
level matrices, identifying levels, performance evaluation, and model control.
Training, loading, deleting, saving, etc., are handled by
the model itself, but the agent must maintain its instance.

- I may used definitions of a state vs a current observation incorrectly.
'''
from datetime import datetime

# NOTE: Python caches modules after the first import, so repeated imports incur virtually no overhead.
import torch as th
import numpy as np
from levelStructure import (
    get_level_data,
    update_stored_matrix,
    get_addition_i
)
from constants import (
    BATCH_SIZE,
    STATE_HEIGHT_BLOCKS,
    STATE_WIDTH_BLOCKS,
    SHIP_STATE_POSY,
    PLAYER_STATE_POSY,
    ONE_BLOCK_SIZE,
    MODEL_PATH,
    LOG_PATH,
    PADDING_X_BLOCKS,
    BLOCKS_PER_CUBE,
    MINIMAL_FRAME_VALUE
)
from env import GameEnv
from stable_baselines3 import PPO, DQN, A2C

# isShip (and likely others) should be the first (ordered) for get_level_frame_state function
STATE_PARAMS = ["isShip", "_touchedRingObject", "isGravityFlipped"] 
COMMON_ENV_DATA = {
    "state_shape": (STATE_HEIGHT_BLOCKS, STATE_WIDTH_BLOCKS),
    "state_dtype": np.float32, # np is required for stable baselines model
    "other_params_len": len(STATE_PARAMS)
}

# NOTE: names should be the same as in PlayLayer.cpp! Otherwise
# they won't be defined in the tracking params in GameData.
TRACKING_PARAMS = ["isShip", "isGravityFlipped", "_touchedRingObject"]

# TODO: maybe algorithmic improvement: not to calculate an action on dead (obstacles)
# positions. But I think the difference is too low, because their widths are small.

def create_model_id(model_name="unknown"):
    return "{}_{}".format(model_name, datetime.now().strftime("%Y%m%d%H%M%S"))

class Agent:
    def __init__(self, model_params, lvl_id, additional_env_data={}, rl_data={}):
        """
        Initializes the Agent with the given environment and model.
        - Order of the self variables matter!

        Args:
            env (gym.Env): The environment instance.
            model: The model used for predictions.
        """
        
        self.play_layer_speed = rl_data.get("play_layer_speed", 1)
        self.cur_epoch = 0
        self.epochs = rl_data.get("epochs", 2)
        self.batch_size = rl_data.get("batch_size", BATCH_SIZE)
        self.env = None
        self.model = None
        self.lvl_id = lvl_id
        self.lvl_matrix = get_level_data(lvl_id, is_hit=True, is_csv=False, is_init=False)
        self.ncols = self.lvl_matrix.shape[1] - PADDING_X_BLOCKS
        self.min_max = self.get_min_max()
        self.init_env(additional_env_data)
        self.model_id = None # is initialised in self.load_model()
        self.load_model(model_params=model_params)
        self.model_n_steps = model_params.get("n_steps", 1)
        self.A = None
        self.status = "ready" # or "training", "observing" and "done" (all epoches ran)
        #print("Model name after init is:", self.model.__class__.__name__)

    def init_env(self, additional_env_data={}):
        env_data = {
            **COMMON_ENV_DATA, 
            **additional_env_data,
            "min_max": self.min_max
        }
        
        self.env = GameEnv(env_data=env_data)


    def process_params(self, data):
        filt_params = {}
        for p in TRACKING_PARAMS: filt_params[p] = []

        # V is a dict of prev and values.
        for key, V in data.items():
            #print(key, V)
            values_per_row = []
            prev = -1
            values_cur = []
            n = 0
            for val in V["values"]:
                if val > prev:
                    values_cur.append(val)
                    prev = val
                else:
                    n = max(len(values_cur), n)
                    values_per_row.append(values_cur)
                    values_cur = [val]
                    prev = val

            i = 0
            values = []
            while i < n:
                values.append(min(values_per_row[j][i] for j in \
                                  range(self.batch_size)))

            filt_params[key] = values[::-1] # to pop more efficiently

        #print(filt_params)
        return filt_params
        '''maxval = -1
        i = 0
        for val in V["values"]:
            if val > maxval:
                filt_params[key].append(val)
                
            else:
                filt_params[i] = min(filt_params[i], val)
                i += 1'''
        
    def handle_params_update(self, data):
        """
        Updates ypos.
        Updates tracked parameters.

        ''' NOTE: Updating ypos functionality
            To get a current result. If it's larger than max,
            then to get the id of the action array used, play the same
            level once again while tracking all yposes. This method is 
            better because I don't need to track yposes per action array,
            but only per batch (or whevever I want to), for instance.
            - Update ypos only after learning!
            TODO: I need to know where the player is jumping (the moment in indeces) for a reward function. 
                But I can't know the length of a jump, because it depends on level's speed. 
                So I should track this too and send as a result.
        '''
        """
        
        filt_params = self.process_params(data["tracking_params_dict"])
        params = [0] * len(TRACKING_PARAMS)
        yposi = get_addition_i("yPos")

        for j in range(self.ncols):
            if j == data["dead_pos"]: break

            # otherwise it's on jump or game wasn't able to track it (and it's zero)
            # TODO: should be more more robust because yPos from level is is too biased.
            # Maybe to track the nest yposes according to prev. Anyway you should lead from results (fix when neccessary).
            if data["can_act_matrix"][j] and data["y_pos_array_best"][j]: 
                self.lvl_matrix[yposi][j] = data["y_pos_array_best"][j]
            
            for i, key in enumerate(TRACKING_PARAMS):
                if filt_params[key] and filt_params[key][0] == j:
                    params[i] = not params[i]
                    filt_params.pop() # NOTE: the lowest value is popped

                self.lvl_matrix[get_addition_i(key)][j] = params[i]   

    def get_action(self, obs, random=False):
        """
        Determines the next action to take.

        Args:
            random (bool): If True, selects a random action for testing.

        Returns:
            int: The selected action.
        """
        '''r = np.random.random()
        if r < 0.1:
            random = True'''

        if random: # this is only needed if you want to test random actions (and use as a comparison)
            return self.env.action_space.sample()
    
        action, _states = self.model.predict(obs, deterministic=True)
        return action


    # TODO: maybe to create cache for getting states. I think it's not a reasonable 
    # boost (accessing matrix isn't so hard as memory required to save every state)
    def get_lvl_frame_state(self, coli=None, params=None, positions=None):
        """
        Retrieves the level frame state from the matrix.

        Args:
            matrix (torch.Tensor): The level matrix.
            coli (int): The current column index.
            params (torch.Tensor): Additional parameters.

        Returns:
            tuple: The level frame tensor and its bounding indices.
        """
        if positions is None:
            is_ship = params[0] # ! is ship.
            level_ypos_i = get_addition_i("yPos")
            ypos = self.lvl_matrix[level_ypos_i, coli] # in blocks

            ypos = int(ypos.item()) # to be used as an index later

            starting_block = ypos - (SHIP_STATE_POSY if is_ship else PLAYER_STATE_POSY) 

            # TODO: if the a state overlaps (> x max) ending of a layer?
            endingX = coli + STATE_WIDTH_BLOCKS # exclusive
            endingY = starting_block + STATE_HEIGHT_BLOCKS # exclusive

        else:
            starting_block, coli, endingX, endingY = positions
        
        #print(starting_block, coli, endingX, endingY)
        return self.lvl_matrix[starting_block:endingY, coli:endingX], ((starting_block, coli), (endingX, endingY)) 

    # NOTE: a state consists of many parameters, not only level data.
    # NOTE: state is a tuple with tensor matrix and tensor array
    def get_state(self, coli):
        """
        Constructs the state from the current matrix and column index.

        Args:
            matrix (torch.Tensor): The level matrix.
            coli (int): The current column index.

        Returns:
            tuple: A tuple containing the level frame and other parameters.
        """
        np_other_params = np.zeros(len(STATE_PARAMS), dtype=np.bool_)
        for i, arg in enumerate(STATE_PARAMS):
            arg_i = get_addition_i(arg)
            np_other_params[i] = self.lvl_matrix[arg_i, coli]
            
        lvl_frame, pos = self.get_lvl_frame_state(coli, np_other_params) 

        # Normalising to non-negativity and flatenning are required
        # steps for environment MultiDescrete space.
        lvl_frame -= MINIMAL_FRAME_VALUE

        np_lvl_frame = self.convert_torch_numpy(lvl_frame).flatten()

        #print("Level frame shape: ", np_lvl_frame.shape)
        if np_lvl_frame.shape == (0, ):
            print(pos)
            print(lvl_frame.shape)
            print(lvl_frame)

        # Return dict for MultiInputPolicy
        return {
            "lvl_frame": np_lvl_frame,
            "other_params": np_other_params,
            "layer_speed": self.play_layer_speed
        }

    def get_min_max(self):
        """
        Computes the minimum and maximum values of a tensor.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing (min_value, max_value).
        """
        # NOTE: item converts from a tensor to a scalar 
        tensor_flat = self.lvl_matrix.view(-1)
        min_val = th.min(tensor_flat).item()
        max_val = th.max(tensor_flat).item()
        return (min_val, max_val)

    @staticmethod
    # NOTE: I can only take action per column (x block position).
    def create_dummy_action_matrix(ncols, batch_size):
        """
        Creates a dummy action matrix filled with zeros.

        Args:
            ncols (int): Number of columns.
            batch_size (int): Size of the batch.

        Returns:
            torch.Tensor: The dummy action matrix.
        """
        return th.zeros((batch_size, ncols), dtype=th.int16)

    def create_actions_matrix(self):
        """
        Generates the actions matrix based on the level matrix.

        Args:
            lvl_matrix (torch.Tensor): The level matrix.
            ncols (int): Number of columns.
            batch_size (int, optional): Size of the batch. Defaults to BATCH_SIZE.

        Returns:
            torch.Tensor: The actions matrix.
        """
        A = self.create_dummy_action_matrix(self.ncols, batch_size=self.batch_size)

        for rowi in range(self.batch_size):
            for coli in range(self.ncols):
                np_obs = self.get_state(coli)
                #print(np_obs)
                action = self.get_action(np_obs)
                #print("Action is ", action)
                # state can always be derived, so no need to store it.
                # NOTE: must use tensor with th matrix
                A[rowi, coli] = th.tensor(action, dtype=th.int16)

        #print(A)
        self.A = A
        #print(self.A)
    
    def get_actions_matrix(self):
        if self.A is None:
            self.create_actions_matrix()
        #print([e for e in self.A[0, :100]])
        #print("Agent actions matrix. ", self.A)
        return self.A
    
    def get_game_input(self):
        actionsMatrix = self.get_actions_matrix()

        # Maybe, into a dict?
        return {
            "actionsMatrix": actionsMatrix,
            "oneBlockSize": ONE_BLOCK_SIZE,
            "trackingParams": TRACKING_PARAMS
        }
    
    def handle_game_observations(self, data):
        print("Agent. Data is received.")
        '''print(data["deadPositions"])
        print(len(data["matrices"]["yPosMatrix"]))'''
        print(data["matrices"]["yPosMatrix"][0][:100])

        self.handle_model_learning(data={
            "canActMatrix": data["matrices"]["canActMatrix"],
            "deadPositions": data["deadPositions"]
        })

        # Params shouldn't be updated before model learning otherwise
        # state-action-reward combination will be inaccurate.
        max_result = np.max(data["deadPositions"])
        prev_max_result = self.lvl_matrix[get_addition_i('maxResult'), 0]
        
        if max_result > prev_max_result:
            best_result_i = np.argmax(data["deadPositions"])
            
            tracking_data = {
                "dead_pos": data["deadPositions"][best_result_i],
                "y_pos_array_best": data["matrices"]["yPosMatrix"][best_result_i],
                "tracking_params_dict": data["trackingParams"],
                "can_act_matrix": data["matrices"]["canActMatrix"][best_result_i]
            }
            
            self.handle_params_update(tracking_data)
        print("Received data has been processed successfully. ")

        update_stored_matrix(self.lvl_matrix, self.lvl_id)

        self.cur_epoch += 1
        if self.cur_epoch != self.epochs:
            if self.cur_epoch % 10 == 0:
                print(f"Epoch {self.cur_epoch} is completed. Saving a model. ")
                self.save_model()
            self.status = "ready"
        else:
            print("Epochs are over. Closing agent. ")
            self.save_model()
            self.status = "done"

    def handle_model_learning(self, data):
        observation_data = []
        steps = 0
        #print(data["canActMatrix"][:100])

        for j in range(self.ncols):
            np_obs = self.get_state(j)
            for i in range(self.batch_size):

                dead_pos = data["deadPositions"][i]
                if j == dead_pos: break
                

                if not data["canActMatrix"][i][j]:
                    continue
                
                action = self.A[i][j].item()
                death_dict = dead_pos - STATE_WIDTH_BLOCKS

                # state_death_dict is an ideal dictance
                state_death_dict = STATE_WIDTH_BLOCKS - (dead_pos - j)
                done = j >= death_dict # in the current frame or not
                if done:
                    #print(np_obs)
                    pass
                observation_data.append((np_obs, action, done, state_death_dict))
                steps += 1

        print(f"Starting learning on {steps} steps")
        obs_len = len(observation_data)
        batches = obs_len // self.model_n_steps
        self.env.observation_data = observation_data[:batches * self.model_n_steps][::-1] # for popping
        print("Env observation data length is:", len(self.env.observation_data))
        self.train_model(n_steps=steps)


    def train_model(self, n_steps=1000, n_eval_episodes=10):
        """
        Trains the model.

        Args:
            n_steps (int, optional): Number of training steps. Defaults to 1000.
            n_eval_episodes (int, optional): Number of evaluation episodes. Defaults to 10.
        """
        self.model.learn(total_timesteps=n_steps)

    def init_model(self, model_params):
        model_name = model_params["name"].lower()
        
        del model_params["name"]
        del model_params["to_init"]
        
        if model_name == "ppo": 
            model = PPO(policy="MultiInputPolicy", env=self.env, tensorboard_log=LOG_PATH, **model_params)
        elif model_name == "dqn":
            model = DQN(policy="MultiInputPolicy", env=self.env, tensorboard_log=LOG_PATH, **model_params)
        elif model_name == "a2c":
            model = A2C(policy="MultiInputPolicy", env=self.env, tensorboard_log=LOG_PATH, **model_params)

        self.model_id = create_model_id(model_name)
        self.model = model

    #TODO: entire mdoel or state dict
    def save_model(self, overwrite=False):
        """
        Saves the model to the specified path.

        Args:
            overwrite (bool): If True, overwrites the existing model.
        """
        model_name = self.model.__class__.__name__
        #print("Model name is:", model_name)
        if not overwrite:
            model_id = create_model_id(model_name)
            path = MODEL_PATH / f"{model_id}.zip"
        else:
            path = MODEL_PATH / f"{self.model_id}.zip"
        
        self.model.save(str(path))
        self.model_id = path.name
        print("Model has been saved. ")

    def load_model_by_name(self, path, model_name):
        models = {
            "ppo": lambda: PPO.load(str(path), env=self.env),
            "dqn": lambda: DQN.load(str(path), env=self.env),
            "a2c": lambda: A2C.load(str(path), env=self.env)
        }
        self.model = models[model_name.lower()]()
        print(f"Model {model_name} loaded from {path}.")

    def load_model(self, model_params=None):
        """
        Loads the model from the specified path.

        Args:
            model_params (dict, optional): Parameters for loading the model.
        """
        to_init = model_params.get("to_init", False)

        if to_init:
            self.init_model(model_params)
            print(f"Model initialized: {self.model_id}")
        else:
            model_id = model_params["id"]
            model_name = model_id.split("_")[0]
            self.model_id = model_id
            path = MODEL_PATH / f"{model_id}.zip"
            self.load_model_by_name(path, model_name=model_name)

    def convert_torch_numpy(self, obj):
        """
        - I have to use numpy for stable baselines model (even with a cuda setup).
        - The function should be pretty fast.
        """
        #print(obj)
        #print(obj.cpu().numpy())
        return obj.cpu().numpy()

    def evaluate_model(self, n_eval_episodes=10):
        """
        Evaluates the model's performance.

        Args:
            n_eval_episodes (int, optional): Number of evaluation episodes. Defaults to 10.
        """
        pass

    def replace_model(self):
        pass


