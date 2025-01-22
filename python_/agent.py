# filepath: /c:/Users/Dmitry/Documents/Coding/VSCFiles/IndependentProjects/GeometryDashAI/python_/agent.py

''' NOTE: Agent functionality
The purpose of the Agent class is to handle state creation,
level matrices, identifying levels, performance evaluation, and model control.
Training, loading, deleting, saving, etc., are handled by
the model itself, but the agent must maintain its instance.
'''
from datetime import datetime
from levelStructure import (
    get_block_index_x,
    get_block_index_y,
    get_max_x,
    get_max_y,
    get_level_data,
    update_stored_matrix,
    get_addition_i
)
import torch as th
from constants import (
    BATCH_SIZE,
    STATE_HEIGHT_BLOCKS,
    STATE_WIDTH_BLOCKS,
    SHIP_STATE_POSY,
    PLAYER_STATE_POSY,
    BLOCKS_PER_CUBE,
    MODEL_PATH,
    LOG_PATH
)
from env import GameEnv
from stable_baselines3 import PPO

# isShip (and likely others) should be the first (ordered) for get_level_frame_state function
STATE_PARAMS = ["isShip", "isBackwards", "isTopDown"]

# TODO: maybe algorithmic improvement: not to calculate an action on dead (obstacles)
# positions. But I think the difference is too low, because their widths are small.

def create_model_id(model_name="unknown"):
    return "{}_{}".format(model_name, datetime.now().strftime("%Y%m%d%H%M%S"))


def init_agent(lvl_id, model_id):
    return Agent(GameEnv, model_id, lvl_id=lvl_id)


class Agent:
    def __init__(self, env, model_params, lvl_id):
        """
        Initializes the Agent with the given environment and model.

        Args:
            env (gym.Env): The environment instance.
            model: The model used for predictions.
        """
        self.env = env
        self.model_id = None # is initialised in self.load_model()
        self.model = self.load_model(model_params=model_params)
        self.lvl_id = lvl_id
        self.lvl_matrix = get_level_data(lvl_id, is_hit=True, is_csv=False, is_init=False)
        self.A = None

    def update_ypos(self):
        """
        Updates the Y position in the matrix.
        """
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
        pass

    def handle_ypos(self, result, action_array):
        """
        Handles the Y position based on the result and action array.

        Args:
            matrix (torch.Tensor): The level matrix.
            result (float): The current result to evaluate.
            action_array (list): The array of actions taken.
        """
        maxResult = self.lvl_matrix[get_addition_i('maxResult'), 0]
        if result > maxResult:
            self.update_ypos(self.lvl_matrix, action_array)

    def get_action(self, state, random=False):
        """
        Determines the next action to take.

        Args:
            random (bool): If True, selects a random action for testing.

        Returns:
            int: The selected action.
        """
        if random: # this is only needed if you want to test random actions (and use as a comparison)
            return self.env.action_space.sample()
    
        return self.model.predict(self.env, state)


    # TODO: maybe to create cache for getting states. I think it's not a reasonable 
    # boost (accessing matrix isn't so hard as memory required to save every state)
    def get_lvl_frame_state(self, coli, params):
        """
        Retrieves the level frame state from the matrix.

        Args:
            matrix (torch.Tensor): The level matrix.
            coli (int): The current column index.
            params (torch.Tensor): Additional parameters.

        Returns:
            tuple: The level frame tensor and its bounding indices.
        """
        is_ship = params[0] # ! is ship.
        level_ypos_i = get_addition_i("levelYPos")[0]
        ypos = self.lvl_matrix[level_ypos_i, coli]
        starting_block = ypos - (SHIP_STATE_POSY if is_ship else PLAYER_STATE_POSY) 

        # TODO: if the a state overlaps (> x max) ending of a layer?
        endingX = coli + STATE_WIDTH_BLOCKS # exclusive
        endingY = starting_block + STATE_HEIGHT_BLOCKS # exclusive

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
        other_params = th.zeros(len(STATE_PARAMS))
        for i, arg in enumerate(STATE_PARAMS):
            arg_i = get_addition_i(arg)[0]
            other_params[i] = self.lvl_matrix[arg_i, coli]
            
        lvl_frame = self.get_lvl_frame_state(self.lvl_matrix, coli, other_params) 

        return (lvl_frame, other_params)


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
        return th.zeros((batch_size, ncols))

    def create_actions_matrix(self, ncols, batch_size=BATCH_SIZE):
        """
        Generates the actions matrix based on the level matrix.

        Args:
            lvl_matrix (torch.Tensor): The level matrix.
            ncols (int): Number of columns.
            batch_size (int, optional): Size of the batch. Defaults to BATCH_SIZE.

        Returns:
            torch.Tensor: The actions matrix.
        """
        A = self.create_dummy_action_matrix(ncols)
        done_A = A.clone() # matrix to track done actions in cpp

        for rowi in range(batch_size):
            for coli in range(ncols):
                state = self.get_state(coli)
                action = self.get_action(state)
            A[rowi, coli] = action # state can always be derived, so no need to store it.

        self.A = A
    
    def get_actions_matrix(self, ncols, batch_size=BATCH_SIZE):
        if self.A is None:
            self.create_actions_matrix(ncols, batch_size)
        return self.A

    def train_model(self, n_steps=1000, n_eval_episodes=10):
        """
        Trains the model.

        Args:
            n_steps (int, optional): Number of training steps. Defaults to 1000.
            n_eval_episodes (int, optional): Number of evaluation episodes. Defaults to 10.
        """
        pass

    def init_model(self, model_params):
        model_name = model_params["name"]
        if model_name == "PPO": model = PPO()

        model_id = create_model_id(model_name)
        return model, model_id

    #TODO: entire mdoel or state dict
    def save_model(self, overwrite=False):
        """
        Saves the model to the specified path.

        Args:
            overwrite (bool): If True, overwrites the existing model.
        """
        model_name = self.model.__class__.__name__
        if not overwrite:
            model_id = create_model_id(model_name)
            path = MODEL_PATH / f"{model_id}.zip"
        else:
            path = MODEL_PATH / f"{self.model_id}.zip"
        
        self.model.save(str(path))
        self.model_id = path.name

    def load_model_by_name(self, path, model_name):
        models = {
            "ppo": lambda: PPO.load(str(path), env=self.env)
        }
        self.model = models[model_name]()

    def load_model(self, model_params=None):
        """
        Loads the model from the specified path.

        Args:
            model_params (dict, optional): Parameters for loading the model.
        """
        to_init = model_params.get("to_init", False)

        if to_init:
            self.model, self.model_id = self.init_model(model_params)
        else:
            model_id = model_params["id"]
            model_name = model_id.split("_")[0]
            self.model_id = model_id
            path = MODEL_PATH / f"{model_id}.zip"
            self.load_model_by_name(path, model_name=model_name)
            

    def load_model(self, model_params=None):
        """
        Loads the model from the specified path.

        Args:
            path (str): The file path from which to load the model.
        """
        to_init = model_params.get("to_init", False)

        if to_init:
            self.model, self.model_id = self.init_model(model_params)
        else:
            model_id = model_params["id"]
            self.model, self.model_id = th.load(MODEL_PATH / model_id), model_id

    def evaluate_model(self, n_eval_episodes=10):
        """
        Evaluates the model's performance.

        Args:
            n_eval_episodes (int, optional): Number of evaluation episodes. Defaults to 10.
        """
        pass

    def replace_model(self):
        pass
