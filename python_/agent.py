# filepath: /c:/Users/Dmitry/Documents/Coding/VSCFiles/IndependentProjects/GeometryDashAI/python_/agent.py

''' NOTE: Agent functionality
The purpose of the Agent class is to handle state creation,
level matrices, identifying levels, performance evaluation, and model control.
Training, loading, deleting, saving, etc., are handled by
the model itself, but the agent must maintain its instance.
'''

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
    BLOCKS_PER_CUBE
)

# isShip (and likely others) should be the first (ordered) for get_level_frame_state function
STATE_PARAMS = ["isShip", "isBackwards", "isTopDown"]

# TODO: maybe algorithmic improvement: not to calculate an action on dead (obstacles)
# positions. But I think the difference is too low, because their widths are small.


def train_model(env, model, n_steps=1000, n_eval_episodes=10):
    pass

def save_model(model, path):
    pass

def load_model(path):
    pass

def evaluate_model(env, model, n_eval_episodes=10):
    pass





class Agent:
    def __init__(self, env, model):
        """
        Initializes the Agent with the given environment and model.

        Args:
            env (gym.Env): The environment instance.
            model: The model used for predictions.
        """
        self.env = env
        self.model = model

    def update_ypos(self, matrix):
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

    def handle_ypos(self, matrix, result, action_array):
        """
        Handles the Y position based on the result and action array.

        Args:
            matrix (torch.Tensor): The level matrix.
            result (float): The current result to evaluate.
            action_array (list): The array of actions taken.
        """
        maxResult = matrix[get_addition_i('maxResult'), 0]
        if result > maxResult:
            self.update_ypos(matrix, action_array)

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
    def get_lvl_frame_state(self, matrix, coli, params):
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
        ypos = matrix[level_ypos_i, coli]
        starting_block = ypos - (SHIP_STATE_POSY if is_ship else PLAYER_STATE_POSY) 

        # TODO: if the a state overlaps (> x max) ending of a layer?
        endingX = coli + STATE_WIDTH_BLOCKS # exclusive
        endingY = starting_block + STATE_HEIGHT_BLOCKS # exclusive

        return matrix[starting_block:endingY, coli:endingX], ((starting_block, coli), (endingX, endingY)) 

    # NOTE: a state consists of many parameters, not only level data.
    # NOTE: state is a tuple with tensor matrix and tensor array
    def get_state(self, matrix, coli):
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
            other_params[i] = matrix[arg_i, coli]
            
        lvl_frame = self.get_lvl_frame_state(matrix, coli, other_params) 

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

    def get_actions_matrix(self, lvl_matrix, ncols, batch_size=BATCH_SIZE):
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

        for rowi in range(batch_size):
            for coli in range(ncols):
                state = self.get_state(lvl_matrix, coli)
                action = self.get_action(state)
            A[rowi, coli] = action # state can always be derived, so no need to store it.

        return A

    def train_model(self, n_steps=1000, n_eval_episodes=10):
        """
        Trains the model.

        Args:
            n_steps (int, optional): Number of training steps. Defaults to 1000.
            n_eval_episodes (int, optional): Number of evaluation episodes. Defaults to 10.
        """
        pass

    def save_model(self, path):
        """
        Saves the model to the specified path.

        Args:
            path (str): The file path to save the model.
        """
        pass

    def load_model(self, path):
        """
        Loads the model from the specified path.

        Args:
            path (str): The file path from which to load the model.
        """
        pass

    def evaluate_model(self, n_eval_episodes=10):
        """
        Evaluates the model's performance.

        Args:
            n_eval_episodes (int, optional): Number of evaluation episodes. Defaults to 10.
        """
        pass
