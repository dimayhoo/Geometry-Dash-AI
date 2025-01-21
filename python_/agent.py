''' NOTE: agent functionality
The purpose my agent mostly is to handle states creation,
level matrices, identifying levels, performance evaluation, etc and model control.
Training, loading, deleting, saving, etc. are handled by
the model itself, but I must have it instance somewhere.
'''


from levelStructure import get_block_index_x, get_block_index_y, get_max_x, get_max_y, get_level_data, update_stored_matrix, get_addition_i
import torch as th
from constants import BATCH_SIZE, STATE_HEIGHT_BLOCKS, STATE_WIDTH_BLOCKS, SHIP_STATE_POSY, PLAYER_STATE_POSY, BLOCKS_PER_CUBE

# isShip (and likely other ones) should be the first(ordered) (for get level frame state fn)!
STATE_PARAMS = ["isShip", "isBackwards", "isTopDown"]

# TODO: maybe algorithmic improvement: not to calculate an action on dead (obstacles)
# positions. But I think the difference is too low, because their widths are small.

''' NOTE: Updating ypos functionality
To get a current result. If it's larger than max,
then to get the id of the action array used, play the same
level once again while tracking all yposes. This method is 
better because I don't need to track yposes per action array,
but only per batch (or whevever I want to), for instance.
- Update ypos only after learning!
TODO: I need to know where the player is jumping (the moment in indeces) for a reward function. But I can't know the length of a jump, because it depends on level's speed. So I should track this too and send as a result.
'''
def update_ypos(matrix):
    pass

def handle_ypos(matrix, result, action_array):
    maxResult = matrix[get_addition_i('maxResult'), 0]
    if result > maxResult:
        update_ypos(matrix, action_array)

def get_action(env, model, random=False):
    if random: # this is only needed if you want to test random actions (and use as a comparison)
        return env.action_space.sample()
    
    return model.predict()

# TODO: maybe to create cache for getting states. I think it's not a reasonable 
# boost (accessing matrix isn't so hard as memory required to save every state)
def get_lvl_frame_state(matrix, coli, params):
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
def get_state(matrix, coli):
    other_params = th.zeros(len(STATE_PARAMS))
    for i, arg in enumerate(STATE_PARAMS):
        arg_i = get_addition_i(arg)[0]
        other_params[i] = matrix[arg_i, coli]
        
    lvl_frame = get_lvl_frame_state(matrix, coli, other_params) 

    return (lvl_frame, other_params)


# NOTE: I can only take action per column (x block position).
def create_dummy_action_matrix(ncols, batch_size):
    return th.zeros((batch_size, ncols))

def create_actions_matrix(lvl_matrix, ncols, batch_size=BATCH_SIZE):
    A = create_dummy_action_matrix(ncols)

    for rowi in range(batch_size):
        for coli in range(ncols):
            state = get_state(lvl_matrix, coli)
            action = get_action(state)
        A[rowi, coli] = action # state can always be derived, so no need to store it.

    return A

def train_model(env, model, n_steps=1000, n_eval_episodes=10):
    pass

def save_model(model, path):
    pass

def load_model(path):
    pass

def evaluate_model(env, model, n_eval_episodes=10):
    pass

