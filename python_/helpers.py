#TODO: make helpers if yo uahve time
from constants import ONE_CUBE_SIZE, BLOCKS_PER_CUBE, CUBE_TIMES_JUMPER_JUMP, LAST_GROUND_BLOCK_INDEX, ONE_BLOCK_SIZE
import os
import pickle
import datetime
import pathlib

# If not ending and an object underlap, round to the full.  
# We should start from the first and shouldn't overlap to the ending.
# For instance, if x is 15, x + w = 24 and block_size is 6.
# I should start from 2 and end at 3 instaed of 4.
def get_block_index_x(x, block_size=ONE_BLOCK_SIZE[0], ending=False):
    return int(x // block_size - (ending and not x % block_size))

def get_block_index_y(y, block_size=ONE_BLOCK_SIZE[1], ending=False):
    return int(y // block_size - (ending and not y % block_size))

# top value without remainders
def get_max_x(Nc, x_block=ONE_BLOCK_SIZE[0]):
    return Nc * x_block

def get_max_y(Nr, y_block=ONE_BLOCK_SIZE[1]):
    return Nr * y_block

def determine_level_ypos(column, j, prev_ypos, max_height, limit=3): #limit=BLOCKS_PER_CUBE[1] * CUBE_TIMES_JUMPER_JUMP
    # I don't use binary search, because the yPos isn't high always.

    i = 0
    while column[i]: # isnt' 0; I don't care about obstacles because player is out there.
        i += 1
    #if 20 <= j <= 30: print(i)

    if i - 1 > LAST_GROUND_BLOCK_INDEX: # not right after ground layer
        ypos = i
        #if 20 <= j <= 30: print("Ypos is ", ypos)
    else: # in the air
        while i < max_height and not column[i]:
            i += 1

        if i == max_height:
            #if 20 <= j <= 30: print("Max height is ", max_height, i)
            return prev_ypos # nothing on the level anymore
        
        ypos = i + 1 # +1 because one block upper than solid one
        #if 20 <= j <= 30: print("Ypos, column[i] is ", ypos, column[i])
    #if 20 <= j <= 30: print(i)

    # NOTE: limit doesn't account for a column (tower), 
    # but I have to set any one in order to get states later, even bad ones.
    if abs(prev_ypos - ypos) >= limit:
        ypos = prev_ypos
    #print(limit)

    return ypos  

def get_parent_path(postfix):
    file_path = pathlib.Path(__file__).parent / postfix
    return file_path

def flush_batch(path):
    """Force save any remaining data in the batch"""
    if hasattr(save_step, "batch") and save_step.batch:
        save_step.batch_count += 1
        name = f"batch_{save_step.session_id}_{save_step.batch_count}"
        file_path = pathlib.Path(__file__).parent / path / f"{name}.pkl"
        with open(file_path, "wb") as f:  
            print(f"Saving final batch with path {file_path}. ")
            print(save_step.batch)
            pickle.dump(save_step.batch, f)
        save_step.batch = []

def save_step(step_data, path='observations/', max_batch_count=3):
    return
    """
    Save step data to a file.
    
    Args:
        step_data (dict): Dictionary containing step information
        path (str): Directory path to save the file
    """
    
    # Ensure directory exists
    os.makedirs(path, exist_ok=True)
    
    # Initialize static variables if first time
    if not hasattr(save_step, "batch"):
        save_step.batch = []
        save_step.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_step.batch_count = 0
    
    if save_step.batch_count >= max_batch_count:
        return 
    
    # Add current step to batch
    save_step.batch.append(step_data)
    
    # Only save when batch reaches sufficient size
    batch_size = 100  # Adjust based on your data size
    if len(save_step.batch) >= batch_size:
        print("Batch size is full. ")
        flush_batch(path)
    

def load_from_pickle(path):
    parent = pathlib.Path(__file__).parent
    path = parent / path
    with open(path, 'rb') as f:  # Open file in binary mode
        file = pickle.load(f)
    return file

def save_pkl_as_txt(path):
    list_step_data = load_from_pickle(path)

    txt_path = path[:-4] + ".txt"
    txt_path = get_parent_path(txt_path)
    
    with open(txt_path, "w") as file:
        file.write(str(list_step_data))

