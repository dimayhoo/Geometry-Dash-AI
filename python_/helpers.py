#TODO: make helpers if yo uahve time
from constants import ONE_CUBE_SIZE, CUBE_TIMES_JUMPER_JUMP, LAST_GROUND_BLOCK_INDEX, ONE_BLOCK_SIZE


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

def determine_level_ypos(column, prev_ypos, limit=get_block_index_y(ONE_CUBE_SIZE[1] * CUBE_TIMES_JUMPER_JUMP)):
    # I don't use binary search, because the Ypos isn't high always.

    i = 0
    while column[i]: # isnt' 0; I don't care about obstacles because player is out there.
        i += 1

    if i - 1 > LAST_GROUND_BLOCK_INDEX: # not right after ground layer
        ypos = i
    else: # in the air
        while not column[i]:
            i += 1
        ypos = i + 1 # +1 because one block upper than solid one

    if abs(prev_ypos - ypos) >= limit:
        ypos = prev_ypos  

    return ypos  


