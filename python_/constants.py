import pathlib

DIR_PATH = pathlib.Path(__file__).parent 
LEVELS_PATH = DIR_PATH / "levels"
OBJ_COLUMNS = ['x', 'y', 'rot', 'id']
HITBOX_COLUMNS = ['x', 'y', 'id', 'w', 'h']

'''
My game is using hd assets. Dimesnsions of the hd balck cube image are 160x160. 
Default window is 23 sm in width. Although I didn't find any scaling transformation form PlayLayer while logging the file. 
But I test the program with hitboxes and I am pretty sure the geometry dash player cube (which should be equal to every other one) is 30x30 axmol positions. 
This suggestion is solidified wtih the frequency of 30x30 in the objects data (csv file).
Besides, I see the number of pixels in 1 axmol position is much greater than 1.
So the images are downscaled probably.
'''
# NOTE: be cautious with module 0 division. Some calculations don't account for remainders.
ONE_CUBE_SIZE = (30, 30)
ONE_BLOCK_SIZE = (6, 15) # Turned out GROUND_LAYER_Y is exactly 7 blocks * 15 axmol units!
BLOCKS_PER_CUBE = (ONE_CUBE_SIZE[0] // ONE_BLOCK_SIZE[0], ONE_CUBE_SIZE[1] // ONE_BLOCK_SIZE[1])
GROUND_LAYER_Y = 105
GROUND_OBJ_ID = -1
PADDING_OBJ_ID = -25 # Random. I don't know a better number. They are all integers of 4 bytes!!! You can use -10000 and it will takes the same size. The only restriction is low/high bounds in the observation space.
CUBE_TIMES_JUMPER_JUMP = 5 # about 4.5-4.85, definitely less than 5
LAST_GROUND_BLOCK_INDEX = GROUND_LAYER_Y // ONE_BLOCK_SIZE[1] - (not GROUND_LAYER_Y % ONE_BLOCK_SIZE[1]) # 7 blocks

MINIMAL_FRAME_VALUE = -25

BATCH_SIZE = 10

# NOTE: the height of one jump is 3 cubes. But there are also ring and high jumpers!!! 
STATE_WIDTH_BLOCKS = ONE_CUBE_SIZE[0] * 5 // ONE_BLOCK_SIZE[0]
STATE_HEIGHT_BLOCKS = ONE_CUBE_SIZE[1] * 7 // ONE_BLOCK_SIZE[1] # 3 cubes original jump + 3 cubes possible jumper or ring jumper + 1 for ground
PADDING_X_BLOCKS = STATE_WIDTH_BLOCKS

# NOTE: this is player's pos, but the cube of the player only touches
# a state with right side. It's not inside one of the state's blocks.
PLAYER_STATE_POSY = ONE_CUBE_SIZE[1] * 1 // ONE_BLOCK_SIZE[1] # NOTE: 2 index is the third block from the bottom
SHIP_STATE_POSY = STATE_HEIGHT_BLOCKS // 2 - 1 # -1 is irrelevant because all numbers are even, but still I leave a "middle" here 

LOG_PATH = DIR_PATH / "training" / "logs"
MODEL_PATH = DIR_PATH / "training" / "models"


BASIC_LEVEL_MAX_HEIGHT_PIXELS = ONE_BLOCK_SIZE[1] * 20 
COMMUNITY_LEVEL_MAX_HEIGHT_PIXELS = ONE_BLOCK_SIZE[1] * 200 # I found 199 is max
LEVEL_MIN_HEIGHT = PADDING_OBJ_ID
