PATH = pathlib.Path(__file__).parent / "levels"
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

BATCH_SIZE = 10

# NOTE: the height of one jump is 3 cubes. But there are also ring and high jumpers!!! 
STATE_WIDTH_BLOCKS = ONE_CUBE_SIZE[0] * 5 // ONE_BLOCK_SIZE[0]
STATE_HEIGHT_BLOCKS = ONE_CUBE_SIZE[1] * 4 // ONE_BLOCK_SIZE[1] # 3 cubes + 1 for ground

# NOTE: this is player's pos, but the cube of the player only touches
# a state with right side. It's not inside one of the state's blocks.
PLAYER_STATE_POSY = ONE_CUBE_SIZE[1] * 1 // ONE_BLOCK_SIZE[1] # NOTE: 2 index is the third block from the bottom
SHIP_STATE_POSY = STATE_HEIGHT_BLOCKS // 2 - 1 # -1 is irrelevant because all numbers are even, but still I leave a "middle" here 