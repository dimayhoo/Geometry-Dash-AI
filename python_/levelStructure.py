
import seaborn as sns
import torch as th
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

# Display more rows (default is 60)
pd.set_option('display.max_rows', 200)

# Display more columns (default is 20)
pd.set_option('display.max_columns', 100)

# Widen the console output
pd.set_option('display.width', 1000)

# Show more characters in each column
pd.set_option('display.max_colwidth', 2000)

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
ONE_CUBE_SIZE = (30, 30)
ONE_BLOCK_SIZE = (6, 15) # Turned out GROUND_LAYER_Y is exactly 7 blocks * 15 axmol units!
BLOCKS_PER_CUBE = (ONE_CUBE_SIZE[0] // ONE_BLOCK_SIZE[0], ONE_CUBE_SIZE[1] // ONE_BLOCK_SIZE[1])
GROUND_LAYER_Y = 105
GROUND_OBJ_ID = -1


''' Additions

isShip()
isBackwards()
isDown()
Ypos()
YposMax() - a condition to know when to update Ypos (only when the final result is more than previous one)

'''

ADDITIONS = { # matrix index | default value to fill
    'isShip': (-1,False),
    'isBackwards': (-2,False),
    'isDown': (-3,False),
    'Ypos': (-4,-2),
    'YposMax': (-5,-2) # tensors cannot store None
}

# TODO: remove this function and update which used it
basic_levels = {
    1: "1"
}

# Game's positions are axmol units. They are counted
# from bottom left corner. They are usually approx pixels.

def get_data_type(obj):
    if 'w' in obj:
        return 'hitbox'
    else:
        return 'object'

# If not ending and an object underlap, round to the full.  
# We should start from the first and shouldn't overlap to the ending.
# For instance, if x is 15, x + w = 24 and block_size is 6.
# I should start from 2 and end at 3 instaed of 4.
def get_block_index_x(x, block_size=ONE_BLOCK_SIZE[0], ending=False):
    return int(x // block_size - (ending and not x % block_size))

def get_block_index_y(y, block_size=ONE_BLOCK_SIZE[1], ending=False):
    return int(y // block_size - (ending and not y % block_size))

def decode_level_data(data, columns, to_sort=True):
    # TODO: maybe, to normilise y and x positions, but then all positoins in every game should be handled. 
    # All this effort for what?
    # y min in a groundLayerY - one_block_size: let y min be so for now 

    if not data or not columns:
        raise ValueError("Data is incomplete.")

    proc_data = [[float(getattr(obj, c, None)) for c in columns] for obj in data]
    dec_data = pd.DataFrame(proc_data, columns=columns)
    if to_sort:
        dec_data = dec_data.sort_values(by=columns)

    #print(decoded_data)
    return dec_data

def get_level_data(level_id, columns=None):
    name = basic_levels.get(level_id)
    if not name: 
        raise ValueError(f"Level with id {level_id} doesn't exist.")
    
    path = f"{PATH}/{name}.csv"
    df = pd.read_csv(path)

    if columns is None: columns = df.columns
    
    return df[columns]
    

def visualise_level(dec_data=None, level_id=None, matrix_path=None):
    """
    Displays level data as a scatter plot.
    Accepts either:
    - dec_data (a DataFrame),
    - level_id (to read from CSV),
    - matrix_path (to load a PyTorch .pt matrix).
    """

    if matrix_path:
        # 1) Load the matrix from disk
        mat = th.load(matrix_path)
        Nr, Nc = mat.shape

        # 2) Create DataFrame of (y, x, id) for plotting
        data_list = []
        for x in range(Nc):
            for y in range(Nr):
                val = mat[y, x].item()  # Convert to Python float/int
                # Skip if val is ground or default, if you only want to show actual objects
                # if val == -1 or val == 0: continue
                data_list.append((y, x, val))

        df_plot = pd.DataFrame(data_list, columns=["y", "x", "id"])

        #print(df_plot.head(100))

        '''add_df = df_plot.iloc[-len(ADDITIONS):]
        print(add_df.describe())'''

        # 3) Plot with seaborn
        sns.scatterplot(x="x", y="y", data=df_plot, hue="id")
        #plt.gca().invert_yaxis()  # Optional: flip Y for 2D grid
        plt.show()
        return df_plot

    # Otherwise, fallback to original logic for DataFrame or level_id
    if dec_data is None and level_id is not None:
        dec_data = get_level_data(level_id, columns=["x", "y", "id"])

    if dec_data is not None:
        sns.scatterplot(x="x", y="y", data=dec_data, hue="id")
        plt.show()

def create_default_matrix(df, number_of_additions=len(ADDITIONS)):
    # Starting from (0, 0). I think memory overtake is purely okay in
    # comparison to handling additional states.
    x_max, y_max = (df['x'] + df['w']).max(), (df['y'] + df['h']).max() 
    x_block, y_block = ONE_BLOCK_SIZE
    
    Nc = x_max // x_block + bool(x_max % x_block) 
    Nr = y_max // y_block + bool(y_max % y_block)

    x_max, y_max = Nc * x_block, Nr * y_block # updating to the top value without remainders

    Nr += number_of_additions # for special params

    # y is the number of rows and x is the number of columns
    return th.zeros((int(Nr), int(Nc))), x_max, y_max

def fill_groud_layer(matrix, ground_layer_y=GROUND_LAYER_Y, ground_obj_id=GROUND_OBJ_ID):
    matrix[:get_block_index_y(ground_layer_y) + 1, :] = ground_obj_id # y index is exclusive

    return matrix

def create_level_matrix(df, columns=HITBOX_COLUMNS):
    # Cube cannot be lower than ground layer (In Geometry Dash, the cube's position relative to the ground layer is fixed by the game's mechanics. The cube generally stays on or above the ground layer, which is the surface the player interacts with.)
    matrix, x_max, y_max = create_default_matrix(df)

    # Filling objects
    matrix = fill_groud_layer(matrix)

    for _, obj in df[columns].iterrows():
        x, y, id, w, h = obj
        xi, yi = get_block_index_x(x), get_block_index_y(y) # starting indeces
        xwi, yhi = get_block_index_x(x + w, ending=True), get_block_index_y(y + h, ending=True)
        matrix[yi:(yhi + 1), xi:(xwi + 1)] = id # indeces are inclusive and exclusive accordingly
    
    # Filling additions.
    for key, (index_y, def_value) in ADDITIONS.items():
        matrix[index_y, :] = def_value

    return matrix, x_max, y_max

def store_level(data, lvl_id, hitboxes=True, rewrite=True):
    basic_levels[lvl_id] = str(lvl_id)
    lvl_name = basic_levels[lvl_id]
    
    if hitboxes:
        columns = HITBOX_COLUMNS
        pt_path = f"{PATH}/{lvl_name}-hit.pt"
        csv_path = f"{PATH}/{lvl_name}-hit-init.csv"
    else:
        columns = OBJ_COLUMNS
        pt_path = f"{PATH}/{lvl_name}-obj.pt"
        csv_path = f"{PATH}/{lvl_name}-obj-init.csv"

    if not rewrite:
        if pathlib.Path(pt_path).exists() and pathlib.Path(csv_path).exists():
            return 0
    
    df = decode_level_data(data, columns, to_sort=True)
    df.to_csv(csv_path)

    matrix, x_max, y_max = create_level_matrix(df, columns)
    th.save(matrix, pt_path)
    
    return 1


visualise_level(matrix_path=f"{PATH}/1-hit.pt")
#print(create_level_matrix(get_level_data(1)))
