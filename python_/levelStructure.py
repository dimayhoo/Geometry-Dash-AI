
import seaborn as sns
import torch as th
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from constants import ONE_BLOCK_SIZE, GROUND_LAYER_Y, GROUND_OBJ_ID, HITBOX_COLUMNS, OBJ_COLUMNS, PATH, LAST_GROUND_BLOCK_INDEX
from helpers import determine_level_ypos, get_block_index_x, get_block_index_y, get_max_x, get_max_y

# Display more rows (default is 60)
pd.set_option('display.max_rows', 200)

# Display more columns (default is 20)
pd.set_option('display.max_columns', 100)

# Widen the console output
pd.set_option('display.width', 1000)

# Show more characters in each column
pd.set_option('display.max_colwidth', 2000)



''' Additions

isShip()
isBackwards()
isDown()
Ypos()
maxResult() - tracking the best case. At least necessary for Ypos update.
levelYPos() - ypos which is derived from the level data.

'''

ADDITIONS = { # matrix row index | default value to fill
    'isShip': (-1,False),
    'isBackwards': (-2,False),
    'isDown': (-3,False),
    'Ypos': (-4,-2), # value-index in blocks!
    'maxResult': (-5,-2), # tensors cannot store None
    "levelYPos": (-6,-2)
}

# TODO: remove this function and update which used it
basic_levels = {
    1: "1"
}

# Game's positions are axmol units. They are counted
# from bottom left corner. They are usually approx pixels.

def get_addition_i(key): # Will be broken automatically if there is no key.
    return ADDITIONS[key][0]

def create_path(lvl_name, is_hitbox=True, is_init=False, is_csv=False):
    hit = "-hit" if is_hitbox else "-obj"
    init = "-init" if is_init else ""
    ext = "csv" if is_csv else "pt"
    return f"{PATH}/{lvl_name}{hit}{init}.{ext}"

def get_data_type(obj):
    if 'w' in obj:
        return 'hitbox'
    else:
        return 'object'

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

def get_level_data(level_id, is_hit=True, is_csv=False, columns=None, is_init=True):
    lvl_name = basic_levels.get(level_id)
    if not lvl_name: 
        raise ValueError(f"Level with id {level_id} doesn't exist.")

    path = create_path(lvl_name, is_hit, is_init, is_csv)
    
    if is_csv:
        df = pd.read_csv(path)
        if columns is None: columns = df.columns
        return df[columns]
    
    else:
        return th.load(path)
    
    

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
        dec_data = get_level_data(level_id, is_csv=True, columns=["x", "y", "id"])

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

    Nr += number_of_additions # for special params

    # y is the number of rows and x is the number of columns
    return th.zeros((int(Nr), int(Nc)))

def fill_groud_layer(matrix, ground_layer_y=GROUND_LAYER_Y, ground_obj_id=GROUND_OBJ_ID):
    matrix[:get_block_index_y(ground_layer_y) + 1, :] = ground_obj_id # y index is exclusive

    return matrix

def fill_level_ypos(matrix):
    level_ypos_rowi, default_level_ypos = get_addition_i('levelYPos')
    ypos_rowi, default_ypos = get_addition_i('Ypos')
    prev_ypos = LAST_GROUND_BLOCK_INDEX + 1 # prevpos is for a slight optimisation.

    for j in range(matrix.shape[1]):
        ypos = matrix[ypos_rowi, j]
        if ypos == default_ypos:
            ypos = determine_level_ypos(matrix[:, j], prev_ypos)
        matrix[level_ypos_rowi, j] = ypos
        prev_ypos = ypos

    return matrix

def create_level_matrix(df, columns=HITBOX_COLUMNS):
    # Cube cannot be lower than ground layer (In Geometry Dash, the cube's position relative to the ground layer is fixed by the game's mechanics. The cube generally stays on or above the ground layer, which is the surface the player interacts with.)
    matrix = create_default_matrix(df)

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

    # Preprocessing y poses for states.
    matrix = fill_level_ypos(matrix)

    return matrix

def update_stored_matrix(matrix, path):
    if not pathlib.Path(path).exists():
        raise ValueError(f"Matrix file {path} doesn't exist.")
    th.save(matrix, path)
    return 1

def store_level(data, lvl_id, hitboxes=True, overwrite=True):
    basic_levels[lvl_id] = str(lvl_id)
    lvl_name = basic_levels[lvl_id]
    
    if hitboxes:
        columns = HITBOX_COLUMNS
        pt_path = create_path(lvl_name, is_hitbox=True)
        csv_path = create_path(lvl_name, is_hitbox=True, is_init=True, is_csv=True)
    else:
        columns = OBJ_COLUMNS
        pt_path = create_path(lvl_name, is_hitbox=False)
        csv_path = create_path(lvl_name, is_hitbox=False, is_init=True, is_csv=True)

    if not overwrite:
        if pathlib.Path(pt_path).exists() and pathlib.Path(csv_path).exists():
            return 1
    
    df = decode_level_data(data, columns, to_sort=True)
    matrix = create_level_matrix(df, columns)

    df.to_csv(csv_path)
    th.save(matrix, pt_path)
    
    return 1


#visualise_level(matrix_path=f"{PATH}/1-hit.pt")
#print(create_level_matrix(get_level_data(1)))
