
import seaborn as sns
import torch as th
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

PATH = pathlib.Path(__file__).parent / "levels"

basic_levels = {
    1: "stereo-madness"
}

# Game's positions are axmol units. They are counted
# from low left corner. They are not pixels.

def decode_cmake_level_data(data):
    decoded_data = sorted([float(obj.x), float(obj.y), float(obj.rotation),  \
                           float(obj.id)] for obj in data)
    ids, X, y, rot = [], [], [], []
    for x0, y0, rot0, id0 in decoded_data:
        ids.append(id0)
        X.append(x0)   
        y.append(y0)
        rot.append(rot0)
    
    #print(decoded_data)
    return ids, X, y, rot, decoded_data
    

def visualise_level(data=None, X=None, y=None, ids=None, path=None):
    if not data and not X and not path: return 0
    elif data:
        ids, X, y, _, _ = decode_cmake_level_data(data)
    elif path:
        df = pd.read_csv(path)
        X = df['X']
        y = df['y']
        ids = df['id']
    
    sns.scatterplot(x=X, y=y, hue=ids)
    plt.show()

def store_level(data, level_id):
    name = basic_levels.get(level_id, level_id)
    ids, X, y, rot, decoded_data = decode_cmake_level_data(data)
    df = pd.DataFrame(decoded_data, columns=['X', 'y', 'rot', 'id'])
    csv_path = f"{PATH}/{name}.csv"
    df.to_csv(csv_path, index=False)
    return 1


#visualise_level(path=f"{PATH}/stereo-madness.csv")
