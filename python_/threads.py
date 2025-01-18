import traceback
import time
from levelStructure import visualise_level, store_level, decode_cmake_level_data

def game_runner(game):
    try:
        game.run()
    except Exception as e:
        print(f"Game encountered an error: {e}")
        traceback.print_exc()

def test_worker(game, app, DONE):
    while not DONE:
        time.sleep(0.016)
        c+= 1
        #print('fps')
        if c == 60:
            c = 0
            #print(game.init(app))
            #print(game.FORCE_LOAD_LEVEL)
            #print(game.add(2, 3))

def level_worker(game, DONE, store=False):
    while not DONE:
        time.sleep(3)
        playLayer = game.PlayLayer.getInstance()
        print("PlayLayer is initialised successfully.")
        if playLayer: 
            DONE = True

            print('PlayLayer isn\'t None. ')
            lvl_data = playLayer.getLevelData()
            if lvl_data: 
                #decode_cmake_level_data(lvl_data)
                #visualise_level(data=lvl_data)
                if store:
                    store_level(lvl_data, 1)
                pass
            return 1
        
        else:
            print("PlayLayer isn't initialised in the game still. ")
    
    return 0
        