import traceback
import time
from levelStructure import visualise_level, store_level, decode_level_data

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

def save_level_worker(game, DONE, hitboxes):
    while not DONE:
        time.sleep(3)
        playLayer = game.PlayLayer.getInstance()
        print("PlayLayer is initialised successfully.")
        if playLayer: 
            DONE = True

            lvl_id = playLayer.getLevelId()
            if lvl_id == -1:
                raise ValueError("Level ID is invalid. ")

            print('PlayLayer isn\'t None. ')
            if hitboxes:
                lvl_data = playLayer.getLevelHitboxData()
            else:
                lvl_data = playLayer.getLevelData()

            store_level(lvl_data, lvl_id=lvl_id, hitboxes=hitboxes)
            return 1
        
        else:
            print("PlayLayer isn't initialised in the game still. ")
    
    return 0


def show_object_position(game, DONE):
    while not DONE:
        time.sleep(3)
        playLayer = game.PlayLayer.getInstance()
        print("PlayLayer is initialised successfully.")
        if playLayer: 
            while not DONE:
                time.sleep(0.016)
                posX = game.getObjectPositionX(playLayer)
                posY = game.getObjectPositionY(playLayer)
                #positions = game_object.getPositionX()
                print("Object X: {}, Y: {}".format(posX, posY))
                
        
        else:
            print("PlayLayer isn't initialised in the game still. ")

        