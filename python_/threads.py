import traceback
import time
from levelStructure import visualise_level, store_level, decode_level_data
from agent import Agent
import queue
import numpy as np

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
            print("Level was stored successfully.")
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

def play_layer_test(game, DONE):
    while not DONE:
        time.sleep(3)
        playLayer = game.PlayLayer.getInstance()
        print("PlayLayer is initialised successfully.")
        if playLayer: 
            while not DONE:
                time.sleep(2)
                playLayer._pauseUpdate = True if not playLayer._pauseUpdate else False
                #playLayer.resetLevel()

                #print("HERE'S:", playLayer.m_bOnGround)
                
        
        else:
            print("PlayLayer isn't initialised in the game still. ")

    
def main_learning_thread(game, DONE):
    """ NOTE: Functionality (game loop)
    This is continuous game loop. Observing, learning, etc works by agent status.
    In other words, I call callback -> agent starts learning -> agent completed that ->
    changes status to "ready" -> loop continues -> ... -> epoches completed (done) ->
    model and any important info saved -> closure.
    """

    print("Main learning thread is initialised. ")

    def agent_worker():
        while not DONE:
            if agent.status == "done":
                return 1
            
            elif agent.status == "ready":
                game_data = agent.get_game_input()
                gameDataInstance.initGameData(game_data, agent_callback)
                print("Starting the observing process.")
                agent.status = "observing"
            
            else: # agent.status == "observing" or "training"
                try:
                    data = observation_queue.get(timeout=1)
                    # TODO: pausing playlayer (I suspect by setting _pauseUpdate to true)
                    agent.handle_game_observations(data) # here will be training
                except queue.Empty:
                    continue

    def agent_callback(data):
        observation_queue.put(data)

    gameDataInstance = None
    while not DONE: 
        time.sleep(1)
        # O(1) fn's computation
        gameDataInstance = game.GameData.getInstance()
        if gameDataInstance is not None:
            break

    playLayer = None
    while not DONE:
        time.sleep(1)
        # O(1) fn's computation
        playLayer = game.PlayLayer.getInstance()
        if playLayer is not None:
            break
    
    if playLayer is None: return 0
    print("PlayLayer is initialised successfully. ")

    observation_queue = queue.Queue()

    model_params = {
        "name": "ppo",
        "to_init": True
    }
    lvl_id = playLayer.getLevelId() 

    # NOTE: if PlayLayer breaks, then the entire obervation process,
    # learning process and other agent functions break completely.
    agent = Agent(model_params=model_params, lvl_id=lvl_id)

    status = agent_worker() # in any way, agent stop working
    # TODO: ensuring agent saved everything (and closed everything) + callback to stop game from running (close all threads)...
    
            
def test_gameData(game, DONE):
    while not DONE:
        time.sleep(2)
        gameData = game.GameData.getInstance()
        print("GameData is initialised successfully.")
        break

    my_dict = {
        "key1": {
            "prev": False,
            "values": [1, 2, 3]
        },
        "key2": {
            "prev": True,
            "values": [234, 234, 234]
        }
    }

    actionsMatrix = np.random.randint(0, 1, (3, 3))
    oneBlockSize = (6, 15)


    '''gameData.setWholeDict({
        "key1": [False, [1, 2, 3]],
        "key2": {
            "prev": True,
            "values": [234, 234, 234]
        },
        "key3": {
            "prev": True,
            "values": np.random.randint(0, 10, (5, ))
        }
    })
    print("Whole dict is set successfully.")

    my_dict_cpp = gameData.getWholeDict()
    print(my_dict_cpp.get("key3"))
    print(my_dict_cpp.get("key3").get("values")[0])

    matA = np.random.randint(0, 10, (3, 3)) 
    matB = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    matC = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    gameData.setAllMatrices(matA, matB, matC)
    print("All matrices are set successfully.")

    print(gameData.getAllMatrices())'''
    
    color = "red"
    def simple_callback():
        print("The color is:", color)
    
    gameData.setCallback(simple_callback)
    print("Callback is set successfully.")

    gameData.doCallback()


    '''while not DONE:
        time.sleep(3)
        playLayer = game.PlayLayer.getInstance()
        print("PlayLayer is initialised successfully.")
        if playLayer: 
            while not DONE:
                time.sleep(0.016)
                game_data = playLayer.getGameData()
                print(game_data)
                
        
        else:
            print("PlayLayer isn't initialised in the game still. ")'''



        