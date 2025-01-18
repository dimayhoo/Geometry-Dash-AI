# python_/app.py
import time
import traceback
import threading
from package_init import package_init

game = package_init()
DONE = False

def game_runner():
    try:
        game.run()
    except Exception as e:
        print(f"Game encountered an error: {e}")
        traceback.print_exc()

def test_worker(app):
    global DONE
    c= 0 
    while not DONE:
        time.sleep(2)
        playLayer = game.PlayLayer.getInstance()
        print("PlayLayer is initialised successfully.")
        if playLayer: 
            print('PlayLayer isn\'t None. It\'s:', playLayer.__dir__())
            print(playLayer.getLevelData())
        else:
            print("PlayLayer isn't initialised in the game still. ")
    
    while not DONE:
        time.sleep(0.016)
        c+= 1
        #print('fps')
        if c == 60:
            c = 0
            #print(game.init(app))
            #print(game.FORCE_LOAD_LEVEL)
            #print(game.add(2, 3))
    

def main():
    global DONE
    # Instantiate AppDelegate
    try:
        app_delegate = game.AppDelegate()
    except Exception as e:
        print(f"Failed to create AppDelegate instance: {e}")
        traceback.print_exc()
        return

    print('Starting the game...')
    # Start game_runner in a separate thread
    #game_thread = threading.Thread(target=game_runner)
    #game_thread.start()
    threading.Thread(target=test_worker, args=(app_delegate, ), daemon=True).start()
    try:
        game.run()
    except KeyboardInterrupt:
        print("Game is terminated by user.")
        DONE = True

    print('Game thread started. You can now control the game.')

    # Initialize the game with the AppDelegate instance
    try:
        success = game.init(app_delegate)
    except Exception as e:
        print(f"Initialization encountered an error: {e}")
        traceback.print_exc()
        return

    if success:
        print("Initialization successful.")
    
    # Example control loop
    '''try:
        while not DONE:
            # Here, implement your control logic, e.g., AI decisions or keyboard input handling
            time.sleep(0.016)  # Placeholder for actual logic
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt. Shutting down.")
        DONE = True
        game.stop()  # Assuming you have a method to gracefully stop the game
        game_thread.join()'''

if __name__ == "__main__":
    main()