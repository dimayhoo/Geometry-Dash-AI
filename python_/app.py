# python_/app.py
import traceback
import threading
from package_init import package_init
from threads import *
import sys

game = package_init()
DONE = threading.Event()

def termination_callback():
    print("Terminating all threads...")
    DONE.set()
    sys.exit()

def initialise_threads(game, app):
    threading.Thread(target=main_learning_thread, args=(game, DONE, termination_callback), daemon=True).start()
    #threading.Thread(target=test_worker, args=(game, app, DONE), daemon=True).start()
    #threading.Thread(target=save_level_worker, args=(game, DONE, True), daemon=True).start()
    #threading.Thread(target=show_object_position, args=(game, DONE), daemon=True).start()
    #threading.Thread(target=play_layer_test, args=(game, DONE), daemon=True).start()
    #threading.Thread(target=test_gameData, args=(game, DONE), daemon=True).start()
    
    # Register the termination callback
    #threading.Thread(target=termination_callback, daemon=True).start()
    return 1

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
    #threading.Thread(target=test_worker, args=(app_delegate, ), daemon=True).start()
    initialise_threads(game, app_delegate)
    
    try:
        game.run()
    except KeyboardInterrupt:
        print("Game is terminated by user.")
        termination_callback()

if __name__ == "__main__":
    main()