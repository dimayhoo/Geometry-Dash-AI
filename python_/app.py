# python_/app.py
import time
import traceback
import threading
from package_init import package_init
from levelStructure import decode_cmake_level_data, store_level, visualise_level
from threads import game_runner, test_worker, level_worker, show_object_position

game = package_init()
DONE = False

def initialise_threads(game, app):
    # threading.Thread(target=test_worker, args=(game, app, DONE), daemon=True).start()
    # threading.Thread(target=level_worker, args=(game, DONE, True), daemon=True).start()
    get_pos = threading.Thread(target=show_object_position, args=(game, DONE), daemon=True)
    
    get_pos.start()
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