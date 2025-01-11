import os
import sys
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import cmake_example as game

def main():
    # Instantiate AppDelegate
    try:
        app_delegate = game.AppDelegate()
    except Exception as e:
        print(f"Failed to create AppDelegate instance: {e}")
        return

    # Initialize the game with the AppDelegate instance
    success = app_delegate.applicationDidFinishLaunching()
    '''try:
        success = game.init(app_delegate)
    except Exception as e:
        print(f"Initialization encountered an error: {e}")
        return'''
    
    if success:
        print("Initialization successful.")
        try:
            # Keep the Python script running to maintain the game loop
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Game terminated by user.")
    else:
        print("Initialization failed.")

if __name__ == "__main__":
    main()