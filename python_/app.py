import os
import sys
import inspect
import time
import traceback
import pathlib

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

path = os.path.join(pathlib.Path(__file__).parent.parent, "cmake_example/build/lib.win-amd64-cpython-312")
mesa_dll_path = r"C:/tools/mesa/x64"  # Update this path as needed
os.environ['PATH'] = path + ";" + os.environ.get('PATH', '')

import cmake_example as game

def main():
    # Instantiate AppDelegate
    try:
        app_delegate = game.AppDelegate()
    except Exception as e:
        print(f"Failed to create AppDelegate instance: {e}")
        traceback.print_exc()
        return

    # Initialize the game with the AppDelegate instance
    try:
        print('something')
        success = game.init(app_delegate)
        print('something')
    except Exception as e:
        print(f"Initialization encountered an error: {e}")
        traceback.print_exc()
        return
    
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