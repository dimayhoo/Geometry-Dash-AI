import os
import sys
import inspect
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

mesa_dll_path = r"C:/tools/mesa/x64"  # Update this path as needed
os.environ['PATH'] = mesa_dll_path + ";" + os.environ.get('PATH', '')
print(os.environ['PATH'])

import cmake_example as game

def main():
    # Instantiate AppDelegate
    try:
        app_delegate = game.AppDelegate()
    except Exception as e:
        print(f"Failed to create AppDelegate instance: {e}")
        return

    print(app_delegate.__dir__())
    app_delegate.applicationDidFinishLaunching()
    # Initialize the game with the AppDelegate instance
    '''try:
        success = game.init(app_delegate)
    except Exception as e:
        print(f"Initialization encountered an error: {e}")
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
        print("Initialization failed.")'''

if __name__ == "__main__":
    main()