# qq How to import module which is accessible in the parent folder (two ways in python)? https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder
# qq How to check the existence of a file at some path?
# TODO: Move this function to access game from parent into the universal, used file. Create a package, maybe.
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import cmake_example as game

def test_main():
    assert game.__version__ == "dev"
    assert game.add(1, 2) == 3
    assert game.subtract(1, 2) == -1
    assert game.FORCE_LOAD_LEVEL == True

if __name__ == "__main__":
    test_main()