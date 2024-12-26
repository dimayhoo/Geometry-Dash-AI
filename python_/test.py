# qq How to import module accessible in parent folder (two ways) in python? https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import cmake_example as m

def test_main():
    assert m.__version__ == "0.0.1"
    assert m.add(1, 2) == 3
    assert m.subtract(1, 2) == -1

if __name__ == "__main__":
    test_main()