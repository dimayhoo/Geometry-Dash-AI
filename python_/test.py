from package_init import package_init
game = package_init()
from levelStructure import get_block_index_x

def test_main():
    assert game.__version__ == "dev"
    assert game.add(1, 2) == 3
    assert game.subtract(1, 2) == -1
    assert game.FORCE_LOAD_LEVEL == True

def test_level():
    assert get_block_index_x(15, 6) == 2
    assert get_block_index_x(24, 6, True) == 3

if __name__ == "__main__":
    test_main()
    test_level()