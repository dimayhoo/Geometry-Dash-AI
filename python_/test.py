from package_init import package_init
game = package_init()

def test_main():
    assert game.__version__ == "dev"
    assert game.add(1, 2) == 3
    assert game.subtract(1, 2) == -1
    assert game.FORCE_LOAD_LEVEL == True

if __name__ == "__main__":
    test_main()