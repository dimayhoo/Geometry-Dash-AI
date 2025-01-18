from app import game

def inspect_level():
    # Suppose you have a way to get the current PlayLayer instance, e.g., a function getCurrentPlayLayer()
    #current_layer = game.getCurrentPlayLayer()  # this is hypothetical; depends on your design

    # Now retrieve the parsed objects
    objects = game.getLevelData()
    for obj_info in objects:
        print(f"Object ID={obj_info.id}, x={obj_info.x}, y={obj_info.y}, rot={obj_info.rotation}")

inspect_level()
