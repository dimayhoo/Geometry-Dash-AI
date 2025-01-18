# python_/package_init.py
# Have to create this function because ordinal packaging doesn't work.

def package_init():
    import sys
    import os
    import pathlib

    # Determine the path to the built C++ Python module
    current_dir = pathlib.Path(__file__).parent
    build_dir = current_dir.parent / "cmake_example" / "build" / "lib.win-amd64-cpython-312"

    # Initialising working directory (axmol can't find it otherwise)
    os.chdir(build_dir)

    # Add the build directory to sys.path if not already present
    build_path_str = str(build_dir.resolve())
    if build_path_str not in sys.path:
        sys.path.append(build_path_str)

    try:
        import cmake_example as game
    except ImportError as e:
        print(f"Failed to import cmake_example: {e}")
        raise

    return game