cmake_minimum_required(VERSION 3.10)

# Project name
project(GeometryDashAI)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source and header files
file(GLOB_RECURSE SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS hdr/*.hpp)

# Include directories
include_directories(hdr)
include_directories("C:/Tools/Glass-4.0/include")
#include_directories("C:/Tools/boost_1_87_0")

#link_directories("C:/Tools/libglass-2.0.0")

# Find SFML package

# SFML with 2.4.0 even doesn't have a cmake config file
#set(SFML_DIR "C:/Tools/SFML-2.4.0/include/SFML")
#find_package(SFML 2.4.0 REQUIRED COMPONENTS system window graphics)

set(SFML_DIR "C:/Tools/SFML-2.6.0/lib/cmake/SFML")
find_package(SFML 2.6.0 REQUIRED COMPONENTS system window graphics)

#set(SFML_DIR "C:/Tools/SFML-2.5.1/lib/cmake/SFML")
#find_package(SFML 2.5.1 REQUIRED COMPONENTS system window graphics)

# Other 
#set(SFML_DIR "C:/Tools/SFML-3.0.0/lib/cmake/SFML")
#find_package(SFML 3.0 REQUIRED COMPONENTS System Window Graphics Audio)

link_directories(
    "C:/Tools/Glass-4.0/lib"           # Link Glass library
    "C:/Tools/SFML-2.6.0/lib"          # Link SFML libraries
)

# Add executable
add_executable(GeometryDash ${SOURCES} ${HEADERS})

# Link SFML libraries
#target_link_libraries(GeometryDash SFML:graphics SFML:system SFML:Window SFML:Audio)
target_link_libraries(GeometryDash sfml-graphics sfml-system sfml-window Glass)

