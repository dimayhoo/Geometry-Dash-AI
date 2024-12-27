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
include_directories("C:/Tools/libglass-2.0.0/include")

#link_directories("C:/Tools/libglass-2.0.0")

# Find SFML package
set(SFML_DIR "C:/Tools/SFML-3.0.0/lib/cmake/SFML")
find_package(SFML 3.0 REQUIRED COMPONENTS System Window Graphics Audio)

# Add executable
add_executable(GeometryDash ${SOURCES} ${HEADERS})

# Link SFML libraries
target_link_libraries(GeometryDash SFML::System SFML::Window SFML::Graphics SFML::Audio)

