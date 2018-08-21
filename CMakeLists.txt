cmake_minimum_required(VERSION 3.0)
if(NOT CMAKE_VERSION VERSION_LESS 3.1)
    set(CMAKE_CXX_STANDARD 11)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -O3")
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(NCooper)

include_directories(
    include
)

add_executable(
    NCooper
    main.cpp
    src/math/linalg/vector.cpp
    src/math/linalg/matrix.cpp
    src/ai/nn/neuron.cpp
    src/ai/nn/layer/layer.cpp
    src/ai/nn/network/network.cpp
)
