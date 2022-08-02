cmake_minimum_required(VERSION 3.16)
project(PlayingCardReader)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(card-reader main.cpp )
target_link_libraries(card-reader ${OpenCV_LIBS})

