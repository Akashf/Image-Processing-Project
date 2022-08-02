cmake_minimum_required(VERSION 3.16)
project(PlayingCardReader)
find_package(OpenCV REQUIRED)

file(GLOB_RECURSE SOURCES src/*)
file(GLOB_RECURSE HEADERS include/*)

include_directories(${OpenCV_INCLUDE_DIRS} include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(card-reader ${SOURCES} ${HEADERS})
target_link_libraries(card-reader ${OpenCV_LIBS})

