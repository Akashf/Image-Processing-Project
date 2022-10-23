cmake_minimum_required(VERSION 3.16)
project(PlayingCardReader)
find_package(OpenCV REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE SOURCES src/*)
file(GLOB_RECURSE HEADERS include/*)

source_group("src" FILES ${SOURCES})
source_group("inc" FILES ${HEADERS})

add_executable(card-reader ${SOURCES} ${HEADERS})
include_directories(${OpenCV_INCLUDE_DIRS} include)

target_link_libraries(card-reader cvui)
target_link_libraries(card-reader ${OpenCV_LIBS})
set_property(TARGET card-reader PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

add_subdirectory(vendor/cvui)

# Copy all assets to install directory 
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/assets/ 
        DESTINATION ${INSTALL_PATH}/assets
        FILES_MATCHING PATTERN "*.jpg"  PATTERN "*.jpg")
    