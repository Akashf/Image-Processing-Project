cmake_minimum_required(VERSION 3.1)
project(
    cvui 
    LANGUAGES CXX
    DESCRIPTION "Header-only C++ UI library built on openCV drawing primitives"
    HOMEPAGE_URL "https://github.com/Dovyski/cvui/"
)

file(GLOB_RECURSE HEADERS include/*)
source_group("inc" FILES ${HEADERS})
add_library(cvui INTERFACE ${HEADERS})
target_include_directories(cvui INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
