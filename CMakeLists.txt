cmake_minimum_required(VERSION 3.15)
project(cc-neuralNetwork)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/_build)

file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

include_directories(include)

set(CMAKE_CXX_STANDARD 14)

file(GLOB SOURCES "src/*.cc" "src/**/*.cc")

add_library(neuralNetwork SHARED ${SOURCES})

add_executable(main ${SOURCES})

#install(FILES ${CMAKE_BINARY_DIR}/)

set(CATCH_INCLUDE_DIR extern/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

file(GLOB TEST_SOURCES "tests/*.cc" "src/**/*.cc" "src/*.cc")
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests Catch)
