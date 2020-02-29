# Welcome

Welcome to this project!

## External

This project uses the Eigen3 library.

## Usage

This is intended to be used with the script build.sh.

## Warning

Don't declare a main function in src.
If this is your use case you need to change the Cmake file to stop using GLOB.
(This isnt recommended anyway:
We do not recommend using GLOB to collect a list of source files from your source tree. If no CMakeLists.txt file changes when a source is added or removed then the generated build system cannot know when to ask CMake to regenerate.)

Why is it in here anyway?
Because I just tested the code and never actually needed to run it (its for my and your understanding).
