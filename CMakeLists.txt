# Program details
cmake_minimum_required(VERSION 3.10)
project(VeinDeep_exe)
set(CMAKE_CXX_STANDARD 20)

# Libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Compile the following files
add_executable(VeinDeep_exe main.cpp hausdorff.cpp hausdorff.h kerneldist.cpp kerneldist.h veindeep.cpp veindeep.h veinfeature.cpp veinfeature.h)
target_link_libraries(VeinDeep_exe ${OpenCV_LIBS})
