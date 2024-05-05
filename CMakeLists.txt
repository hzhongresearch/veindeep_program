# Program details
cmake_minimum_required(VERSION 3.10)
project(VeinDeep_exe)
set(CMAKE_CXX_STANDARD 20)

# Libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV4 REQUIRED opencv4)

# Compile the following files
add_executable(VeinDeep_exe main.cpp hausdorff.cpp hausdorff.h kerneldist.cpp kerneldist.h veindeep.cpp veindeep.h veinfeature.cpp veinfeature.h)
target_link_libraries(VeinDeep_exe ${OPENCV4_LIBRARIES})
target_include_directories(VeinDeep_exe PUBLIC ${OPENCV4_INCLUDE_DIRS})
target_compile_options(VeinDeep_exe PUBLIC ${OPENCV4_CFLAGS_OTHER})
