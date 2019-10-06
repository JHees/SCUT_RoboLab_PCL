cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Debug)
project(SCUT_RoboLab_PCL)

find_package(PCL 1.8 REQUIRED)

#include_directories(${PCL_INCLUDE_DIRS}})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

add_executable(SCUT_RoboLab_PCL SCUT_RoboLab_PCL.cpp)
target_link_libraries(SCUT_RoboLab_PCL ${PCL_LIBRARIES})