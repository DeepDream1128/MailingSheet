cmake_minimum_required(VERSION 3.15)


project(msd)
# 设置O2优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(OpenCV 4 REQUIRED)
# find_package(Eigen3 REQUIRED)
# find_package(Ceres REQUIRED COMPONENTS EigenSparse)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(msd msd_v1.1.cpp)

target_link_libraries(msd ${OpenCV_LIBRARIES})