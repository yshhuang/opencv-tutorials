cmake_minimum_required(VERSION 3.13)
project(opencv-tutorials)
find_package(OpenCV)
set(CMAKE_CXX_STANDARD 14)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv-tutorials main.cpp)
target_link_libraries(opencv-tutorials ${OpenCV_LIBS})

add_executable(display-image introduction-to-opencv/display-image.cpp)
target_link_libraries(display-image ${OpenCV_LIBS})

add_executable(modify-image introduction-to-opencv/modify-image.cpp)
target_link_libraries(modify-image ${OpenCV_LIBS})