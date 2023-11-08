cmake_minimum_required(VERSION 3.20)
project(cloudpoint2depth)

set(CMAKE_CXX_STANDARD 14)

add_executable(cloudpoint2depth
        mian.cpp
        )

# opencv
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})