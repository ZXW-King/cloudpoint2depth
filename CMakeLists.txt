cmake_minimum_required(VERSION 3.20)
project(cloudpoint2depth)

set(CMAKE_CXX_STANDARD 14)

add_executable(cloudpoint2depth
        mian.cpp
        include/TofDepthData.h
        include/Resolution.h
        include/utils.h
        include/CameraMoudleParam.h
        include/CameraParam.h
        )

# opencv
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/)
# yaml
include(${ROOT_DIR}/cmake/find_yaml.cmake)
include_directories(${YAML_CPP_INCLUDE_DIR})
target_link_libraries(cloudpoint2depth ${YAMLCPP_LIBRARY})


find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries (cloudpoint2depth ${PCL_LIBRARIES})
install(TARGETS cloudpoint2depth RUNTIME DESTINATION bin)