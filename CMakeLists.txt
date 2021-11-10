cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
IF(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    message("${CMAKE_CXX_FLAGS_DEBUG} -O0")
    message("${CMAKE_C_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)
find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)
message(STATUS ${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
#include_directories(${PROJECT_SOURCE_DIR}/include/serialize.hpp)
#include_directories(${PROJECT_SOURCE_DIR}/include/torch_unfold.h)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
message('hello world')
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall - Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,stack-size=100000000")
cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
cuda_add_library(involution2dPlugin SHARED ${PROJECT_SOURCE_DIR}/include/involution2d.cu)
cuda_add_library(unfoldplugin SHARED ${PROJECT_SOURCE_DIR}/include/torch_unfold.cu)



find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov5 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/yolov5.cpp ${PROJECT_SOURCE_DIR}/unfold.cpp ${PROJECT_SOURCE_DIR}/Involution2dPlugin.cpp)

target_link_libraries(yolov5 involution2dPlugin)
target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 unfoldplugin)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 ${OpenCV_LIBS})



