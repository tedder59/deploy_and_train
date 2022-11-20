cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(trt_tools LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS}")

if(Xavier)
    set(CMAKE_CUDA_ARCHITECTURES 72)
endif()

if(Orin)
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
find_library(CUDART_LIB cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                               PATH_SUFFIXES lib64)
add_library(cudart SHARED IMPORTED)
set_property(TARGET cudart PROPERTY IMPORTED_LOCATION ${CUDART_LIB})

# TensorRT
if(EXISTS "/usr/local/TensorRT")
    set(TRT_LIB_DIR "/usr/local/TensorRT/lib")
    set(TRT_INC_DIR "/usr/local/TensorRT/include")
else(EXISTS "/usr/local/TensorRT")
    if(Xavier OR Orin)
        set(TRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
        set(TRT_INC_DIR "/usr/include/aarch64-linux-gnu")
    else(Xavier OR Orin)
        set(TRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")
        set(TRT_INC_DIR "/usr/include/x86_64-linux-gnu")
    endif(Xavier OR Orin)
endif(EXISTS "/usr/local/TensorRT")

file(STRINGS "${TRT_INC_DIR}/NvInferVersion.h" VERSION_STRINGS REGEX "#define NV_TENSORRT_.*")
string(REGEX MATCH "NV_TENSORRT_MAJOR [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
string(REGEX MATCH "[0-9]" TRT_MAJOR ${TRT_TYPE_STRING})
string(REGEX MATCH "NV_TENSORRT_MINOR [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
string(REGEX MATCH "[0-9]" TRT_MINOR ${TRT_TYPE_STRING})

if(TRT_MAJOR GREATER_EQUAL 8)
    add_definitions(-DTRT8)
    if(TRT_MINOR GREATER 3)
        add_definitions(-DTENSORRT_MEMORY_POOL)
    endif(TRT_MINOR GREATER 3)
endif(TRT_MAJOR GREATER_EQUAL 8)

include_directories("${TRT_INC_DIR}")
find_library(NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR})
add_library(nvinfer SHARED IMPORTED)
set_property(TARGET nvinfer PROPERTY IMPORTED_LOCATION ${NVINFER_LIB})

find_library(NVONNXPARSER_LIB nvonnxparser HINTS ${TRT_LIB_DIR})
add_library(nvonnxparser SHARED IMPORTED)
set_property(TARGET nvonnxparser PROPERTY IMPORTED_LOCATION ${NVONNXPARSER_LIB})

find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS ${TRT_LIB_DIR})
add_library(nvinfer_plugin SHARED IMPORTED)
set_property(TARGET nvinfer_plugin PROPERTY IMPORTED_LOCATION ${NVINFER_PLUGIN_LIB})

add_subdirectory(plugins)
add_subdirectory(builder)
add_subdirectory(infer)
add_subdirectory(app)
