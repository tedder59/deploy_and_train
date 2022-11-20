# benchmark
find_package(benchmark REQUIRED)

# opencv
if(Orin OR Xavier)
    include_directories("/usr/include/opencv4")
    link_directories("/usr/lib/aarch64-linux-gnu")
    set(OpenCV_LIBS opencv_imgcodecs opencv_core opencv_imgproc)
else(Orin OR Xavier)
    find_package(OpenCV REQUIRED)
    include_directories("${OpenCV_INCLUDE_DIRS}")
    link_directories("${OpenCV_LIBRARY_DIRS}")
endif(Orin OR Xavier)

# dt
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

# bench
add_executable(bench benchmark.cpp)
target_link_libraries(bench dt_infer dt_builder dt_plugin
                            nvinfer nvonnxparser
                            nvinfer_plugin cudart
                            ${OpenCV_LIBS} benchmark
                            pthread)

# centernet
add_executable(cdet centernet/main.cpp)
target_link_libraries(cdet dt_infer dt_builder dt_plugin
                           nvinfer nvonnxparser
                           nvinfer_plugin cudart
                           ${OpenCV_LIBS} benchmark
                           pthread)
