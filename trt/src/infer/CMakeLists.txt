add_library(dt_infer SHARED trt_engine.cpp)
target_include_directories(dt_infer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(dt_infer dt_builder dt_plugin nvinfer nvonnxparser nvinfer_plugin cudart)
