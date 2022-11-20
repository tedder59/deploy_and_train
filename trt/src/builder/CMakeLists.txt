add_library(dt_builder SHARED builder.cpp calibrator.cpp)
target_include_directories(dt_builder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(dt_builder dt_plugin nvinfer)

add_executable(build.out main.cpp)
target_include_directories(build.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(build.out dt_builder dt_plugin nvinfer nvonnxparser nvinfer_plugin cudart)
