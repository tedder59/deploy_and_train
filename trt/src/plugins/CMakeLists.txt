# file(GLOB LANEATT_PLUGIN_SRCS laneatt/*.cpp laneatt/*.cu)
file(GLOB CENTERNET_PLUGIN_SRCS centernet/*.cpp centernet/*.cu)

set(PLUGIN_SRCS)
list(APPEND PLUGIN_SRCS plugin.cpp
                        ${CENTERNET_PLUGIN_SRCS})

add_library(dt_plugin SHARED ${PLUGIN_SRCS})
target_include_directories(dt_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dt_plugin nvinfer cudart)

if(Xavier)
target_compile_options(dt_plugin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_72,code=[compute_72,sm_72]>)
endif()
