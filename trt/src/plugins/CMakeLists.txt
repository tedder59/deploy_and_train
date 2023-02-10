# file(GLOB LANEATT_PLUGIN_SRCS laneatt/*.cpp laneatt/*.cu)
file(GLOB CENTERNET_PLUGIN_SRCS centernet/*.cpp centernet/*.cu)
file(GLOB BEVDET_PLUGIN_SRCS bevdet/*.cpp bevdet/*.cu)

set(PLUGIN_SRCS)
list(APPEND PLUGIN_SRCS plugin.cpp
                        ${CENTERNET_PLUGIN_SRCS}
			${BEVDET_PLUGIN_SRCS})

add_library(dt_plugin SHARED ${PLUGIN_SRCS})
target_include_directories(dt_plugin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dt_plugin nvinfer cudart)

if(Xavier)
target_compile_options(dt_plugin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_72,code=[compute_72,sm_72]>)
else()
target_compile_options(dt_plugin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_87,code=[compute_87,sm_87]>)
endif()
