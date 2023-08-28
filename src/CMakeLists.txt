cmake_minimum_required(VERSION 3.4.1)
project(roop.cpp)
# if(NOT DEFINED ENV{VENUS_PATH})
#    message(FATAL_ERROR "environment variable:VENUS_PATH not found")
# endif()
# message(STATUS "venus path $ENV{VENUS_PATH}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

option(BUILD_WITH_OPENVINO "Enable OpenVINO" ON)

if (BUILD_WITH_OPENVINO)
	if(NOT DEFINED ENV{OPENVINO_ROOT_PATH})
		message(FATAL_ERROR "not defined environment variable:OPENVINO_ROOT_PATH")
	endif()
	message("openvino directory: $ENV{OPENVINO_ROOT_PATH}")
	include_directories($ENV{OPENVINO_ROOT_PATH}/include)
	include_directories($ENV{OPENVINO_ROOT_PATH}/include/ie)
	include_directories($ENV{OPENVINO_ROOT_PATH}/3rdparty/tbb/include)
	include_directories($ENV{OPENVINO_ROOT_PATH}/deployment_tools/inference_engine/include)
	include_directories($ENV{OPENVINO_ROOT_PATH}/deployment_tools/ngraph/include)
	include_directories($ENV{OPENVINO_ROOT_PATH}/deployment_tools/inference_engine/external/tbb/include)
	link_directories($ENV{OPENVINO_ROOT_PATH}/deployment_tools/inference_engine/lib/intel64)
	link_directories($ENV{OPENVINO_ROOT_PATH}/deployment_tools/inference_engine/external/tbb/lib)
	link_directories($ENV{OPENVINO_ROOT_PATH}/deployment_tools/ngraph/lib)
	link_directories($ENV{OPENVINO_ROOT_PATH}/runtime/lib/intel64)
	link_directories($ENV{OPENVINO_ROOT_PATH}/3rdparty/tbb/lib)
	link_directories($ENV{OPENVINO_ROOT_PATH}/lib/intel64)	
endif()

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/engine ENGINE_SRC)

file(GLOB ENGINE_SOURCE
    "engine/*.h"
    "engine/*.cpp"
    )

message(STATUS "venus path ${ENGINE_SOURCE}")

add_executable(roop roop.cpp ${ENGINE_SOURCE})
