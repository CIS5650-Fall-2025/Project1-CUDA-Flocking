cmake_minimum_required(VERSION 3.18)
project(cis5650_boids LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(OpenGL REQUIRED)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(GL_LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(GL_LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

find_package(CCCL REQUIRED)

set(headers
    src/cudaMat4.hpp
    src/glslUtility.hpp
    src/kernel.h
    src/main.hpp
    src/utilityCore.hpp
    )

set(sources
    src/glslUtility.cpp
    src/kernel.cu
    src/main.cpp
    src/utilityCore.cpp
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_link_libraries(${CMAKE_PROJECT_NAME} ${GL_LIBRARIES} CCCL::Thrust)
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
endif()
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:-G>
)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

set(N_FOR_VIS 5000 CACHE STRING "Number of boids to simulate")
option(VISUALIZE "Turn on OpenGL visualization" ON)
option(FINE_GRAINED_CELLS "Use fine-grained cells with 27 neighbors instead of 8" ON)
option(UNIFORM_GRID "Use uniform grids for efficient neighbor checks" OFF)
option(COHERENT_GRID "Use semi-coherent memory access for uniform grids" OFF)
set(CUDA_BLOCK_SIZE 128 CACHE STRING "CUDA block size")
option(FPS_MEASURE "Enable frame rate measurements" OFF)
set(FPS_MEASURE_START 2 CACHE STRING "Seconds after start to begin frame rate measurements")
set(FPS_MEASURE_DURATION 20 CACHE STRING "Number of seconds to measure frame rates for")
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    N_FOR_VIS=${N_FOR_VIS}
    VISUALIZE=$<IF:$<BOOL:${VISUALIZE}>,1,0>
    FINE_GRAINED_CELLS=$<IF:$<BOOL:${FINE_GRAINED_CELLS}>,1,0>
    UNIFORM_GRID=$<IF:$<BOOL:${UNIFORM_GRID}>,1,0>
    COHERENT_GRID=$<IF:$<BOOL:${COHERENT_GRID}>,1,0>
    CUDA_BLOCK_SIZE=${CUDA_BLOCK_SIZE}
    FPS_MEASURE=$<IF:$<BOOL:${FPS_MEASURE}>,1,0>
    FPS_MEASURE_START=${FPS_MEASURE_START}
    FPS_MEASURE_DURATION=${FPS_MEASURE_DURATION}
)

add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
    )
