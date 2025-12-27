cmake_minimum_required(VERSION 3.19)
project(OpticalSimulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_DOUBLE_PRECISION "Use double precision complex ops" OFF)

file(GLOB_RECURSE CPP_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CUDA_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_library(complex_ops STATIC ${CPP_SRC_FILES} ${CUDA_SRC_FILES})
target_include_directories(complex_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(complex_ops PUBLIC CUDA::cublas CUDA::cusolver)

add_executable(basic_arithm_test ${CMAKE_CURRENT_SOURCE_DIR}/examples/basic_arithm_test.cu)
target_link_libraries(basic_arithm_test PRIVATE complex_ops)

add_executable(T_matrix_test ${CMAKE_CURRENT_SOURCE_DIR}/examples/T_matrix_test.cu)
target_link_libraries(T_matrix_test PRIVATE complex_ops)

if(USE_DOUBLE_PRECISION)
  set(PRECISION_DEF USE_DOUBLE_PRECISION=1)
else()
  set(PRECISION_DEF USE_DOUBLE_PRECISION=0)
endif()

target_compile_definitions(complex_ops PUBLIC ${PRECISION_DEF})
target_compile_definitions(basic_arithm_test PRIVATE ${PRECISION_DEF})
target_compile_definitions(T_matrix_test PRIVATE ${PRECISION_DEF})

# Optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-trapping-math -fno-math-errno -fno-signed-zeros")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fassociative-math -freciprocal-math -fno-rounding-math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ftz=true")
