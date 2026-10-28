cmake_minimum_required(VERSION 3.19)
find_program(RCWA_NVCC_EXECUTABLE nvcc)
if(RCWA_NVCC_EXECUTABLE)
  set(CMAKE_CUDA_COMPILER "${RCWA_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
endif()
# Set the project default before CMake initializes the CUDA compiler.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build" FORCE)
else()
  set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures to build")
endif()
project(OpticalSimulator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_DOUBLE_PRECISION "Use double precision complex ops" ON)

file(GLOB_RECURSE CPP_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CUDA_SRC_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_library(complex_ops STATIC ${CPP_SRC_FILES} ${CUDA_SRC_FILES})
target_include_directories(complex_ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(complex_ops PUBLIC CUDA::cublas CUDA::cusolver CUDA::cufft)

add_executable(homogeneous_test ${CMAKE_CURRENT_SOURCE_DIR}/example/homogeneous_test.cu)
target_link_libraries(homogeneous_test PRIVATE complex_ops)

add_executable(method_test ${CMAKE_CURRENT_SOURCE_DIR}/example/method_test.cu)
target_link_libraries(method_test PRIVATE complex_ops)

add_executable(pipeline_test ${CMAKE_CURRENT_SOURCE_DIR}/example/pipeline_test.cu)
target_link_libraries(pipeline_test PRIVATE complex_ops)

add_executable(TSV ${CMAKE_CURRENT_SOURCE_DIR}/example/TSV.cu)
target_link_libraries(TSV PRIVATE complex_ops)
option(BUILD_TRANSFER_EXPERIMENTS "Build transfer profiling experiments" OFF)
if(BUILD_TRANSFER_EXPERIMENTS)
  add_executable(transfer_gemm_test ${CMAKE_CURRENT_SOURCE_DIR}/example/transfer_gemm_test.cu)
  target_link_libraries(transfer_gemm_test PRIVATE complex_ops)
  enable_testing()
  add_test(NAME transfer_gemm_test COMMAND transfer_gemm_test)
  add_executable(transfer_gemm_bench ${CMAKE_CURRENT_SOURCE_DIR}/example/transfer_gemm_bench.cu)
  target_link_libraries(transfer_gemm_bench PRIVATE complex_ops)
endif()

add_executable(lalanne_convergence_test ${CMAKE_CURRENT_SOURCE_DIR}/example/lalanne_convergence_test.cu)
target_link_libraries(lalanne_convergence_test PRIVATE complex_ops)

add_executable(normal_vector_field_test ${CMAKE_CURRENT_SOURCE_DIR}/example/normal_vector_field_test.cu)
target_link_libraries(normal_vector_field_test PRIVATE complex_ops)

if(USE_DOUBLE_PRECISION)
  set(PRECISION_DEF USE_DOUBLE_PRECISION=1)
else()
  set(PRECISION_DEF USE_DOUBLE_PRECISION=0)
endif()

target_compile_definitions(complex_ops PUBLIC ${PRECISION_DEF})
target_compile_definitions(homogeneous_test PRIVATE ${PRECISION_DEF})
target_compile_definitions(method_test PRIVATE ${PRECISION_DEF})
target_compile_definitions(pipeline_test PRIVATE ${PRECISION_DEF})
target_compile_definitions(TSV PRIVATE ${PRECISION_DEF})
target_compile_definitions(lalanne_convergence_test PRIVATE ${PRECISION_DEF})
target_compile_definitions(normal_vector_field_test PRIVATE ${PRECISION_DEF})

# Optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-trapping-math -fno-math-errno -fno-signed-zeros")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fassociative-math -freciprocal-math -fno-rounding-math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ftz=true")
