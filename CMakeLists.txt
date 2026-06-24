cmake_minimum_required(VERSION 3.19)

set(SIMULOK_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_ARCHITECTURES "${SIMULOK_CUDA_ARCHITECTURES}")

project(Simulok LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(Freetype REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PNG REQUIRED)
find_package(glfw3 REQUIRED)

set(SIMULOK_GUI_SOURCES
    gui/main.cpp
    gui/material_database.cpp
    gui/png_mask.cpp
    gui/result_io.cpp
    gui/simulation_config.cpp)

set(SIMULOK_WORKER_SOURCES
    gui/material_database.cpp
    gui/png_mask.cpp
    gui/result_io.cpp
    gui/simulation_backend.cu
    gui/simulation_config.cpp
    gui/worker_main.cpp
    src/complex_matrix.cu
    src/farfield_abbe.cu
    src/rcwa.cu)

add_executable(simulok_gui ${SIMULOK_GUI_SOURCES})
target_include_directories(simulok_gui PRIVATE gui)
target_link_libraries(simulok_gui PRIVATE Freetype::Freetype OpenGL::GL glfw PNG::PNG)
target_compile_options(simulok_gui PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>)
add_custom_command(TARGET simulok_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:simulok_gui>/resources/fonts
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/resources/fonts/LiberationSans-Bold.ttf
            $<TARGET_FILE_DIR:simulok_gui>/resources/fonts/LiberationSans-Bold.ttf
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/resources/fonts/LiberationSans-Regular.ttf
            $<TARGET_FILE_DIR:simulok_gui>/resources/fonts/LiberationSans-Regular.ttf
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/resources/fonts/LICENSE-liberation.txt
            $<TARGET_FILE_DIR:simulok_gui>/resources/fonts/LICENSE-liberation.txt
    COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:simulok_gui>/resources/config
    COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:simulok_gui>/resources/geometry
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/config/default_simulation.json
            $<TARGET_FILE_DIR:simulok_gui>/resources/config/default_simulation.json
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/resources/tztek_materials.csv
            $<TARGET_FILE_DIR:simulok_gui>/resources/tztek_materials.csv
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/resources/geometry
            $<TARGET_FILE_DIR:simulok_gui>/resources/geometry)

function(add_simulok_worker precision use_double)
    set(target simulok_worker_${precision})
    add_executable(${target} ${SIMULOK_WORKER_SOURCES})
    target_include_directories(${target} PRIVATE gui include)
    target_compile_definitions(${target} PRIVATE USE_DOUBLE_PRECISION=${use_double})
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --ftz=true>)
    target_link_libraries(${target} PRIVATE
        CUDA::cublas
        CUDA::cufft
        CUDA::cusolver
        PNG::PNG)
endfunction()

add_simulok_worker(double 1)
add_simulok_worker(single 0)
