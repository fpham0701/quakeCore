cmake_minimum_required(VERSION 3.23)
project(quakeCore LANGUAGES CXX)

option(QUAKECORE_ENABLE_AVX2 "Enable AVX2 instructions" ON)
option(QUAKECORE_ENABLE_OPENMP "Enable OpenMP parallel backend" ON)
option(QUAKECORE_ENABLE_CUDA "Enable CUDA backend when available" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(quakecore_core
  src/core/frustum.cpp
  src/bench/camera_path.cpp
)

add_library(quakecore_io
  src/io/bsp_parser.cpp
)

add_library(quakecore_engines
  src/engines/baseline_engine.cpp
  src/engines/cpu_opt_engine.cpp
  src/engines/gpu_opt_engine_stub.cpp
)

target_link_libraries(quakecore_engines PUBLIC quakecore_core quakecore_io)

if(QUAKECORE_ENABLE_AVX2)
  target_compile_options(quakecore_engines PRIVATE -mavx2)
endif()

if(QUAKECORE_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(quakecore_engines PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if(QUAKECORE_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    target_sources(quakecore_engines PRIVATE src/engines/gpu_opt_engine.cu)
    set_source_files_properties(src/engines/gpu_opt_engine_stub.cpp PROPERTIES HEADER_FILE_ONLY ON)
    set_target_properties(quakecore_engines PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES 80
    )
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(quakecore_engines PUBLIC CUDA::cudart)
  endif()
endif()

add_executable(quakecore_baseline src/apps/baseline_main.cpp)
target_link_libraries(quakecore_baseline PRIVATE quakecore_engines quakecore_core quakecore_io)

add_executable(quakecore_cpu_opt src/apps/cpu_opt_main.cpp)
target_link_libraries(quakecore_cpu_opt PRIVATE quakecore_engines quakecore_core quakecore_io)

add_executable(quakecore_gpu_opt src/apps/gpu_opt_main.cpp)
target_link_libraries(quakecore_gpu_opt PRIVATE quakecore_engines quakecore_core quakecore_io)
