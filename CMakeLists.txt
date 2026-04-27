cmake_minimum_required(VERSION 3.23)
project(quakeCore LANGUAGES CXX)

option(QUAKECORE_ENABLE_AVX2 "Enable AVX2 instructions" ON)
option(QUAKECORE_ENABLE_OPENMP "Enable OpenMP parallel backend" ON)
option(QUAKECORE_BUILD_BASELINE "Build baseline target" ON)
option(QUAKECORE_BUILD_CPU_OPT "Build CPU-optimized target" OFF)
option(QUAKECORE_BUILD_GPU_OPT "Build GPU-optimized target (requires CUDA)" OFF)
option(QUAKECORE_BUILD_BENCH "Build universal benchmark target" OFF)

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

add_library(quakecore_baseline_engine
  src/engines/baseline_engine.cpp
)
target_link_libraries(quakecore_baseline_engine PUBLIC quakecore_core quakecore_io)

if(QUAKECORE_BUILD_CPU_OPT OR QUAKECORE_BUILD_BENCH)
  add_library(quakecore_cpu_engine
    src/engines/cpu_opt_engine.cpp
  )
  target_link_libraries(quakecore_cpu_engine PUBLIC quakecore_core quakecore_io)
endif()

if(QUAKECORE_ENABLE_AVX2 AND (QUAKECORE_BUILD_CPU_OPT OR QUAKECORE_BUILD_BENCH))
  target_compile_options(quakecore_cpu_engine PRIVATE -mavx2)
endif()

if(QUAKECORE_ENABLE_OPENMP AND (QUAKECORE_BUILD_CPU_OPT OR QUAKECORE_BUILD_BENCH))
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(quakecore_cpu_engine PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if(QUAKECORE_BUILD_GPU_OPT OR QUAKECORE_BUILD_BENCH)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(quakecore_gpu_engine
    src/engines/gpu_opt_engine.cu
  )
  set_target_properties(quakecore_gpu_engine PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 80
  )
  target_link_libraries(quakecore_gpu_engine PUBLIC quakecore_core quakecore_io CUDA::cudart)
endif()

if(QUAKECORE_BUILD_BASELINE)
  add_executable(quakecore_baseline src/apps/baseline_main.cpp)
  target_link_libraries(quakecore_baseline PRIVATE quakecore_baseline_engine quakecore_core quakecore_io)
endif()

if(QUAKECORE_BUILD_CPU_OPT)
  add_executable(quakecore_cpu_opt src/apps/cpu_opt_main.cpp)
  target_link_libraries(quakecore_cpu_opt PRIVATE quakecore_cpu_engine quakecore_core quakecore_io)
endif()

if(QUAKECORE_BUILD_GPU_OPT)
  add_executable(quakecore_gpu_opt src/apps/gpu_opt_main.cpp)
  target_link_libraries(quakecore_gpu_opt PRIVATE quakecore_gpu_engine quakecore_core quakecore_io)
endif()

if(QUAKECORE_BUILD_BENCH)
  if(NOT QUAKECORE_BUILD_CPU_OPT)
    message(FATAL_ERROR "QUAKECORE_BUILD_BENCH=ON requires QUAKECORE_BUILD_CPU_OPT=ON")
  endif()
  if(NOT QUAKECORE_BUILD_GPU_OPT)
    message(FATAL_ERROR "QUAKECORE_BUILD_BENCH=ON requires QUAKECORE_BUILD_GPU_OPT=ON")
  endif()
  add_executable(quakecore_bench src/apps/bench_main.cpp)
  target_link_libraries(quakecore_bench PRIVATE quakecore_baseline_engine quakecore_cpu_engine quakecore_gpu_engine quakecore_core quakecore_io)
endif()
