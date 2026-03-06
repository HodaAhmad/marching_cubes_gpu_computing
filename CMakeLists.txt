# Require a CMake new enough to support `CMAKE_CUDA_ARCHITECTURES native`
cmake_minimum_required(VERSION 3.24)

# Prefer GLVND OpenGL (gets rid of the warning)
cmake_policy(SET CMP0072 NEW)
set(OpenGL_GL_PREFERENCE GLVND)

project(GPU_Triangles LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Let CMake/NVCC pick the right SM for this machine
# (Alternatively: set it explicitly, e.g. set(CMAKE_CUDA_ARCHITECTURES 89))
set(CMAKE_CUDA_ARCHITECTURES native)


option(BUILD_VIEWER "Build the OpenGL viewer (requires GLFW + OpenGL dev libs)" ON)

# ---- GLFW / glm / OpenGL ----
if (BUILD_VIEWER)
  # ---- GLAD (C source -> needs C enabled in project) ----
  add_library(glad STATIC src/glad.c)
  target_include_directories(glad PUBLIC include)

  find_package(PkgConfig REQUIRED)

  # ---- GLM (header-only) ----
  # Prefer vendored GLM (external/glm), fall back to system headers if available.
  if (EXISTS "${CMAKE_SOURCE_DIR}/external/glm/glm/glm.hpp")
      message(STATUS "Using vendored GLM: ${CMAKE_SOURCE_DIR}/external/glm")
      include_directories(${CMAKE_SOURCE_DIR}/external/glm)
  else()
      message(STATUS "Using system GLM headers (if installed)")
      include_directories(/usr/include)
  endif()

  pkg_search_module(GLFW REQUIRED glfw3)
  find_package(OpenGL REQUIRED)

  # ---- Viewer executable ----
  add_executable(viewer src/viewer.cpp)
  target_include_directories(viewer PRIVATE include ${GLFW_INCLUDE_DIRS})
  
  target_link_libraries(viewer PRIVATE glad ${GLFW_LIBRARIES} OpenGL::GL dl)
else()
  message(STATUS "BUILD_VIEWER=OFF: skipping viewer (GLFW/OpenGL not required).")
endif()

# ---- Generator (CUDA) executable ----
add_executable(generator
  src/generator.cu
  src/mc_tables.cu
  src/k_scan.cu
  src/k_mc.cu
  src/k_chunk.cu
  src/scan.cu
)

# Enable device/host lambdas for CUDA 13+
target_compile_options(generator PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
target_include_directories(generator PRIVATE ${CMAKE_SOURCE_DIR}/include)

set_target_properties(generator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---- Baseline generator (CUDA) executable ----
add_executable(generator_baseline src/generator_baseline.cu)
target_include_directories(generator_baseline PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_compile_options(generator_baseline PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

set_target_properties(generator_baseline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(generator_baseline PRIVATE CUDA::cudart)


# Link CUDA runtime (modern, via CUDAToolkit package)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(generator PRIVATE CUDA::cudart)

enable_testing()
add_subdirectory(tests)