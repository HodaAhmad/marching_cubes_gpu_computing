cmake_minimum_required(VERSION 3.24)

find_package(CUDAToolkit REQUIRED)

# Test 1: scan

add_executable(test_scan test_scan.cu)

target_include_directories(test_scan PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(test_scan PRIVATE CUDA::cudart)

# Compile the scan implementation into the test binary
target_sources(test_scan PRIVATE
  ${CMAKE_SOURCE_DIR}/src/scan.cu
  ${CMAKE_SOURCE_DIR}/src/k_scan.cu
)

set_target_properties(test_scan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME test_scan COMMAND test_scan)

# Test 2: scatter chunks

add_executable(test_scatter_chunks test_scatter_chunks.cu)

target_include_directories(test_scatter_chunks PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(test_scatter_chunks PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

target_link_libraries(test_scatter_chunks PRIVATE CUDA::cudart)

target_sources(test_scatter_chunks PRIVATE
  ${CMAKE_SOURCE_DIR}/src/k_chunk.cu
  ${CMAKE_SOURCE_DIR}/src/mc_tables.cu
)



set_target_properties(test_scatter_chunks PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_test(NAME test_scatter_chunks COMMAND $<TARGET_FILE:test_scatter_chunks>)
