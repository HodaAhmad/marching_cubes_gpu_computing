#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(expr) do {                              \
  cudaError_t _err = (expr);                               \
  if (_err != cudaSuccess) {                               \
    fprintf(stderr, "CUDA error %s at %s:%d\n",            \
            cudaGetErrorString(_err), __FILE__, __LINE__); \
    std::exit(1);                                          \
  }                                                        \
} while(0)
