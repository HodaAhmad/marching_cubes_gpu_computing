#pragma once
#include <cstdint>
#include <cuda_runtime.h>

template<int BLOCK_THREADS>
__global__ void scanBlocksKernel(
    const uint32_t* in,
    uint32_t* out,
    uint32_t* blockSums,
    int n
);

template<int BLOCK_THREADS>
__global__ void addBlockOffsetsKernel(
    uint32_t* out,
    const uint32_t* blockOffsets,
    int n
);

__global__ void computeTotalKernel(
    const uint32_t* triCount,
    const uint32_t* triOffset,
    uint32_t* totalOut,
    int n
);
