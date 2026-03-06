#include <cstdint>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include "common_cuda.h"
#include "k_scan_kernels.cuh"

//block-exclusive scan
template<int BLOCK_THREADS>
__global__ void scanBlocksKernel(
    const uint32_t* in,
    uint32_t* out,        // exclusive offsets (block-local for now)
    uint32_t* blockSums,  // sum per block
    int n)
{
    using BlockScan = cub::BlockScan<uint32_t, BLOCK_THREADS>;
    __shared__ typename BlockScan::TempStorage temp;

    int gid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    uint32_t x = (gid < n) ? in[gid] : 0;

    uint32_t prefix;
    uint32_t blockTotal;
    BlockScan(temp).ExclusiveSum(x, prefix, blockTotal);

    if (gid < n) out[gid] = prefix;
    if (threadIdx.x == BLOCK_THREADS - 1) blockSums[blockIdx.x] = blockTotal;
}



//scanned block offsets
template<int BLOCK_THREADS>
__global__ void addBlockOffsetsKernel(
    uint32_t* out,
    const uint32_t* blockOffsets,
    int n)
{
    int gid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (gid < n) out[gid] += blockOffsets[blockIdx.x];
}



//cmompute total triangles
__global__ void computeTotalKernel(
    const uint32_t* triCount,
    const uint32_t* triOffset,
    uint32_t* totalOut,
    int n)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (n == 0) { *totalOut = 0; return; }
        *totalOut = triOffset[n - 1] + triCount[n - 1];
    }
}

template __global__ void scanBlocksKernel<256>(const uint32_t*, uint32_t*, uint32_t*, int);
template __global__ void addBlockOffsetsKernel<256>(uint32_t*, const uint32_t*, int);
