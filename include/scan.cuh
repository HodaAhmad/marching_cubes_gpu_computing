#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>


//scan workspace for memory allocations
struct ScanWorkspace {
    static constexpr int BLOCK = 256;

    // Level l scans an input of length sizes[l]
    std::vector<int> sizes;            // sizes[l]
    std::vector<int> numBlocks;        // ceil(sizes[l]/BLOCK)

    // For each level l:
    // scanBlocksKernel produces:
    //   prefix[l] (exclusive prefix for that level's input)
    //   blockSums[l] (length numBlocks[l])
    std::vector<uint32_t*> d_prefix;   // d_prefix[0] will be the user output pointer (not owned)
    std::vector<uint32_t*> d_blockSums;

    int levels = 0;
};


// allocate/free workspace
ScanWorkspace createScanWorkspace(int n);
void destroyScanWorkspace(ScanWorkspace& ws);

// run scan using the workspace
void exclusiveScanUint32_ws(uint32_t* d_in, uint32_t* d_out, int n,
                            const ScanWorkspace& ws);


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