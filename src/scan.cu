#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "common_cuda.h"
#include "scan.cuh"


ScanWorkspace createScanWorkspace(int n) {
    ScanWorkspace ws;

    int size = n;
    while (true) {
        ws.sizes.push_back(size);
        int nb = (size + ScanWorkspace::BLOCK - 1) / ScanWorkspace::BLOCK;
        ws.numBlocks.push_back(nb);

        // allocate block sums for this level (length nb)
        uint32_t* d_sums = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sums, nb * sizeof(uint32_t)));
        ws.d_blockSums.push_back(d_sums);

        // prefix arrays are needed only for levels >= 1
        // (level 0 prefix is the final output array you pass in)
        if ((int)ws.sizes.size() >= 2) {
            uint32_t* d_pref = nullptr;
            CUDA_CHECK(cudaMalloc(&d_pref, size * sizeof(uint32_t)));
            ws.d_prefix.push_back(d_pref);
        }

        // next level input size is nb (block sums)
        if (nb <= 1) break;
        size = nb;
    }

    ws.levels = (int)ws.sizes.size();

    // ws.d_prefix currently holds prefix arrays for levels 1..L-1, but stored starting at index 0
    // We'll map them carefully in the scan function.

    return ws;
}

void destroyScanWorkspace(ScanWorkspace& ws) {
    for (auto p : ws.d_blockSums) {
        if (p) CUDA_CHECK(cudaFree(p));
    }
    for (auto p : ws.d_prefix) {
        if (p) CUDA_CHECK(cudaFree(p));
    }
    ws = {};
}


//launching kernels
void exclusiveScanUint32_ws(uint32_t* d_in, uint32_t* d_out, int n,
                                   const ScanWorkspace& ws)
{
    // Handle degenerate cases (avoid <<<0, ...>>> launches)
    if (n <= 0) return;
    if (n == 1) {
        // exclusive scan of one element is always 0
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(uint32_t)));
        return;
    }

    constexpr int BLOCK = ScanWorkspace::BLOCK;

    // ---- Up-sweep: scan each level and produce block sums ----
    // Level 0: input = d_in, prefix = d_out, sums = ws.d_blockSums[0]
    scanBlocksKernel<BLOCK><<<ws.numBlocks[0], BLOCK>>>(d_in, d_out, ws.d_blockSums[0], n);
    CUDA_CHECK(cudaGetLastError());

    // Levels 1..L-1: input = blockSums[l-1], prefix = ws.d_prefix[l-1], sums = blockSums[l]
    for (int l = 1; l < ws.levels; ++l) {
        uint32_t* levelIn  = ws.d_blockSums[l - 1];
        uint32_t* levelOut = ws.d_prefix[l - 1];        // prefix buffer for this level
        uint32_t* levelSums= ws.d_blockSums[l];

        int size = ws.sizes[l];
        scanBlocksKernel<BLOCK><<<ws.numBlocks[l], BLOCK>>>(levelIn, levelOut, levelSums, size);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Down-sweep: add scanned block offsets back down ----
    // We need to add level (l) prefix offsets into level (l-1) prefix outputs.
    // For l = levels-1 down to 1:
    //   add offsets from ws.d_prefix[l-1] into ws.d_prefix[l-2] (or d_out for l==1)
    for (int l = ws.levels - 1; l >= 1; --l) {
        uint32_t* blockOffsets = ws.d_prefix[l - 1]; // scanned offsets for blocks of level (l-1)
        int prevSize = ws.sizes[l - 1];
        int prevBlocks = ws.numBlocks[l - 1];

        if (l == 1) {
            // add to final output array (d_out)
            addBlockOffsetsKernel<BLOCK><<<prevBlocks, BLOCK>>>(d_out, blockOffsets, prevSize);
        } else {
            // add to previous level's prefix array
            uint32_t* prevPrefix = ws.d_prefix[l - 2];
            addBlockOffsetsKernel<BLOCK><<<prevBlocks, BLOCK>>>(prevPrefix, blockOffsets, prevSize);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}
