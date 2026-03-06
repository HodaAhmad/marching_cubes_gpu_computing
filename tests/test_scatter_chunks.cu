#include <cassert>
#include <cstdint>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#include "common_cuda.h"
#include "scan.cuh"
#include "k_chunk_kernels.cuh"  

static std::vector<uint32_t> cpu_exclusive_scan_u32(const std::vector<uint32_t>& in) {
    std::vector<uint32_t> out(in.size(), 0);
    uint32_t sum = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = sum;
        sum += in[i];
    }
    return out;
}

static void run_one(int nChunks, uint32_t seed) {
    // Make a deterministic active mask
    std::vector<uint32_t> h_active(nChunks, 0);
    for (int i = 0; i < nChunks; ++i) {
        // ~25% active, deterministic
        uint32_t x = seed * 1664525u + 1013904223u + (uint32_t)i * 2654435761u;
        h_active[i] = ((x >> 28) & 3u) == 0u ? 1u : 0u;
    }

    // CPU offsets + expected IDs
    auto h_offset = cpu_exclusive_scan_u32(h_active);
    uint32_t totalActive = (nChunks > 0) ? (h_offset.back() + h_active.back()) : 0;

    std::vector<uint32_t> h_expected(totalActive, 0xDEADBEEF);
    for (int i = 0; i < nChunks; ++i) {
        if (h_active[i]) {
            h_expected[h_offset[i]] = (uint32_t)i;
        }
    }

    // Device buffers
    uint32_t *d_active = nullptr, *d_offset = nullptr, *d_outIds = nullptr;

    CUDA_CHECK(cudaMalloc(&d_active, (size_t)nChunks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_offset, (size_t)nChunks * sizeof(uint32_t)));

    if (totalActive > 0) {
        CUDA_CHECK(cudaMalloc(&d_outIds, (size_t)totalActive * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_outIds, 0xCD, (size_t)totalActive * sizeof(uint32_t)));
    }

    CUDA_CHECK(cudaMemcpy(d_active, h_active.data(),
                          (size_t)nChunks * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset, h_offset.data(),
                          (size_t)nChunks * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Launch
    int threads = 256;
    int blocks  = (nChunks + threads - 1) / threads;

    // Only launch if there is any work AND output buffer exists
    if (nChunks > 0 && blocks > 0 && totalActive > 0) {
        scatterActiveChunksKernel<<<blocks, threads>>>(d_active, d_offset, d_outIds, nChunks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copy back
    std::vector<uint32_t> h_out(totalActive, 0);
    if (totalActive > 0) {
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_outIds,
                              (size_t)totalActive * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
    }

    assert(h_out == h_expected && "scatterActiveChunksKernel output IDs mismatch");

    if (d_outIds) CUDA_CHECK(cudaFree(d_outIds));
    CUDA_CHECK(cudaFree(d_active));
    CUDA_CHECK(cudaFree(d_offset));
}


int main() {
    // test various sizes, including edge cases
    const int sizes[] = {0, 1, 2, 7, 31, 256, 257, 1024, 4096};

    for (uint32_t seed = 1; seed <= 20; ++seed) {
        for (int n : sizes) {
            run_one(n, seed);
        }
    }

    std::cout << "[OK] test_scatter_chunks passed\n";
    return 0;
}
