#include <cassert>
#include <cstdint>
#include <vector>
#include <numeric>
#include <iostream>
#include <cuda_runtime.h>

#include "common_cuda.h"
#include "scan.cuh"

// CPU reference exclusive scan
static std::vector<uint32_t> cpu_exclusive_scan(const std::vector<uint32_t>& in) {
    std::vector<uint32_t> out(in.size(), 0);
    uint32_t sum = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = sum;
        sum += in[i];
    }
    return out;
}

static void run_one(int n, uint32_t seed) {
    // deterministic input
    std::vector<uint32_t> h_in(n);
    for (int i = 0; i < n; ++i) {
        h_in[i] = (seed * 1664525u + 1013904223u + (uint32_t)i * 2654435761u) & 1023u;
    }
    auto h_ref = cpu_exclusive_scan(h_in);

    uint32_t *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    auto ws = createScanWorkspace(n);
    exclusiveScanUint32_ws(d_in, d_out, n, ws);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint32_t> h_out(n);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // verify
    assert(h_out == h_ref && "GPU scan does not match CPU reference");

    destroyScanWorkspace(ws);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    // a mix of sizes, including non-multiples of 256
    const int sizes[] = {0, 1, 2, 3, 7, 31, 255, 256, 257, 511, 512, 777, 1024, 2049};

    for (uint32_t s = 1; s <= 20; ++s) {
        for (int n : sizes) {
            run_one(n, s);
        }
    }

    std::cout << "[OK] test_scan passed\n";
    return 0;
}
