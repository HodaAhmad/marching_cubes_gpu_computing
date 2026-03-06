#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "mc_tables.cuh"

// Linearly interpolate between points p1 and p2 given their scalar values valp1 and
// valp2 relative to the isosurface threshold isoLevel.
__device__ __forceinline__ float3 vertexInterp(float isoLevel,
                                               const float3 &p1, const float3 &p2,
                                               float valp1, float valp2)
{
    float t = (isoLevel - valp1) / (valp2 - valp1);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return make_float3(p1.x + t * (p2.x - p1.x),
                       p1.y + t * (p2.y - p1.y),
                       p1.z + t * (p2.z - p1.z));
}


//chunks kernel helper
__device__ __forceinline__ uint8_t cubeIndexAtCell(
    const float* scalarField,
    int Nx, int Ny,
    int x, int y, int z,
    float isoLevel)
{
    auto idx = [=] __device__ (int xi, int yi, int zi) {
        return xi + yi * Nx + zi * Nx * Ny;
    };

    float val[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int cx = x + d_cornerOffsets[i][0];
        int cy = y + d_cornerOffsets[i][1];
        int cz = z + d_cornerOffsets[i][2];
        val[i] = scalarField[idx(cx, cy, cz)];
    }

    int cubeIndex = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (val[i] < isoLevel) cubeIndex |= (1 << i);
    }

    return (uint8_t)cubeIndex;
}