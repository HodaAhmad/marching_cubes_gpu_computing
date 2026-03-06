#pragma once
#include <cstdint>
#include <cuda_runtime.h>

__global__ void countTrisKernel(
    const float* scalarField,
    int Nx, int Ny, int Nz,
    float isoLevel,
    uint8_t* cubeIndexOut,
    uint32_t* triCountOut
);

__global__ void emitKernel(
    const float* scalarField,
    int Nx, int Ny, int Nz,
    float minX, float minY, float minZ,
    float dx, float dy, float dz,
    float isoLevel,
    const uint8_t* cubeIndex,
    const uint32_t* triCount,
    const uint32_t* triOffset,
    float4* vertices,
    uint32_t* indices
);

__global__ void marchingCubesKernel(
    const float* scalarField,
    unsigned int Nx, unsigned int Ny, unsigned int Nz,
    float minX, float minY, float minZ,
    float dx, float dy, float dz,
    float4* vertices,
    unsigned int* vertexCounter,
    unsigned int* indices,
    unsigned int* indexCounter,
    unsigned int maxTriangles
);
