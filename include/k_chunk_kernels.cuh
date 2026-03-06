#pragma once
#include <cstdint>
#include <cuda_runtime.h>

//Mask build / list construction 
__global__ void buildChunkMaskKernel(
    const float* sf,
    int Nx, int Ny, int Nz,
    float isoLevel,
    int cellsX, int cellsY, int cellsZ,
    int chunksX, int chunksY,
    uint32_t* chunkActive
);

__global__ void mergeChunkMaskKernel(
    const uint32_t* chunkActive,
    uint32_t* chunkNonEmpty,
    int nChunks
);

__global__ void scatterActiveChunksKernel(
    const uint32_t* __restrict__ active,
    const uint32_t* __restrict__ offset,
    uint32_t* __restrict__ outIds,
    int nChunks
);

//Active chunks: count + emit (non-tiled) 
template<int CHUNK>
__global__ void countTrisActiveChunksKernel(
    const float* scalarField,
    int Nx, int Ny, int Nz,
    float isoLevel,
    int chunksX, int chunksY,
    const uint32_t* activeChunkIds,
    uint32_t numActiveChunks,
    uint8_t* cubeIndexOut,
    uint32_t* triCountOut
);

template<int CHUNK>
__global__ void emitActiveChunksKernel(
    const float* scalarField,
    int Nx, int Ny, int Nz,
    float minX, float minY, float minZ,
    float dx, float dy, float dz,
    float isoLevel,
    int chunksX, int chunksY,
    const uint32_t* activeChunkIds,
    uint32_t numActiveChunks,
    const uint8_t* cubeIndex,
    const uint32_t* triCount,
    const uint32_t* triOffset,
    float4* vertices,
    uint32_t* indices
);

//Active chunks: clear + tiled count + tiled emit 
template<int CHUNK>
__global__ void clearTriCountActiveChunksKernel(
    uint32_t* triCount,
    int cellsX, int cellsY, int cellsZ,
    int chunksX, int chunksY,
    const uint32_t* activeChunkIds,
    uint32_t numActiveChunks
);

template<int CHUNK>
__global__ void countTrisActiveChunksKernel_tiled(
    const float* scalarField,
    int Nx, int Ny, int Nz,
    float isoLevel,
    int chunksX, int chunksY,
    const uint32_t* activeChunkIds,
    uint32_t numActiveChunks,
    uint8_t* cubeIndexOut,
    uint32_t* triCountOut
);

template<int CHUNK>
__global__ void emitActiveChunksKernel_tiled(
    const float* scalarField,
    int Nx, int Ny, int Nz,
    float minX, float minY, float minZ,
    float dx, float dy, float dz,
    float isoLevel,
    int chunksX, int chunksY,
    const uint32_t* activeChunkIds,
    uint32_t numActiveChunks,
    const uint8_t* cubeIndex,
    const uint32_t* triCount,
    const uint32_t* triOffset,
    float4* vertices,
    uint32_t* indices
);
