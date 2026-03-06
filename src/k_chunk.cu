#include <cstdint>
#include <cuda_runtime.h>
#include "mc_tables.cuh"
#include "mc_device_helpers.cuh"
#include "k_chunk_kernels.cuh"


//tiles kernel helper
template<int CHUNK>
__device__ __forceinline__ int tileIndex(int lx, int ly, int lz) {
    // tile coords are in [0..CHUNK] in each dimension
    int S = CHUNK + 1;
    return lx + ly * S + lz * S * S;
}


//build mask kernel
constexpr int CHUNK = 8;
__global__ void buildChunkMaskKernel(const float* sf,
                                     int Nx, int Ny, int Nz,
                                     float isoLevel,
                                     int cellsX, int cellsY, int cellsZ,
                                     int chunksX, int chunksY,
                                     uint32_t* chunkActive)
{
    int chunkId = blockIdx.x;
    int chunksXY = chunksX * chunksY;

    int cz = chunkId / chunksXY;
    int rem = chunkId - cz * chunksXY;
    int cy = rem / chunksX;
    int cx = rem - cy * chunksX;

    int x0 = cx * CHUNK;
    int y0 = cy * CHUNK;
    int z0 = cz * CHUNK;

    __shared__ int any;
    if (threadIdx.x == 0) any = 0;
    __syncthreads();

    for (int idx = threadIdx.x; idx < CHUNK*CHUNK*CHUNK; idx += blockDim.x) {
        int lz = idx / (CHUNK*CHUNK);
        int rem2 = idx - lz * (CHUNK*CHUNK);
        int ly = rem2 / CHUNK;
        int lx = rem2 - ly * CHUNK;

        int x = x0 + lx;
        int y = y0 + ly;
        int z = z0 + lz;

        if (x >= cellsX || y >= cellsY || z >= cellsZ) continue;

        uint8_t cubeIndex = cubeIndexAtCell(sf, Nx, Ny, x, y, z, isoLevel);
        if (cubeIndex != 0 && cubeIndex != 255) {
            atomicExch(&any, 1);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) chunkActive[chunkId] = (uint32_t)any;
}

//presistant chunk non-empty mask update
__global__ void mergeChunkMaskKernel(const uint32_t* chunkActive,
                                     uint32_t* chunkNonEmpty,
                                     int nChunks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nChunks) return;

    //mark the persistent mask
    //chunkNonEmpty here; it persists across extractions
    if (chunkActive[i]) chunkNonEmpty[i] = 1u;
}



//scatter active chunks ID
__global__ void scatterActiveChunksKernel(const uint32_t* __restrict__ active,
                                          const uint32_t* __restrict__ offset,
                                          uint32_t* __restrict__ outIds,
                                          int nChunks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nChunks) return;
    if (active[i]) outIds[offset[i]] = (uint32_t)i;
}



//chunk kernel
template<int CHUNK>
__global__ void countTrisActiveChunksKernel(
    const float* scalarField,
    int Nx, int Ny, int Nz,
    float isoLevel,
    int chunksX, int chunksY,
    const uint32_t* activeChunkIds,
    uint32_t numActiveChunks,
    uint8_t* cubeIndexOut,
    uint32_t* triCountOut)

{
    int cellsX = Nx - 1, cellsY = Ny - 1, cellsZ = Nz - 1;

    uint32_t a = (uint32_t)blockIdx.x;
    if (a >= numActiveChunks) return;

    uint32_t chunkId = activeChunkIds[a];

    int chunksXY = chunksX * chunksY;
    int cz = chunkId / chunksXY;
    int rem = chunkId - cz * chunksXY;
    int cy = rem / chunksX;
    int cx = rem - cy * chunksX;

    int x0 = cx * CHUNK;
    int y0 = cy * CHUNK;
    int z0 = cz * CHUNK;

    auto idx = [=] __device__ (int xi, int yi, int zi) {
        return xi + yi * Nx + zi * Nx * Ny;
    };

    for (int local = threadIdx.x; local < CHUNK*CHUNK*CHUNK; local += blockDim.x) {
        int lz = local / (CHUNK * CHUNK);
        int rem2 = local - lz * (CHUNK * CHUNK);
        int ly = rem2 / CHUNK;
        int lx = rem2 - ly * CHUNK;

        int x = x0 + lx;
        int y = y0 + ly;
        int z = z0 + lz;

        if (x >= cellsX || y >= cellsY || z >= cellsZ) continue;

        int tid = x + y * cellsX + z * (cellsX * cellsY);

        float val[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int cx2 = x + d_cornerOffsets[i][0];
            int cy2 = y + d_cornerOffsets[i][1];
            int cz2 = z + d_cornerOffsets[i][2];
            val[i] = scalarField[idx(cx2, cy2, cz2)];
        }

        int cubeIndex = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (val[i] < isoLevel) cubeIndex |= (1 << i);
        }

        cubeIndexOut[tid] = (uint8_t)cubeIndex;

        const int* row = d_triTable[cubeIndex];
        uint32_t t = 0;
        #pragma unroll
        for (int tri = 0; tri < 5; ++tri) {
            int e0 = row[3 * tri + 0];
            if (e0 < 0) break;
            t++;
        }
        triCountOut[tid] = t;
    }
}

//chunk emit kernel
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
    uint32_t* indices)
{
    int cellsX = Nx - 1, cellsY = Ny - 1, cellsZ = Nz - 1;

    uint32_t a = (uint32_t)blockIdx.x;
    if (a >= numActiveChunks) return;

    uint32_t chunkId = activeChunkIds[a];

    int chunksXY = chunksX * chunksY;
    int cz = chunkId / chunksXY;
    int rem = chunkId - cz * chunksXY;
    int cy = rem / chunksX;
    int cx = rem - cy * chunksX;

    int x0 = cx * CHUNK;
    int y0 = cy * CHUNK;
    int z0 = cz * CHUNK;

    auto idx = [=] __device__ (int xi, int yi, int zi) {
        return xi + yi * Nx + zi * Nx * Ny;
    };

    for (int local = threadIdx.x; local < CHUNK*CHUNK*CHUNK; local += blockDim.x) {
        int lz = local / (CHUNK * CHUNK);
        int rem2 = local - lz * (CHUNK * CHUNK);
        int ly = rem2 / CHUNK;
        int lx = rem2 - ly * CHUNK;

        int x = x0 + lx;
        int y = y0 + ly;
        int z = z0 + lz;

        if (x >= cellsX || y >= cellsY || z >= cellsZ) continue;

        int tid = x + y * cellsX + z * (cellsX * cellsY);

        uint32_t tcount = triCount[tid];
        if (tcount == 0) continue;

        float3 pos[8];
        float  val[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int cx2 = x + d_cornerOffsets[i][0];
            int cy2 = y + d_cornerOffsets[i][1];
            int cz2 = z + d_cornerOffsets[i][2];
            pos[i] = make_float3(dx * (float)cx2 + minX,
                                 dy * (float)cy2 + minY,
                                 dz * (float)cz2 + minZ);
            val[i] = scalarField[idx(cx2, cy2, cz2)];
        }

        int ci = (int)cubeIndex[tid];
        int edges = d_edgeTable[ci];
        if (edges == 0) continue;

        float3 vertList[12];
        #pragma unroll
        for (int e = 0; e < 12; ++e) {
            if (edges & (1 << e)) {
                int a2 = d_edgeIndexMap[e][0];
                int b2 = d_edgeIndexMap[e][1];
                vertList[e] = vertexInterp(isoLevel, pos[a2], pos[b2], val[a2], val[b2]);
            }
        }

        const int* row = d_triTable[ci];

        uint32_t baseTri = triOffset[tid];
        uint32_t baseV   = baseTri * 3;

        #pragma unroll
        for (int tri = 0; tri < 5; ++tri) {
            int e0 = row[3 * tri + 0];
            int e1 = row[3 * tri + 2];
            int e2 = row[3 * tri + 1];
            if (e0 < 0) break;

            uint32_t v = baseV + tri * 3;

            vertices[v + 0] = make_float4(vertList[e0].x, vertList[e0].y, vertList[e0].z, 0.f);
            vertices[v + 1] = make_float4(vertList[e1].x, vertList[e1].y, vertList[e1].z, 0.f);
            vertices[v + 2] = make_float4(vertList[e2].x, vertList[e2].y, vertList[e2].z, 0.f);

            indices[v + 0] = v + 0;
            indices[v + 1] = v + 1;
            indices[v + 2] = v + 2;
        }
    }
}


//write zeros only for active chunks
template<int CHUNK>
__global__ void clearTriCountActiveChunksKernel(
    uint32_t* triCount,
    int cellsX, int cellsY, int cellsZ,
    int chunksX, int chunksY,
    const uint32_t* activeChunkIds,
    uint32_t numActiveChunks)
{
    uint32_t chunkListIdx = (uint32_t)blockIdx.x;
    if (chunkListIdx >= numActiveChunks) return;

    uint32_t chunkId = activeChunkIds[chunkListIdx];

    int cx = (int)(chunkId % (uint32_t)chunksX);
    int cy = (int)((chunkId / (uint32_t)chunksX) % (uint32_t)chunksY);
    int cz = (int)(chunkId / (uint32_t)(chunksX * chunksY));

    int x0 = cx * CHUNK;
    int y0 = cy * CHUNK;
    int z0 = cz * CHUNK;

    int local = (int)threadIdx.x; // 0..255
    if (local >= CHUNK*CHUNK*CHUNK) return;

    int lx = local % CHUNK;
    int ly = (local / CHUNK) % CHUNK;
    int lz = local / (CHUNK * CHUNK);

    int x = x0 + lx;
    int y = y0 + ly;
    int z = z0 + lz;

    if (x >= cellsX || y >= cellsY || z >= cellsZ) return;

    int cellId = x + y * cellsX + z * cellsX * cellsY;
    triCount[cellId] = 0;
}

//load chunk scalar values in shared memory and process
template<int CHUNK>
__global__ void countTrisActiveChunksKernel_tiled(
    const float* scalarField,
    int Nx, int Ny, int Nz,
    float isoLevel,
    int chunksX, int chunksY,
    const uint32_t* activeChunkIds,
    uint32_t numActiveChunks,
    uint8_t* cubeIndexOut,
    uint32_t* triCountOut)
{
    int cellsX = Nx - 1, cellsY = Ny - 1, cellsZ = Nz - 1;

    uint32_t a = (uint32_t)blockIdx.x;
    if (a >= numActiveChunks) return;

    uint32_t chunkId = activeChunkIds[a];

    int chunksXY = chunksX * chunksY;
    int cz = (int)(chunkId / (uint32_t)chunksXY);
    int rem = (int)(chunkId - (uint32_t)cz * (uint32_t)chunksXY);
    int cy = rem / chunksX;
    int cx = rem - cy * chunksX;

    int x0 = cx * CHUNK;
    int y0 = cy * CHUNK;
    int z0 = cz * CHUNK;

    auto idx = [=] __device__ (int xi, int yi, int zi) {
        return xi + yi * Nx + zi * Nx * Ny;
    };

    // Load scalar tile (CHUNK+1)^3 into shared memory
    __shared__ float s_tile[(CHUNK + 1) * (CHUNK + 1) * (CHUNK + 1)];

    const int S = CHUNK + 1;
    const int tileCount = S * S * S;

    for (int t = (int)threadIdx.x; t < tileCount; t += (int)blockDim.x) {
        int lz = t / (S * S);
        int remt = t - lz * (S * S);
        int ly = remt / S;
        int lx = remt - ly * S;

        int gx = x0 + lx;
        int gy = y0 + ly;
        int gz = z0 + lz;

        float v = 0.0f;
        if (gx < Nx && gy < Ny && gz < Nz) {
            v = scalarField[idx(gx, gy, gz)];
        }
        s_tile[t] = v;
    }
    __syncthreads();

    // Process cells, fetch corners from shared memory
    for (int local = (int)threadIdx.x; local < CHUNK * CHUNK * CHUNK; local += (int)blockDim.x) {
        int lz = local / (CHUNK * CHUNK);
        int rem2 = local - lz * (CHUNK * CHUNK);
        int ly = rem2 / CHUNK;
        int lx = rem2 - ly * CHUNK;

        int x = x0 + lx;
        int y = y0 + ly;
        int z = z0 + lz;

        if (x >= cellsX || y >= cellsY || z >= cellsZ) continue;

        int tid = x + y * cellsX + z * (cellsX * cellsY);

        float val[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int ox = d_cornerOffsets[i][0];
            int oy = d_cornerOffsets[i][1];
            int oz = d_cornerOffsets[i][2];
            val[i] = s_tile[tileIndex<CHUNK>(lx + ox, ly + oy, lz + oz)];
        }

        int cubeIndex = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (val[i] < isoLevel) cubeIndex |= (1 << i);
        }

        cubeIndexOut[tid] = (uint8_t)cubeIndex;

        const int* row = d_triTable[cubeIndex];
        uint32_t tcount = 0;
        #pragma unroll
        for (int tri = 0; tri < 5; ++tri) {
            int e0 = row[3 * tri + 0];
            if (e0 < 0) break;
            tcount++;
        }
        triCountOut[tid] = tcount;
    }
}


//emit tiled kernel
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
    uint32_t* indices)
{
    int cellsX = Nx - 1, cellsY = Ny - 1, cellsZ = Nz - 1;

    uint32_t a = (uint32_t)blockIdx.x;
    if (a >= numActiveChunks) return;

    uint32_t chunkId = activeChunkIds[a];

    int chunksXY = chunksX * chunksY;
    int cz = (int)(chunkId / (uint32_t)chunksXY);
    int rem = (int)(chunkId - (uint32_t)cz * (uint32_t)chunksXY);
    int cy = rem / chunksX;
    int cx = rem - cy * chunksX;

    int x0 = cx * CHUNK;
    int y0 = cy * CHUNK;
    int z0 = cz * CHUNK;

    auto idx = [=] __device__ (int xi, int yi, int zi) {
        return xi + yi * Nx + zi * Nx * Ny;
    };

    // Load scalar tile (CHUNK+1)^3 into shared memory
    __shared__ float s_tile[(CHUNK + 1) * (CHUNK + 1) * (CHUNK + 1)];

    const int S = CHUNK + 1;
    const int tileCount = S * S * S;

    for (int t = (int)threadIdx.x; t < tileCount; t += (int)blockDim.x) {
        int lz = t / (S * S);
        int remt = t - lz * (S * S);
        int ly = remt / S;
        int lx = remt - ly * S;

        int gx = x0 + lx;
        int gy = y0 + ly;
        int gz = z0 + lz;

        float v = 0.0f;
        if (gx < Nx && gy < Ny && gz < Nz) {
            v = scalarField[idx(gx, gy, gz)];
        }
        s_tile[t] = v;
    }
    __syncthreads();

    for (int local = (int)threadIdx.x; local < CHUNK * CHUNK * CHUNK; local += (int)blockDim.x) {
        int lz = local / (CHUNK * CHUNK);
        int rem2 = local - lz * (CHUNK * CHUNK);
        int ly = rem2 / CHUNK;
        int lx = rem2 - ly * CHUNK;

        int x = x0 + lx;
        int y = y0 + ly;
        int z = z0 + lz;

        if (x >= cellsX || y >= cellsY || z >= cellsZ) continue;

        int tid = x + y * cellsX + z * (cellsX * cellsY);

        uint32_t tcount = triCount[tid];
        if (tcount == 0) continue;

        // Corners: positions are computed from global coords (cheap), values from shared tile
        float3 pos[8];
        float  val[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int ox = d_cornerOffsets[i][0];
            int oy = d_cornerOffsets[i][1];
            int oz = d_cornerOffsets[i][2];

            int cx2 = x + ox;
            int cy2 = y + oy;
            int cz2 = z + oz;

            pos[i] = make_float3(dx * (float)cx2 + minX,
                                 dy * (float)cy2 + minY,
                                 dz * (float)cz2 + minZ);

            val[i] = s_tile[tileIndex<CHUNK>(lx + ox, ly + oy, lz + oz)];
        }

        int ci = (int)cubeIndex[tid];
        int edges = d_edgeTable[ci];
        if (edges == 0) continue;

        float3 vertList[12];
        #pragma unroll
        for (int e = 0; e < 12; ++e) {
            if (edges & (1 << e)) {
                int a2 = d_edgeIndexMap[e][0];
                int b2 = d_edgeIndexMap[e][1];
                vertList[e] = vertexInterp(isoLevel, pos[a2], pos[b2], val[a2], val[b2]);
            }
        }

        const int* row = d_triTable[ci];

        uint32_t baseTri = triOffset[tid];
        uint32_t baseV   = baseTri * 3;

        // Preserve your winding/order (e1/e2 swapped)
        #pragma unroll
        for (int tri = 0; tri < 5; ++tri) {
            int e0 = row[3 * tri + 0];
            int e1 = row[3 * tri + 2];
            int e2 = row[3 * tri + 1];
            if (e0 < 0) break;

            uint32_t v = baseV + (uint32_t)tri * 3u;

            vertices[v + 0] = make_float4(vertList[e0].x, vertList[e0].y, vertList[e0].z, 0.f);
            vertices[v + 1] = make_float4(vertList[e1].x, vertList[e1].y, vertList[e1].z, 0.f);
            vertices[v + 2] = make_float4(vertList[e2].x, vertList[e2].y, vertList[e2].z, 0.f);

            indices[v + 0] = v + 0;
            indices[v + 1] = v + 1;
            indices[v + 2] = v + 2;
        }
    }
}


template __global__ void countTrisActiveChunksKernel<CHUNK>(
    const float*, int, int, int, float, int, int,
    const uint32_t*, uint32_t, uint8_t*, uint32_t*
);

template __global__ void emitActiveChunksKernel<CHUNK>(
    const float*, int, int, int, float, float, float,
    float, float, float, float,
    int, int,
    const uint32_t*, uint32_t,
    const uint8_t*, const uint32_t*, const uint32_t*,
    float4*, uint32_t*
);

template __global__ void clearTriCountActiveChunksKernel<CHUNK>(
    uint32_t*, int, int, int, int, int,
    const uint32_t*, uint32_t
);

template __global__ void countTrisActiveChunksKernel_tiled<CHUNK>(
    const float*, int, int, int, float, int, int,
    const uint32_t*, uint32_t, uint8_t*, uint32_t*
);

template __global__ void emitActiveChunksKernel_tiled<CHUNK>(
    const float*, int, int, int, float, float, float,
    float, float, float, float,
    int, int,
    const uint32_t*, uint32_t,
    const uint8_t*, const uint32_t*, const uint32_t*,
    float4*, uint32_t*
);
