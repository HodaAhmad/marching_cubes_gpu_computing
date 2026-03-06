#include <cstdint>
#include <cuda_runtime.h>
#include "mc_tables.cuh"
#include "k_mc_kernels.cuh"
#include "mc_device_helpers.cuh"

__device__ unsigned int insertVertexIndex(const float3 &vertex,
                                          float4 *vertices,
                                          unsigned int *vertexCounter)
{
    // Atomically get the next available vertex index.
    unsigned int index = atomicAdd(vertexCounter, 1);
    // Store the vertex position as a float4 (w component unused).
    vertices[index] = make_float4(vertex.x, vertex.y, vertex.z, 0.0f);
    return index;
}



//count per-cell triangles kernel
__global__ void countTrisKernel(
    const float* scalarField,
    int Nx, int Ny, int Nz,
    float isoLevel,
    uint8_t* cubeIndexOut,
    uint32_t* triCountOut)
{
    int cellsX = Nx - 1, cellsY = Ny - 1, cellsZ = Nz - 1;
    int nCells = cellsX * cellsY * cellsZ;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nCells) return;

    int z = tid / (cellsX * cellsY);
    int rem = tid - z * (cellsX * cellsY);
    int y = rem / cellsX;
    int x = rem - y * cellsX;

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

    cubeIndexOut[tid] = (uint8_t)cubeIndex;

    // Count triangles by scanning triTable row (max 5)
    const int* row = d_triTable[cubeIndex];
    uint32_t t = 0;
    #pragma unroll
    for (int tri = 0; tri < 5; ++tri) {
        int e0 = row[3 * tri + 0];
        if (e0 < 0) break;
        t++;
    }
    triCountOut[tid] = t; // 0..5
}


//emit triangles to global memory
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
    uint32_t* indices)
{
    int cellsX = Nx - 1, cellsY = Ny - 1, cellsZ = Nz - 1;
    int nCells = cellsX * cellsY * cellsZ;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nCells) return;

    uint32_t tcount = triCount[tid];
    if (tcount == 0) return;

    int z = tid / (cellsX * cellsY);
    int rem = tid - z * (cellsX * cellsY);
    int y = rem / cellsX;
    int x = rem - y * cellsX;

    auto idx = [=] __device__ (int xi, int yi, int zi) {
        return xi + yi * Nx + zi * Nx * Ny;
    };

    float3 pos[8];
    float  val[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int cx = x + d_cornerOffsets[i][0];
        int cy = y + d_cornerOffsets[i][1];
        int cz = z + d_cornerOffsets[i][2];
        pos[i] = make_float3(dx * (float)cx + minX,
                             dy * (float)cy + minY,
                             dz * (float)cz + minZ);
        val[i] = scalarField[idx(cx, cy, cz)];
    }

    int ci = (int)cubeIndex[tid];
    int edges = d_edgeTable[ci];
    if (edges == 0) return;

    float3 vertList[12];
    #pragma unroll
    for (int e = 0; e < 12; ++e) {
        if (edges & (1 << e)) {
            int a = d_edgeIndexMap[e][0];
            int b = d_edgeIndexMap[e][1];
            vertList[e] = vertexInterp(isoLevel, pos[a], pos[b], val[a], val[b]);
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

        // indices become trivial (0..N-1), still writes your file format
        indices[v + 0] = v + 0;
        indices[v + 1] = v + 1;
        indices[v + 2] = v + 2;
    }
}

// Base naive implementation of the Marching Cubes algorithm
__global__ void marchingCubesKernel(const float *scalarField,
                                     unsigned int Nx, unsigned int Ny, unsigned int Nz,
                                     float minX, float minY, float minZ,
                                     float dx, float dy, float dz,
                                     float4 *vertices,
                                     unsigned int *vertexCounter,
                                     unsigned int *indices,
                                     unsigned int *indexCounter,
                                     unsigned int maxTriangles)
{
    // Compute 3D thread indices.
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Only process interior cubes; ignore cells on the far boundary.
    if (x >= Nx - 1 || y >= Ny - 1 || z >= Nz - 1)
        return;

    // Local storage for the eight corner positions and scalar values.
    float3 pos[8];
    float val[8];

    // Helper lambda to compute 1D index into scalar field array.
    auto idx = [=] __device__ (unsigned int xi, unsigned int yi, unsigned int zi) -> unsigned int
    {
        return xi + yi * Nx + zi * Nx * Ny;
    };

    // Compute position and scalar value at each of the eight cube corners.
    for (int i = 0; i < 8; ++i)
    {
        unsigned int cx = x + d_cornerOffsets[i][0];
        unsigned int cy = y + d_cornerOffsets[i][1];
        unsigned int cz = z + d_cornerOffsets[i][2];
        pos[i] = make_float3(dx * (float)cx + minX,
                             dx * (float)cy + minY,
                             dx * (float)cz + minZ);
        val[i] = scalarField[idx(cx, cy, cz)];


        // // Debug: uncomment to use a synthetic scalar field instead of input data.
        // float xf = dx * (float)cx + minX;
        // float yf = dx * (float)cy + minY;
        // float zf = dx * (float)cz + minZ;
        // val[i] = yf + sin(xf + zf) * 0.5f; // Example scalar field function
    }

    // Determine the cube configuration bitfield by comparing each corner value
    // against the isosurface value (0.0f).  A bit i is set if the scalar at
    // corner i is below the isosurface.
    int cubeIndex = 0;
    for (int i = 0; i < 8; ++i)
    {
        if (val[i] < 0.0f) cubeIndex |= (1 << i);
    }

    int edges = d_edgeTable[cubeIndex];
    if (edges == 0) return; // no surface crosses this cube

    // For each possible edge store the interpolated vertex and its key.
    float3 vertList[12];

    // For each edge compute interpolation and unique key when needed.
    for (int i = 0; i < 12; ++i)
    {
        if ((edges & (1 << i)) != 0)
        {
            int a = d_edgeIndexMap[i][0];
            int b = d_edgeIndexMap[i][1];
            
            vertList[i] = vertexInterp(0.0f, pos[a], pos[b], val[a], val[b]);
        }
    }

    // Emit triangles based on the triangulation table.
    const int *tableRow = d_triTable[cubeIndex];
    for (int tri = 0; tri < 5; ++tri)
    {
        int i0 = tableRow[3 * tri + 0];
        int i2 = tableRow[3 * tri + 1];
        int i1 = tableRow[3 * tri + 2];
        if (i0 < 0 || i1 < 0 || i2 < 0) break;

        // Insert vertices.
        unsigned int v0 = insertVertexIndex(vertList[i0], vertices, vertexCounter);
        unsigned int v1 = insertVertexIndex(vertList[i1], vertices, vertexCounter);
        unsigned int v2 = insertVertexIndex(vertList[i2], vertices, vertexCounter);

        // Insert triangle indices.
        unsigned int index = atomicAdd(indexCounter, 3u);
        if (index < (maxTriangles) * 3u)
        {
            indices[index + 0] = v0;
            indices[index + 1] = v1;
            indices[index + 2] = v2;
        }
    }
}

