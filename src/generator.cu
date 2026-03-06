#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <string>
#include <cub/block/block_scan.cuh>
#include "mc_tables.cuh"
#include "common_cuda.h"
#include "k_scan_kernels.cuh"
#include "mc_device_helpers.cuh"
#include "k_mc_kernels.cuh"
#include "k_chunk_kernels.cuh"
#include "scan.cuh"
#include "cli.h"

__global__ void scanBlocksKernel(const uint32_t* d_in, uint32_t* d_out, uint32_t* d_blockSums, int n);
__global__ void addBlockOffsetsKernel(uint32_t* d_data, const uint32_t* d_offsets, int n);
__global__ void computeTotalKernel(const uint32_t* d_counts, const uint32_t* d_offsets, uint32_t* d_total, int n);



void writeTrianglesToFile(const char *filename, const std::vector<float4> &vertices,
                         const std::vector<unsigned int> &indices)
{
    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        return;
    }

    unsigned int vertexCount = vertices.size();
    unsigned int indexCount = indices.size();

    // Write vertex count and index count
    fwrite(&vertexCount, sizeof(unsigned int), 1, file);
    fwrite(&indexCount, sizeof(unsigned int), 1, file);

    // Write vertices
    fwrite(vertices.data(), sizeof(float4), vertexCount, file);

    // Write indices
    fwrite(indices.data(), sizeof(unsigned int), indexCount, file);

    fclose(file);
}


enum Axis { X = 0, Y = 1, Z = 2 };

// -------- 3D Grid Class with Dynamic Resizing --------
template<typename T>
class Grid3D {
public:
    Grid3D(int nx, int ny, int nz) : nx(nx), ny(ny), nz(nz), data(nx * ny * nz, T()) {}

    T& operator()(int x, int y, int z) {
        assert(x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz);
        return data[z * nx * ny + y * nx + x];
    }

    const T& operator()(int x, int y, int z) const {
        assert(x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz);
        return data[z * nx * ny + y * nx + x];
    }

    void resize(int newX, int newY, int newZ) {
        nx = newX;
        ny = newY;
        nz = newZ;
        data.resize(nx * ny * nz);
    }

    // Cuts the n values of a side: the side on dim "dim" and direction "dir" (dire = 0 for min, 1 for max)
    void cut(int dim, int dir, int n) {
        assert(dim >= 0 && dim < 3 && n >= 0);

        if (n == 0) return;

        Grid3D<T> result(
            dim == 0 ? nx - n : nx,
            dim == 1 ? ny - n : ny,
            dim == 2 ? nz - n : nz
        );

        int xOffset = (dim == 0 && dir == 0) ? n : 0;
        int yOffset = (dim == 1 && dir == 0) ? n : 0;
        int zOffset = (dim == 2 && dir == 0) ? n : 0;

        for (int z = 0; z < result.sizeZ(); ++z)
            for (int y = 0; y < result.sizeY(); ++y)
                for (int x = 0; x < result.sizeX(); ++x)
                    result(x, y, z) = (*this)(x + xOffset, y + yOffset, z + zOffset);

        *this = std::move(result);
    }


    int sizeX() const { return nx; }
    int sizeY() const { return ny; }
    int sizeZ() const { return nz; }

    const std::vector<T>& getData() const { return data; }
    std::vector<T>& getData() { return data; }


private:
    int nx, ny, nz;
    std::vector<T> data;
};


// Helper function to generate a random normal variable
static float random_normal(int seed)
{
    srand(seed);
    float u1 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float u2 = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * static_cast<float>(M_PI) * u2);
}


// Add values and noise to a 1D array
std::vector<float> expand_and_noise_1D(const std::vector<float>& input, int step, int seed) {
    int N = static_cast<int>(input.size());
    // if (N <= 3) return input;  // Pas assez de points pour interpoler

    assert(N >= 4 && "Input vector must have at least 3 elements for the expansion.");

    std::vector<float> even(N), odd(N);
    for (int i = 1; i < N - 1; ++i) {
        even[i] = input[i];
        odd[i] = -0.25f * (input[i - 1] - input[i + 1]) + 0.02f * random_normal(i + step*N + seed) / static_cast<float>(1+step);
    }

    for (int i = 1; i < N - 1; ++i) {
        even[i] -= 0.5f * odd[i];
        odd[i] += even[i];
    }

    std::vector<float> result((N - 2) * 2);
    for (int i = 1; i < N - 1; ++i) {
        result[(i - 1) * 2 + 0] = even[i];
        result[(i - 1) * 2 + 1] = odd[i];
    }

    return result;
}

// -------- Slice Access --------
void extract_line(const Grid3D<float>& grid, std::vector<float>& line,
                  Axis axis, int i, int j) {
    int N = (axis == X) ? grid.sizeX() :
            (axis == Y) ? grid.sizeY() : grid.sizeZ();
    line.resize(N);
    for (int k = 0; k < N; ++k) {
        int x = (axis == X) ? k : (axis == Y) ? i : i;
        int y = (axis == Y) ? k : (axis == X) ? i : j;
        int z = (axis == Z) ? k : (axis == Y) ? j : j;
        line[k] = grid(x, y, z);
    }
}

void write_line(Grid3D<float>& grid, const std::vector<float>& line,
                Axis axis, int i, int j) {
    int N = static_cast<int>(line.size());
    for (int k = 0; k < N; ++k) {
        int x = (axis == X) ? k : (axis == Y) ? i : i;
        int y = (axis == Y) ? k : (axis == X) ? i : j;
        int z = (axis == Z) ? k : (axis == Y) ? j : j;
        grid(x, y, z) = line[k];
    }
}

// Expand and add noise to a 3D grid
void expand_and_noise_3D(Grid3D<float>& grid, int step) {
    for (Axis axis : {X, Y, Z}) {
        int oldLength = (axis == X) ? grid.sizeX() :
                        (axis == Y) ? grid.sizeY() : grid.sizeZ();
        if (oldLength < 3)
            continue;

        int newLength = (oldLength - 2) * 2;

        int newX = (axis == X) ? newLength : grid.sizeX();
        int newY = (axis == Y) ? newLength : grid.sizeY();
        int newZ = (axis == Z) ? newLength : grid.sizeZ();
        Grid3D<float> newGrid(newX, newY, newZ);

        std::vector<float> line, expanded;

        int I = (axis == X) ? grid.sizeY() :
                (axis == Y) ? grid.sizeX() : grid.sizeX();
        int J = (axis == X) ? grid.sizeZ() :
                (axis == Y) ? grid.sizeZ() : grid.sizeY();

        int stride = ((axis == X)? grid.sizeY() * grid.sizeZ() :
                      (axis == Y) ? grid.sizeX() * grid.sizeZ() :
                                    grid.sizeX() * grid.sizeY());
        int seed = step * stride * stride;

        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                extract_line(grid, line, axis, i, j);
                expanded = expand_and_noise_1D(line, step, seed + i + j * stride);
                write_line(newGrid, expanded, axis, i, j);
            }
        }

        grid = std::move(newGrid);
    }
}

// OPT3b: GPU scalar field fill

__device__ __forceinline__ uint32_t hash_u32(uint32_t x) {
    // simple mix hash
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float u32_to_unit_float(uint32_t x) {
    // [0,1)
    return (x & 0x00FFFFFF) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ __forceinline__ float fade(float t) {
    // smoothstep-ish
    return t * t * (3.f - 2.f * t);
}

__device__ float valueNoise3D(float x, float y, float z, uint32_t seed) {
    // Value noise at integer lattice with trilinear interpolation
    int xi = (int)floorf(x);
    int yi = (int)floorf(y);
    int zi = (int)floorf(z);

    float tx = x - xi;
    float ty = y - yi;
    float tz = z - zi;

    float fx = fade(tx);
    float fy = fade(ty);
    float fz = fade(tz);

    auto lattice = [&](int ix, int iy, int iz) -> float {
        uint32_t h = seed;
        h ^= hash_u32((uint32_t)ix * 73856093u);
        h ^= hash_u32((uint32_t)iy * 19349663u);
        h ^= hash_u32((uint32_t)iz * 83492791u);
        h = hash_u32(h);
        return u32_to_unit_float(h) * 2.f - 1.f; // [-1,1]
    };

    float c000 = lattice(xi + 0, yi + 0, zi + 0);
    float c100 = lattice(xi + 1, yi + 0, zi + 0);
    float c010 = lattice(xi + 0, yi + 1, zi + 0);
    float c110 = lattice(xi + 1, yi + 1, zi + 0);

    float c001 = lattice(xi + 0, yi + 0, zi + 1);
    float c101 = lattice(xi + 1, yi + 0, zi + 1);
    float c011 = lattice(xi + 0, yi + 1, zi + 1);
    float c111 = lattice(xi + 1, yi + 1, zi + 1);

    float x00 = lerp(c000, c100, fx);
    float x10 = lerp(c010, c110, fx);
    float x01 = lerp(c001, c101, fx);
    float x11 = lerp(c011, c111, fx);

    float y0 = lerp(x00, x10, fy);
    float y1 = lerp(x01, x11, fy);

    return lerp(y0, y1, fz);
}

__device__ float fbm3D(float x, float y, float z, uint32_t seed) {
    // few octaves is enough
    float sum = 0.f;
    float amp = 1.f;
    float freq = 1.f;

    #pragma unroll
    for (int o = 0; o < 4; ++o) {
        sum += amp * valueNoise3D(x * freq, y * freq, z * freq, seed + 101u * o);
        freq *= 2.f;
        amp *= 0.5f;
    }
    return sum;
}

__global__ void generateScalarFieldKernel(
    float* out,
    int Nx, int Ny, int Nz,
    float minX, float minY, float minZ,
    float dx, float dy, float dz,
    bool planeMode,
    float planeY,
    uint32_t seed)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int N = Nx * Ny * Nz;
    if (tid >= N) return;

    // decode (x,y,z) from linear index with your layout: z*Nx*Ny + y*Nx + x
    int x = tid % Nx;
    int tmp = tid / Nx;
    int y = tmp % Ny;
    int z = tmp / Ny;

    float worldX = minX + dx * (float)x;
    float worldY = minY + dy * (float)y;
    float worldZ = minZ + dz * (float)z;

    float v = 0.f;

    if (planeMode) {
        // Match your CPU plane field: terrain(x,y,z) = worldY - planeY
        v = worldY - planeY;
    } else {
        // Terrain-like: a sloped density + fBm noise (procedural)
        // (This is not identical to your CPU expand_and_noise_3D, but it is GPU procedural generation.)
        float base = 0.2f + (worldY - 0.0f); // a simple vertical ramp
        float n = fbm3D(worldX * 0.15f, worldY * 0.15f, worldZ * 0.15f, seed);
        v = base + 0.6f * n;
    }

    out[tid] = v;
}



int main(int argc, char** argv)
{
    Options opt = parseArgs(argc, argv);

    //default values for runs
    int WARMUP = opt.warmup;
    int ITERS  = opt.iters;

    bool deterministic = opt.deterministic;
    int  CULL_CHUNKS   = opt.cullChunks ? 1 : 0;

    bool PLANE_FIELD = opt.planeMode;
    float planeY     = opt.planeY;

    int  TILE_SMEM = opt.tiled ? 1 : 0;
    bool GPU_FIELD = opt.gpuField;

    int Nx = opt.Nx, Ny = opt.Ny, Nz = opt.Nz;
    float iso = opt.iso;


    /*for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--warmup" && i + 1 < argc) {
            WARMUP = std::stoi(argv[++i]);
        }
        else if (arg == "--iters" && i + 1 < argc) {
            ITERS = std::stoi(argv[++i]);
        }
        else if (arg == "--help") {
            std::cout << "Usage: ./generator [--warmup N] [--iters N]\n";
            return 0;
        }
        else if (arg == "--deterministic") {
            deterministic = true;
        }
        else if (arg == "--cull") {
            CULL_CHUNKS = 1;
        }
        else if (arg == "--plane") {
            PLANE_FIELD = true;
        }
        else if (arg == "--planeY" && i + 1 < argc) {
            planeY = std::stof(argv[++i]);
        }
        else if (arg == "--tile") {
            TILE_SMEM = 1;
        }
        else if (arg == "--gpuField") {
            GPU_FIELD = true;
        }
    }
        */
    
    std::cout << "Chunk culling(OPT2): " << (CULL_CHUNKS ? "ON" : "OFF") << "\n";
    std::cout << "Shared-memory tiling (OPT3a): " << (TILE_SMEM ? "ON" : "OFF") << "\n";
    std::cout << "GPU scalar field (OPT3b): " << (GPU_FIELD ? "ON" : "OFF") << "\n";

    if (PLANE_FIELD) {
        std::cout << "Scalar field mode: PLANE (planeY=" << planeY << ")\n";
    }

    // Grid bounds in world space
    const float minX = -15.0f, minY = -2.1f, minZ = -15.0f;
    const float maxX = 15.0f, maxY = 2.4f, maxZ = 15.0f;

    if (deterministic) {
        srand(0);
        std::cout << "Using deterministic seed (0)\n";
    } else {
        srand(static_cast<unsigned int>(time(nullptr)));
    }


    /////////////////////////////////////////////////////////////////
    // Generate the scalar field data to have something interesting to
    // extract an isosurface from.
    /////////////////////////////////////////////////////////////////

    // Initial coarse grid
    Grid3D<float> terrain(16, 6, 16);

    // Decide final dimensions
    //int Nx = 0, Ny = 0, Nz = 0;

    if (!GPU_FIELD) {
        // CPU path
        if (PLANE_FIELD) {
            const int Nx_plane = 388;
            const int Ny_plane = 68;
            const int Nz_plane = 388;

            terrain = Grid3D<float>(Nx_plane, Ny_plane, Nz_plane);
            float dy_plane = (maxY - minY) / float(Ny_plane - 1);

            for (int z = 0; z < Nz_plane; ++z)
            for (int y = 0; y < Ny_plane; ++y)
            for (int x = 0; x < Nx_plane; ++x)
            {
                float worldY = minY + y * dy_plane;
                terrain(x, y, z) = worldY - planeY;
            }
        } else {
            // original noise terrain code 
            for (int z = 0; z < terrain.sizeZ(); ++z)
            for (int y = 0; y < terrain.sizeY(); ++y)
            for (int x = 0; x < terrain.sizeX(); ++x)
            {
                int yGroundInt = 3;
                terrain(x, y, z) =
                    0.2f + float(y - yGroundInt)
                    + 0.08f * random_normal(x + y * 500 + z * 10000);
            }

            expand_and_noise_3D(terrain, 0);
            expand_and_noise_3D(terrain, 1);
            expand_and_noise_3D(terrain, 5);
            expand_and_noise_3D(terrain, 10);
            expand_and_noise_3D(terrain, 20);
        }

        Nx = terrain.sizeX();
        Ny = terrain.sizeY();
        Nz = terrain.sizeZ();

    } else {
        // GPU path: generate directly on device
        Nx = 388; Ny = 68; Nz = 388;
    }
    size_t scalarCount = (size_t)Nx * Ny * Nz;

    std::cout << "Generated scalar field size: " << Nx << " x " << Ny << " x " << Nz << std::endl;

    float *d_scalarField = nullptr;

    // Compute grid spacing
    float dx = (maxX - minX) / static_cast<float>(Nx - 1);
    float dy = (maxY - minY) / static_cast<float>(Ny - 1);
    float dz = (maxZ - minZ) / static_cast<float>(Nz - 1);

    // Maximum number of triangles allowed (arbitrary limit to avoid overruns)
    //const unsigned int maxTriangles = 32u * 1024u * 1024u;

    // Allocate device memory for the scalar field.
    //cudaMalloc(&d_scalarField, scalarField.size() * sizeof(float));
    //cudaMemcpy(d_scalarField, scalarField.data(), scalarField.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (!GPU_FIELD) {
        std::vector<float> scalarField = terrain.getData();
        cudaMalloc(&d_scalarField, scalarField.size() * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_scalarField, scalarField.data(),
                            scalarField.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
    } else {
        int N = (int)scalarCount;
        int threadsGen = 256;
        int blocksGen  = (N + threadsGen - 1) / threadsGen;

        uint32_t seed = deterministic ? 0u : (uint32_t)time(nullptr);
        CUDA_CHECK(cudaMalloc(&d_scalarField, scalarCount * sizeof(float)));

        generateScalarFieldKernel<<<blocksGen, threadsGen>>>(
            d_scalarField, Nx, Ny, Nz,
            minX, minY, minZ,
            dx, dy, dz,
            PLANE_FIELD, planeY,
            seed
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    //new (dynamic allocation based on counting)
    int cellsX = Nx - 1, cellsY = Ny - 1, cellsZ = Nz - 1;
    int nCells = cellsX * cellsY * cellsZ;

    int threads = 256;
    int blocks  = (nCells + threads - 1) / threads;

    //chunks
    constexpr int CHUNK = 8;
    int chunksX = (cellsX + CHUNK - 1) / CHUNK;
    int chunksY = (cellsY + CHUNK - 1) / CHUNK;
    int chunksZ = (cellsZ + CHUNK - 1) / CHUNK;
    int nChunks = chunksX * chunksY * chunksZ;

    uint8_t*  d_cubeIndex = nullptr;
    uint32_t* d_triCount  = nullptr;
    uint32_t* d_triOffset = nullptr;
    uint32_t* d_totalTris = nullptr;

    ScanWorkspace scanWS = createScanWorkspace(nCells);
    uint32_t h_numActiveChunks = 0;   // cached on host
    bool activeListBuilt = false;
    
    cudaMalloc(&d_cubeIndex, nCells * sizeof(uint8_t));
    cudaMalloc(&d_triCount,  nCells * sizeof(uint32_t));
    cudaMalloc(&d_triOffset, nCells * sizeof(uint32_t));
    cudaMalloc(&d_totalTris, sizeof(uint32_t));

    //chunks
    uint32_t* d_chunkActive = nullptr;   // use uint32 so you can reuse your scan
    uint32_t* d_chunkOffset = nullptr;
    uint32_t* d_activeChunkIds = nullptr;
    uint32_t* d_numActiveChunks = nullptr;
    uint32_t* d_chunkNonEmpty = nullptr;  

    ScanWorkspace chunkScanWS{};

    if (CULL_CHUNKS) {
        chunkScanWS = createScanWorkspace(nChunks);
        CUDA_CHECK(cudaMalloc(&d_chunkActive, nChunks * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_chunkOffset, nChunks * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_activeChunkIds, nChunks * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_numActiveChunks, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_chunkNonEmpty, nChunks * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_chunkNonEmpty, 0, nChunks * sizeof(uint32_t))); // once
    }

    cudaEvent_t startEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);

    //one time sizing run 
    //float iso = 0.0f; 

    //helper for chunks
    auto discoverNonEmptyChunks = [&]() {
        // Temporary mask for this discovery pass
        CUDA_CHECK(cudaMemset(d_chunkActive, 0, nChunks * sizeof(uint32_t)));

        // 1 block per chunk
        buildChunkMaskKernel<<<nChunks, 256>>>(
            d_scalarField, Nx, Ny, Nz, iso,
            cellsX, cellsY, cellsZ,
            chunksX, chunksY,
            d_chunkActive
        );
        CUDA_CHECK(cudaGetLastError());

        // Merge into persistent mask (OR semantics)
        int t = 256;
        int b = (nChunks + t - 1) / t;
        mergeChunkMaskKernel<<<b, t>>>(d_chunkActive, d_chunkNonEmpty, nChunks);
        CUDA_CHECK(cudaGetLastError());
    };

    //presistant build
    auto buildActiveChunkListFromPersistentMask = [&]() {
        // 1) scan persistent mask -> offsets
        exclusiveScanUint32_ws(d_chunkNonEmpty, d_chunkOffset, nChunks, chunkScanWS);

        // 2) compute total active chunks
        computeTotalKernel<<<1,1>>>(d_chunkNonEmpty, d_chunkOffset, d_numActiveChunks, nChunks);
        CUDA_CHECK(cudaGetLastError());

        // 3) scatter IDs
        scatterActiveChunksKernel<<<(nChunks + 255)/256, 256>>>(
            d_chunkNonEmpty, d_chunkOffset, d_activeChunkIds, nChunks
        );
        CUDA_CHECK(cudaGetLastError());
    };


    if (CULL_CHUNKS) {
        // First extraction / discovery step: mark non-empty chunks persistently
        discoverNonEmptyChunks();

        // Build the active chunk list from the persistent mask
        buildActiveChunkListFromPersistentMask();
        CUDA_CHECK(cudaMemcpy(&h_numActiveChunks, d_numActiveChunks, sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));

        //CUDA_CHECK(cudaMemset(d_triCount, 0xCD, nCells * sizeof(uint32_t)));
       //CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_triCount, 0, nCells * sizeof(uint32_t)));
        std::cout << "Active chunks: " << h_numActiveChunks << " / " << nChunks << "\n";
        activeListBuilt = true; // optional, if you keep the variable
    }
    
    

    countTrisKernel<<<blocks, threads>>>(d_scalarField, Nx, Ny, Nz, iso,
                                        d_cubeIndex, d_triCount);

    exclusiveScanUint32_ws(d_triCount, d_triOffset, nCells, scanWS);

    computeTotalKernel<<<1,1>>>(d_triCount, d_triOffset, d_totalTris, nCells);

    uint32_t totalTris = 0;

    cudaMemcpy(&totalTris, d_totalTris, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint32_t totalTris_after = 0;

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t totalVerts = totalTris * 3;
    std::cout << "Total triangles: " << totalTris << "\n";
    
    //new
    float4* d_vertices = nullptr;
    unsigned int* d_indices = nullptr;

    cudaMalloc(&d_vertices, totalVerts * sizeof(float4));
    cudaMalloc(&d_indices,  totalVerts * sizeof(unsigned int));

    std::vector<float> times_ms;
    times_ms.reserve(ITERS);

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        if (CULL_CHUNKS) {
            //buildActiveChunkList();
            uint32_t numActiveChunks = h_numActiveChunks;
            clearTriCountActiveChunksKernel<CHUNK><<<h_numActiveChunks, 256>>>(
                d_triCount,
                cellsX, cellsY, cellsZ,
                chunksX, chunksY,
                d_activeChunkIds, h_numActiveChunks
            );
            CUDA_CHECK(cudaGetLastError());

             //Count
            if (TILE_SMEM) {
                countTrisActiveChunksKernel_tiled<CHUNK><<<numActiveChunks, 256>>>(
                    d_scalarField, Nx, Ny, Nz, iso,
                    chunksX, chunksY,
                    d_activeChunkIds, numActiveChunks,
                    d_cubeIndex, d_triCount
                );
            } else {
                countTrisActiveChunksKernel<CHUNK><<<numActiveChunks, 256>>>(
                    d_scalarField, Nx, Ny, Nz, iso,
                    chunksX, chunksY,
                    d_activeChunkIds, numActiveChunks,
                    d_cubeIndex, d_triCount
                );
            }
            CUDA_CHECK(cudaGetLastError());
            //scan
            exclusiveScanUint32_ws(d_triCount, d_triOffset, nCells, scanWS);

            //emit
            if (TILE_SMEM) {
                emitActiveChunksKernel_tiled<CHUNK><<<numActiveChunks, 256>>>(
                    d_scalarField, Nx, Ny, Nz,
                    minX, minY, minZ,
                    dx, dy, dz,
                    iso,
                    chunksX, chunksY,
                    d_activeChunkIds, numActiveChunks,
                    d_cubeIndex, d_triCount, d_triOffset,
                    d_vertices, d_indices
                );
            } else {
                emitActiveChunksKernel<CHUNK><<<numActiveChunks, 256>>>(
                    d_scalarField, Nx, Ny, Nz,
                    minX, minY, minZ,
                    dx, dy, dz,
                    iso,
                    chunksX, chunksY,
                    d_activeChunkIds, numActiveChunks,
                    d_cubeIndex, d_triCount, d_triOffset,
                    d_vertices, d_indices
                );
            }
            CUDA_CHECK(cudaGetLastError());

        } else {
            countTrisKernel<<<blocks, threads>>>(d_scalarField, Nx, Ny, Nz, iso,
                                                d_cubeIndex, d_triCount);
            exclusiveScanUint32_ws(d_triCount, d_triOffset, nCells, scanWS);

            emitKernel<<<blocks, threads>>>(d_scalarField, Nx, Ny, Nz,
                                            minX, minY, minZ,
                                            dx, dy, dz,
                                            iso,
                                            d_cubeIndex, d_triCount, d_triOffset,
                                            d_vertices, d_indices);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }


    // Timed runs
    for (int i = 0; i < ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(startEvent));

        if (CULL_CHUNKS) {
            //buildActiveChunkList();
            uint32_t numActiveChunks = h_numActiveChunks;
            clearTriCountActiveChunksKernel<CHUNK><<<h_numActiveChunks, 256>>>(
                d_triCount,
                cellsX, cellsY, cellsZ,
                chunksX, chunksY,
                d_activeChunkIds, h_numActiveChunks
            );
            CUDA_CHECK(cudaGetLastError());

             //Count
            if (TILE_SMEM) {
                countTrisActiveChunksKernel_tiled<CHUNK><<<numActiveChunks, 256>>>(
                    d_scalarField, Nx, Ny, Nz, iso,
                    chunksX, chunksY,
                    d_activeChunkIds, numActiveChunks,
                    d_cubeIndex, d_triCount
                );
            } else {
                countTrisActiveChunksKernel<CHUNK><<<numActiveChunks, 256>>>(
                    d_scalarField, Nx, Ny, Nz, iso,
                    chunksX, chunksY,
                    d_activeChunkIds, numActiveChunks,
                    d_cubeIndex, d_triCount
                );
            }
            CUDA_CHECK(cudaGetLastError());
            //scan
            exclusiveScanUint32_ws(d_triCount, d_triOffset, nCells, scanWS);
           
            //Debug check
            /*if (i == 0) {
                computeTotalKernel<<<1,1>>>(d_triCount, d_triOffset, d_totalTris, nCells);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(&totalTris_after, d_totalTris, sizeof(uint32_t),
                                    cudaMemcpyDeviceToHost));
                std::cout << "[CHECK] totalTris this iter: " << totalTris_after << "\n";
            }
            */

            //emit
            if (TILE_SMEM) {
                emitActiveChunksKernel_tiled<CHUNK><<<numActiveChunks, 256>>>(
                    d_scalarField, Nx, Ny, Nz,
                    minX, minY, minZ,
                    dx, dy, dz,
                    iso,
                    chunksX, chunksY,
                    d_activeChunkIds, numActiveChunks,
                    d_cubeIndex, d_triCount, d_triOffset,
                    d_vertices, d_indices
                );
            } else {
                emitActiveChunksKernel<CHUNK><<<numActiveChunks, 256>>>(
                    d_scalarField, Nx, Ny, Nz,
                    minX, minY, minZ,
                    dx, dy, dz,
                    iso,
                    chunksX, chunksY,
                    d_activeChunkIds, numActiveChunks,
                    d_cubeIndex, d_triCount, d_triOffset,
                    d_vertices, d_indices
                );
            }
            CUDA_CHECK(cudaGetLastError());

        } else {
            // OPT1 path (full grid)
            countTrisKernel<<<blocks, threads>>>(
                d_scalarField, Nx, Ny, Nz, iso,
                d_cubeIndex, d_triCount
            );
            CUDA_CHECK(cudaGetLastError());

            exclusiveScanUint32_ws(d_triCount, d_triOffset, nCells, scanWS);

            emitKernel<<<blocks, threads>>>(
                d_scalarField, Nx, Ny, Nz,
                minX, minY, minZ,
                dx, dy, dz,
                iso,
                d_cubeIndex, d_triCount, d_triOffset,
                d_vertices, d_indices
            );
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaEventRecord(endEvent));
        CUDA_CHECK(cudaEventSynchronize(endEvent));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, endEvent));
        times_ms.push_back(ms);
    }



    auto min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();

    double var = 0.0;
    for (float t : times_ms) {
        double d = t - mean_ms;
        var += d * d;
    }
    var /= times_ms.size();
    double std_ms = std::sqrt(var);

    //stats
    std::cout << "OPT1 pipeline timing (" << ITERS << " runs, " << WARMUP << " warmup):\n"
            << "  mean: " << mean_ms << " ms\n"
            << "  min : " << min_ms  << " ms\n"
            << "  std : " << std_ms  << " ms\n";


    //final run for output
    if (CULL_CHUNKS) {
        //buildActiveChunkList();
        uint32_t numActiveChunks = h_numActiveChunks;
        clearTriCountActiveChunksKernel<CHUNK><<<h_numActiveChunks, 256>>>(
            d_triCount,
            cellsX, cellsY, cellsZ,
            chunksX, chunksY,
            d_activeChunkIds, h_numActiveChunks
        );
        CUDA_CHECK(cudaGetLastError());

            //Count
        if (TILE_SMEM) {
            countTrisActiveChunksKernel_tiled<CHUNK><<<numActiveChunks, 256>>>(
                d_scalarField, Nx, Ny, Nz, iso,
                chunksX, chunksY,
                d_activeChunkIds, numActiveChunks,
                d_cubeIndex, d_triCount
            );
        } else {
            countTrisActiveChunksKernel<CHUNK><<<numActiveChunks, 256>>>(
                d_scalarField, Nx, Ny, Nz, iso,
                chunksX, chunksY,
                d_activeChunkIds, numActiveChunks,
                d_cubeIndex, d_triCount
            );
        }
        CUDA_CHECK(cudaGetLastError());
        //scan
        exclusiveScanUint32_ws(d_triCount, d_triOffset, nCells, scanWS);

        //emit
        if (TILE_SMEM) {
            emitActiveChunksKernel_tiled<CHUNK><<<numActiveChunks, 256>>>(
                d_scalarField, Nx, Ny, Nz,
                minX, minY, minZ,
                dx, dy, dz,
                iso,
                chunksX, chunksY,
                d_activeChunkIds, numActiveChunks,
                d_cubeIndex, d_triCount, d_triOffset,
                d_vertices, d_indices
            );
        } else {
            emitActiveChunksKernel<CHUNK><<<numActiveChunks, 256>>>(
                d_scalarField, Nx, Ny, Nz,
                minX, minY, minZ,
                dx, dy, dz,
                iso,
                chunksX, chunksY,
                d_activeChunkIds, numActiveChunks,
                d_cubeIndex, d_triCount, d_triOffset,
                d_vertices, d_indices
            );
        }
        CUDA_CHECK(cudaGetLastError());

    } else {
        countTrisKernel<<<blocks, threads>>>(
            d_scalarField, Nx, Ny, Nz, iso,
            d_cubeIndex, d_triCount);

        exclusiveScanUint32_ws(d_triCount, d_triOffset, nCells, scanWS);

        emitKernel<<<blocks, threads>>>(
            d_scalarField, Nx, Ny, Nz,
            minX, minY, minZ,
            dx, dy, dz,
            iso,
            d_cubeIndex, d_triCount, d_triOffset,
            d_vertices, d_indices);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    unsigned int vertexCount = (unsigned int) totalVerts;
    unsigned int indexCount  = (unsigned int) totalVerts;

    std::vector<float4> hostVertices(vertexCount);
    CUDA_CHECK(cudaMemcpy(hostVertices.data(), d_vertices,
                        vertexCount * sizeof(float4), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> hostIndices(indexCount);
    CUDA_CHECK(cudaMemcpy(hostIndices.data(), d_indices,
                        indexCount * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::cout << "Vertices generated: " << vertexCount << std::endl;
    std::cout << "Indices generated:  " << indexCount << std::endl;

    int numToPrint = std::min<int>(5, static_cast<int>(vertexCount));
    std::cout << "First " << numToPrint << " vertices:" << std::endl;
    for (int i = 0; i < numToPrint; ++i)
    {
        float4 v = hostVertices[i];
        std::cout << "Vertex " << i << ": (" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")\n";
    }
    numToPrint = std::min<int>(5, static_cast<int>(indexCount));
    std::cout << "First " << numToPrint << " indices:" << std::endl;
    for (int i = 0; i < numToPrint; ++i)
    {
        std::cout << "Index " << i << ": " << hostIndices[i] << std::endl;
    }

    // Write the triangles to "triangles.txt" file.
    //writeTrianglesToFile("triangles.txt", hostVertices, hostIndices);
    writeTrianglesToFile(opt.output.c_str(), hostVertices, hostIndices);


    // Free device memory.
    cudaFree(d_scalarField);
    cudaFree(d_vertices);
    cudaFree(d_indices);
    cudaFree(d_cubeIndex);
    cudaFree(d_triCount);
    cudaFree(d_triOffset);
    cudaFree(d_totalTris);

    if (CULL_CHUNKS) {
        CUDA_CHECK(cudaFree(d_chunkActive));
        CUDA_CHECK(cudaFree(d_chunkOffset));
        CUDA_CHECK(cudaFree(d_activeChunkIds));
        CUDA_CHECK(cudaFree(d_numActiveChunks));
        CUDA_CHECK(cudaFree(d_chunkNonEmpty));
        destroyScanWorkspace(chunkScanWS);
    }
    destroyScanWorkspace(scanWS);

    return 0;
}

