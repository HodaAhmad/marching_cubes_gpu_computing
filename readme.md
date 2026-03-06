# GPU-Based Marching Cubes (CUDA)

## Overview
This project implements and optimizes a GPU-based Marching Cubes pipeline using CUDA. The focus is on correctness, performance measurement, and memory-aware GPU optimization. Multiple optimization stages can be enabled at runtime to compare baseline and optimized variants.

## Build & Run
Build steps (headless mode, OpenGL viewer disabled due to OS and headless environment limitation, this project was built on MacOS):

`mkdir build && cd build`  
`cmake .. -DBUILD_VIEWER=OFF`  
`make -j`  

Executable:  
`./generator`

## Command-Line Options
- `--warmup N` – Warm-up iterations  
- `--iters N` – Measured iterations  
- `--deterministic` – Fixed random seed  
- `--cull` – Chunk-based empty-region culling (OPT2)  
- `--tile` – Shared-memory tiling inside chunks (OPT3a)  
- `--plane` – Plane scalar field  
- `--planeY Y` – Plane height  
- `--gpuField` – GPU-side scalar field generation (OPT3b)  

Flags are composable.

## Baseline
The baseline generates the scalar field on the CPU, uploads it to the GPU, and uses global atomic operations for triangle output allocation. This implementation is correct but memory-bound and suffers from atomic contention.

## Optimizations
**OPT1 - Atomic-Free Output:** Global atomics are replaced by a multi-pass GPU pipeline (Count -->  Prefix Sum --> Emit), assigning deterministic output ranges per cell. Achieves ~1.7× speedup.

**OPT2 – Chunk-Based Culling:** The volume is partitioned into 8×8×8 cell chunks. Chunks without surface crossings are skipped using a persistent mask, yielding up to ~1.6× speedup for sparse fields.

**OPT3a – Shared-Memory Tiling:** Each active chunk is processed by one CUDA block. At kernel start, the block cooperatively loads a `(CHUNK+1)^3` scalar tile into shared memory so cells in the chunk can reuse scalar corner values without repeatedly loading them from global memory. In practice, this optimization did **not** consistently improve performance in our pipeline. 

**OPT3b – GPU-Side Scalar Field Generation:** Scalar field generation is moved entirely to the GPU using a procedural kernel, eliminating host–device transfers. The GPU-generated terrain is intentionally not numerically identical to the CPU version. Enabled via `--gpuField`.

## Command-Line Interface (CLI)

### Core Options

- `--output FILE`  
  Output file for generated triangles (default: `triangles.txt`)

- `--iso V`  
  Iso-value for surface extraction (default: `0.0`)

- `--warmup N`  
  Number of warm-up iterations (default: `1`)

- `--iters N`  
  Number of timed iterations (default: `10`)

- `--deterministic`  
  Fix random seed for reproducible runs

### Grid Size

- `--Nx N --Ny N --Nz N`  
  Requested grid resolution  

  **Note:**  
  These values are ignored when using the default CPU terrain generation.  
  Grid size is fixed internally unless `--gpuField` or `--plane` is used.


### Scalar Field Modes

- `--plane`  
  Use a plane scalar field instead of noise terrain

- `--planeY V`  
  Plane height (default: `0.0`)

- `--gpuField`  
  Generate scalar field directly on the GPU (OPT3b)


### Optimization Flags

- `--chunks`  
  Enable chunk-based pipeline

- `--cull`  
  Enable empty-region chunk culling (OPT2)  
  **Requires:** `--chunks`

- `--tiled`  
  Enable shared-memory tiling inside chunks (OPT3a)  
  **Requires:** `--chunks`


### Important Notes

- The chunk-based pipeline is activated **only** when `--chunks` is specified.
- In the current implementation, shared-memory tiling is executed **only when chunk culling is enabled**.

**Correct flag combination for OPT3a:**

```bash
--chunks --cull --tiled
```

## Example Runs
Baseline:  
`./generator_baseline --deterministic --warmup 5 --iters 50`

Atomic-free Multi-pass (OPT1): 
`./generator --deterministic --warmup 5 --iters 50`

Chunk culling (OPT2):  
`./generator --deterministic --warmup 5 --iters 50 --cull --chunks`

Chunk Culling + Tiling (OPT3a):
`./generator --chunks --cull --tiled --deterministic --warmup 5 --iters 50`

Plane field (sparse surface):  
`./generator --plane --planeY 0.0 --chunks --cull --deterministic`

GPU-Generated Scalar Field (OPT3b): 
`./generator --gpuField --chunks --cull --deterministic`


## Performance & Profiling
Timing is performed using CUDA events with warm-up iterations to remove cold-start effects. Memory allocation, transfers, and I/O are excluded. Mean, minimum, and standard deviation are reported. Profiling with NVIDIA Nsight Compute shows the pipeline is primarily memory-bound, with output writes dominating DRAM traffic.

## Correctness
- Triangle counts and output sizes are stable across repeated runs for the same configuration.
- `--deterministic` fixes the random seed to make results reproducible.
- Output ordering may differ due to parallel execution (valid mesh but different ordering).
- **Plane field:** GPU and CPU match exactly (both compute `worldY - planeY` at the final resolution).
- **GPU terrain (`--gpuField`):** the generated terrain is **not identical** to the original CPU terrain generation (different noise/procedural model), so triangle counts and geometry are expected to differ from the CPU terrain case.


## Running Tests

This project includes GPU unit tests for key kernels (scan and chunk scattering).
Tests are built and run using **CMake + CTest**.

### Build with tests enabled
```bash
mkdir build
cd build
cmake .. -DBUILD_VIEWER=OFF
cmake --build . -j
```

## Code Structure & File Layout

### Baseline file with minor changes

- `src/generator_baseline.cu`  
  Main program entry point.  
  Handles CLI parsing, pipeline orchestration, timing, and output.


### Main Control

- `src/generator.cu`  
  Main program entry point.  
  Handles CLI parsing, pipeline orchestration, timing, and output.


### Scan / Prefix Sum

- `src/k_scan.cu`  
  CUDA kernels for block-level scan and offset propagation.

- `src/scan.cu`  
  Host-side scan orchestration and management (`ScanWorkspace`).

- `include/scan.cuh`  
  Scan interface definitions and workspace structures.


### Marching Cubes (Full Grid)

- `src/k_mc.cu`  
  Marching Cubes kernels:
  - `countTrisKernel`
  - `emitKernel`
  - `marchingCubesKernel` 


### Chunk-Based Pipeline

- `src/k_chunk.cu`  
  Chunk-related CUDA kernels:
  - `buildChunkMaskKernel`
  - `mergeChunkMaskKernel`
  - `scatterActiveChunksKernel`
  - `countTrisActiveChunksKernel`
  - `emitActiveChunksKernel`
  - Tiled variants (`*_tiled`)

- `include/k_chunk_kernels.cuh`  
  Declarations and interfaces for chunk-based kernels.



### Lookup Tables & Helpers

- `src/mc_tables.cu`  
  Marching Cubes lookup tables (`edgeTable`, `triTable`, etc.).

- `include/mc_tables.cuh`  
  Declarations for Marching Cubes lookup tables.

- `include/mc_device_helpers.cuh`  
  Device-side helper functions (interpolation, indexing utilities).


### Tests

- `tests/test_scan.cu`  
  Unit test for exclusive prefix scan correctness.

- `tests/test_scatter_chunks.cu`  
  Unit test for chunk ID scattering and compaction.


