#pragma once
#include <cuda_runtime.h>

extern __constant__ int d_edgeTable[256];
extern __constant__ int d_triTable[256][16];
extern __constant__ int d_cornerOffsets[8][3];
extern __constant__ int d_edgeIndexMap[12][2];
