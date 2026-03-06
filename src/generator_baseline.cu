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

#define CUDA_CHECK(expr) do {                              \
  cudaError_t _err = (expr);                               \
  if (_err != cudaSuccess) {                               \
    fprintf(stderr, "CUDA error %s at %s:%d\n",            \
            cudaGetErrorString(_err), __FILE__, __LINE__); \
    std::exit(1);                                          \
  }                                                        \
} while(0)

// The edgeTable maps the 256 possible cube configurations to a 12-bit
// number, each bit corresponding to one edge (see edge indexing in the
// d_edgeIndexMap).  A bit is set if the isosurface intersects that edge.
__constant__ int d_edgeTable[256]={
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };

// The triTable maps the 256 possible cube configurations to up to
// 5 triangles (15 vertex indices, -1 as sentinel).
// E.g., triTable[3] = {1, 8, 3, 9, 8, 1, -1, ...} means that
// for cube configuration 3, there are two triangles:
//  - triangle 1: vertices 1, 8, 3
//  - triangle 2: vertices 9, 8, 1
__constant__ int d_triTable[256][16] =
    {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

// Offsets of the cube corners relative to the cube origin
__constant__ int d_cornerOffsets[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
};

// Maps edge indices to the two endpoints (corner indices) that define the edge.
// The corners are numbered according to d_cornerOffsets.
__constant__ int d_edgeIndexMap[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7}
};



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


int main(int argc, char** argv)
{
    //default values for runs
    int WARMUP = 5;
    int ITERS  = 50;
    bool deterministic = false;

    for (int i = 1; i < argc; ++i) {
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

    srand(static_cast<unsigned int>(time(nullptr)));

    /////////////////////////////////////////////////////////////////
    // Generate the scalar field data to have something interesting to
    // extract an isosurface from.
    /////////////////////////////////////////////////////////////////

    // Initial coarse grid
    Grid3D<float> terrain(16, 6, 16);
    for (int z = 0; z < terrain.sizeZ(); ++z)
    for (int y = 0; y < terrain.sizeY(); ++y)
    for (int x = 0; x < terrain.sizeX(); ++x)
    {
        // Assign base scalar field values with some noise
        int yGroundInt = 3;
        terrain(x, y, z) =
                    // Y axis bias: lower -> negative
                    0.2f + static_cast<float>(y - yGroundInt)
                    // Add some noise
                    + 0.08f * random_normal(x + y * 500 + z * 10000);
    }

    // Apply expansion and noise
    expand_and_noise_3D(terrain, 0);
    expand_and_noise_3D(terrain, 1);
    expand_and_noise_3D(terrain, 5);
    expand_and_noise_3D(terrain, 10);
    expand_and_noise_3D(terrain, 20);

    std::vector<float> scalarField = terrain.getData();

    const int Nx = terrain.sizeX();
    const int Ny = terrain.sizeY();
    const int Nz = terrain.sizeZ();

    std::cout << "Generated scalar field size: " << Nx << " x " << Ny << " x " << Nz << std::endl;

    // Compute grid spacing
    float dx = (maxX - minX) / static_cast<float>(Nx - 1);
    float dy = (maxY - minY) / static_cast<float>(Ny - 1);
    float dz = (maxZ - minZ) / static_cast<float>(Nz - 1);

    // Maximum number of triangles allowed (arbitrary limit to avoid overruns)
    const unsigned int maxTriangles = 32u * 1024u * 1024u;

    // Allocate device memory for the scalar field.
    float *d_scalarField = nullptr;
    cudaMalloc(&d_scalarField, scalarField.size() * sizeof(float));
    cudaMemcpy(d_scalarField, scalarField.data(), scalarField.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate device memory for the output vertices.
    float4 *d_vertices = nullptr;
    cudaMalloc(&d_vertices, maxTriangles * sizeof(float4));

    // Allocate device memory for indices
    // Each triangle has 3 indices.
    unsigned int *d_indices = nullptr;
    cudaMalloc(&d_indices, maxTriangles * 3u * sizeof(unsigned int));

    // Allocate atomic counters and performance counters.
    unsigned int *d_vertexCounter = nullptr;
    unsigned int *d_indexCounter  = nullptr;
    cudaMalloc(&d_vertexCounter, sizeof(unsigned int));
    cudaMalloc(&d_indexCounter,  sizeof(unsigned int));

    cudaEvent_t startEvent, endEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);


    // Launch the Marching Cubes kernel
    dim3 blockDim(4, 4, 4);
    dim3 gridDim((Nx - 1 + blockDim.x - 1) / blockDim.x,
                    (Ny - 1 + blockDim.y - 1) / blockDim.y,
                    (Nz - 1 + blockDim.z - 1) / blockDim.z);

    // Start the timer for performance measurement.
    /*
    cudaEventRecord(startEvent);

    marchingCubesKernel<<<gridDim, blockDim>>>(d_scalarField,
                                                Nx, Ny, Nz,
                                                minX, minY, minZ,
                                                dx, dy, dz,
                                                d_vertices,
                                                d_vertexCounter,
                                                d_indices,
                                                d_indexCounter,
                                                maxTriangles);

    // Stop the timer.
    cudaEventRecord(endEvent);
    cudaEventSynchronize(endEvent);

    cudaDeviceSynchronize();

    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, startEvent, endEvent);
    std::cout << "Marching Cubes kernel executed in " << elapsedTime << " ms." << std::endl;
    */

    //new check with mean and stddev
    auto resetCounters = [&]() {
        CUDA_CHECK(cudaMemset(d_vertexCounter, 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(d_indexCounter,  0, sizeof(unsigned int)));
    };

    std::vector<float> times_ms;
    times_ms.reserve(ITERS);

    // Warmup iterations
    for (int i = 0; i < WARMUP; ++i) {
        resetCounters();

        marchingCubesKernel<<<gridDim, blockDim>>>(d_scalarField,
                                                Nx, Ny, Nz,
                                                minX, minY, minZ,
                                                dx, dy, dz,
                                                d_vertices,
                                                d_vertexCounter,
                                                d_indices,
                                                d_indexCounter,
                                                maxTriangles);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    //kernel measured runs
    for (int i = 0; i < ITERS; ++i) {
        resetCounters();

        CUDA_CHECK(cudaEventRecord(startEvent));
        marchingCubesKernel<<<gridDim, blockDim>>>(d_scalarField,
                                                Nx, Ny, Nz,
                                                minX, minY, minZ,
                                                dx, dy, dz,
                                                d_vertices,
                                                d_vertexCounter,
                                                d_indices,
                                                d_indexCounter,
                                                maxTriangles);
        CUDA_CHECK(cudaEventRecord(endEvent));

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(endEvent));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, endEvent));
        times_ms.push_back(ms);
    }

    // Stats
    auto min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    double mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();

    double var = 0.0;
    for (float t : times_ms) {
        double d = t - mean_ms;
        var += d * d;
    }
    var /= times_ms.size();
    double std_ms = std::sqrt(var);

    //printing outputs
    std::cout << "Marching Cubes kernel timing (" << ITERS << " runs, " << WARMUP << " warmup):\n"
            << "  mean: " << mean_ms << " ms\n"
            << "  min : " << min_ms  << " ms\n"
            << "  std : " << std_ms  << " ms\n";

   
    //final run used for output file writing
    resetCounters();

    marchingCubesKernel<<<gridDim, blockDim>>>(d_scalarField,
                                            Nx, Ny, Nz,
                                            minX, minY, minZ,
                                            dx, dy, dz,
                                            d_vertices,
                                            d_vertexCounter,
                                            d_indices,
                                            d_indexCounter,
                                            maxTriangles);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // Fetch the counters back to host.
    unsigned int vertexCount = 0, indexCount = 0;
    cudaMemcpy(&vertexCount, d_vertexCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&indexCount,  d_indexCounter,  sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::vector<float4> hostVertices(vertexCount);
    if (vertexCount > 0)
    {
        cudaMemcpy(hostVertices.data(), d_vertices, vertexCount * sizeof(float4), cudaMemcpyDeviceToHost);
    }
    std::vector<unsigned int> hostIndices(indexCount);
    if (indexCount > 0)
    {
        cudaMemcpy(hostIndices.data(), d_indices, indexCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }

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
    writeTrianglesToFile("triangles.txt", hostVertices, hostIndices);


    // Free device memory.
    cudaFree(d_scalarField);
    cudaFree(d_vertices);
    cudaFree(d_indices);
    cudaFree(d_vertexCounter);
    cudaFree(d_indexCounter);

    return 0;
}
