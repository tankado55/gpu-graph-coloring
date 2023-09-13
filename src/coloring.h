
#pragma once

#include <curand_kernel.h>
#include "utils/graph/graph.h"
#include "utils/common.h"

/**
 *  graph coloring struct (colors are: 1,2,3,..,k)
 */

struct Coloring {
	bool		uncoloredNodes;
	uint		numOfColors;
	uint* coloring;   // each element denotes the color of the node at the correspondent index
};

struct ColoringUtils {
	int* availableColorIndex{ nullptr };
	int* availableColors{ nullptr };
};

Coloring* LubyGreedy(GraphStruct*);
void printColoring(Coloring*, GraphStruct*, bool);
__global__ void init(uint seed, curandState_t*, uint*, uint);
void LubyJPcolorer(Coloring*, GraphStruct*, uint*);
__global__ void findIS(Coloring*, GraphStruct*, uint*);
__global__ void print_d(GraphStruct*, bool);