#pragma once

#include <curand_kernel.h>
#include <bitset>
#include "graph/graph.h"
#include "utils/common.h"

#define THREADxBLOCK 128

struct Coloring {
	uint iterationCount;
	uint* coloring;   // each element denotes the color of the node at the correspondent index
	bool* coloredNodes;
};

void printColoring(Coloring*, GraphStruct*, bool);
__global__ void print_d(GraphStruct*, bool);
__global__ void calculateInbounds(GraphStruct* graphStruct, unsigned int* inboundCounts, unsigned int* priorities, int n);
__global__ void colorWithInboundCountersBitmaps(uint*, bool*, GraphStruct*, uint*, uint*, bool*, bool*, uint*, bool*);
__global__ void applyBufferWithInboundCountersBitmaps(uint*, bool*, GraphStruct*, unsigned*, unsigned*, unsigned*, bool*, bool*, uint*);
__global__ void colorWithoutInbounds(bool* isColored, GraphStruct* graphStruct, uint* buffer, bool* filledBuffer, bool* bitmaps, uint* bitmapIndex, uint* priorities, bool* uncoloredFlag);

namespace global
{
	Coloring* color(Graph& graph, uint* d_priorities);
};
