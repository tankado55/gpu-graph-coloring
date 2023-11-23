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

class Colorer
{
public:
	Coloring* LDFColoring();
};

Coloring* RandomPriorityColoring(Graph& graph);

void test(Graph& graph);
Coloring* RandomPriorityColoringV2(Graph& graph);
Coloring* RandomPriorityColoringV3(Graph& graph);
void printColoring(Coloring*, GraphStruct*, bool);
__global__ void InitRandomPriorities(uint seed, curandState_t*, uint*, uint);
__global__ void colorIS(Coloring*, GraphStruct*, uint*);
__global__ void print_d(GraphStruct*, bool);
__global__ void applyBufferWithInboundCountersBitmaps(uint*, bool*, GraphStruct*, unsigned*, unsigned*, unsigned*, bool*, bool*, uint*);
__global__ void colorWithInboundCountersBitmaps(uint*, bool*, GraphStruct*, uint*, uint*, bool*, bool*, uint*, bool*);
__global__ void calculateInbounds(GraphStruct* graphStruct, unsigned int* inboundCounts, unsigned int* priorities, int n);
