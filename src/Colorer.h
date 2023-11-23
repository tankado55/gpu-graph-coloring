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
private:
	virtual uint* calculatePriority(Graph& graph, GraphStruct* d_graphStruct) = 0;
public:
	Coloring* color(Graph& graph);
	virtual ~Colorer();
};

void printColoring(Coloring*, GraphStruct*, bool);
__global__ void print_d(GraphStruct*, bool);
__global__ void calculateInbounds(GraphStruct* graphStruct, unsigned int* inboundCounts, unsigned int* priorities, int n);
__global__ void colorWithInboundCountersBitmaps(uint*, bool*, GraphStruct*, uint*, uint*, bool*, bool*, uint*, bool*);
__global__ void applyBufferWithInboundCountersBitmaps(uint*, bool*, GraphStruct*, unsigned*, unsigned*, unsigned*, bool*, bool*, uint*);
