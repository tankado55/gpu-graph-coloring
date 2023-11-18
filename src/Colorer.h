
#pragma once

#include <curand_kernel.h>
#include <bitset>
#include "graph/graph.h"
#include "utils/common.h"

#define THREADxBLOCK 128

struct Coloring {
	bool uncoloredFlag;
	uint iterationCount;
	uint* coloring;   // each element denotes the color of the node at the correspondent index
	bool* coloredNodes;
};

enum priorityEnum
{
	LDF = 0,
	SDF = 1
};

class Colorer
{
private:
	Coloring* m_Coloring;
	GraphStruct* m_GraphStruct;
	Graph* m_Graph;
	uint* m_InboundCounts;

public:
	Colorer(Graph*);
	Coloring* LDFColoring();
	Coloring* RandomPriorityColoringCPUSequential();
	~Colorer();
};

Coloring* RandomPriorityColoring(Graph& graph);

void test(Graph& graph);
Coloring* RandomPriorityColoringV2(Graph& graph);
Coloring* RandomPriorityColoringV3(Graph& graph);
Coloring* DegreePriorityColoringV3(Graph& graph, priorityEnum);
void printColoring(Coloring*, GraphStruct*, bool);
__global__ void InitRandomPriorities(uint seed, curandState_t*, uint*, uint);
__global__ void colorIS(Coloring*, GraphStruct*, uint*);
__global__ void print_d(GraphStruct*, bool);
__global__ void applyBufferWithInboundCountersBitmaps(uint*, bool*, GraphStruct*, unsigned*, unsigned*, unsigned*, bool*, bool*, uint*);
__global__ void colorWithInboundCountersBitmaps(uint*, bool*, GraphStruct*, uint*, uint*, bool*, bool*, uint*, bool*);
__global__ void calculateInbounds(GraphStruct* graphStruct, unsigned int* inboundCounts, unsigned int* priorities, int n);
