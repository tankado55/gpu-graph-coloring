
#pragma once

#include <curand_kernel.h>
#include <bitset>
#include "graph/graph.h"
#include "utils/common.h"

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

// probabilmente lo cancello, non ha molto senso tenersi lo stato dei colori fuori dalla GPU
struct ColoringUtils {
	int* availableColorIndex{ nullptr };
	int* availableColors{ nullptr };
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

Coloring* RandomPriorityColoringCPUSequentialV2();
Coloring* RandomPriorityColoring(Graph& graph);

void test(Graph& graph);
Coloring* RandomPriorityColoringV2(Graph& graph);
Coloring* RandomPriorityColoringV3(Graph& graph);
Coloring* DegreePriorityColoringV3(GraphStruct*, int, int, priorityEnum);
void printColoring(Coloring*, GraphStruct*, bool);
__global__ void InitRandomPriorities(uint seed, curandState_t*, uint*, uint);
__global__ void findIS(Coloring*, GraphStruct*, uint*);
__global__ void print_d(GraphStruct*, bool);
__global__ void applyBufferWithInboundCountersBitmaps(uint*, bool*, GraphStruct*, unsigned*, unsigned*, unsigned*, bool*, bool*, uint*);
__global__ void colorWithInboundCountersBitmaps(uint*, bool*, GraphStruct*, uint*, uint*, bool*, bool*, uint*, bool*);