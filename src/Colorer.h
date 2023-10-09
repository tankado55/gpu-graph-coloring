
#pragma once

#include <curand_kernel.h>
#include <bitset>
#include "graph/graph.h"
#include "utils/common.h"

struct Coloring {
	bool uncoloredFlag;
	uint numOfColors;
	uint* coloring;   // each element denotes the color of the node at the correspondent index
	bool* coloredNodes;
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
Coloring* RandomPriorityColoring(GraphStruct*);
void printColoring(Coloring*, GraphStruct*, bool);
__global__ void init(uint seed, curandState_t*, uint*, uint);
void LubyJPcolorer(Coloring*, GraphStruct*, uint*);
__global__ void findIS(Coloring*, GraphStruct*, uint*);
__global__ void print_d(GraphStruct*, bool);