
#pragma once

#include <curand_kernel.h>
#include <bitset>
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

class Colorer
{
private:

	Coloring m_Coloring;
	GraphStruct* m_GraphStruct;

public:
	Colorer(GraphStruct*);
	Coloring* LDFColoring();
	~Colorer();
};

Coloring* RandomPriorityColoring(GraphStruct*);
void printColoring(Coloring*, GraphStruct*, bool);
__global__ void init(uint seed, curandState_t*, uint*, uint);
void LubyJPcolorer(Coloring*, GraphStruct*, uint*);
__global__ void findIS(Coloring*, GraphStruct*, uint*);
__global__ void print_d(GraphStruct*, bool);