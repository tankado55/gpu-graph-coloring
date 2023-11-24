#pragma once
#include "Colorer.h"

__global__ void InitRandomPriorities(uint seed, curandState_t* states, uint* priorities, uint n);

namespace RandomPriorityColorer
{
	uint* calculatePriority(Graph& graph, GraphStruct* d_graphStruct);
	Coloring* color(Graph& graph);
};

