#pragma once
#include "Colorer.h"

__global__ void assignPrioritySmallestDesgreeLast(uint* priorities, GraphStruct* graphStruct, double avgDeg, uint priority, int* remainingCount, int* sumDeg);

namespace SmallestDegreeLast
{
	uint* calculatePriority(Graph& graph, GraphStruct* d_graphStruct);
	Coloring* color(Graph& graph);
}