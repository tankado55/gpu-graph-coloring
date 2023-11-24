#pragma once

#include "Colorer.h"

namespace LargestDegreeFirst
{
	uint* calculatePriority(Graph& graph, GraphStruct* d_graphStruct);
	Coloring* color(Graph& graph);
};

__global__ void InitLDFPriorities(GraphStruct* graphStruct, uint* priorities, uint n);
