#pragma once
#include "Colorer.h"

namespace SequentialGreedyColorer
{
	uint* calculatePriority(Graph& graph, GraphStruct* d_graphStruct);
	Coloring* color(Graph& graph);
};