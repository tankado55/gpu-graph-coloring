#pragma once
#include "Colorer.h"

class RandomPriorityColorer : Colorer
{

private:
	uint* calculatePriority(Graph& graph, GraphStruct* d_graphStruct);

public:
	Coloring* color(Graph& graph);
};