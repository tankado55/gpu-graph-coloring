#pragma once
#include "Colorer.h"

class SmallestDegreeLast : Colorer
{

private:
	static uint* calculatePriority(Graph& graph, GraphStruct* d_graphStruct);

public:
	static Coloring* color(Graph& graph);
};