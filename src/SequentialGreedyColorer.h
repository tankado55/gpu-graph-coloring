#pragma once
#include "Colorer.h"

class SequentialGreedyColorer : public Colorer
{
private:
	uint* calculatePriority(Graph& graph, GraphStruct* d_graphStruct) override;
public:
	Coloring* color(Graph& graph);
};