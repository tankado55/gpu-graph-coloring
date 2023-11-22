#pragma once
#include "Colorer.h"

class SequentialGreedyColorer : Colorer
{
public:
	static Coloring* color(Graph& graph);
};