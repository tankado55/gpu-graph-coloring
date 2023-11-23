#pragma once

#include "Colorer.h"

class LargestDegreeFirst : public Colorer
{

private:
	uint* calculatePriority(Graph& graph, GraphStruct* d_graphStruct);

};
