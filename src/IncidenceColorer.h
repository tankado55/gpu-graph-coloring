#pragma once

#include "Colorer.h"

__global__ void applyBufferIncidence(uint* coloring, bool* isColored, GraphStruct* graphStruct, uint* buffer,
	bool* filledBuffer, uint* priorities, bool* bitmaps, uint* bitmapIndex, unsigned n);

namespace IncidenceColorer
{
	Coloring* color(Graph& graph);
};