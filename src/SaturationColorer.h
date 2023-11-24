#pragma once
#include "Colorer.h"

__global__ void applyBufferSaturation(uint* coloring, bool* isColored, GraphStruct* graphStruct, uint* buffer,
	bool* filledBuffer, uint* priorities, bool* bitmaps, uint* bitmapIndex, unsigned n);
__global__ void updatePriorities(bool* isColored, GraphStruct* graphStruct, uint* priorities, bool* bitmaps, uint* bitmapIndex, unsigned n);

namespace SaturationColorer
{
	Coloring* color(Graph& graph);
};