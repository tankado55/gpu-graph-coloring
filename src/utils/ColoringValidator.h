#include "../graph/graph.h"
#include "../Colorer.h"

void validateColoring(Coloring* coloring, GraphStruct* graph)
{
	int n = graph->nodeCount;
	for (int i = 0; i < n; ++i)
	{
		int offset = graph->neighIndex[i];
		int deg = graph->deg(i);
		int color = coloring->coloring[i];
		for (int j = 0; j < deg; ++j)
		{
			int neigh = graph->neighs[offset + j];
			int neighColor = coloring->coloring[neigh];
			if (color == neighColor) {
				printf("coloring FAILED, node: [%d] is neigh of [%d] with color [%d] \n", i, neigh, color);
			}
		}
	}
}