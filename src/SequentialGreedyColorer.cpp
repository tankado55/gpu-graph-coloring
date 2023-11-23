#include "SequentialGreedyColorer.h"
#include <vector>
#include <iostream>

uint* SequentialGreedyColorer::calculatePriority(Graph& graph, GraphStruct* d_graphStruct)
{
	return nullptr;
}

Coloring* SequentialGreedyColorer::color(Graph& graph)
{
	double start = seconds();
	// Init
	int n = graph.GetNodeCount();
	int edgeCount = graph.GetEdgeCount();
	GraphStruct* graphStruct = graph.getStruct();
	uint* coloring = (uint*)malloc(n * sizeof(uint));
	bool* coloredNodes = (bool*)malloc(n * sizeof(bool));
	memset(coloring, 0, n * sizeof(uint));
	memset(coloredNodes, 0, n * sizeof(bool));

	coloredNodes[0] = true; // assign first color to first vertex
	double stop = seconds();
	std::cout << "Initialization: " << elapsedTime(start, stop) << std::endl;

	start = seconds();
	for (int i = 1; i < n; ++i)
	{
		int deg = graph.deg(i);
		std::vector<bool> availableColors(deg + 1, 1);

		
		int offset = graphStruct->neighIndex[i];

		// calculate availables colors
		for (int j = 0; j < deg; j++)
		{
			int neighId = graphStruct->neighs[j + offset];
			if (coloredNodes[neighId] == true)
			{
				if (coloring[neighId] < availableColors.size())
				{
					availableColors[coloring[neighId]] = false;
				}
			}
		}
		// best color
		int bestColor = deg+1;
		for (int j = 0; j < deg; j++)
		{
			if (availableColors[j])
			{
				bestColor = j;
				break;
			}
		}
		coloring[i] = bestColor;
		coloredNodes[i] = true;
	}
	Coloring* coloringStruct = (Coloring*)malloc(sizeof(Coloring));
	coloringStruct->coloring = coloring;
	coloringStruct->iterationCount = n;
	stop = seconds();
	std::cout << "Processing: " << elapsedTime(start, stop) << std::endl;
	free(coloredNodes);
	return coloringStruct;
}
