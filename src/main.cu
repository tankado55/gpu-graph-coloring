
#include <iostream>
#include "Colorer.h"
#include "utils/common.h"
#include "graph/graph.h"
#include "utils/ColoringValidator.h"
#include "IncidenceColorer.h"
#include "SaturationColorer.h"
#include "SmallestDegreeLast.h"
#include "SequentialGreedyColorer.h"
#include "RandomPriorityColorer.h"
#include "LargestDegreeFirst.h"

int main(int argc, char* argv[]) {
	// Random generated graph
	//unsigned int n = 20000;
	//float prob = .018;				    // density (percentage) for random graphs
	//std::default_random_engine engine{ 0 };  // fixed seed
	//graph.randGraph(prob, engine, n);

	const char* path;
	int mod;

	if (argc < 3) {
		path = "inputData/soc-youtube-snap/soc-youtube-snap.mtx";
		mod = 2;
	}
	else
	{
		path = argv[1];
		mod = atoi(argv[2]);
	}

	Graph graph;
	graph.ReadFromMtxFile(path);
	GraphStruct* graphStruct = graph.getStruct();

	printf("start, edgeCount: %d\n", graphStruct->edgeCount);
	printf("start, nodeCount: %d\n", graphStruct->nodeCount);

	Coloring* coloring;
	switch (mod)
	{
	case 1:
		coloring = SequentialGreedyColorer::color(graph);
		break;
	case 2:
		coloring = RandomPriorityColorer::color(graph);
		break;
	case 3:
		coloring = LargestDegreeFirst::color(graph);
		break;
	case 4:
		coloring = SmallestDegreeLast::color(graph);
		break;
	case 5:
		coloring = SaturationColorer::color(graph);
		break;
	case 6:
		coloring = IncidenceColorer::color(graph);
		break;
	default:
		break;
	}

	std::cout << "Iterations: " << coloring->iterationCount << std::endl;
	validateColoring(coloring, graphStruct);

	free(coloring->coloring);
	return EXIT_SUCCESS;
}
