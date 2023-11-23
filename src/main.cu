
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
		mod = 1;
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

	switch (mod)
	{
	case 1:
		SequentialGreedyColorer colorer;
		break;
	default:
		break;
	}

	

	
	//Colorer colorer(&graph);
	//Coloring* coloring = RandomPriorityColoring(graph);     // 0.375 20k 1.509 no inbound
	//Coloring* coloring = RandomPriorityColoringV2(graph); // 0.352     20k 1.424 con inbounds 0.97 msi
	//Coloring* coloring = RandomPriorityColoringV3(graph);
	//Coloring* coloring = DegreePriorityColoringV3(graph, priorityEnum);              // bitmaps 20k 0.018 1.029sec/0.914sec
	//Coloring* coloring = IncidenceColorer::color(graph);
	//Coloring* coloring = SaturationColorer::color(graph);
	//LargestDegreeFirst colorer; //TODO virtual destructor
	Coloring* coloring = colorer->color(graph);
	//Coloring* coloring = SequentialGreedyColorer::color(graph);
	//Coloring* coloring = RandomPriorityColorer::color(graph);
	//test(graph);

	

	//printColoring(coloring, graphStruct, 1);
	std::cout << "Iterations: " << coloring->iterationCount << std::endl;

	validateColoring(coloring, graphStruct);
	free(coloring->coloring);
	free(colorer);
	return EXIT_SUCCESS;
}
