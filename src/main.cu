
#include <iostream>
#include "Colorer.h"
#include "utils/common.h"
#include "graph/graph.h"
#include "utils/ColoringValidator.h"
#include "IncidenceColorer.h"
#include "SaturationColorer.h"
#include "SmallestDegreeLast.h"

int main(void) {
	unsigned int n = 20000;
	float prob = .018;				    // density (percentage) for random graphs
	std::default_random_engine engine{ 0 };  // fixed seed

	// new graph with n nodes
	Graph graph(Graph::MemoryEnum::HostAllocated);
	graph.ReadFromMtxFile("inputData/soc-youtube-snap/soc-youtube-snap.mtx");
	//graph.ReadFromMtxFile("inputData/kron_g500-logn21/kron_g500-logn21.mtx");

	// generate a random graph
	//graph.randGraph(prob, engine, n);

	// get the graph struct
	GraphStruct* graphStruct = graph.getStruct();

	printf("start, edgeCount: %d\n", graphStruct->edgeCount);
	printf("start, nodeCount: %d\n", graphStruct->nodeCount);

	// print small graph
	if (n <= 128) {
		//graph.print(true);  // CPU print
		print_d <<<1, 1 >>> (graphStruct, true);  // GPU print
		cudaDeviceSynchronize();
	}

	//-------------- START TIME ----------------//
	

	
	//Colorer colorer(&graph);
	//Coloring* coloring = RandomPriorityColoring(graph);     // 0.375 20k 1.509 no inbound
	//Coloring* coloring = RandomPriorityColoringV2(graph); // 0.352     20k 1.424 con inbounds 0.97 msi
	//Coloring* coloring = RandomPriorityColoringV3(graph);
	priorityEnum priorityEnum = LDF;
	//Coloring* coloring = DegreePriorityColoringV3(graph, priorityEnum);              // bitmaps 20k 0.018 1.029sec/0.914sec
	//Coloring* coloring = IncidenceColorer::color(graph);
	//Coloring* coloring = SaturationColorer::color(graph);
	Coloring* coloring = SmallestDegreeLast::color(graph);
	//test(graph);

	
	//-------------- END TIME ----------------//

	//printColoring(coloring, graphStruct, 1);
	std::cout << "Iterations: " << coloring->iterationCount << std::endl;

	validateColoring(coloring, graphStruct);

	return EXIT_SUCCESS;
}
