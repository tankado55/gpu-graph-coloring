
#include "Colorer.h"
#include "utils/common.h"
#include "graph/graph.h"
#include <iostream>

int main(void) {
	unsigned int n = 10000;		 // number of nodes for random graphs
	float prob = .02;				    // density (percentage) for random graphs
	std::default_random_engine engine{ 0 };  // fixed seed

	// new graph with n nodes
	Graph graph(Graph::MemoryEnum::ManagedAllocated);

	// generate a random graph
	graph.randGraph(prob, engine, n);

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
	double start = seconds();

	
	//Colorer colorer(&graph);
	//Coloring* coloring = RandomPriorityColoring(graphStruct); //0.35 10000 .02  msi: 0.532
	Coloring* coloring = RandomPriorityColoringV2(graph); // 
	//Coloring* coloring = colorer.LDFColoring(); //2.585 10000 .02
	//printColoring(coloring, graphStruct, 1);

	double stop = seconds();
	//-------------- END TIME ----------------//

	printColoring(coloring, graphStruct, 1);

	std::cout << elapsedTime(start, stop);

	return EXIT_SUCCESS;
}
