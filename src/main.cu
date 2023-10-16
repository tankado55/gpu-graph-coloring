
#include "Colorer.h"
#include "utils/common.h"
#include "graph/graph.h"
#include "utils/ColoringValidator.h"
#include <iostream>

int main(void) {
	unsigned int n = 17000;		 // number of nodes for random graphs 16k 5.117.258 17k 5.777.572
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
	printf("start, maxDeg: %d\n", graphStruct->maxDeg);

	// print small graph
	if (n <= 128) {
		//graph.print(true);  // CPU print
		print_d <<<1, 1 >>> (graphStruct, true);  // GPU print
		cudaDeviceSynchronize();
	}

	///////////////////DEBUG
	//uint offset = graphStruct->neighIndex[6277];
	//uint deg = graphStruct->neighIndex[6277 + 1] - graphStruct->neighIndex[6277];
	
	//printf("node: %d, deg %d: , index: %d,%d\n", 6277, deg, graphStruct->neighIndex[6277], graphStruct->neighIndex[6277 + 1]);
	//for (int i = 0; i < deg; ++i)
	//{
	//	printf("neigh: %d\n", graphStruct->neighs[offset + i]);
	//}
	

	//offset = graphStruct->neighIndex[9985];
	//deg = graphStruct->neighIndex[9985 + 1] - graphStruct->neighIndex[9985];

	//printf("node: %d, deg: %d\n", 9985, deg);
	//for (int i = 0; i < deg; ++i)
	//{
	//	printf("neigh: %d\n", graphStruct->neighs[offset + i]);
	//}

	//-------------- START TIME ----------------//
	double start = seconds();

	
	//Colorer colorer(&graph);
	//Coloring* coloring = RandomPriorityColoring(graph);     // 0.375 20k 1.509 no inbound
	//Coloring* coloring = RandomPriorityColoringV2(graph); // 0.352     20k 1.424 con inbounds 0.97 msi
	//Coloring* coloring = RandomPriorityColoringV3(graph); //                                0.96 72 colors
	Coloring* coloring = LDFColoringV3(graph);              //                                     70 colors

	double stop = seconds();
	//-------------- END TIME ----------------//

	//printColoring(coloring, graphStruct, 1);
	std::cout << coloring->numOfColors << std::endl;

	std::cout << elapsedTime(start, stop) << std::endl;

	validateColoring(coloring, graphStruct);

	return EXIT_SUCCESS;
}
