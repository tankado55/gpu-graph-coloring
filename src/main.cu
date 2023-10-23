
#include "Colorer.h"
#include "utils/common.h"
#include "graph/graph.h"
#include "utils/ColoringValidator.h"
#include <iostream>

int main(void) {
	unsigned int n = 10000;		 // 80k ok 115 mln edge
	float prob = .018;				    // density (percentage) for random graphs
	std::default_random_engine engine{ 0 };  // fixed seed

	// new graph with n nodes
	Graph graph(Graph::MemoryEnum::ManagedAllocated);
	//graph.ReadFromMtxFile("inputData/kron_g500-logn21.mtx");
	//GraphStruct* d_GraphStruct;
	//graph.copyToDevice(d_GraphStruct);

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
	//Coloring* coloring = RandomPriorityColoring(graph);     // 0.375 20k 1.509 no inbound
	//Coloring* coloring = RandomPriorityColoringV2(graph); // 0.352     20k 1.424 con inbounds 0.97 msi
	//Coloring* coloring = RandomPriorityColoringV3(graph); //                                0.96 72 colors
	Coloring* coloring = LDFColoringV3(graph);              //                                     70 colors
	//test(graph);

	double stop = seconds();
	//-------------- END TIME ----------------//

	//printColoring(coloring, graphStruct, 1);
	std::cout << "Iterations: " << coloring->iterationCount << std::endl;

	std::cout << elapsedTime(start, stop) << std::endl;

	validateColoring(coloring, graphStruct);

	return EXIT_SUCCESS;
}
