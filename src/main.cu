
#include "Colorer.h"
#include "utils/common.h"
#include "graph/graph.h"
#include <iostream>

int main(void) {
	unsigned int n = 1000;		 // number of nodes for random graphs
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

	// GPU Luby-JP greedy coloring
	//Coloring* coloring = RandomPriorityColoring(graphStruct);
	Colorer colorer(&graph);
	Coloring* coloring2 = colorer.LDFColoring();
	//printColoring(coloring, graphStruct, 1);
	printColoring(coloring2, graphStruct, 1);

	return EXIT_SUCCESS;
}


// il bool memory lo metto nel randgrapg
