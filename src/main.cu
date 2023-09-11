
#include "coloring.h"
#include "utils/common.h"
#include <iostream>

int main(void) {
	unsigned int n = 128;		 // number of nodes for random graphs
	float prob = .5;				    // density (percentage) for random graphs
	std::default_random_engine engine{ 0 };  // fixed seed

	// new graph with n nodes
	Graph graph(n, 1);

	// generate a random graph
	graph.randGraph(prob, engine);

	// get the graph struct
	GraphStruct* graphStruct = graph.getStruct();

	// print small graph
	if (n <= 128) {
		//graph.print(true);  // CPU print
		print_d <<<1, 1 >>> (graphStruct, true);  // GPU print
		cudaDeviceSynchronize();
	}

	// GPU Luby-JP greedy coloring
	Coloring* col = LubyGreedy(graphStruct);
	printColoring(col, graphStruct, 1);

	return EXIT_SUCCESS;
}
