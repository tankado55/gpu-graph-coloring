#include <stdio.h>
#include <iostream>
#include "graph.h"
#include "../common.h"

using namespace std;

/**
 * Set the CUDA Unified Memory for nodes and edges
 * @param memType node or edge memory type
 */
void Graph::memsetGPU(node_sz n, string memType) {
	if (!memType.compare("nodes")) {
		CHECK(cudaMallocManaged(&graphStruct, sizeof(GraphStruct)));
		CHECK(cudaMallocManaged(&(graphStruct->cumDegs), (n+1)*sizeof(node)));
		CHECK(cudaMallocManaged(&(graphStruct->inCount), n * sizeof(uint)));
	}
	else if (!memType.compare("edges")) {
		CHECK(cudaMallocManaged(&(graphStruct->neighs), graphStruct->edgeSize*sizeof(node)));
	}
}

/**
 * Print the graph on device (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
__global__ void print_d(GraphStruct* graphStruct, bool verbose) {
	printf("** Graph (num node: %d, num edges: %d)\n", graphStruct->nodeCount,graphStruct->edgeSize);

	if (verbose) {
		for (int i = 0; i < graphStruct->nodeCount; i++) {
			printf(" node(%d)[%d]-> ",i,graphStruct->cumDegs[i+1]-graphStruct->cumDegs[i]);
			for (int j = 0; j < graphStruct->cumDegs[i+1] - graphStruct->cumDegs[i]; j++) {
				printf("%d ", graphStruct->neighs[graphStruct->cumDegs[i]+j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}


