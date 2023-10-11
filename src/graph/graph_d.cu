#include <stdio.h>
#include <iostream>
#include "graph.h"
#include "../utils/common.h"

using namespace std;

/**
 * Set the CUDA Unified Memory for nodes and edges
 * @param memType node or edge memory type
 */




/**
 * Print the graph on device (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
__global__ void print_d(GraphStruct* graphStruct, bool verbose) {
	printf("** Graph (num node: %d, num edges: %d)\n", graphStruct->nodeCount,graphStruct->edgeCount);

	if (verbose) {
		for (int i = 0; i < graphStruct->nodeCount; i++) {
			printf(" node(%d)[%d]-> ",i,graphStruct->neighIndex[i+1]-graphStruct->neighIndex[i]);
			for (int j = 0; j < graphStruct->neighIndex[i+1] - graphStruct->neighIndex[i]; j++) {
				printf("%d ", graphStruct->neighs[graphStruct->neighIndex[i]+j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}