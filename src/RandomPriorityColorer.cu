#include <iostream>
#include "RandomPriorityColorer.h"

__global__ void InitRandomPriorities(uint seed, curandState_t* states, uint* priorities, uint n) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;
	curand_init(seed, idx, 0, &states[idx]);
	priorities[idx] = curand(&states[idx]) % n * n;
}

uint* RandomPriorityColorer::calculatePriority(Graph& graph, GraphStruct* d_graphStruct)
{
	int n = graph.GetNodeCount();
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);

	uint* d_priorities;
	cudaMalloc((void**)&d_priorities, graph.GetNodeCount() * sizeof(uint));

	curandState_t* states;
	cudaMalloc((void**)&states, n * sizeof(curandState_t));
	uint seed = 0;

	InitRandomPriorities <<<gridDim, blockDim >>> (seed, states, d_priorities, n);

	return d_priorities;
}

Coloring* RandomPriorityColorer::color(Graph& graph)
{
	setStartTime();

	GraphStruct* d_graphStruct;
	graph.getDeviceStruct(d_graphStruct);
	uint* d_priorities = calculatePriority(graph, d_graphStruct);
	Coloring* coloring = global::color(graph, d_priorities);
	cudaFree(d_priorities);
	return coloring;
}
