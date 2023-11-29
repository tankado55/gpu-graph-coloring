#include "LargestDegreeFirst.h"

__global__ void InitLDFPriorities(GraphStruct* graphStruct, uint* priorities, uint n) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;
	priorities[idx] = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];
}

uint* LargestDegreeFirst::calculatePriority(Graph& graph, GraphStruct* d_graphStruct)
{
	int n = graph.GetNodeCount();

	uint* d_priorities;
	cudaMalloc((void**)&d_priorities, n * sizeof(uint));
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);
	InitLDFPriorities << <gridDim, blockDim >> > (d_graphStruct, d_priorities, n);
	cudaDeviceSynchronize();
	return d_priorities;
}

Coloring* LargestDegreeFirst::color(Graph& graph)
{
	setStartTime();

	GraphStruct* d_graphStruct;
	graph.getDeviceStruct(d_graphStruct);
	uint* d_priorities = calculatePriority(graph, d_graphStruct);
	Coloring* coloring = global::color(graph, d_priorities);
	cudaFree(d_priorities);
	return coloring;
}
