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
