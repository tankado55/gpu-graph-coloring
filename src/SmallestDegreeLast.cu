#include <iostream>
#include "SmallestDegreeLast.h"
#include "utils/common.h"

__global__ void assignPrioritySmallestDesgreeLast(uint* priorities, GraphStruct* graphStruct, double avgDeg, uint priority, int* remainingCount, int* sumDeg)
{
    uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= graphStruct->nodeCount)
		return;
    if (priorities[idx])
        return;

    uint deg = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];

	uint offset = graphStruct->neighIndex[idx];
	uint currentDeg = 0;
	for (uint i = 0; i < deg; ++i)
	{
		int neighId = graphStruct->neighs[offset + i];
		if (priorities[neighId] == 0 || priorities[neighId] == priority)
		{
			currentDeg++;
		}
	}

    if (currentDeg <= avgDeg)
    {
        priorities[idx] = priority;
        atomicSub(remainingCount, 1);
        //atomicSub(sumDeg, deg); // not properly correct, possible solution: keep a buffer of current deg and sum it in parallel after each step
    }
}


uint* SmallestDegreeLast::calculatePriority(Graph& graph, GraphStruct* d_graphStruct)
{
    int n = graph.GetNodeCount();
    dim3 blockDim(THREADxBLOCK);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);

    double avgDeg = 0;
    
    int* sumDeg;
    CHECK(cudaMallocManaged(&(sumDeg), sizeof(int)));
    *sumDeg = graph.GetEdgeCount();

    int* remainingCount;
    CHECK(cudaMallocManaged(&(remainingCount), sizeof(int)));
    *remainingCount = graph.GetNodeCount();

    uint* d_priorities;
    cudaMalloc((void**)&d_priorities, graph.GetNodeCount() * sizeof(uint));

    int i = 1;
    while (*remainingCount > 0) // TODO: if I don't use the average I can use a flag
    {
        //avgDeg = *sumDeg / *remainingCount;
		avgDeg++;
		while (true)
		{
			int prevRemainingCount = *remainingCount;
			assignPrioritySmallestDesgreeLast << <gridDim, blockDim >> > (d_priorities, d_graphStruct, avgDeg, i, remainingCount, sumDeg);
			cudaDeviceSynchronize();
			if (prevRemainingCount == *remainingCount)
				break;
			++i;
		}
    }
    return d_priorities;
}

Coloring* SmallestDegreeLast::color(Graph& graph)
{
    GraphStruct* d_graphStruct;
    graph.copyToDevice(d_graphStruct);
    uint* d_priorities = calculatePriority(graph, d_graphStruct);
    Coloring* coloring = global::color(graph, d_priorities);
    cudaFree(d_priorities);
    return coloring;
}


