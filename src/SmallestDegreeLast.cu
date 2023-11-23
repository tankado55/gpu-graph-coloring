#include <iostream>
#include "SmallestDegreeLast.h"
#include "utils/common.h"
#include "utils/MyDebug.h"

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
	// Init
	int n = graph.GetNodeCount();
	int edgeCount = graph.GetEdgeCount();

	std::cout << "Copying graph to device ..." << std::endl;
	GraphStruct* d_graphStruct;
	graph.copyToDevice(d_graphStruct);

	// Alloc and Init returning struct
	double start = seconds();
	uint* coloring = (uint*)malloc(n * sizeof(uint));
	bool* coloredNodes = (bool*)malloc(n * sizeof(bool));
	memset(coloring, 0, n * sizeof(uint));
	memset(coloredNodes, 0, n * sizeof(bool));
	uint* d_coloring;
	bool* d_coloredNodes;
	CHECK(cudaMalloc((void**)&(d_coloring), n * sizeof(uint)));
	CHECK(cudaMalloc((void**)&(d_coloredNodes), n * sizeof(bool)));
	cudaMemcpy(d_coloring, coloring, n * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(d_coloredNodes, coloredNodes, n * sizeof(bool), cudaMemcpyHostToDevice);

	// Generate priorities using degrees
	uint* d_priorities = calculatePriority(graph, d_graphStruct);

	// Calculate inbound counters
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);
	uint* inboundCounts;
	CHECK(cudaMalloc((void**)&inboundCounts, n * sizeof(uint)));
	cudaMemset(inboundCounts, 0, n * sizeof(uint));
	calculateInbounds <<<gridDim, blockDim >> > (d_graphStruct, inboundCounts, d_priorities, n);
	cudaDeviceSynchronize();

	// inizialize bitmaps, every node has a bitmap with a length of inbound edges + 1 TODO: alloc on gpu
	// vision: allocare tutto in un array come al solito ma serve la prefix sum
	// alternativa1: sequenziale O(n)
	uint bitCount = (n + (int)(edgeCount + 1) / 2);
	uint* bitmapIndex = (uint*)malloc((n + 1) * sizeof(uint));
	uint* h_InboundCounts;
	h_InboundCounts = (uint*)malloc(n * sizeof(uint));
	cudaMemcpy(h_InboundCounts, inboundCounts, n * sizeof(uint), cudaMemcpyDeviceToHost);
	bitmapIndex[0] = 0;
	for (int i = 1; i < n + 1; i++)
		bitmapIndex[i] = bitmapIndex[i - 1] + h_InboundCounts[i - 1] + 1; //the inbound should be only in gpu mem TODO: parallelize with scan
	bool* d_bitmaps;
	uint* d_bitmapIndex;
	CHECK(cudaMalloc((void**)&d_bitmaps, bitCount * sizeof(bool)));
	CHECK(cudaMalloc((void**)&d_bitmapIndex, (n + 1) * sizeof(uint)));
	cudaMemset(d_bitmaps, 1, bitCount * sizeof(bool));
	cudaMemcpy(d_bitmapIndex, bitmapIndex, (n + 1) * sizeof(uint), cudaMemcpyHostToDevice);
	delete(bitmapIndex);

	// Alloc buffer needed to synchronize the coloring
	unsigned* buffer;
	cudaMalloc((void**)&buffer, n * sizeof(unsigned));
	cudaMemset(buffer, 0, n * sizeof(unsigned));
	bool* filledBuffer;
	cudaMalloc((void**)&filledBuffer, n * sizeof(bool));
	cudaMemset(filledBuffer, 0, n * sizeof(bool));

	// Color TODO: tieni il flag sulla gpu e itera con gli stream
	int iterationCount = 0;
	bool* uncoloredFlag = (bool*)malloc(sizeof(bool));
	*uncoloredFlag = true;
	bool* d_uncoloredFlag;
	cudaMalloc((void**)&d_uncoloredFlag, sizeof(bool));
	double stop = seconds();
	std::cout << "Initialization: " << elapsedTime(start, stop) << std::endl;
	start = seconds();
	while (*uncoloredFlag) {
		*uncoloredFlag = false;
		cudaMemcpy(d_uncoloredFlag, uncoloredFlag, sizeof(bool), cudaMemcpyHostToDevice);
		colorWithInboundCountersBitmaps << <gridDim, blockDim >> > (d_coloring, d_coloredNodes, d_graphStruct, inboundCounts, buffer, filledBuffer, d_bitmaps, d_bitmapIndex, d_uncoloredFlag);
		cudaDeviceSynchronize();
		applyBufferWithInboundCountersBitmaps << <gridDim, blockDim >> > (d_coloring, d_coloredNodes, d_graphStruct, d_priorities, inboundCounts, buffer, filledBuffer, d_bitmaps, d_bitmapIndex);
		cudaDeviceSynchronize();
		cudaMemcpy(uncoloredFlag, d_uncoloredFlag, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		iterationCount++;
	}
	stop = seconds();
	std::cout << "Processing: " << elapsedTime(start, stop) << std::endl;


	// Free
	cudaFree(d_priorities);
	cudaFree(inboundCounts);
	cudaFree(buffer);
	cudaFree(filledBuffer);

	//cudaMemcpy(coloring, d_coloring, sizeof(Coloring), cudaMemcpyDeviceToHost);
	cudaMemcpy(coloring, d_coloring, n * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(coloredNodes, d_coloredNodes, n * sizeof(bool), cudaMemcpyDeviceToHost);

	Coloring* coloringStruct = (Coloring*)malloc(sizeof(Coloring));
	coloringStruct->coloring = coloring;
	coloringStruct->coloredNodes = coloredNodes;
	coloringStruct->iterationCount = iterationCount;
	return coloringStruct;
}


