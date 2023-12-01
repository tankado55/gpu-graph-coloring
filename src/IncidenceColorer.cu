#include <iostream>
#include "IncidenceColorer.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>

__global__ void applyBufferIncidence(uint* coloring, bool* isColored, GraphStruct* graphStruct, uint* buffer, 
	bool* filledBuffer, uint* priorities, bool* bitmaps, uint* bitmapIndex, unsigned n)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n)
		return;

	if (isColored[idx])
		return;

	if (!filledBuffer[idx])
		return;

	coloring[idx] = buffer[idx];
	isColored[idx] = true;
	filledBuffer[idx] = false;

	uint offset = graphStruct->neighIndex[idx];
	uint deg = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];

	for (uint i = 0; i < deg; i++)
	{
		uint neighID = graphStruct->neighs[offset + i];

		atomicAdd(&priorities[neighID], 1);
		int neighColorCount = bitmapIndex[neighID + 1] - bitmapIndex[neighID];
		if (buffer[idx] < neighColorCount)
			bitmaps[bitmapIndex[neighID] + buffer[idx]] = 0;
	}
}

Coloring* IncidenceColorer::color(Graph& graph)
{
	setStartTime();

	// Init
	unsigned n = graph.GetNodeCount();
	int edgeCount = graph.GetEdgeCount();
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);
	GraphStruct* d_graphStruct;
	graph.getDeviceStruct(d_graphStruct);

	// Alloc and Init returning struct
	double start = seconds();
	uint* coloring = (uint*)malloc(n * sizeof(uint));
	bool* isColored = (bool*)malloc(n * sizeof(bool));
	memset(coloring, 0, n * sizeof(uint));
	memset(isColored, 0, n * sizeof(bool));
	uint* d_coloring;
	bool* d_isColored;
	CHECK(cudaMalloc((void**)&(d_coloring), n * sizeof(uint)));
	CHECK(cudaMalloc((void**)&(d_isColored), n * sizeof(bool)));
	cudaMemcpy(d_coloring, coloring, n * sizeof(uint), cudaMemcpyHostToDevice);
	cudaMemcpy(d_isColored, isColored, n * sizeof(bool), cudaMemcpyHostToDevice);

	// Incidence priority
	uint* d_priorities;
	cudaMalloc((void**)&d_priorities, n * sizeof(uint));
	cudaMemset(d_priorities, 0, n * sizeof(uint));

	// inizialize bitmaps
	bool* bitmaps;
	uint bitCount = (n + edgeCount);
	CHECK(cudaMallocManaged(&(bitmaps), bitCount * sizeof(bool)));
	memset(bitmaps, 1, bitCount * sizeof(bool));
	uint* bitmapIndex;
	CHECK(cudaMallocManaged(&bitmapIndex, (n + 1) * sizeof(uint)));
	bitmapIndex[0] = 0;
	GraphStruct* graphStruct = graph.getStruct();
	for (int i = 1; i < n + 1; i++) // can be paralelized with a scan
	{
		int prevDeg = graphStruct->neighIndex[i] - graphStruct->neighIndex[i - 1];
		bitmapIndex[i] = bitmapIndex[i - 1] + prevDeg + 1;
	}

	// Alloc buffer needed to synchronize the coloring
	uint* buffer;
	cudaMalloc((void**)&buffer, n * sizeof(uint));
	cudaMemset(buffer, 0, n * sizeof(uint));
	bool* filledBuffer;
	cudaMalloc((void**)&filledBuffer, n * sizeof(bool));
	cudaMemset(filledBuffer, 0, n * sizeof(bool));

	// Main algo
	int iterationCount = 0;
	bool* uncoloredFlag = (bool*)malloc(sizeof(bool));
	*uncoloredFlag = true;
	bool* d_uncoloredFlag;
	cudaMalloc((void**)&d_uncoloredFlag, sizeof(bool));
	double lap = getLapTime();
	std::cout << "Initialization: " << lap << std::endl;
	start = seconds();
	while (*uncoloredFlag) {
		*uncoloredFlag = false;
		cudaMemcpy(d_uncoloredFlag, uncoloredFlag, sizeof(bool), cudaMemcpyHostToDevice);
		colorWithoutInbounds <<<gridDim, blockDim >>> (d_isColored, d_graphStruct, buffer, filledBuffer, bitmaps, bitmapIndex, d_priorities, d_uncoloredFlag);
		cudaDeviceSynchronize();
		cudaMemcpy(uncoloredFlag, d_uncoloredFlag, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		applyBufferIncidence <<<gridDim, blockDim >>> (
			d_coloring, d_isColored, d_graphStruct, buffer, filledBuffer, d_priorities, bitmaps, bitmapIndex, n);
		cudaDeviceSynchronize();
		cudaMemcpy(uncoloredFlag, d_uncoloredFlag, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		iterationCount++;
	}
	lap = getLapTime();
	std::cout << "Processing: " << lap << std::endl;

	//copy and build results
	cudaMemcpy(coloring, d_coloring, n * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(isColored, d_isColored, n * sizeof(bool), cudaMemcpyDeviceToHost);
	Coloring* coloringStruct = (Coloring*)malloc(sizeof(Coloring));
	coloringStruct->coloring = coloring;
	coloringStruct->coloredNodes = isColored;
	coloringStruct->iterationCount = iterationCount;

	// Free
	cudaFree(buffer);
	cudaFree(filledBuffer);
	cudaFree(d_coloring);

	return coloringStruct;
}
