#include <iostream>
#include "IncidenceColorer.h"
#include "device_launch_parameters.h"
#include "utils/common.h"
#include <cooperative_groups.h>

__global__ void colorIncidence(bool* isColored, GraphStruct* graphStruct, uint* buffer, bool* filledBuffer, bool* bitmaps, uint* bitmapIndex, uint* priorities, bool* uncoloredFlag)
{//d_coloredNodes, d_graphStruct, buffer, filledBuffer, bitmaps, bitmapIndex, d_priorities, d_uncoloredFlag
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graphStruct->nodeCount)
		return;

	if (isColored[idx])
		return;

	uint offset = graphStruct->neighIndex[idx];
	uint deg = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];
	/*if (idx == 3 || idx == 114) {
		printf("id: %d, priority: %d\n", idx, priorities[idx]);
	}*/

	bool candidate = true;
	for (uint j = 0; j < deg; j++) {
		uint neighID = graphStruct->neighs[offset + j];

		if (!isColored[neighID] &&
			((priorities[idx] < priorities[neighID]) || ((priorities[idx] == priorities[neighID]) && idx < neighID))) {
			candidate = false;
		}
	}
	if (candidate) {
		/*if (idx == 3 || idx == 114) {
			printf("id: %d, CANDIDATE\n", idx);
		}*/
		int colorCount = bitmapIndex[idx + 1] - bitmapIndex[idx];
		int bestColor = 0;
		for (int i = 0; i < colorCount; ++i)
		{
			if (bitmaps[bitmapIndex[idx] + i])
			{
				bestColor = i;
				break;
			}
		}
		buffer[idx] = bestColor;
		filledBuffer[idx] = true;
	}
	else
	{
		*uncoloredFlag = true;
	}
}

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
		/*if (idx == 3 && neighID == 114) {
			printf("id: %d, %d, color: %d, neigh color count: %d\n", idx, neighID, buffer[idx], neighColorCount);
		}*/
	}
	/*if (idx == 3 || idx == 114) {
		printf("id: %d, buffer applied, color: %d, colorCount: %d, Degree: %d\n", idx, coloring[idx], bitmapIndex[idx + 1] - bitmapIndex[idx], graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx]);
	}*/
}

__global__ void applyBufferSaturation(uint* coloring, bool* isColored, GraphStruct* graphStruct, uint* buffer, 
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
		/*if (idx == 3 && neighID == 114) {
			printf("id: %d, %d, color: %d, neigh color count: %d\n", idx, neighID, buffer[idx], neighColorCount);
		}*/
	}
	/*if (idx == 3 || idx == 114) {
		printf("id: %d, buffer applied, color: %d, colorCount: %d, Degree: %d\n", idx, coloring[idx], bitmapIndex[idx + 1] - bitmapIndex[idx], graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx]);
	}*/
}

Coloring* IncidenceColorer::color(Graph& graph)
{
	// Init
	unsigned n = graph.GetNodeCount();
	int edgeCount = graph.GetEdgeCount();
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);

	std::cout << "Copying graph to device ..." << std::endl;
	GraphStruct* d_graphStruct;
	graph.copyToDevice(d_graphStruct);

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

	// inizialize bitmaps, every node has a bitmap with a length of inbound edges + 1 TODO: alloc on gpu
	// vision: allocare tutto in un array come al solito ma serve la prefix sum
	// alternativa1: sequenziale O(n)
	bool* bitmaps;
	uint bitCount = (n + edgeCount);
	CHECK(cudaMallocManaged(&(bitmaps), bitCount * sizeof(bool)));
	memset(bitmaps, 1, bitCount * sizeof(bool));
	uint* bitmapIndex;
	CHECK(cudaMallocManaged(&bitmapIndex, (n + 1) * sizeof(uint)));
	bitmapIndex[0] = 0;
	GraphStruct* graphStruct = graph.getStruct();
	for (int i = 1; i < n + 1; i++)
	{
		int prevDeg = graphStruct->neighIndex[i] - graphStruct->neighIndex[i - 1];
		bitmapIndex[i] = bitmapIndex[i - 1] + prevDeg + 1; //the inbound should be only in gpu mem TODO: parallelize with scan
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
	double stop = seconds();
	std::cout << "Initialization: " << elapsedTime(start, stop) << std::endl;
	start = seconds();
	while (*uncoloredFlag) {
		*uncoloredFlag = false;
		cudaMemcpy(d_uncoloredFlag, uncoloredFlag, sizeof(bool), cudaMemcpyHostToDevice);
		colorIncidence <<<gridDim, blockDim >>> (d_isColored, d_graphStruct, buffer, filledBuffer, bitmaps, bitmapIndex, d_priorities, d_uncoloredFlag);
		cudaDeviceSynchronize();
		cudaMemcpy(uncoloredFlag, d_uncoloredFlag, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		applyBufferIncidence << <gridDim, blockDim >> > (
			d_coloring, d_isColored, d_graphStruct, buffer, filledBuffer, d_priorities, bitmaps, bitmapIndex, n);
		cudaDeviceSynchronize();
		cudaMemcpy(uncoloredFlag, d_uncoloredFlag, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		iterationCount++;
	}
	stop = seconds();
	std::cout << "Processing: " << elapsedTime(start, stop) << std::endl;

	//copy and build results
	cudaMemcpy(coloring, d_coloring, n * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(isColored, d_isColored, n * sizeof(bool), cudaMemcpyDeviceToHost);
	Coloring* coloringStruct = (Coloring*)malloc(sizeof(Coloring));
	coloringStruct->coloring = coloring;
	coloringStruct->coloredNodes = isColored;
	coloringStruct->iterationCount = iterationCount;

	// Free
	cudaFree(d_priorities);
	cudaFree(buffer);
	cudaFree(filledBuffer);

	return coloringStruct;
}


