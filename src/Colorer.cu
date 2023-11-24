
#include "device_launch_parameters.h"
#include <iostream>
#include "graph/graph_d.h"
#include "utils/common.h"
#include <cooperative_groups.h>
#include "colorer.h"

//TODO: evita atomic add facendo il confronto al contrario
__global__ void calculateInbounds(GraphStruct* graphStruct, unsigned int* inboundCounts, unsigned int* priorities, int n) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;

	uint degree = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];
	//printf("node(%d [myDegree: %d] \n", idx, degree);

	//inboundCounts[idx] = 0;
	for (uint i = 0; i < degree; ++i)
	{
		uint neighID = graphStruct->neighs[graphStruct->neighIndex[idx] + i];
		if (priorities[idx] > priorities[neighID])
		{
			atomicAdd(&inboundCounts[neighID], 1);
		}
		else if (priorities[idx] == priorities[neighID] && idx > neighID)
		{
			atomicAdd(&inboundCounts[neighID], 1);
		}
	}
}

__global__ void colorWithInboundCountersBitmaps(uint* coloring, bool* coloredNodes, GraphStruct* graphStruct, uint* inboundCounts, uint* buffer, bool* filledBuffer, bool* bitmaps, uint* bitmapIndex, bool* uncoloredFlag)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	//printf("GPU - I'm %d, test: %d\n", idx, *uncoloredFlag);
	if (idx >= graphStruct->nodeCount)
		return;
	//printf("GPU - I'm %d, colored?: %d\n", idx, coloredNodes[idx]);
	if (coloredNodes[idx])
		return;

	//printf("GPU - I'm %d, myInbound: %d\n", idx, inboundCounts[idx]);

	//if (idx == 8984)
	//	printf("uncolored: %d, still: %d\n", idx, inboundCounts[idx]);
	if (inboundCounts[idx] == 0) // Ready node
	{
		int colorCount = bitmapIndex[idx + 1] - bitmapIndex[idx];
		//if (idx == 18836)
		//	printf("I'm %d, total colors: %d\n", idx, colorCount);

		int bestColor = colorCount;
		for (int i = 0; i < colorCount; ++i)
		{
			if (bitmaps[bitmapIndex[idx] + i])
			{
				bestColor = i;
				//if (idx == 18836)
				//	printf("I'm: %d, ---------best color: %d\n", idx, bestColor);
				break;
			}
		}
		buffer[idx] = bestColor;
		filledBuffer[idx] = true;
		//printf("I'm %d, filled buffer: %d\n", idx, bestColor);
	}
	else
	{
		*uncoloredFlag = true;
	}
}

__global__ void applyBufferWithInboundCountersBitmaps(uint* coloring, bool* coloredNodes, GraphStruct* graphStruct, unsigned* priorities, unsigned* inboundCounts, unsigned* buffer, bool* filledBuffer, bool* bitmaps, uint* bitmapIndex)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graphStruct->nodeCount)
		return;

	if (coloredNodes[idx])
		return;

	if (!filledBuffer[idx])
		return;

	uint offset = graphStruct->neighIndex[idx];
	uint deg = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];

	for (uint i = 0; i < deg; i++)
	{
		uint neighID = graphStruct->neighs[offset + i];

		if (priorities[idx] > priorities[neighID])
		{
			atomicAdd(&inboundCounts[neighID], -1);
			int colorCount = bitmapIndex[neighID + 1] - bitmapIndex[neighID];
			if (buffer[idx] < colorCount)
				bitmaps[bitmapIndex[neighID] + buffer[idx]] = 0;
		}
		else if (priorities[idx] == priorities[neighID] && idx > neighID)
		{
			atomicAdd(&inboundCounts[neighID], -1);
			int colorCount = bitmapIndex[neighID + 1] - bitmapIndex[neighID];
			if (buffer[idx] < colorCount)
				bitmaps[bitmapIndex[neighID] + buffer[idx]] = 0;

			//if (neighID == 18836) {
			//	printf("I'm: %d, ---------removed arc to: %d, still: %d\n", idx, neighID, inboundCounts[neighID]);
			//	printf("%d, %d, %d, %d, %d\n", bitmaps[bitmapIndex[neighID] + 0], bitmaps[bitmapIndex[neighID] + 1], bitmaps[bitmapIndex[neighID] + 2], bitmaps[bitmapIndex[neighID] + 3], bitmaps[bitmapIndex[neighID] + 4]);
			//}
		}
		else {
			//if (neighID == 3)
			//	printf("I'm: %d, ---------NOT removed arc to: %d, still: %d\n", idx, neighID, inboundCounts[neighID]);
		}

	}
	//if (idx == 18836)
	//	printf("I'm: %d, colored: %d \n", idx, buffer[idx]);

	coloring[idx] = buffer[idx];
	coloredNodes[idx] = true;
	filledBuffer[idx] = false;
	//printf("buffer applied: from %d, color: %d\n", idx, coloring[idx]);
}

__global__ void colorWithoutInbounds(bool* isColored, GraphStruct* graphStruct, uint* buffer, bool* filledBuffer, bool* bitmaps, uint* bitmapIndex, uint* priorities, bool* uncoloredFlag)
{
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

/**
 * Print the graph (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
void printColoring(Coloring* col, GraphStruct* graphStruct, bool verbose) {
	unsigned n = graphStruct->nodeCount;
	std::cout << "** Graph (num node: " << n << ", num edges: " << graphStruct->edgeCount << ")" << std::endl;
	std::cout << "** Coloring (num colors: " << col->iterationCount + 1 << ")" << std::endl;
	if (verbose) {
		for (uint i = 0; i <= col->iterationCount; i++) {
			std::cout << "   color(" << i << ")" << "-> ";
			for (uint j = 0; j < n; j++)
				if (col->coloring[j] == i)
					std::cout << j << " ";
			std::cout << "\n";
		}
		std::cout << "\n";
	}
}

Coloring* global::color(Graph& graph, uint* d_priorities)
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

	// Calculate inbound counters
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);
	uint* inboundCounts;
	CHECK(cudaMalloc((void**)&inboundCounts, n * sizeof(uint)));
	cudaMemset(inboundCounts, 0, n * sizeof(uint));
	calculateInbounds << <gridDim, blockDim >> > (d_graphStruct, inboundCounts, d_priorities, n);
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
