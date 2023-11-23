
#include "device_launch_parameters.h"
#include <iostream>
#include "graph/graph_d.h"
#include "utils/common.h"
#include <cooperative_groups.h>
#include "colorer.h"




void mallocOnHost(Coloring* coloring, unsigned n)
{
	coloring = (Coloring*)malloc(sizeof(Coloring));
	coloring->coloring = (uint*)calloc(n, sizeof(uint));
	coloring->coloredNodes = (bool*)calloc(n, sizeof(bool));
}

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


__global__ void applyBuffer(Coloring* coloring, unsigned* buffer, bool* filledBuffer, unsigned n)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= n)
		return;

	if (coloring->coloredNodes[idx])
		return;

	if (!filledBuffer[idx])
		return;

	coloring->coloring[idx] = buffer[idx];
	coloring->coloredNodes[idx] = true;
	filledBuffer[idx] = false;
	//printf("buffer applied: %d, color: %d\n", idx, coloring->coloring[idx]);

}

__global__ void applyBufferWithInboundCounters(Coloring* coloring, GraphStruct* graphStruct, unsigned* priorities, unsigned* inboundCounts,unsigned* buffer, bool* filledBuffer)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graphStruct->nodeCount)
		return;

	if (coloring->coloredNodes[idx])
		return;

	if (!filledBuffer[idx])
		return;

	uint offset = graphStruct->neighIndex[idx];
	uint deg = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];

	for (uint i = 0; i < deg; i++)
	{
		//TODO: check if there is arc in the dag, it could cause troubles when I will implement shortcuts
		uint neighID = graphStruct->neighs[offset + i];
		if (!coloring->coloredNodes[neighID] &&
			((priorities[idx] > priorities[neighID]) || ((priorities[idx] == priorities[neighID]) && idx > neighID)))
		{
			atomicAdd(&inboundCounts[neighID], -1);
			//if (neighID == 750)
			//	printf("I'm: %d, removed arc to: %d: \n", idx, neighID);
		}
		//printf("I'm: %d, removed arc to: %d: \n", idx, neighID);
		
	}
	coloring->coloring[idx] = buffer[idx];
	coloring->coloredNodes[idx] = true;
	filledBuffer[idx] = false;
	//printf("buffer applied: %d, color: %d\n", idx, coloring->coloring[idx]);

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
		//if (idx == 0)
		//{
		//	printf("GPU - I'm %d, flag true, still: %d\n", idx, inboundCounts[idx]);
		//}
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

			//if (neighID == 18836) {
			//	printf("I'm: %d, ---------removed arc to: %d, still: %d\n", idx, neighID, inboundCounts[neighID]);
			//	printf("%d, %d, %d, %d, %d\n", bitmaps[bitmapIndex[neighID] + 0], bitmaps[bitmapIndex[neighID] + 1], bitmaps[bitmapIndex[neighID] + 2], bitmaps[bitmapIndex[neighID] + 3], bitmaps[bitmapIndex[neighID] + 4]);
			//}
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
