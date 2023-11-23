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
