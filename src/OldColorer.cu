Coloring* RandomPriorityColoringV2(Graph& graph) // Versione senza dag, solo con inbound count e senza bitmaps
{
	// Alloc and Init returning struct
	Coloring* coloring;
	int n = graph.getStruct()->nodeCount;
	CHECK(cudaMallocManaged(&coloring, sizeof(Coloring)));
	CHECK(cudaMallocManaged(&(coloring->coloring), n * sizeof(uint)));
	CHECK(cudaMallocManaged(&(coloring->coloredNodes), n * sizeof(bool)));
	memset(coloring->coloring, 0, n * sizeof(uint));
	memset(coloring->coloredNodes, 0, n * sizeof(bool));
	coloring->uncoloredFlag = true;
	coloring->iterationCount = 0;
	GraphStruct* graphStruct = graph.getStruct();

	// Generate random node priorities
	curandState_t* states;
	uint* priorities;
	cudaMalloc((void**)&states, n * sizeof(curandState_t));
	cudaMalloc((void**)&priorities, n * sizeof(uint));
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);
	uint seed = 0;
	InitRandomPriorities << <gridDim, blockDim >> > (seed, states, priorities, n);
	cudaDeviceSynchronize();

	// Calculate inbound counters
	uint* inboundCounts;
	CHECK(cudaMalloc((void**)&inboundCounts, n * sizeof(uint)));
	cudaMemset(inboundCounts, 0, n * sizeof(uint));
	calculateInbounds << <gridDim, blockDim >> > (graphStruct, inboundCounts, priorities, n);
	cudaDeviceSynchronize();

	// Alloc buffer needed to synchronize the coloring
	unsigned* buffer;
	cudaMalloc((void**)&buffer, n * sizeof(unsigned));
	cudaMemset(buffer, 0, n * sizeof(unsigned));
	bool* filledBuffer;
	cudaMalloc((void**)&filledBuffer, n * sizeof(bool));
	cudaMemset(filledBuffer, 0, n * sizeof(bool));

	// Color TODO: tieni il flag sulla gpu e itera con gli stream
	coloring->iterationCount = 0;
	while (coloring->uncoloredFlag) {
		coloring->uncoloredFlag = false;
		colorWithInboundCounters << <gridDim, blockDim >> > (coloring, graphStruct, inboundCounts, buffer, filledBuffer);
		cudaDeviceSynchronize();
		applyBufferWithInboundCounters << <gridDim, blockDim >> > (coloring, graphStruct, priorities, inboundCounts, buffer, filledBuffer);
		cudaDeviceSynchronize();
		coloring->iterationCount++;
	}

	// Free
	cudaFree(states);
	cudaFree(priorities);
	cudaFree(inboundCounts);
	cudaFree(buffer);
	cudaFree(filledBuffer);
	//cudaFree(coloring);
	//cudaFree(coloring->coloring);
	//cudaFree(coloring->coloredNodes);

	return coloring;

}

Coloring* RandomPriorityColoringV3(Graph& graph) // V2 + bitmaps
{
	// Alloc and Init returning struct
	Coloring* coloring;
	int n = graph.getStruct()->nodeCount;
	CHECK(cudaMallocManaged(&coloring, sizeof(Coloring)));
	CHECK(cudaMallocManaged(&(coloring->coloring), n * sizeof(uint)));
	CHECK(cudaMallocManaged(&(coloring->coloredNodes), n * sizeof(bool)));
	memset(coloring->coloring, 0, n * sizeof(uint));
	memset(coloring->coloredNodes, 0, n * sizeof(bool));
	coloring->uncoloredFlag = true;
	coloring->iterationCount = 0;
	GraphStruct* graphStruct = graph.getStruct();

	// Generate random node priorities
	curandState_t* states;
	uint* priorities;
	cudaMalloc((void**)&states, n * sizeof(curandState_t));
	cudaMalloc((void**)&priorities, n * sizeof(uint));
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);
	uint seed = 0;
	InitRandomPriorities << <gridDim, blockDim >> > (seed, states, priorities, n);
	cudaDeviceSynchronize();

	// Calculate inbound counters
	uint* inboundCounts;
	CHECK(cudaMalloc((void**)&inboundCounts, n * sizeof(uint)));
	cudaMemset(inboundCounts, 0, n * sizeof(uint));
	calculateInbounds << <gridDim, blockDim >> > (graphStruct, inboundCounts, priorities, n);
	cudaDeviceSynchronize();

	// inizialize bitmaps, every node has a bitmap with a length of inbound edges + 1 TODO: aloc on gpu
	// vision: allocare tutto in un array come al solito ma serve la prefix sum
	// alternativa1: sequenziale O(n)
	// alternativa2: le bitmap vengono allocate staticamente nel kernel, basterebbe poi costruire un index, non sono sequenziali ma penso sia ok
	bool* bitmaps;
	uint bitCount = (n + (int)(graphStruct->edgeCount + 1) / 2);
	CHECK(cudaMallocManaged(&(bitmaps), bitCount * sizeof(bool)));
	memset(bitmaps, 1, bitCount * sizeof(bool));
	uint* bitmapIndex;
	CHECK(cudaMallocManaged(&bitmapIndex, (n + 1) * sizeof(uint)));
	bitmapIndex[0] = 0;
	uint* h_InboundCounts;
	h_InboundCounts = (uint*)malloc(n * sizeof(uint));
	cudaMemcpy(h_InboundCounts, inboundCounts, n * sizeof(uint), cudaMemcpyDeviceToHost);
	for (int i = 1; i < n + 1; i++)
		bitmapIndex[i] = bitmapIndex[i - 1] + h_InboundCounts[i - 1] + 1; //the inbound should be only in gpu mem TODO: parallelize with scan

	// Alloc buffer needed to synchronize the coloring
	unsigned* buffer;
	cudaMalloc((void**)&buffer, n * sizeof(unsigned));
	cudaMemset(buffer, 0, n * sizeof(unsigned));
	bool* filledBuffer;
	cudaMalloc((void**)&filledBuffer, n * sizeof(bool));
	cudaMemset(filledBuffer, 0, n * sizeof(bool));

	// Color TODO: tieni il flag sulla gpu e itera con gli stream
	coloring->iterationCount = 0;
	while (coloring->uncoloredFlag) {
		coloring->uncoloredFlag = false;
		colorWithInboundCountersBitmaps << <gridDim, blockDim >> > (coloring, graphStruct, inboundCounts, buffer, filledBuffer, bitmaps, bitmapIndex);
		cudaDeviceSynchronize();
		applyBufferWithInboundCountersBitmaps << <gridDim, blockDim >> > (coloring, graphStruct, priorities, inboundCounts, buffer, filledBuffer, bitmaps, bitmapIndex);
		cudaDeviceSynchronize();
		coloring->iterationCount++;
	}

	// Free
	cudaFree(states);
	cudaFree(priorities);
	cudaFree(inboundCounts);
	cudaFree(buffer);
	cudaFree(filledBuffer);
	//cudaFree(coloring);
	//cudaFree(coloring->coloring);
	//cudaFree(coloring->coloredNodes);

	return coloring;
}