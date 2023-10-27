
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

__global__ void initLDF(GraphStruct* graphStruct, int* inboundCounts, int n) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;

	uint degree = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];

	inboundCounts[idx] = 0;
	for (uint i = 0; i < degree; ++i)
	{
		uint neighID = graphStruct->neighs[graphStruct->neighIndex[idx] + i];
		uint neighDegree = graphStruct->neighIndex[neighID + 1] - graphStruct->neighIndex[neighID]; // ottimizzabile su CPU
		if (degree > neighDegree)
		{
			atomicAdd(&inboundCounts[neighID], 1);
			printf(" atomicAdd node(%d -> %d [count: %d] \n", idx, neighID, inboundCounts[neighID]);
		}
		else if (degree == neighDegree && idx > neighID)
		{
			atomicAdd(&inboundCounts[neighID], 1);
			printf(" atomicAdd node(%d -> %d [count: %d] \n", idx, neighID, inboundCounts[neighID]);
		}
	}
}

__global__ void initLDF2(GraphStruct* graphStruct, uint* inboundCounts, int n) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;

	uint degree = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];

	inboundCounts[idx] = 0;
	for (uint i = 0; i < degree; ++i) //TODO: ciclo inutile, basta mettere piú 1 a ogni elemento della lista
	{
		uint neighID = graphStruct->neighs[graphStruct->neighIndex[idx] + i];

		atomicAdd(&inboundCounts[neighID], 1);
		//printf(" atomicAdd node(%d -> %d [count: %d] \n", idx, neighID, inboundCounts[neighID]);

	}
}

Coloring* Colorer::LDFColoring()
{
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((m_GraphStruct->nodeCount + blockDim.x - 1) / blockDim.x, 1, 1);

	// Init DAG TODO: refactorare
	GraphStruct* dag;
	CHECK(cudaMallocManaged(&dag, sizeof(GraphStruct)));
	CHECK(cudaMallocManaged(&(dag->neighIndex), (m_GraphStruct->nodeCount + 1) * sizeof(int)));
	CHECK(cudaMallocManaged(&(dag->neighs), (m_GraphStruct->edgeCount + 1) / 2 * sizeof(int)));
	m_Graph->getLDFDag(dag);

	//initLDF <<<gridDim, blockDim>>> (m_GraphStruct, m_InboundCounts, m_GraphStruct->nodeCount);
	initLDF2 << <gridDim, blockDim >> > (dag, m_InboundCounts, m_GraphStruct->nodeCount);
	cudaDeviceSynchronize();

	// inizialize bitmaps
	// Every node has a bitmap with a length of inbound edges + 1
	bool* bitmaps;
	uint bitCount = (m_GraphStruct->nodeCount + (int)(m_GraphStruct->edgeCount + 1) / 2);
	CHECK(cudaMallocManaged(&(bitmaps), bitCount * sizeof(bool)));
	memset(bitmaps, 1, bitCount * sizeof(bool));
	uint* bitmapIndex;
	CHECK(cudaMallocManaged(&bitmapIndex, (m_GraphStruct->nodeCount + 1) * sizeof(uint)));
	cudaDeviceSynchronize();
	bitmapIndex[0] = 0;
	for (int i = 1; i < m_GraphStruct->nodeCount + 1; i++)
		bitmapIndex[i] = bitmapIndex[i - 1] + m_InboundCounts[i - 1] + 1; //this info should be taken by the dag and the inbound should be only in gpu mem

	uint iterationCount = 0;
	while (m_Coloring->uncoloredFlag) {
		m_Coloring->uncoloredFlag = false;
		iterationCount++;
		printf("------------ Sequential iteration: %d \n", iterationCount);
		int deb_inBoundSum = 0;
		for (int i = 0; i < m_GraphStruct->nodeCount; ++i)
		{
			deb_inBoundSum += m_InboundCounts[i];
		}
		printf("------------ inboundsum: %d \n", deb_inBoundSum);
		printf("edges: %d", m_GraphStruct->edgeCount);
		int deb_ready = 0;
		for (int i = 0; i < m_GraphStruct->nodeCount; ++i)
		{
			if (m_InboundCounts[i] == 0 && m_Coloring->coloredNodes[i] == false)
				++deb_ready;
		}
		if (deb_ready == 0)
			printf("------------ ready: %d \n", deb_ready);
		findISLDF << < gridDim, blockDim >> > (m_Coloring, dag, bitmaps, bitmapIndex, m_InboundCounts);
		cudaDeviceSynchronize();
	}

	return m_Coloring;
}

void test(Graph& graph)
{
	GraphStruct* graphStruct = graph.getStruct();
	int n = graphStruct->nodeCount;
	uint* h_InboundCounts;
	h_InboundCounts = (uint*)malloc(n * sizeof(uint));

	//priorities
	uint* priorities;
	cudaMalloc((void**)&priorities, n * sizeof(uint));
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);
	InitLDFPriorities << <gridDim, blockDim >> > (graphStruct, priorities, n);
	cudaDeviceSynchronize();

	//inbounds
	uint* inboundCounts;
	CHECK(cudaMalloc((void**)&inboundCounts, n * sizeof(uint)));
	cudaMemset(inboundCounts, 0, n * sizeof(uint));
	calculateInbounds << <gridDim, blockDim >> > (graphStruct, inboundCounts, priorities, n);
	cudaDeviceSynchronize();

	cudaMemcpy(h_InboundCounts, inboundCounts, n * sizeof(uint), cudaMemcpyDeviceToHost);

	testAtomicAdd << <gridDim, blockDim >> > (graphStruct, priorities, inboundCounts);
	cudaDeviceSynchronize();


	cudaMemcpy(h_InboundCounts, inboundCounts, n * sizeof(uint), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; ++i)
	{
		if (h_InboundCounts[0] != 0)
			std::cout << "error" << std::endl;
	}
	std::cout << "end" << std::endl;
}