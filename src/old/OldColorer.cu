
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

Coloring* SequentialColorer::color(Graph& graph)
{
	// DAG
	Graph dag;
	m_Graph->BuildRandomDAG(dag);

	// temp data inizialization
	uint bitCount = (m_GraphStruct->nodeCount + (int)(m_GraphStruct->edgeCount + 1) / 2);
	std::vector<bool> bitmaps(bitCount, true);
	std::vector<uint> bitmapIndex(m_GraphStruct->nodeCount + 1);
	std::vector<uint> inboundCounts(m_GraphStruct->nodeCount, 0);
	GraphStruct* dagStruct = dag.getStruct();
	for (int i = 0; i < dag.GetEdgeCount(); ++i)
		inboundCounts[dagStruct->neighs[i]]++;
	for (int i = 1; i < m_GraphStruct->nodeCount + 1; i++)
		bitmapIndex[i] = bitmapIndex[i - 1] + m_InboundCounts[i - 1] + 1;

	// JP Coloring
	m_Coloring->iterationCount = 0;
	while (m_Coloring->uncoloredFlag)
	{
		m_Coloring->uncoloredFlag = false;
		for (int i = 0; i < m_GraphStruct->nodeCount; ++i)
		{
			if (m_Coloring->coloring[i])
				continue;

			uint offset = dagStruct->neighIndex[i];
			uint deg = dagStruct->neighIndex[i + 1] - dagStruct->neighIndex[i];

			if (inboundCounts[i] == 0) // Ready node
			{
				int colorCount = bitmapIndex[i + 1] - bitmapIndex[i];
				printf("I'm %d, total colors: %d\n", i, colorCount);

				int bestColor = colorCount;
				for (int j = 0; j < colorCount; ++j)
				{
					if (bitmaps[bitmapIndex[i] + j])
					{
						if (j < bestColor)
						{
							//TODO: find another way
							bestColor = j;
							break;
						}
					}
				}
				m_Coloring->coloring[i] = bestColor;
				m_Coloring->coloredNodes[i] = true;
				printf("colored: %d, best color: %d: \n", i, m_Coloring->coloring[i]);
				if (bestColor > m_Coloring->iterationCount)
				{
					m_Coloring->iterationCount = bestColor; // possibile race, potrei computarlo nella print
				}
				for (uint j = 0; j < deg; j++) {
					uint neighID = dagStruct->neighs[offset + j];
					inboundCounts[neighID]--;
					bitmaps[bitmapIndex[neighID] + bestColor] = 0;

				}
			}
			else
			{
				m_Coloring->uncoloredFlag = true;
			}
		}
	}
	return m_Coloring;
}

__global__ void findISLDF(Coloring* coloring, GraphStruct* graphStruct, bool* bitmaps, uint* bitmapIndex, uint* inboundCounts)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graphStruct->nodeCount)
		return;

	if (coloring->coloredNodes[idx])
		return;

	//printf("GPU - I'm %d, myInbound: %d\n", idx, inboundCounts[idx]);

	uint offset = graphStruct->neighIndex[idx];
	uint deg = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];

	if (inboundCounts[idx] == 0) // Ready node
	{
		int colorCount = bitmapIndex[idx + 1] - bitmapIndex[idx];
		printf("I'm %d, total colors: %d\n", idx, colorCount);

		int bestColor = colorCount;
		for (int i = 0; i < colorCount; ++i)
		{
			if (bitmaps[bitmapIndex[idx] + i])
			{
				bestColor = i;
				break;
			}
		}
		coloring->coloring[idx] = bestColor;
		coloring->coloredNodes[idx] = true;
		printf("colored: %d, best color: %d: \n", idx, coloring->coloring[idx]);

		for (uint i = 0; i < deg; i++) {
			uint neighID = graphStruct->neighs[offset + i];
			if (!coloring->coloredNodes[neighID])
			{
				atomicAdd(&inboundCounts[neighID], -1);
				bitmaps[bitmapIndex[neighID] + bestColor] = 0;
			}
		}
	}
	else
	{
		coloring->uncoloredFlag = true;
	}
}

Coloring* RandomPriorityColoring(Graph& graph) // no inboundsCount, no bitmap no dag
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
		colorIS << <gridDim, blockDim >> > (coloring, graphStruct, priorities, buffer, filledBuffer);
		cudaDeviceSynchronize();
		applyBuffer << <gridDim, blockDim >> > (coloring, buffer, filledBuffer, n);
		cudaDeviceSynchronize();
		coloring->iterationCount++;
	}

	// Free
	cudaFree(states);
	cudaFree(priorities);
	cudaFree(buffer);
	cudaFree(filledBuffer);
	//cudaFree(coloring);
	//cudaFree(coloring->coloring);
	//cudaFree(coloring->coloredNodes);

	return coloring;
}

__global__ void InitSDFPriorities(GraphStruct* graphStruct, uint* priorities, uint n) { //TODO: passa direttamente neighIndex
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;
	priorities[idx] = UINT_MAX - (graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx]);
}

/**
 * find an IS
 */
__global__ void colorIS(Coloring* col, GraphStruct* graphStruct, uint* weights, unsigned* buffer, bool* filledBuffer)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graphStruct->nodeCount)
		return;

	if (col->coloredNodes[idx])
		return;

	uint offset = graphStruct->neighIndex[idx];
	uint deg = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];

	bool candidate = true;
	for (uint j = 0; j < deg; j++) {
		uint neighID = graphStruct->neighs[offset + j];

		if (!col->coloredNodes[neighID] &&
			((weights[idx] < weights[neighID]) || ((weights[idx] == weights[neighID]) && idx < neighID))) {
			candidate = false;
		}
	}
	if (candidate) {
		buffer[idx] = col->iterationCount;
		filledBuffer[idx] = true;
		//printf("candidate: %d, color: %d\n", idx, col->numOfColors);
	}
	else
	{
		col->uncoloredFlag = true;
		//printf("not candidate: %d, color: %d\n", idx, col->numOfColors);
	}
}

__global__ void colorWithInboundCounters(Coloring* coloring, GraphStruct* graphStruct, uint* inboundCounts, uint* buffer, bool* filledBuffer)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graphStruct->nodeCount)
		return;

	if (coloring->coloredNodes[idx])
		return;

	//printf("GPU - I'm %d, myInbound: %d\n", idx, inboundCounts[idx]);



	//if (idx == 750)
	//	printf("uncolored: %d, still: %d -- best color: %d: \n", idx, inboundCounts[idx], coloring->numOfColors);
	if (inboundCounts[idx] == 0) // Ready node
	{
		buffer[idx] = coloring->iterationCount;
		filledBuffer[idx] = true;
	}
	else
	{
		coloring->uncoloredFlag = true;
	}
}

Coloring* DegreePriorityColoringV3(Graph& graph, priorityEnum priorityEnum)
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
	uint* d_priorities = calculateDegreePriority(d_graphStruct, priorityEnum, n);

	// Calculate inbound counters
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);
	uint* inboundCounts;
	uint* outboundCounts;
	CHECK(cudaMalloc((void**)&inboundCounts, n * sizeof(uint)));
	CHECK(cudaMalloc((void**)&outboundCounts, n * sizeof(uint)));
	cudaMemset(inboundCounts, 0, n * sizeof(uint));
	cudaMemset(outboundCounts, 0, n * sizeof(uint));
	calculateInbounds << <gridDim, blockDim >> > (d_graphStruct, inboundCounts, d_priorities, n);
	cudaDeviceSynchronize();

	// inizialize bitmaps, every node has a bitmap with a length of inbound edges + 1 TODO: alloc on gpu
	// vision: allocare tutto in un array come al solito ma serve la prefix sum
	// alternativa1: sequenziale O(n)
	bool* bitmaps;
	uint bitCount = (n + (int)(edgeCount + 1) / 2);
	CHECK(cudaMallocManaged(&(bitmaps), bitCount * sizeof(bool)));
	memset(bitmaps, 1, bitCount * sizeof(bool));
	uint* bitmapIndex;
	CHECK(cudaMallocManaged(&bitmapIndex, (n + 1) * sizeof(uint)));
	uint* h_InboundCounts;
	h_InboundCounts = (uint*)malloc(n * sizeof(uint));
	cudaMemcpy(h_InboundCounts, inboundCounts, n * sizeof(uint), cudaMemcpyDeviceToHost);
	bitmapIndex[0] = 0;
	for (int i = 1; i < n + 1; i++)
		bitmapIndex[i] = bitmapIndex[i - 1] + h_InboundCounts[i - 1] + 1; //the inbound should be only in gpu mem TODO: parallelize with scan

	// Alloc buffer needed to synchronize the coloring
	unsigned* buffer;
	cudaMalloc((void**)&buffer, n * sizeof(unsigned));
	cudaMemset(buffer, 0, n * sizeof(unsigned));
	bool* filledBuffer;
	cudaMalloc((void**)&filledBuffer, n * sizeof(bool));
	cudaMemset(filledBuffer, 0, n * sizeof(bool));

	// DEBUG
	uint* h_priorities = (uint*)malloc(n * sizeof(uint));
	//cudaMemcpy(h_priorities, priorities, n * sizeof(uint), cudaMemcpyDeviceToHost);

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
		colorWithInboundCountersBitmaps << <gridDim, blockDim >> > (d_coloring, d_coloredNodes, d_graphStruct, inboundCounts, buffer, filledBuffer, bitmaps, bitmapIndex, d_uncoloredFlag);
		cudaDeviceSynchronize();
		applyBufferWithInboundCountersBitmaps << <gridDim, blockDim >> > (d_coloring, d_coloredNodes, d_graphStruct, d_priorities, inboundCounts, buffer, filledBuffer, bitmaps, bitmapIndex);
		cudaDeviceSynchronize();
		//cudaMemcpy(h_priorities, priorities, n * sizeof(uint), cudaMemcpyDeviceToHost); //TODO: remove
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
	//cudaFree(coloring);
	//cudaFree(coloring->coloring);
	//cudaFree(coloring->coloredNodes);

	//cudaMemcpy(coloring, d_coloring, sizeof(Coloring), cudaMemcpyDeviceToHost);
	cudaMemcpy(coloring, d_coloring, n * sizeof(uint), cudaMemcpyDeviceToHost);
	cudaMemcpy(coloredNodes, d_coloredNodes, n * sizeof(bool), cudaMemcpyDeviceToHost);

	Coloring* coloringStruct = (Coloring*)malloc(sizeof(Coloring));
	coloringStruct->coloring = coloring;
	coloringStruct->coloredNodes = coloredNodes;
	coloringStruct->iterationCount = iterationCount;
	return coloringStruct;
}