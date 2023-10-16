
#include "device_launch_parameters.h"
#include <iostream>
#include "graph/graph_d.h"
#include "utils/common.h"
#include <cooperative_groups.h>

#include "colorer.h"

#define THREADxBLOCK 128

Colorer::Colorer(Graph* graph)
{
	m_Graph = graph;
	m_GraphStruct = graph->getStruct();
	CHECK(cudaMallocManaged(&m_Coloring, sizeof(Coloring)));
	m_Coloring->uncoloredFlag = true;
	m_Coloring->numOfColors = 0;

	uint n = m_GraphStruct->nodeCount;

	CHECK(cudaMallocManaged(&m_Coloring->coloring, n * sizeof(uint)));
	memset(m_Coloring->coloring, 0, n * sizeof(uint));
	CHECK(cudaMallocManaged(&m_Coloring->coloredNodes, n * sizeof(bool)));
	memset(m_Coloring->coloredNodes, 0, n * sizeof(bool));

	//init inbound counts
	CHECK(cudaMallocManaged(&m_InboundCounts, n * sizeof(uint)));
}

Colorer::~Colorer(){
	cudaFree(m_InboundCounts);
}

__global__ void initLDF(GraphStruct* graphStruct, int* inboundCounts, int n) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;

	uint degree = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];
	printf("node(%d [myDegree: %d] \n", idx, degree);

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
	//printf("node(%d [myDegree: %d] \n", idx, degree);

	inboundCounts[idx] = 0;
	for (uint i = 0; i < degree; ++i) //TODO: ciclo inutile, basta mettere piú 1 a ogni elemento della lista
	{
		uint neighID = graphStruct->neighs[graphStruct->neighIndex[idx] + i];
		
		atomicAdd(&inboundCounts[neighID], 1);
		//printf(" atomicAdd node(%d -> %d [count: %d] \n", idx, neighID, inboundCounts[neighID]);
		
	}
}

__global__ void findISLDF(Coloring* coloring, GraphStruct* graphStruct, bool* bitmaps, uint* bitmapIndex, uint* inboundCounts)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graphStruct->nodeCount) //é giusto
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



Coloring* Colorer::LDFColoring()
{
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((m_GraphStruct->nodeCount + blockDim.x - 1) / blockDim.x, 1, 1);
	
	// Init DAG TODO: refactorare
	GraphStruct* dag;
	CHECK(cudaMallocManaged(&dag, sizeof(GraphStruct)));
	CHECK(cudaMallocManaged(&(dag->neighIndex), (m_GraphStruct->nodeCount + 1) * sizeof(int)));
	CHECK(cudaMallocManaged(&(dag->neighs), (m_GraphStruct->edgeCount+1)/2 * sizeof(int)));
	m_Graph->getLDFDag(dag);

	//initLDF <<<gridDim, blockDim>>> (m_GraphStruct, m_InboundCounts, m_GraphStruct->nodeCount);
	initLDF2 <<<gridDim, blockDim>>> (dag, m_InboundCounts, m_GraphStruct->nodeCount);
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
		findISLDF <<< gridDim, blockDim >>> (m_Coloring, dag, bitmaps, bitmapIndex, m_InboundCounts);
		cudaDeviceSynchronize();
	}

	return m_Coloring;
}

Coloring* Colorer::RandomPriorityColoringCPUSequential()
{
	// DAG
	Graph dag(Graph::MemoryEnum::HostAllocated);
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
	m_Coloring->numOfColors = 0;
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
				if (bestColor > m_Coloring->numOfColors)
				{
					m_Coloring->numOfColors = bestColor; // possibile race, potrei computarlo nella print
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

	inboundCounts[idx] = 0;
	for (uint i = 0; i < degree; ++i)
	{
		uint neighID = graphStruct->neighs[graphStruct->neighIndex[idx] + i];
		if (priorities[idx] > priorities[neighID])
		{
			atomicAdd(&inboundCounts[neighID], 1);
			//printf(" atomicAdd node(%d -> %d [count: %d] \n", idx, neighID, inboundCounts[neighID]);
		}
		else if (priorities[idx] == priorities[neighID] && idx > neighID)
		{
			atomicAdd(&inboundCounts[neighID], 1);
			//printf(" atomicAdd node(%d -> %d [count: %d] \n", idx, neighID, inboundCounts[neighID]);
		}
	}
}





__global__ void buildParallelDag()
{

}





/**
 * find an IS
 */
__global__ void findIS(Coloring* col, GraphStruct* graphStruct, uint* weights, unsigned* buffer, bool* filledBuffer)
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
		buffer[idx] = col->numOfColors;
		filledBuffer[idx] = true;
		//printf("candidate: %d, color: %d\n", idx, col->numOfColors);
	}
	else
	{
		col->uncoloredFlag = true;
		//printf("not candidate: %d, color: %d\n", idx, col->numOfColors);
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
		buffer[idx] = coloring->numOfColors;
		filledBuffer[idx] = true;
	}
	else
	{
		coloring->uncoloredFlag = true;
	}
}



/**
 *  this GPU kernel takes an array of states, and an array of ints, and puts a random int into each
 */
__global__ void InitRandomPriorities(uint seed, curandState_t* states, uint* priorities, uint n) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;
	curand_init(seed, idx, 0, &states[idx]);
	priorities[idx] = curand(&states[idx]) % n * n;
}

__global__ void InitLDFPriorities(GraphStruct* graphStruct, uint* priorities, uint n) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;
	priorities[idx] = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];
}



/**
 * Print the graph (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
void printColoring(Coloring* col, GraphStruct* graphStruct, bool verbose) {
	unsigned n = graphStruct->nodeCount;
	std::cout << "** Graph (num node: " << n << ", num edges: " << graphStruct->edgeCount << ")" << std::endl;
	std::cout << "** Coloring (num colors: " << col->numOfColors + 1 << ")" << std::endl;
	if (verbose) {
		for (uint i = 0; i <= col->numOfColors; i++) {
			std::cout << "   color(" << i << ")" << "-> ";
			for (uint j = 0; j < n; j++)
				if (col->coloring[j] == i)
					std::cout << j << " ";
			std::cout << "\n";
		}
		std::cout << "\n";
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
	coloring->numOfColors = 0;
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
	coloring->numOfColors = 0;
	while (coloring->uncoloredFlag) {
		coloring->uncoloredFlag = false;
		findIS << <gridDim, blockDim >> > (coloring, graphStruct, priorities, buffer, filledBuffer);
		cudaDeviceSynchronize();
		applyBuffer << <gridDim, blockDim >> > (coloring, buffer, filledBuffer, n);
		cudaDeviceSynchronize();
		coloring->numOfColors++;
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
	coloring->numOfColors = 0;
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
	coloring->numOfColors = 0;
	while (coloring->uncoloredFlag) {
		coloring->uncoloredFlag = false;
		colorWithInboundCounters <<<gridDim, blockDim>>> (coloring, graphStruct, inboundCounts, buffer, filledBuffer);
		cudaDeviceSynchronize();
		applyBufferWithInboundCounters <<<gridDim, blockDim>>>(coloring, graphStruct, priorities, inboundCounts, buffer, filledBuffer);
		cudaDeviceSynchronize();
		coloring->numOfColors++;
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
	coloring->numOfColors = 0;
	GraphStruct* graphStruct = graph.getStruct();

	// Generate random node priorities
	curandState_t* states;
	uint* priorities;
	cudaMalloc((void**)&states, n * sizeof(curandState_t));
	cudaMalloc((void**)&priorities, n * sizeof(uint));
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);
	uint seed = 0;
	InitRandomPriorities <<<gridDim, blockDim >> > (seed, states, priorities, n);
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
	coloring->numOfColors = 0;
	while (coloring->uncoloredFlag) {
		coloring->uncoloredFlag = false;
		colorWithInboundCountersBitmaps <<<gridDim, blockDim>>> (coloring, graphStruct, inboundCounts, buffer, filledBuffer, bitmaps, bitmapIndex);
		cudaDeviceSynchronize();
		applyBufferWithInboundCountersBitmaps <<<gridDim, blockDim>>>(coloring, graphStruct, priorities, inboundCounts, buffer, filledBuffer, bitmaps, bitmapIndex);
		cudaDeviceSynchronize();
		coloring->numOfColors++;
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

Coloring* LDFColoringV3(Graph& graph)
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
	coloring->numOfColors = 0;
	GraphStruct* graphStruct = graph.getStruct();

	// Generate LDF priorities
	uint* priorities;
	cudaMalloc((void**)&priorities, n * sizeof(uint));
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, 1, 1);
	InitLDFPriorities <<<gridDim, blockDim >>> (graphStruct, priorities, n);
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

	// DEBUG
	uint* h_priorities = (uint*)malloc(n * sizeof(uint));

	// Color TODO: tieni il flag sulla gpu e itera con gli stream
	coloring->numOfColors = 0;
	while (coloring->uncoloredFlag) {
		coloring->uncoloredFlag = false;
		colorWithInboundCountersBitmaps <<<gridDim, blockDim>>> (coloring, graphStruct, inboundCounts, buffer, filledBuffer, bitmaps, bitmapIndex);
		cudaDeviceSynchronize();
		applyBufferWithInboundCountersBitmaps <<<gridDim, blockDim>>>(coloring, graphStruct, priorities, inboundCounts, buffer, filledBuffer, bitmaps, bitmapIndex);
		cudaDeviceSynchronize();
		coloring->numOfColors++;
		cudaMemcpy(h_priorities, priorities, n * sizeof(uint), cudaMemcpyDeviceToHost); //TODO: remove
		cudaDeviceSynchronize();
	}

	// Free
	cudaFree(priorities);
	cudaFree(inboundCounts);
	cudaFree(buffer);
	cudaFree(filledBuffer);
	//cudaFree(coloring);
	//cudaFree(coloring->coloring);
	//cudaFree(coloring->coloredNodes);

	return coloring;
}

__global__ void colorWithInboundCountersBitmaps(Coloring* coloring, GraphStruct* graphStruct, uint* inboundCounts, uint* buffer, bool* filledBuffer, bool* bitmaps, uint* bitmapIndex)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graphStruct->nodeCount)
		return;

	if (coloring->coloredNodes[idx])
		return;

	//printf("GPU - I'm %d, myInbound: %d\n", idx, inboundCounts[idx]);

	if (idx == 8984)
		printf("uncolored: %d, still: %d -- iteration: %d: \n", idx, inboundCounts[idx], coloring->numOfColors);
	if (inboundCounts[idx] == 0) // Ready node
	{
		int colorCount = bitmapIndex[idx + 1] - bitmapIndex[idx];
		//printf("I'm %d, total colors: %d\n", idx, colorCount);

		int bestColor = colorCount;
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
		//printf("I'm %d, filled buffer: %d\n", idx, bestColor);
	}
	else
	{
		coloring->uncoloredFlag = true;
	}
}

__global__ void applyBufferWithInboundCountersBitmaps(Coloring* coloring, GraphStruct* graphStruct, unsigned* priorities, unsigned* inboundCounts, unsigned* buffer, bool* filledBuffer, bool* bitmaps, uint* bitmapIndex)
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
		uint neighID = graphStruct->neighs[offset + i];
		
		if (priorities[idx] > priorities[neighID])
		{
			atomicAdd(&inboundCounts[neighID], -1);
			bitmaps[bitmapIndex[neighID] + buffer[idx]] = 0;

			if (neighID == 8984)
				printf("I'm: %d, removed arc to: %d: \n", idx, neighID);
			//printf("I'm: %d, removed arc to: %d: \n", idx, neighID);
		}
		else if (priorities[idx] == priorities[neighID] && idx > neighID)
		{
			atomicAdd(&inboundCounts[neighID], -1);
			bitmaps[bitmapIndex[neighID] + buffer[idx]] = 0;

			if (neighID == 8984)
				printf("I'm: %d, removed arc to: %d: \n", idx, neighID);
			//printf("I'm: %d, removed arc to: %d: \n", idx, neighID);
		}

	}
	coloring->coloring[idx] = buffer[idx];
	coloring->coloredNodes[idx] = true;
	filledBuffer[idx] = false;
	//printf("buffer applied: %d, color: %d\n", idx, coloring->coloring[idx]);
}

__global__ void testAtomicAdd(GraphStruct* graphStruct, unsigned* priorities, unsigned* inboundCounts)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graphStruct->nodeCount)
		return;

	uint offset = graphStruct->neighIndex[idx];
	uint deg = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];

	for (uint i = 0; i < deg; i++)
	{
		uint neighID = graphStruct->neighs[offset + i];

		if (priorities[idx] > priorities[neighID])
		{
			atomicAdd(&inboundCounts[neighID], -1);

			if (neighID == 8984)
				printf("I'm: %d, removed arc to: %d: \n", idx, neighID);
			//printf("I'm: %d, removed arc to: %d: \n", idx, neighID);
		}
		else if (priorities[idx] == priorities[neighID] && idx > neighID)
		{
			atomicAdd(&inboundCounts[neighID], -1);

			//if (neighID == 8984)
			//	printf("I'm: %d, removed arc to: %d: \n", idx, neighID);
			//printf("I'm: %d, removed arc to: %d: \n", idx, neighID);
		}

	}

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



