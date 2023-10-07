
#include "device_launch_parameters.h"
#include <iostream>
#include "graph/graph_d.h"
#include "utils/common.h"
#include <cooperative_groups.h>

#include "colorer.h"

using namespace std;

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
	for (uint i = 0; i < degree; ++i)
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
				if (i < bestColor) //TODO: find another way
					bestColor = i;
					//break;
			}
		}
		coloring->coloring[idx] = bestColor;
		coloring->coloredNodes[idx] = true;
		printf("colored: %d, best color: %d: \n", idx, coloring->coloring[idx]);
		if (bestColor > coloring->numOfColors)
		{
			coloring->numOfColors = bestColor; // possibile race, potrei computarlo nella print
		}
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

Coloring* RandomPriorityColoring(GraphStruct* graphStruct) {
	// set coloring struct

	Coloring* col;
	CHECK(cudaMallocManaged(&col, sizeof(Coloring)));
	uint n = graphStruct->nodeCount;
	col->uncoloredFlag = true;

	// cudaMalloc for arrays of struct Coloring
	CHECK(cudaMallocManaged(&(col->coloring), n * sizeof(uint)));
	memset(col->coloring, 0, n);

	// allocate space on the GPU for the random states
	curandState_t* states;
	uint* weigths;
	cudaMalloc((void**)&states, n * sizeof(curandState_t));
	cudaMalloc((void**)&weigths, n * sizeof(uint));
	dim3 threads(THREADxBLOCK);
	dim3 blocks((graphStruct->nodeCount + threads.x - 1) / threads.x, 1, 1);
	uint seed = 0;
	init <<< blocks, threads >>> (seed, states, weigths, n);
	cudaDeviceSynchronize();
	// start coloring (dyn. parall.)
	LubyJPcolorer(col, graphStruct, weigths);

	cudaFree(states);
	cudaFree(weigths);
	return col;
}

/**
 * find an IS
 */
__global__ void findIS(Coloring* col, GraphStruct* graphStruct, uint* weights) {
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graphStruct->nodeCount)
		return;

	if (col->coloring[idx])
		return;

	uint offset = graphStruct->neighIndex[idx];
	uint deg = graphStruct->neighIndex[idx + 1] - graphStruct->neighIndex[idx];

	bool candidate = true;
	for (uint j = 0; j < deg; j++) {
		uint neighID = graphStruct->neighs[offset + j];
		if (!col->coloring[neighID] &&
			((weights[idx] < weights[neighID]) ||
				((weights[idx] == weights[neighID]) && idx < neighID))) {
			candidate = false;
		}
	}
	if (candidate) {
		col->coloring[idx] = col->numOfColors;
	}
	else
		col->uncoloredFlag = true;
}

/**
 *  this GPU kernel takes an array of states, and an array of ints, and puts a random int into each
 */
__global__ void init(uint seed, curandState_t* states, uint* numbers, uint n) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx > n)
		return;
	curand_init(seed, idx, 0, &states[idx]);
	numbers[idx] = curand(&states[idx]) % n * n;
}






/**
 * Luby IS & Jones−Plassmann colorer
 */
void LubyJPcolorer(Coloring* col, GraphStruct* graphStruct, uint* weights) {
	dim3 threads(THREADxBLOCK);
	dim3 blocks((graphStruct->nodeCount + threads.x - 1) / threads.x, 1, 1);

	// loop on ISs covering the graph
	col->numOfColors = 0;
	while (col->uncoloredFlag) {
		col->uncoloredFlag = false;
		col->numOfColors++;
		findIS <<< blocks, threads >>> (col, graphStruct, weights);
		cudaDeviceSynchronize();
	}
}


/**
 * Print the graph (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
void printColoring(Coloring* col, GraphStruct* graphStruct, bool verbose) {
	unsigned n = graphStruct->nodeCount;
	cout << "** Graph (num node: " << n << ", num edges: " << graphStruct->edgeCount << ")" << endl;
	cout << "** Coloring (num colors: " << col->numOfColors + 1 << ")" << endl;
	if (verbose) {
		for (uint i = 0; i <= col->numOfColors; i++) {
			cout << "   color(" << i << ")" << "-> ";
			for (uint j = 0; j < n; j++)
				if (col->coloring[j] == i)
					cout << j << " ";
			cout << "\n";
		}
		cout << "\n";
	}
}

