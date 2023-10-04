
#include "device_launch_parameters.h"
#include <iostream>
#include "utils/graph/graph_d.h"
#include "utils/common.h"
#include <cooperative_groups.h>

#include "colorer.h"

using namespace std;

#define THREADxBLOCK 128

Colorer::Colorer(const GraphStruct* graphStruct)
{
	m_GraphStruct = graphStruct;
	CHECK(cudaMallocManaged((void **) &m_Coloring, sizeof(Coloring)));
	m_Coloring.uncoloredNodes = true;
	m_Coloring.numOfColors = 0;

	uint n = graphStruct->nodeCount;

	CHECK(cudaMallocManaged(&(m_Coloring.coloring), n * sizeof(uint)));
	memset(m_Coloring.coloring, 0, n);
}

__global__ void initLDF(GraphStruct* graphStruct, uint n) {
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;

	uint degree = graphStruct->cumDegs[idx + 1] - graphStruct->cumDegs[idx];
	printf("node(%d [myDegree: %d] \n", idx, degree);

	graphStruct->inCounts[idx] = 0;
	for (uint i = 0; i < degree; ++i)
	{
		uint neighID = graphStruct->neighs[graphStruct->cumDegs[idx] + i];
		uint neighDegree = graphStruct->cumDegs[neighID + 1] - graphStruct->cumDegs[neighID]; // ottimizzabile su CPU
		if (degree > neighDegree)
		{
			atomicAdd(&graphStruct->inCounts[neighID], 1);
			printf(" atomicAdd node(%d -> %d [count: %d] \n", idx, neighID, graphStruct->inCounts[neighID]);
		}
		else if (degree == neighDegree && idx > neighID)
		{
			atomicAdd(&graphStruct->inCounts[neighID], 1);
			printf(" atomicAdd node(%d -> %d [count: %d] \n", idx, neighID, graphStruct->inCounts[neighID]);
		}
	}
}

Coloring* Colorer::LDFColoring()
{
	// Init DAG calculating inCounts
	dim3 blockDim(THREADxBLOCK);
	dim3 gridDim((m_GraphStruct->nodeCount + blockDim.x - 1) / blockDim.x, 1, 1);

	initLDF <<<gridDim, blockDim>>> (m_GraphStruct, m_GraphStruct->nodeCount);
	cudaDeviceSynchronize();
	for (int i = 0; i < m_GraphStruct->nodeCount; ++i) {
		std::cout << "node" << i << " inCount: " << m_GraphStruct->inCounts[i] << "\n";
	}

	// inizialize bitmaps
	bool* bitmaps;
	CHECK(cudaMallocManaged(&(bitmaps), (m_GraphStruct->nodeCount + m_GraphStruct->edgeCount) * sizeof(bool)));
	int* bitmapIndex;
	CHECK(cudaMallocManaged(&(bitmapIndex), m_GraphStruct->nodeCount * sizeof(int)));




	

	

	
	uint iterationCount = 0;
	while (coloring->uncoloredNodes) {
		coloring->uncoloredNodes = false;
		iterationCount++;
		printf("Sequential iteration: %d \n", iterationCount);
		findISLDF << < blocks, threads >> > (coloring, graphStruct);
		cudaDeviceSynchronize();
	}

	return coloring;
}

Coloring* RandomPriorityColoring(GraphStruct* graphStruct) {
	// set coloring struct

	Coloring* col;
	CHECK(cudaMallocManaged(&col, sizeof(Coloring)));
	uint n = graphStruct->nodeCount;
	col->uncoloredNodes = true;

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

	uint offset = graphStruct->cumDegs[idx];
	uint deg = graphStruct->cumDegs[idx + 1] - graphStruct->cumDegs[idx];

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
		col->uncoloredNodes = true;
}

__global__ void findISLDF(Coloring* col, GraphStruct* graphStruct) {
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graphStruct->nodeCount)
		return;

	if (col->coloring[idx])
		return;

	uint offset = graphStruct->cumDegs[idx];
	uint deg = graphStruct->cumDegs[idx + 1] - graphStruct->cumDegs[idx];

	if (graphStruct->inCounts[idx] == 0)
	{
		int bestColor;
		//TODO: cicla su colore
		
		for (int i = 0; i < graphStruct->maxDeg + 1; ++i)
		{
			if (col->colorBitmaps[idx * (graphStruct->maxDeg + 1)+ i])
			{
				bestColor = i;
				//break;
			}
		}
		col->coloring[idx] = bestColor;
		printf("colored: %d, best color: %d: \n", idx, bestColor);
		if (bestColor > col->numOfColors)
		{
			col->numOfColors = bestColor; // possibile race, potrei computarlo nella print
		}
		for (uint i = 0; i < deg; i++) {
			uint neighID = graphStruct->neighs[offset + i];
			if (!col->coloring[neighID])
			{
				atomicAdd(&graphStruct->inCounts[neighID], -1);
				col->colorBitmaps[neighID * (graphStruct->maxDeg + 1) + bestColor] = 0;
			}
		}
	}
	else
	{
		col->uncoloredNodes = true;
	}
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
	while (col->uncoloredNodes) {
		col->uncoloredNodes = false;
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
	node n = graphStruct->nodeCount;
	cout << "** Graph (num node: " << n << ", num edges: " << graphStruct->edgeCount << ")" << endl;
	cout << "** Coloring (num colors: " << col->numOfColors << ")" << endl;
	if (verbose) {
		for (uint i = 1; i <= col->numOfColors; i++) {
			cout << "   color(" << i << ")" << "-> ";
			for (uint j = 0; j < n; j++)
				if (col->coloring[j] == i)
					cout << j << " ";
			cout << "\n";
		}
		cout << "\n";
	}
}

