﻿
#include "device_launch_parameters.h"
#include <iostream>
#include "coloring.h"
#include "utils/graph/graph_d.h"
#include "utils/common.h"

using namespace std;

#define THREADxBLOCK 128

Coloring* LubyGreedy(GraphStruct* graphStruct) {
	// set coloring struct

	Coloring* col;
	CHECK(cudaMallocManaged(&col, sizeof(Coloring)));
	uint n = graphStruct->nodeSize;
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
	dim3 blocks((graphStruct->nodeSize + threads.x - 1) / threads.x, 1, 1);
	uint seed = 0;
	init <<< blocks, threads >>> (seed, states, weigths, n);
	cudaDeviceSynchronize();
	// start coloring (dyn. parall.)
	//LubyJPcolorer <<< 1, 1 >>> (col, str, weigths);

//#####################
	// loop on CPU

	// loop on ISs covering the graph
	col->numOfColors = 0;
	while (col->uncoloredNodes) {
		col->uncoloredNodes = false;
		col->numOfColors++;
		findIS <<< blocks, threads >>> (col, graphStruct, weigths);
		cudaDeviceSynchronize();
	}
	//#####################


	cudaFree(states);
	cudaFree(weigths);
	return col;
}

/**
 * find an IS
 */
__global__ void findIS(Coloring* col, GraphStruct* graphStruct, uint* weights) {
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= graphStruct->nodeSize)
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
__global__ void LubyJPcolorer(Coloring* col, GraphStruct* graphStruct, uint* weights) {
	dim3 threads(THREADxBLOCK);
	dim3 blocks((graphStruct->nodeSize + threads.x - 1) / threads.x, 1, 1);

	// loop on ISs covering the graph
	col->numOfColors = 0;
	while (col->uncoloredNodes) {
		col->uncoloredNodes = false;
		col->numOfColors++;
		findIS <<< blocks, threads >>> (col, graphStruct, weights);
		//cudaDeviceSynchronize();
	}
}



/**
 * Print the graph (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
void printColoring(Coloring* col, GraphStruct* graphStruct, bool verbose) {
	node n = graphStruct->nodeSize;
	cout << "** Graph (num node: " << n << ", num edges: " << graphStruct->edgeSize << ")" << endl;
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

