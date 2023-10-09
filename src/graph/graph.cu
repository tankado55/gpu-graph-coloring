#include <vector>
#include <cstring>
#include <iostream>
#include <memory>

#include "graph.h"
#include "../utils/common.h"

using namespace std;

/**
 * Generate an Erdos random graph
 * @param n number of nodes
 * @param density probability of an edge (expected density)
 * @param eng seed
 */

Graph::Graph(MemoryEnum mem) {
	memoryEnum = mem;
	Init();
}

void Graph::Init() {
	if (memoryEnum == ManagedAllocated)
	{
		CHECK(cudaMallocManaged(&graphStruct, sizeof(GraphStruct)));
	}
	else
	{
		graphStruct = new GraphStruct();
	}
	graphStruct->nodeCount = graphStruct->edgeCount = graphStruct->maxDeg = 0;
	graphStruct->neighIndex = graphStruct->neighs = NULL;
}

void Graph::randGraph(float prob, std::default_random_engine & eng, unsigned n) {
	/*
		if (useManagedMemory)
		memsetGPU(n, string("nodes"));
	else {
		
		graphStruct->neighIndex = new unsigned[n + 1] {};  // starts by zero
	}
	*/
	
	graphStruct->nodeCount = n;

	if (prob < 0 || prob > 1) {
		printf("[Graph] Warning: Probability not valid (set p = 0.5)!!\n");
	}
	uniform_real_distribution<> randR(0.0, 1.0);

	// gen edges
	vector<int>* edges = new vector<int>[n];
	vector<unsigned> index(n+1);
	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++)
			if (randR(eng) < prob) {
				edges[i].push_back(j);
				edges[j].push_back(i);
				index[i + 1]++;
				index[j + 1]++;
				graphStruct->edgeCount += 2;
			}
	}
	for (int i = 0; i < n; i++)
		index[i + 1] += index[i];

	// manage memory for edges with CUDA Unified Memory
	if (memoryEnum == ManagedAllocated)
		AllocManaged();
	else
		AllocHost();
	
	for (int i = 0; i < n; i++)
	{
		memcpy((graphStruct->neighs + graphStruct->neighIndex[i]), edges[i].data(), sizeof(int) * edges[i].size());
		memcpy(graphStruct->neighIndex, index.data(), sizeof(unsigned) * index.size());
	}

	// max, min, mean deg
	maxDeg = 0;
	minDeg = n;
	for (int i = 0; i < n; i++) {
		if (graphStruct->deg(i) > maxDeg)
		{
			maxDeg = graphStruct->deg(i);
			graphStruct->maxDeg = maxDeg;
		}
		if (graphStruct->deg(i) < minDeg)
			minDeg = graphStruct->deg(i);
	}
	density = (float)graphStruct->edgeCount / (float)(n * (n - 1));
	meanDeg = (float)graphStruct->edgeCount / (float)n;
	if (minDeg == 0)
		connected = false;
	else
		connected = true;
}

void Graph::BuildRandomDAG(Graph& dag)
{
	// Init
	dag.graphStruct->nodeCount = graphStruct->nodeCount;
	dag.graphStruct->edgeCount = (graphStruct->edgeCount + 1) / 2;
	if (dag.memoryEnum == ManagedAllocated)
		dag.AllocManaged();
	else
		dag.AllocHost();
	
	// Create an array or random priorities
	float* priorities = new float[graphStruct->nodeCount];
	std::default_random_engine engine{ 0 };  // fixed seed
	uniform_real_distribution<> randR(0.0, std::numeric_limits<float>::max());
	for (int i = 0; i < graphStruct->nodeCount; ++i)
	{
		float r = randR(engine);
		priorities[i] = r;
	}

	//build dag
	int k = 0;
	for (int i = 0; i < graphStruct->nodeCount; ++i)
	{
		float priority = priorities[i];
		int degree = graphStruct->deg(i);
		for (int j = 0; j < degree; ++j)
		{
			unsigned int neighID = graphStruct->neighs[graphStruct->neighIndex[i] + j];
			float neighPriority = priorities[neighID];
			if (priority > neighPriority || (priority == neighPriority && i > neighID))
			{
				dag.graphStruct->neighs[k] = neighID;
				++k;
			}
		}
		dag.graphStruct->neighIndex[i + 1] = k;
	}
	delete[] priorities;
}

void Graph::getLDFDag(GraphStruct* res)
{
	res->nodeCount = graphStruct->nodeCount;
	res->edgeCount = (graphStruct->edgeCount + 1) / 2;
	int k = 0;
	for (int i = 0; i < graphStruct->nodeCount; ++i)
	{
		int degree = graphStruct->deg(i);
		for (int j = 0; j < degree; ++j)
		{
			unsigned int neighID = graphStruct->neighs[graphStruct->neighIndex[i] + j];
			unsigned int neighDegree = graphStruct->deg(neighID);
			if (degree > neighDegree || (degree == neighDegree && i > neighID))
			{
				res->neighs[k] = neighID;
				++k;
			}
		}
		res->neighIndex[i + 1] = k;
	}
}

int Graph::GetEdgeCount()
{
	return graphStruct->edgeCount;
}

void Graph::AllocManaged()
{
	CHECK(cudaMallocManaged(&(graphStruct->neighIndex), (graphStruct->nodeCount + 1) * sizeof(unsigned)));
	CHECK(cudaMallocManaged(&(graphStruct->neighs), graphStruct->edgeCount * sizeof(unsigned)));
}

void Graph::FreeManaged()
{
	CHECK(cudaFree(graphStruct->neighIndex));
	CHECK(cudaFree(graphStruct->neighs));
}

void Graph::AllocHost()
{
	graphStruct->neighs = new unsigned[graphStruct->edgeCount] {};
	graphStruct->neighIndex = new unsigned[graphStruct->nodeCount + 1] {};  // starts by zero
}

/**
 * Print the graph (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
void Graph::print(bool verbose) {
	unsigned n = graphStruct->nodeCount;
	cout << "** Graph (num node: " << n << ", num edges: " << graphStruct->edgeCount
			<< ")" << endl;
	cout << "         (min deg: " << minDeg << ", max deg: " << maxDeg
		 << ", mean deg: " << meanDeg << ", connected: " << connected << ")"
		 << endl;

	if (verbose) {
		for (int i = 0; i < n; i++) {
			cout << "   node(" << i << ")" << "["
					<< graphStruct->neighIndex[i + 1] - graphStruct->neighIndex[i] << "]-> ";
			for (int j = 0; j < graphStruct->neighIndex[i + 1] - graphStruct->neighIndex[i]; j++) {
				cout << graphStruct->neighs[graphStruct->neighIndex[i] + j] << " ";
			}
			cout << "\n";
		}
		cout << "\n";
	}
}

Graph::~Graph()
{
	if (memoryEnum == ManagedAllocated)
	{
		FreeManaged();
	}
	else
	{
		delete graphStruct;
	}
}
