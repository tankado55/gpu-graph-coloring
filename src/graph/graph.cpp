#include <vector>
#include <cstring>
#include <iostream>
#include <memory>
#include "graph.h"

using namespace std;

/**
 * Generate an Erdos random graph
 * @param n number of nodes
 * @param density probability of an edge (expected density)
 * @param eng seed
 */
void Graph::Init(unsigned n) {
	if (GPUEnabled)
		memsetGPU(n, string("nodes"));
	else {
		graphStruct = new GraphStruct();
		graphStruct->neighIndex = new unsigned[n + 1]{};  // starts by zero
	}
	graphStruct->nodeCount = n;
}

/**
 * Generate a new random graph
 * @param eng seed
 */
void Graph::randGraph(float prob, std::default_random_engine & eng) {
	if (prob < 0 || prob > 1) {
		printf("[Graph] Warning: Probability not valid (set p = 0.5)!!\n");
	}
	uniform_real_distribution<> randR(0.0, 1.0);
	unsigned n = graphStruct->nodeCount;

	// gen edges
	vector<int>* edges = new vector<int>[n];
	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++)
			if (randR(eng) < prob) {
				edges[i].push_back(j);
				edges[j].push_back(i);
				graphStruct->neighIndex[i + 1]++;
				graphStruct->neighIndex[j + 1]++;
				graphStruct->edgeCount += 2;
			}
	}
	for (int i = 0; i < n; i++)
		graphStruct->neighIndex[i + 1] += graphStruct->neighIndex[i];

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
	density = (float) graphStruct->edgeCount / (float) (n * (n - 1));
	meanDeg = (float) graphStruct->edgeCount / (float) n;
	if (minDeg == 0)
		connected = false;
	else
		connected = true;

	// manage memory for edges with CUDA Unified Memory
	if (GPUEnabled)
		memsetGPU(n,"edges");
	else
		graphStruct->neighs = new unsigned[graphStruct->edgeCount] { };

	for (int i = 0; i < n; i++)
		memcpy((graphStruct->neighs + graphStruct->neighIndex[i]), edges[i].data(), sizeof(int) * edges[i].size());
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
