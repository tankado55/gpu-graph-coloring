#include <vector>
#include <cstring>
#include <iostream>
#include <memory>
#include <fstream>
#include <string>
#include <set>

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

void Graph::Init() {
	if (memoryEnum == ManagedAllocated)
	{
		CHECK(cudaMallocManaged(&graphStruct, sizeof(GraphStruct)));
	}
	else
	{
		graphStruct = new GraphStruct();
	}
	graphStruct->nodeCount = graphStruct->edgeCount = 0;
	graphStruct->neighIndex = graphStruct->neighs = NULL;
}

void Graph::ReadFromMtxFile(const char* mtx) {
	printf("Reading .mtx input file %s\n", mtx);
	std::ifstream cfile;
	cfile.open(mtx);
	std::string str;
	getline(cfile, str);
	char c;
	sscanf(str.c_str(), "%c", &c);
	while (c == '%') {
		getline(cfile, str);
		sscanf(str.c_str(), "%c", &c);
	}
	int n;
	sscanf(str.c_str(), "%d %d %d", &graphStruct->nodeCount, &n, &graphStruct->edgeCount);
	graphStruct->edgeCount *= 2;
	if (graphStruct->nodeCount != n) {
		printf("error!\n");
		exit(0);
	}
	printf("num_vertices %d, num_edges %d\n", graphStruct->nodeCount, graphStruct->edgeCount);
	vector<set<int> > svector;
	set<int> s;
	for (int i = 0; i < graphStruct->nodeCount; i++)
		svector.push_back(s);
	int dst, src;
	int sameNodeCounter = 0;
	for (int i = 0; i < graphStruct->edgeCount/2; i++) {
		getline(cfile, str);
		sscanf(str.c_str(), "%d %d", &dst, &src);

		if (src == dst) {
			sameNodeCounter++;
			continue;
		}

		dst--; // -1 because the dataset begins with node id = 1
		src--; // -1 because the dataset begins with node id = 1

		svector[src].insert(dst);
		svector[dst].insert(src);
	}
	cfile.close();
	graphStruct->neighIndex = (uint*)malloc((graphStruct->nodeCount + 1) * sizeof(int));
	int count = 0;
	for (int i = 0; i < graphStruct->nodeCount; i++) {
		graphStruct->neighIndex[i] = count;
		count += svector[i].size();
	}
	graphStruct->neighIndex[graphStruct->nodeCount] = count;
	if (count != graphStruct->edgeCount) {
		printf("The graph is not symmetric\n");
		printf("found %d arc with src and dst equals\n", sameNodeCounter);
		printf("real number of edges: %d\n", count);

		graphStruct->edgeCount = count;
	}
	double avgdeg;
	double variance = 0.0;
	int maxdeg = 0;
	int mindeg = graphStruct->nodeCount;
	avgdeg = (double)graphStruct->edgeCount / graphStruct->nodeCount;
	for (int i = 0; i < graphStruct->nodeCount; i++) {
		int deg_i = graphStruct->neighIndex[i + 1] - graphStruct->neighIndex[i];
		if (deg_i > maxdeg)
			maxdeg = deg_i;
		if (deg_i < mindeg)
			mindeg = deg_i;
		variance += (deg_i - avgdeg) * (deg_i - avgdeg) / graphStruct->nodeCount;
	}
	printf("mindeg %d maxdeg %d avgdeg %.2f variance %.2f\n", mindeg, maxdeg, avgdeg, variance);
	graphStruct->neighs = (uint*)malloc(graphStruct->edgeCount * sizeof(int));
	set<int>::iterator site;
	for (int i = 0, index = 0; i < graphStruct->nodeCount; i++) {
		site = svector[i].begin();
		while (site != svector[i].end()) {
			graphStruct->neighs[index++] = *site;
			site++;
		}
	}
}

void Graph::copyToDevice(GraphStruct*& dest)
{
	CHECK(cudaMalloc((void**)&dest, sizeof(GraphStruct)));
	CHECK(cudaMemcpy(&dest->nodeCount, &graphStruct->nodeCount, sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(&dest->edgeCount, &graphStruct->edgeCount, sizeof(int), cudaMemcpyHostToDevice));

	uint* neighIndexTemp;
	uint* neighsTemp;
	CHECK(cudaMalloc((void**)&neighIndexTemp, (graphStruct->nodeCount + 1) * sizeof(uint)));
	CHECK(cudaMalloc((void**)&neighsTemp, graphStruct->edgeCount * sizeof(uint)));
	CHECK(cudaMemcpy(&(dest->neighIndex), &(neighIndexTemp), sizeof(uint*), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(&(dest->neighs), &(neighsTemp), sizeof(uint*), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(neighIndexTemp, graphStruct->neighIndex, (graphStruct->nodeCount + 1) * sizeof(uint), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(neighsTemp, graphStruct->neighs, graphStruct->edgeCount * sizeof(uint), cudaMemcpyHostToDevice));
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

void Graph::BuildLDFDagV2(Graph& dag) //TODO: check all the code
{
	dag.graphStruct->nodeCount = graphStruct->nodeCount;
	dag.graphStruct->edgeCount = (graphStruct->edgeCount + 1) / 2;
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
				dag.graphStruct->neighs[k] = neighID;
				++k;
			}
		}
		dag.graphStruct->neighIndex[i + 1] = k;
	}
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

int Graph::GetNodeCount()
{
	return graphStruct->nodeCount;
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

void Graph::AllocDagOnDevice(GraphStruct* dag)
{
	CHECK(cudaMalloc((void**)&dag, sizeof(GraphStruct)));
	dag->nodeCount = graphStruct->nodeCount;
	dag->edgeCount = (graphStruct->edgeCount + 1) / 2;
	CHECK(cudaMalloc((void**)&dag->neighIndex, (dag->nodeCount + 1) * sizeof(uint)));
	CHECK(cudaMalloc((void**)&dag->neighs, dag->edgeCount * sizeof(uint)));
}

/**
 * Print the graph (verbose = 1 for "verbose print")
 * @param verbose print the complete graph
 */
void Graph::print(bool verbose)
{
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
