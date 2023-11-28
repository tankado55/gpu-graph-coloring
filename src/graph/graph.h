#pragma once

#include "../utils/common.h"
#include <random>

struct GraphStruct {
	unsigned nodeCount{0};             // num of graph nodes
	unsigned edgeCount{0};             // num of graph edges
	unsigned* neighIndex{nullptr};       // cumsum of node degrees
	unsigned* neighs{nullptr};           // list of neighbors for all nodes (edges)
		
	~GraphStruct() {delete[] neighs; delete[] neighIndex;}
};

class Graph {
	
private:
	GraphStruct* graphStruct{ nullptr };     // graph structure
	GraphStruct* d_graphStruct{ nullptr };
	uint* neighIndexTemp{ nullptr }; // pointer to call cudaFree
	uint* neighsTemp{ nullptr }; // pointer to call cudaFree
	float density{0.0f};
	unsigned maxDeg{0};
	unsigned minDeg{0};
	double avgDeg{0.0f};
	bool connected{true};
	void AllocHost();
	bool deviceAllocated{ false };

public:
	Graph();
	~Graph();
	void Init();
	void readFromMtxFile(const char* mtx);
	void getDeviceStruct(GraphStruct*& dest);
	void randGraph(float, std::default_random_engine&, unsigned);
	void print(bool);
	GraphStruct* getStruct() {return graphStruct;}
	int GetEdgeCount();
	uint GetNodeCount();
	double GetAvgDeg();
	void BuildRandomDAG(Graph&);
	void BuildLDFDagV2(Graph&);
	void AllocDagOnDevice(GraphStruct*);

	//utils
	unsigned int deg(unsigned i);
	bool isNeighbor(unsigned i, unsigned j);
};
