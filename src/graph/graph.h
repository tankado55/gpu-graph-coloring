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
	float density{0.0f};	        // Probability of an edge (Erdos graph)
	GraphStruct * graphStruct{nullptr};     // graph structure
	unsigned maxDeg{0};
	unsigned minDeg{0};
	double avgDeg{0.0f};
	bool connected{true};
	void AllocManaged();                 // use UVA memory on CPU/GPU
	void FreeManaged();
	void AllocHost();

public:
	enum MemoryEnum { ManagedAllocated, HostAllocated } memoryEnum;

	Graph(MemoryEnum);
	~Graph();
	void Init();
	void ReadFromMtxFile(const char* mtx);
	void copyToDevice(GraphStruct*& dest);
	void randGraph(float, std::default_random_engine&, unsigned);  // generate an Erdos random graph
	void BuildRandomDAG(Graph&);
	void print(bool);
	GraphStruct* getStruct() {return graphStruct;}
	void getLDFDag(GraphStruct*);
	void BuildLDFDagV2(Graph&);
	int GetEdgeCount();
	uint GetNodeCount();
	double GetAvgDeg();
	void AllocDagOnDevice(GraphStruct*);

	//utils
	unsigned int deg(unsigned i);
	bool isNeighbor(unsigned i, unsigned j);
};
