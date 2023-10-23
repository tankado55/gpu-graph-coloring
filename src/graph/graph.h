
#pragma once

#include <random>

/**
 * Base structure (array 1D format) of a graph
 */
struct GraphStruct {
	unsigned nodeCount{0};             // num of graph nodes
	unsigned edgeCount{0};             // num of graph edges
	unsigned* neighIndex{nullptr};       // cumsum of node degrees
	unsigned* neighs{nullptr};           // list of neighbors for all nodes (edges)
		
	~GraphStruct() {delete[] neighs; delete[] neighIndex;}

	// check whether node j is a neighbor of node i
	bool isNeighbor(unsigned i, unsigned j) {
		for (unsigned k = 0; k < deg(i); k++) 
			if (neighs[neighIndex[i]+k] == j)
	    	return true;
	  return false;
	}

	// return the degree of node i
	unsigned int deg(unsigned i) {
		return(neighIndex[i+1]-neighIndex[i]);
	}
};

class Graph {
	
private:
	float density{0.0f};	        // Probability of an edge (Erdos graph)
	GraphStruct * graphStruct{nullptr};     // graph structure
	unsigned maxDeg{0};
	unsigned minDeg{0};
	float meanDeg{0.0f};
	bool connected{true};
	void AllocManaged();                 // use UVA memory on CPU/GPU
	void FreeManaged();
	void AllocHost();

public:
	enum MemoryEnum { ManagedAllocated, HostAllocated } memoryEnum;

	Graph(MemoryEnum);
	~Graph();
	void Init(); // CPU/GPU mem setup
	void ReadFromMtxFile(const char* mtx);
	void copyToDevice(GraphStruct*& dest);
	void randGraph(float, std::default_random_engine&, unsigned);  // generate an Erdos random graph
	void BuildRandomDAG(Graph&);
	void print(bool);
	void print_d(GraphStruct *, bool);
	GraphStruct* getStruct() {return graphStruct;}
	void getLDFDag(GraphStruct*);
	void BuildLDFDagV2(Graph&);
	int GetEdgeCount();
	void AllocDagOnDevice(GraphStruct*);
};
