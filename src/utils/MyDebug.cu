#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include "MyDebug.h"

void MyDebug::fWritePriority(uint* d_priority, uint n)
{
	uint* priority = (uint*)malloc(sizeof(uint) * n);
	cudaMemcpy(priority, d_priority, sizeof(uint) * n, cudaMemcpyDeviceToHost);

	std::ofstream myfile;
	myfile.open("priority.txt");
	for (int i = 0; i < n; ++i)
	{
		myfile << priority[i] << std::endl;
	}
	myfile.close();
}
