#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>
#include<math.h>
#include<iomanip>
#define dd double
#define n 8
#define MAX 1
#define MIN -1
using namespace std;
__global__ void crossover(dd*a, dd*b,dd *point) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i > *point) {
		a[i] = a[i] + b[i];
		b[i] = a[i] - b[i];
		a[i] = a[i] - b[i];
	}
}
int main()
{
	dd c1[n], c2[n],crossover_point=3;
	dd *dev_c1, *dev_c2,*dev_crossover_point;
	cudaMalloc((void**)&dev_c1, n * sizeof(dd));
	cudaMalloc((void**)&dev_c2, n * sizeof(dd));
	cudaMalloc(&dev_crossover_point,sizeof(dd));
	for (int i = 0; i < n; i++) {
		c1[i]= MIN + (rand() / (dd)RAND_MAX) *(MAX - MIN);
		c2[i]= MIN + (rand() / (dd)RAND_MAX) *(MAX - MIN);
	}
	cout << "\nChromosome 1: ";
	for (int i = 0; i < n; i++) {
		cout << " " << c1[i];
	}
	cout << "\n\nChromosome 2: ";
	for (int i = 0; i < n; i++) {
		cout << " " << c2[i];
	}
	cudaMemcpy(dev_c1, c1, n * sizeof(dd), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c2, c2, n * sizeof(dd), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_crossover_point, &crossover_point,sizeof(dd), cudaMemcpyHostToDevice);
	crossover << <1, n >> > (dev_c1, dev_c2, dev_crossover_point);

	cudaMemcpy(c1, dev_c1, n * sizeof(dd), cudaMemcpyDeviceToHost);
	cudaMemcpy(c2, dev_c2, n * sizeof(dd), cudaMemcpyDeviceToHost);
	cout << "\n\nAfter crossover\nChromosome 1: ";
	for (int i = 0; i < n; i++) {
		cout << " " << c1[i];
	}
	cout << "\n\nChromosome 2: ";
	for (int i = 0; i < n; i++) {
		cout << " " << c2[i];
	}
	cout << endl << endl;
    return 0;
}