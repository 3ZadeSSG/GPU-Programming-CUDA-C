/*
Design a parallel program in CUDA C++ platform for the following:
Find the transpose, sum, difference, scalar and vector multiplications of matrix of parallel
and randomly initialized with the number between -1.00 to +1.00. 
Mention the parameters: number of processors used, execution time and memory utilization
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<math.h>
#include<iomanip>
#include<iostream>
#define dd double
#define n 4
#define MAX 1
#define MIN -1
using namespace std;
__global__ void matrixAdd(dd a[][n], dd b[][n], dd c[][n]) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	c[i][j] = a[i][j] + b[i][j];
}
__global__ void matrixSub(dd a[][n], dd b[][n], dd c[][n]) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	c[i][j] = a[i][j] - b[i][j];
}
__global__ void matrixTranspose(dd a[][n], dd b[][n]) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	b[i][j] = a[j][i];
}
__global__ void matrixMultiply(dd a[][n], dd b[][n], dd c[][n]) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	for (int k = 0; k < n; k++) {
		c[i][k] += a[i][k] * b[k][j];
	}
}
int main()
{
	dd a[n][n], b[n][n], c[n][n];
	dd(*dev_a)[n], (*dev_b)[n], (*dev_c)[n];
	cudaMalloc((void**)&dev_a, n*n * sizeof(dd));
	cudaMalloc((void**)&dev_b, n*n * sizeof(dd));
	cudaMalloc((void**)&dev_c, n*n * sizeof(dd));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i][j] = MIN + (rand() / (dd)RAND_MAX) *(MAX - MIN);
			b[i][j] = MIN + (rand() / (dd)RAND_MAX) *(MAX - MIN);
		}
	}
	cudaMemcpy(dev_a, a, n*n * sizeof(dd), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n*n * sizeof(dd), cudaMemcpyHostToDevice);
	dim3 block(n / 2, n / 2);  
	dim3 thread(n/(n/2), n/(n/2));
	matrixAdd << <block, thread >> > (dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, n*n * sizeof(dd), cudaMemcpyDeviceToHost);
	cout<< "\nMatrix A: " << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << setw(8) << setprecision(4) << a[i][j];
		}
		cout << endl;
	}
	cout << "\nMatrix B: " << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << setw(8) << setprecision(4) << b[i][j];
		}
		cout << endl;
	}
	cout << "\nAddition Result: " << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << setw(8) << setprecision(4) << c[i][j];
		}
		cout << endl;
	}
	matrixSub << <block, thread >> > (dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, n*n * sizeof(dd), cudaMemcpyDeviceToHost);
	cout << "\nSubtraction Result: " << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << setw(8) << setprecision(4) << c[i][j];
		}
		cout << endl;
	}
	matrixTranspose << <block, thread >> >(dev_a, dev_c);
	cudaMemcpy(c, dev_c, n*n * sizeof(dd), cudaMemcpyDeviceToHost);
	cout << "\nTranspose result of matrix A: " << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << setw(8) << setprecision(4) << c[i][j];
		}
		cout << endl;
	}
	dim3 mBlock(1, 1);
	dim3 mThread(n, n);
	matrixMultiply << <mBlock, mThread >> > (dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, n*n * sizeof(dd), cudaMemcpyDeviceToHost);
	cout << "\nMultiplication AxB: \n" << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << setw(8) << setprecision(4) << c[i][j];
		}
		cout << endl;
	}
    return 0;
}