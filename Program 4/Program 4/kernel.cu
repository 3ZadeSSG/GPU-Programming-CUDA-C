/*
Calculate cosine distance, Manhattan distance and Euclidian Distance between two n-dimensional points considered as vectors.
For cosine distance: https://en.wikipedia.org/wiki/Cosine_similarity
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>
#include<math.h>
#include<iomanip>
#include<random>
#include<curand.h>
#include<curand_kernel.h>
#include<ctime>
#define dd double
#define n 24
using namespace std;
__global__ void initialize(dd *p1, dd*p2) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	curandState state;
	curand_init((unsigned long long)clock() + i, 0, 1, &state);
	p1[i] = curand_uniform_double(&state);
	p2[i] = curand_uniform_double(&state);
}
__global__ void euclidianDistance(dd *p1, dd *p2, dd *result,dd *temp) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	*result = 0;
	temp[i]= pow((p1[i] - p2[i]), 2);
	for (int id = 0; id < n; id++) {
		*result += temp[id];
	}
	*result = sqrt(*result);
}
__global__ void manhattanDistance(dd *p1, dd *p2, dd *result, dd *temp) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	*result = 0;
	temp[i] = abs(p1[i] - p2[i]);
	for (int id = 0; id < n; id++) {
		*result += temp[id];
	}
}
__global__ void dotProduct(dd*a, dd*b, dd*c,dd *result) {  //for calculating a.b
	int i= blockDim.x*blockIdx.x + threadIdx.x;
	c[i] = a[i] * b[i];
	*result = 0;
	for (int id = 0; id < n; id++) {
		*result += c[id];
	}
}
__global__ void euclidianDotProduct(dd*a, dd*b, dd*c,dd*result) { //for calculating ||a||.||b||
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.x*blockIdx.x + threadIdx.x;
	c[i] = (a[i] * a[i]);
	dd temp = 0;
	for (int id = 0; id < n; id++) {
		temp += c[id];
	}
	*result = sqrt(temp);
	temp = 0;
	c[j] = (b[j] * b[j]);
	for (int id = 0; id < n; id++) {
		temp += c[id];
	}
	*result = *result*sqrt(temp);
}
int main()
{
	dd p1[n], p2[n],manhattan_distance=0,euclidian_distance=0,euclidian_dot=0,vector_dot=0;
	dd *dev_p1, *dev_p2,*dev_temp,*dev_result;
	cudaMalloc((void**)&dev_p1, n * sizeof(dd)); //vector 1
	cudaMalloc((void**)&dev_p2, n * sizeof(dd)); //vector 2
	cudaMalloc((void**)&dev_temp, n * sizeof(dd));  
	cudaMalloc(&dev_result, sizeof(dd));
	initialize << <1, n >> > (dev_p1, dev_p2); //initialize vectors using parallel technique
	cudaMemcpy(p1, dev_p1, n * sizeof(dd), cudaMemcpyDeviceToHost);
	cudaMemcpy(p2, dev_p2, n * sizeof(dd), cudaMemcpyDeviceToHost);
	cout << "\nVector 1: ";
	for (int i = 0; i < n; i++) {
		cout << setw(8) << setprecision(5) << p1[i] << " ";
		//cout << p1[i] << " ";
	}
	cout << "\n\nVector 2: ";
	for (int i = 0; i < n; i++) {
		cout <<setw(8)<<setprecision(5)<< p2[i] << " ";
		//cout << p2[i] << " ";
	}

	euclidianDistance << <1, n>> > (dev_p1, dev_p2, dev_result,dev_temp);
	cudaMemcpy(&euclidian_distance, dev_result,sizeof(dd), cudaMemcpyDeviceToHost); //copy euclidian distance from device to host
	cout <<"\n\nEuclidian Distance:"<< euclidian_distance<<endl; //print euclidian distance

	manhattanDistance << <1, n>> > (dev_p1, dev_p2, dev_result, dev_temp);
	cudaMemcpy(&manhattan_distance, dev_result, sizeof(dd), cudaMemcpyDeviceToHost); //copy manhattan distance from device to host
	cout << "\nManhattan Distance:" << manhattan_distance << endl; //print manhattan distance

	euclidianDotProduct << <1, n>> > (dev_p1, dev_p2,dev_temp, dev_result);
	cudaMemcpy(&euclidian_dot, dev_result, sizeof(dd), cudaMemcpyDeviceToHost);
	dotProduct << <1, n >> > (dev_p1, dev_p2, dev_temp, dev_result);
	cudaMemcpy(&vector_dot, dev_result, sizeof(dd), cudaMemcpyDeviceToHost);

	cout << "\nVector dot: " << vector_dot<<"\nEuclidian dot: "<<euclidian_dot;
	cout << "\nCosine Distance: " << (vector_dot / euclidian_dot)<<endl; //cosine distance = vector dot product / euclidian dot product
	
    return 0;
}
