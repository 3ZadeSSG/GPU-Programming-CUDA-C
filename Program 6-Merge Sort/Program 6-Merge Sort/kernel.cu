#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>
#include<iomanip>
#include<math.h>
#define MIN 1
#define MAX 100
#define n 25
using namespace std;
__device__ void merge(int *array, int *result, int l, int r, int u) {
	int i, j, k;
	i = l;
	j = r;
	k = l;
	while (i < r && j < u) {
		if (array[i] <= array[j]) {
			result[k++] = array[i++];
		}
		else {
			result[k++] = array[j++];
		}
	}
	while (i < r) {
		result[k++] = array[i++];
	}
	while (j < u) {
		result[k++] = array[j++];
	}
	for (k = l; k < u; k++) {
		array[k] = result[k];
	}

}
__global__ void mergeSort(int *arr, int *result) {
	int index = threadIdx.x;
	int k, u, i;
	extern __shared__ int shared[];
	shared[index] = arr[index];
	k = 1;
	while (k < n) {
		i = 0;
		while (i + k < n) {
			u = i+k*2;
			if (u > n) {
				u = n + 1;
			}
			merge(shared, result, i, i + k, u);
			i = i + k * 2;
		}
		k = k * 2;
	}
	arr[index] = shared[index];
}
int main()
{
	int array[n];
	int *dev_array, *dev_result;
	for (int i = 0; i < n; i++) {
		array[i] = rand() % MAX + MIN;
	}
	cudaMalloc((void**)&dev_array, n * sizeof(int));
	cudaMalloc((void**)&dev_result, n * sizeof(int));
	cout << "\nArray: ";
	for (int i = 0; i < n; i++) {
		cout << " " << array[i];
	}
	cout << endl;

	cudaMemcpy(dev_array, array, n * sizeof(int), cudaMemcpyHostToDevice);
	mergeSort << <1, n,sizeof(int)*n*2>> > (dev_array, dev_result); //single block with multiple thread, max thread in a block can be 1024 so max size of array will be 1024
	cudaMemcpy(array, dev_result, n*sizeof(int), cudaMemcpyDeviceToHost);

	cout << "\nSorted array: ";
	for (int i = 0; i < n; i++) {
		cout << " " << array[i];
	}
	cout << endl << endl;
    return 0;
}