#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#define ll long
#define n 10
using namespace std;
__global__ void bubbleSort(ll *a) {
	int index = threadIdx.x;
	int prev = 0;
	for (int i = 0; i < n-1; i++) {
		if (index>=prev++ && index< n-1) {
			if (a[index] > a[index + 1])
			{
				a[index] = a[index]+a[index + 1];
				a[index+1] = a[index] - a[index + 1];
				a[index] = a[index] - a[index + 1];
			}
		}
	}
}
int main()
{
	ll a[n];
	ll *dev_a;
	cout << "Initial array: ";
	for (int i = 0; i < n; i++) {
		a[i] = rand();
		cout << a[i] << " ";
	}
	cudaMalloc((void**)&dev_a, n * sizeof(ll));
	cudaMemcpy(dev_a, a, n * sizeof(ll), cudaMemcpyHostToDevice);
	bubbleSort << <1, n >> > (dev_a);
	cudaMemcpy(a, dev_a, n * sizeof(ll), cudaMemcpyDeviceToHost);
	cout << "\nSorted Array: ";
	for (int i = 0; i < n; i++) {
		cout << a[i] << " ";
	}
    return 0;
}
