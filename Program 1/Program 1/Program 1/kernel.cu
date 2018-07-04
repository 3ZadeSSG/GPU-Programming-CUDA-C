/*
Design a parallel program in CUDA C++ platform for the following:
Find the row-sum, column-sum, Row-Maximum and Column-Minimum of n-order square matrix 
populated automatically using random values in between 0.00 to 1.00. Mention the parameters: 
number of processors used, execution time, speed-up and memory utilization
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>
#include<conio.h>
#include<math.h>
#include<iomanip>
#define n 4
using namespace std;
__global__ void matrix_operation(double a[][n],double *c,double *d,double *c_min,double *r_max) {
	int x = threadIdx.x;
	int y = threadIdx.y;
	r_max[x] = a[x][0];
	c_min[y] = a[0][y];
	for (int i = 0; i < n; i++) {
		c[x] = a[x][i] + c[x];
		d[y] = a[i][y] + d[y];
		if (r_max[x] < a[x][i])
			r_max[x] = a[x][i];
		if (c_min[y] > a[i][y])
			c_min[y] = a[i][y];
	}
}
int main()
{
	double a[n][n],col_sum[n],row_sum[n],col_min[n],row_max[n];
	double(*dev_a)[n],(*dev_b)[n],*dev_col_sum,*dev_row_sum,*dev_col_min,*dev_row_max;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			//a[i][j] = i + j * 2;
			a[i][j] = rand() / double(RAND_MAX);
		}
	}
	cudaMalloc((void**)&dev_a, n*n * sizeof(double));
	cudaMalloc((void**)&dev_b, n*n * sizeof(double));
	cudaMalloc((void**)&dev_row_sum, n * sizeof(double));
	cudaMalloc((void**)&dev_col_sum, n * sizeof(double));
	cudaMalloc((void**)&dev_col_min, n * sizeof(double));
	cudaMalloc((void**)&dev_row_max, n * sizeof(double));
	cudaMemcpy(dev_a, a, n*n * sizeof(double), cudaMemcpyHostToDevice);
	dim3 block(1, 1);
	dim3 thread(n, n);
	matrix_operation << <block, thread >> > (dev_a,dev_row_sum,dev_col_sum,dev_col_min,dev_row_max);
	cudaMemcpy(row_sum, dev_row_sum, n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(col_sum, dev_col_sum, n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(col_min, dev_col_min, n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(row_max, dev_row_max, n * sizeof(double), cudaMemcpyDeviceToHost);
	cout << "\nCreated Matrix: " << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << setw(8) << setprecision(4) << a[i][j];
		}
		cout << endl;
	}
	cout <<"\nRow sum of matrix: ";
	for (int i = 0; i < n; i++) {
		cout << setw(8) << setprecision(4) << row_sum[i];
	}
	cout <<"\nColumn sum of matrix: ";
	for (int i = 0; i < n; i++) {
		cout << setw(8) << setprecision(4) << col_sum[i];
	}
	cout << "\nColumn min of matrix: ";
	for (int i = 0; i < n; i++) {
		cout << setw(8) << setprecision(4) << col_min[i];
	}
	cout << "\nRow max of matrix: ";
	for (int i = 0; i < n; i++) {
		cout << setw(8)<<setprecision(4)<< row_max[i];
	}
	cout << endl;
	getch();
    return 0;
}