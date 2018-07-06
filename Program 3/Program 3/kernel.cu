#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<math.h>
#include<iostream>
#include<iomanip>
#define dd double
#define e 2.71828182845904523536
#define p 4
#define m 2
#define n 3
#define tileMAX 64
#define beta 10
#define delta 0.6
#define random (rand() / (double)RAND_MAX)
using namespace std;
__global__ void vectorMultiplyUX(dd a[m][n], dd b[n][1], dd c[m][1]) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	for (int k = 0; k < n; k++) {
		c[i][j] += a[i][k] * b[k][j];
	}
}
__global__ void vectorMultiplyVfUX(dd V[p][m],dd fUX[m][1],dd result[p][1]) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	for (int k = 0; k < m; k++) {
		result[i][j] += V[i][k] * fUX[k][j];
	}
}
__global__ void gX(dd VfUX[p][1], dd result[p][1]) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	result[i][j] = VfUX[i][j] * delta;
}
__global__ void fX(dd X[m][1],dd result[m][1]) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	result[i][j] = 1 / (1 + (pow(e, (-beta*X[i][j]))));
}
int main()
{
	dd U[m][n],V[p][m],X[n][1],UX[m][1],fUX[m][1],VfUX[p][1],gVfUX[p][1];
	dd(*dev_U)[n],(*dev_V)[m], (*dev_X)[1], (*dev_UX)[1],(*dev_fUX)[1],(*dev_VfUX)[1],(*dev_gVfUX)[1];
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			U[i][j] = random;
		}
	}
	for (int i = 0; i < n; i++) {
		X[i][0] = random;
	}
	for (int i = 0; i < p; i++) {
		for (int j = 0; j < m; j++) {
			V[i][j] = random;
		}
	}
	cudaMalloc((void**)&dev_U, m*n * sizeof(dd));
	cudaMalloc((void**)&dev_X, n*1 * sizeof(dd));
	cudaMalloc((void**)&dev_V, p * m * sizeof(dd));
	cudaMalloc((void**)&dev_UX, m*1 * sizeof(dd));
	cudaMalloc((void**)&dev_fUX, m * 1 * sizeof(dd));
	cudaMalloc((void**)&dev_VfUX, p * 1 * sizeof(dd));
	cudaMalloc((void**)&dev_gVfUX, p * 1 * sizeof(dd));

	cudaMemcpy(dev_U, U, m*n * sizeof(dd), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_X, X, n*1* sizeof(dd), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_V, V, p*m* 1 * sizeof(dd), cudaMemcpyHostToDevice);

	vectorMultiplyUX << <1, tileMAX >> > (dev_U, dev_X, dev_UX);   //UX
	cudaMemcpy(UX, dev_UX, m * 1 * sizeof(dd), cudaMemcpyDeviceToHost);

	fX << <1, tileMAX >> > (dev_UX, dev_fUX);						//f(UX)
	cudaMemcpy(fUX, dev_fUX, m * 1 * sizeof(dd), cudaMemcpyDeviceToHost);

	vectorMultiplyVfUX << <1, 8 >> > (dev_V, dev_fUX, dev_VfUX);	//Vf(UX)
	cudaMemcpy(VfUX, dev_VfUX, p * 1 * sizeof(dd), cudaMemcpyDeviceToHost);

	gX << <1, tileMAX >> > (dev_VfUX, dev_gVfUX);					//Z=g(Vf(UX))
	cudaMemcpy(gVfUX, dev_gVfUX, p * 1 * sizeof(dd), cudaMemcpyDeviceToHost);


	cout << "\nU: \n";
	for (int i = 0; i < m; i++) {
		cout <<setw(8)<< " ";
		for (int j = 0; j < n; j++) {
			cout <<setw(8)<< U[i][j]<<" ";
		}
		cout << endl;
	}
	cout << "\nX: \n";
	for (int i = 0; i < n; i++) {
		cout << setw(8)<<" "<< X[i][0]<<endl;
	}
	cout << "\nV: \n";
	for (int i = 0; i < p; i++) {
		cout << setw(8) << " ";
		for (int j = 0; j < m; j++) {
			cout << setw(8)<< V[i][j]<<" ";
		}
		cout << endl;
	}
	cout << "\nUX: \n";
	for (int i = 0; i < m; i++) {
			cout << setw(8)<<" "<< UX[i][0]<<endl;
	}
	cout << "\nf(UX): \n";
	for (int i = 0; i < m; i++) {
		cout << setw(8) << " " << fUX[i][0] << endl;
	}
	cout << "\nVf(UX): \n";
	for (int i = 0; i < p; i++) {
		cout << setw(8) << " " << VfUX[i][0] << endl;
	}
	cout << "\nZ=g(Vf(UX)): \n";
	for (int i = 0; i < p; i++) {
		cout << setw(8) <<" "<< gVfUX[i][0] <<endl;
	}
    return 0;
}