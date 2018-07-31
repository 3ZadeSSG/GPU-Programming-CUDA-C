/* Tested on Device : Asus Nvidia GTX 1080 Ti OC Edition 
  Host: i5-6600K
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<math.h>
#include<iostream>
#include<iomanip>
#include<curand.h>
#include<curand_kernel.h>
#include<conio.h>
#define dd double
#define e 2.71828182845904523536
#define p 4
#define m 2
#define n 3
#define MAX_X 10 //max value of X
#define MIN_X 0  //min value of X
#define MAX_UV 1 //max value of U and V
#define MIN_UV -1 //min value of U and V
#define ll unsigned long long 
#define tileMAX 64 
#define beta 10 
#define delta 0.6
using namespace std;
__global__ void initializeParallelU(dd a[m][n]) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	curandState state;
	curand_init((ll)clock()+i+j, 0, 1, &state);
	a[i][j] = (curand_uniform_double(&state)*(MAX_UV-(MIN_UV)))+(MIN_UV);
}
__global__ void initializeParallelV(dd a[p][m]) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	curandState state;
	curand_init((ll)clock() + i + j, 0, 1, &state);
	a[i][j] = (curand_uniform_double(&state) * (MAX_UV-(MIN_UV))) + (MIN_UV);
}
__global__ void initializeParallelX(dd a[n][1]) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	curandState state;
	curand_init((ll)clock() + i + j, 0, 1, &state);
	a[i][j] = (curand_uniform_double(&state) * (MAX_X-MIN_X) )+ MIN_X;
}
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
	cudaMalloc((void**)&dev_U, m*n * sizeof(dd));
	cudaMalloc((void**)&dev_X, n*1 * sizeof(dd));
	cudaMalloc((void**)&dev_V, p * m * sizeof(dd));
	cudaMalloc((void**)&dev_UX, m*1 * sizeof(dd));
	cudaMalloc((void**)&dev_fUX, m * 1 * sizeof(dd));
	cudaMalloc((void**)&dev_VfUX, p * 1 * sizeof(dd));
	cudaMalloc((void**)&dev_gVfUX, p * 1 * sizeof(dd));

	dim3 block(1, 1);  //single 1 dim3 block with multiple threads, max thread in a block can be 1024
	dim3 thread(m, n);  //dim3 for U
	initializeParallelU << <block,thread>> > (dev_U);
	dim3 thread2(p, m);  //dim3 for V
	initializeParallelV << <block, thread2 >> > (dev_V);
	dim3 thread3(n, 1); //dim3 for X
	initializeParallelX << <block, thread3 >> > (dev_X);

	/*Copy parallely initialized reuslt into the Host matrices*/
	cudaMemcpy(U, dev_U, m*n * sizeof(dd), cudaMemcpyDeviceToHost);
	cudaMemcpy(V, dev_V, p* m * sizeof(dd), cudaMemcpyDeviceToHost);
	cudaMemcpy(X, dev_X, n * 1 * sizeof(dd), cudaMemcpyDeviceToHost);

	/*Operations according to formula and their result is copied into the host variables*/
	vectorMultiplyUX << <1, tileMAX >> > (dev_U, dev_X, dev_UX);   //UX
	cudaMemcpy(UX, dev_UX, m * 1 * sizeof(dd), cudaMemcpyDeviceToHost);
	fX << <1, tileMAX >> > (dev_UX, dev_fUX);						//f(UX)
	cudaMemcpy(fUX, dev_fUX, m * 1 * sizeof(dd), cudaMemcpyDeviceToHost);
	vectorMultiplyVfUX << <1, 8 >> > (dev_V, dev_fUX, dev_VfUX);	//Vf(UX)
	cudaMemcpy(VfUX, dev_VfUX, p * 1 * sizeof(dd), cudaMemcpyDeviceToHost);
	gX << <1, tileMAX >> > (dev_VfUX, dev_gVfUX);					//Z=g(Vf(UX))
	cudaMemcpy(gVfUX, dev_gVfUX, p * 1 * sizeof(dd), cudaMemcpyDeviceToHost);
	cudaMemcpy(U, dev_U, m*n * sizeof(dd), cudaMemcpyDeviceToHost);
	cudaMemcpy(X, dev_X, n * 1 * sizeof(dd), cudaMemcpyDeviceToHost);

	/*Following will print the content of initial matrices as well as result*/
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