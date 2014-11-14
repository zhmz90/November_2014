#include<stdio.h>
#include<cuda_runtime.h>

__global__ 
void hello(int* d_A){
	int i = threadIdx.x;
//	int j = threadIdx.y;
//	d_A[i] = i;
	d_A[i] = i;
	printf("D%d\t",d_A[i]);

}

int main(void){
	
	int *h_A;
	int const LEN = 12;
	size_t const BYTES = LEN*sizeof(int); 
	h_A = (int *)malloc(BYTES);
	int *d_A;
	cudaMalloc(&d_A, BYTES);
	cudaMemcpy(d_A, h_A, BYTES,cudaMemcpyHostToDevice);
	hello<<<1,LEN>>>(d_A);
	cudaMemcpy(h_A, d_A, BYTES,cudaMemcpyDeviceToHost);
	for (int i=0; i<LEN; i++){
		printf("H %d\t", *h_A);
		h_A++;
	}

//	printf("Hello, world!\n");
	cudaFree(d_A);	
	return 0;
}
