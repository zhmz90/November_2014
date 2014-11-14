#include <stdio.h>
#include <cuda_runtime.h>

__global__
void add( int a, int b, int *c ){
	int i = threadIdx.x;
	*(c+i) = a + b;

}

int main(){
	int *h_c;
	int *d_c;
	const int LEN = 100;
	cudaMalloc((void **)&d_c, LEN*sizeof(int));
	
	cudaEvent_t start, stop;
	cudaEventCreate( &start);
	cudaEventCreate( &stop);
	cudaEventRecord( start, 0);	

	add<<<1,LEN>>>(2, 7, d_c);

	cudaEventRecord( stop, 0);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime,start,stop);
	cudaMemcpy(h_c, d_c, LEN*sizeof(int),cudaMemcpyDeviceToHost);
	
//	printf("2 + 7 = %d\n", h_c);
	printf("elapsedTime is: %f", elapsedTime);
	cudaFree(d_c);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
