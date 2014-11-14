#include <stdio.h>
#include <cuda_runtime.h>


__global__ 
void add(int *d_a, int *d_b, int *d_c){
	int i = threadIdx.x;
//	*(d_c+i) = *(d_a+i) + *(d_b+i);
	d_c[i] = d_a[i] + d_b[i];
	int j = blockIdx.x;
	printf("%d\t",j);
}

int main(){
	printf("Hello, world!\n");
	//host mem
	int *h_a, *h_b, *h_c;
	const int LEN = 10;
	size_t SIZE = LEN * sizeof(int);
	h_a = (int *)malloc(SIZE);
	h_b = (int *)malloc(SIZE);
	h_c = (int *)malloc(SIZE);

	for (int i=0; i<LEN;i++){	
		*(h_a+i) = 1;
		*(h_b+i) = 1;
//		*(h_c+i) = 0;
//		printf("%d\n",*(h_a+i));
	}
	//device mem
	int *d_a, *d_b, *d_c;
	cudaMalloc((void **)&d_a, SIZE);
	cudaMalloc((void **)&d_b, SIZE);
	cudaMalloc((void **)&d_c, SIZE);
	//var transfer to device
	cudaMemcpy(d_a,h_a,SIZE,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,SIZE,cudaMemcpyHostToDevice);	

	//kernel
	int threadsPerBlock = LEN;
	int blocksPerGrid = 8;	
	add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);	

	cudaMemcpy(h_c,d_c,SIZE,cudaMemcpyDeviceToHost);	
	for (int i=0;i<LEN;i++){
		
	//	printf("%d\t",*(h_c+i));	
	
	}
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	//printf("\n");
	return 0;

}


