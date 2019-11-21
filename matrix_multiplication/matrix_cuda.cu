#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16
#define MATRIX_SIZE 4
 
//GPU kernel 
__global__ void gpu_matrix_mult(int *device_a, int *device_b, int *device_c, int n = 4) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  if( col < n && row < n) 
  {
    for(int i = 0; i < n; i++) 
    {
      sum += device_a[row * n + i] * device_b[i * n + col];
    }
    device_c[row * n + col] = sum;
  }
}

int main(int argc, char const *argv[])
{
  printf("Begin \n");

  int *host_a, *host_b, *host_c;
  int *device_a, *device_b, *device_c;

  //memory allocation	
  cudaMallocHost((void **) &host_a, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);
  cudaMallocHost((void **) &host_b, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);
  cudaMallocHost((void **) &host_c, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);

  unsigned int grid_rows = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_cols = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(grid_cols, grid_rows);

  printf("Initialize matrix A\n");
  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      host_a[i * MATRIX_SIZE + j] = i + j;
      printf("%i\t", i + j);
    }
    printf("\n");
  }

  printf("Initialize matrix B\n");
  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      host_b[i * MATRIX_SIZE + j] = i + j;
      printf("%i\t", i + j);
    }
    printf("\n");
  }

  printf("Allocating device memory...\n");
   //GPU memory allocation
  cudaMalloc((void **) &device_a, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);
  cudaMalloc((void **) &device_b, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);
  cudaMalloc((void **) &device_c, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE);

  printf("Copying to device..\n");
  cudaMemcpy(device_a, host_a, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE, cudaMemcpyHostToDevice);

  // Launch kernel 
  gpu_matrix_mult<<<dimGrid, dimBlock>>>(device_a, device_b, device_c, MATRIX_SIZE); 

  //Wait for kernel call to finish
  cudaThreadSynchronize();

  // Transefr results from device to host 
  cudaMemcpy(host_c, device_c, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE, cudaMemcpyDeviceToHost);

  printf("Reading matrix C\n");
  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      int aux = host_c[i * MATRIX_SIZE + j];
      printf("%i\t", aux);
    }
    printf("\n");
  }
  
  // free memory
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  cudaFreeHost(host_a);
  cudaFreeHost(host_b);
  cudaFreeHost(host_c);
  return 0;
}