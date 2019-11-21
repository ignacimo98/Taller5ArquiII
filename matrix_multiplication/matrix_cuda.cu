#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 250
#define MATRIX_SIZE 4
 
//GPU kernel 
__global__ void gpu_matrix_mult(int *device_a, int *device_b, int *d_result, int n = 4) {
  __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int tmp = 0;
  int idx;
 
  for (int sub = 0; sub < gridDim.x; ++sub) 
  {
    idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
    if(idx >= n*n)
    {
      // n may not divisible by BLOCK_SIZE
      tile_a[threadIdx.y][threadIdx.x] = 0;
    }
    else
    {
      tile_a[threadIdx.y][threadIdx.x] = device_a[idx];
    }

    idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
    if(idx >= n*n)
    {
      tile_b[threadIdx.y][threadIdx.x] = 0;
    }  
    else
    {
      tile_b[threadIdx.y][threadIdx.x] = device_b[idx];
    }

    __syncthreads();

    // matrix multiplication
    for (int k = 0; k < BLOCK_SIZE; ++k) 
    {
      tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    }
    __syncthreads();
  }
  if(row < n && col < n)
  {
    d_result[row * n + col] = tmp;
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
      printf("%i\t");
    }
    printf("\n");
  }

  printf("Initialize matrix B\n");
  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      host_b[i * MATRIX_SIZE + j] = i + j;
      printf("%i\t");
    }
    printf("\n");
  }

  printf("Allocating device memory...\n");
   //GPU memory allocation
  cudaMalloc((void **) &device_a, sizeof(int)*m*MATRIX_SIZE);
  cudaMalloc((void **) &device_b, sizeof(int)*MATRIX_SIZE*k);
  cudaMalloc((void **) &device_c, sizeof(int)*m*k);

  printf("Copying to device..\n");
  cudaMemcpy(device_a, host_a, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, sizeof(int)*MATRIX_SIZE*MATRIX_SIZE, cudaMemcpyHostToDevice);

  // Launch kernel 
  gpu_matrix_mult<<<dimGrid, dimBlock>>>(device_a, device_b, device_c, MATRIX_SIZE); 

  //Wait for kernel call to finish
  cudaThreadSynchronize();

  // Transefr results from device to host 
  cudaMemcpy(host_c, device_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);

  printf("Reading matrix C\n");
  for (int i = 0; i < MATRIX_SIZE; ++i) {
    for (int j = 0; j < MATRIX_SIZE; ++j) {
      host_c[i * MATRIX_SIZE + j] = i + j;
      printf("%i\t");
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
  cudaFreeHost(h_cc);
  return 0;
}