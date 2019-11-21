#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void saxpy_cpu(int n, float a, float *x, float *y){
    for (int i = 0; i < n; ++i)
        y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  clock_t start_d=clock();
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  cudaThreadSynchronize();
  clock_t end_d = clock();

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);


  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  clock_t start_h = clock();
  saxpy_cpu(N, 2.0f, x, y);
  clock_t end_h = clock();

  //Time computing
  double time_d = (double)(end_d-start_d)/CLOCKS_PER_SEC;
  double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;
  printf("n = %d \t GPU time = %fs \t CPU time = %fs\n", n, time_d, time_h);
  
  free(x);
  free(y);
}