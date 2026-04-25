// Suma de vectores usando la memoria global de la GPU

#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <math.h>

typedef int *vector;

// Timers: kernel and application
StopWatchInterface *hTimer = NULL;
StopWatchInterface *kTimer = NULL;

// Function for generating random values for a vector
void LoadStartValuesIntoVectorRand(vector V, unsigned int n)
{
  unsigned int i;

  for (i = 0; i < n; i++)
    V[i] = (int)(random() % 9);
}

// Function for printing a vector
void PrintVector(vector V, unsigned int n)
{
  unsigned int i;

  for (i = 0; i < n; i++)
    printf("%3d", V[i]);
  printf("\n");
}

// CUDA Kernels
__global__ void SumVectorA(vector A, vector B, vector C, unsigned int n, int comp)
{
  // Each thread will compute the sum of 'comp' consecutive components of the vectors
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * comp;

  // check that we do not access out of bounds elements
  for (int j = 0; j < comp && i + j < n; j++)
    C[i + j] = A[i + j] + B[i + j];
}

__global__ void SumVectorB(vector A, vector B, vector C, unsigned int n, int comp, int dist)
{
  // Each thread will compute the sum of 'comp' components of the vectors separated by 'dist'
  // positions
  int i = blockIdx.x * blockDim.x * comp + threadIdx.x;

  if (i >= n) return;
  for (int j = 0; j < comp; j++) {
    int idx = i + j * dist;
    if (idx < n)
      C[idx] = A[idx] + B[idx];
  }
}

// ------------------------
// MAIN function
// ------------------------
int main(int argc, char **argv)
{

  unsigned int n, size;
  float timerValue;
  double ops;
  int comp;       // number of components per thread
  char kernel;    // kernel to execute
  int debug;      // debug mode
  int block_size; // threads per block

  if (argc == 6)
  {
    n = atoi(argv[1]);
    comp = atoi(argv[2]);
    kernel = argv[3][0];
    debug = atoi(argv[4]);
    block_size = atoi(argv[5]);
  }
  else
  {
    printf("Sintaxis: <ejecutable> <n> <comp> <kernel> <debug> <block_size>\n");
    printf("\t- <kernel>: a o b\n");
    printf("\t- <debug>: 0 o 1\n");
    printf("Ejemplo: ./sumVectors 1024 1 a 0 32\n");
    exit(0);
  }

  srandom(12345);
  size = n * sizeof(int);

  // Define vectors at host
  vector hA, hB, hC;

  // Pointer to vectors into device
  vector dA, dB, dC;

  // timers
  sdkCreateTimer(&hTimer);
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);

  // Load values into hA
  hA = (int *)malloc(size);
  LoadStartValuesIntoVectorRand(hA, n);
  // printf("\nPrinting Vector hA  %d\n",n);
  // PrintVector(hA,n);

  // Load values into hB
  hB = (int *)malloc(size);
  LoadStartValuesIntoVectorRand(hB, n);
  // printf("\nPrinting Vector hB  %d\n",n);
  // PrintVector(hB,n);

  // Start hC
  hC = (int *)malloc(size);

  // Allocate device memory
  checkCudaErrors(cudaMalloc((void **)&dA, size));
  checkCudaErrors(cudaMalloc((void **)&dB, size));
  checkCudaErrors(cudaMalloc((void **)&dC, size));

  // Copy vectors from host to device
  checkCudaErrors(cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice));

  // Setup execution parameters
  dim3 threads(block_size);
  dim3 grid((int)ceil((float)n / ((float)block_size * comp)));

  // Timers
  sdkCreateTimer(&kTimer);
  sdkResetTimer(&kTimer);
  sdkStartTimer(&kTimer);

  // Execute the kernel
  if (kernel == 'a')
    SumVectorA<<<grid, threads>>>(dA, dB, dC, n, comp);
  else if (kernel == 'b')
  {
    int dist = block_size; // stride = threads per block, so each block covers block_size*comp elements
    SumVectorB<<<grid, threads>>>(dA, dB, dC, n, comp, dist);
  }
  else
  {
    printf("Unrecognized kernel\n");
    exit(0);
  }

  cudaDeviceSynchronize();

  sdkStopTimer(&kTimer);

  // Copy result vector from device to host
  checkCudaErrors(cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost));

  // Print its value
  // printf("\nPrinting vector hC  %d\n",n);
  // PrintVector(hC,n);

  sdkStopTimer(&hTimer);

  timerValue = sdkGetTimerValue(&kTimer);
  timerValue = timerValue / 1000;
  sdkDeleteTimer(&kTimer);
  if (debug)
  {
    printf("\n----- Debug mode -----");
    printf("\nPrinting Vector hA [%d]\n", n);
    PrintVector(hA, n);
    printf("\nPrinting Vector hB [%d]\n", n);
    PrintVector(hB, n);
    printf("\nResult (vector hC) [%d]\n", n);
    PrintVector(hC, n);
    printf("----------------------\n\n\n");
  }

  // Free vectors
  free(hA);
  free(hB);
  free(hC);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  printf("----- Stats -----\n");
  printf("Kernel time: %f s\n", timerValue);

  timerValue = sdkGetTimerValue(&hTimer);
  timerValue = timerValue / 1000;
  sdkDeleteTimer(&hTimer);
  printf("Total time: %f s\n", timerValue);
  ops = n / timerValue;
  printf("%f Gops/s \n", ops / 1000000000);

  return 0;
}
