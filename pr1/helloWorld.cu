// Hola Mundo

#include <stdio.h>
#include <stdlib.h>
#include "helper_cuda.h"
#include "helper_timer.h"

__global__ void cuda_hello(){
    printf("¡¡Hello World!! I am the thread %d: block %d, thread %d \n",
            (blockDim.x*blockIdx.x+threadIdx.x), blockIdx.x, threadIdx.x);
}

int main() {
   cuda_hello<<<2,16>>>(); 
   cudaDeviceSynchronize();

   return 0;
}
