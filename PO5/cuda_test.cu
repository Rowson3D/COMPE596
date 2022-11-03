#include <stdio.h>
#include <math.h>
#define cudaCheckError(){                                                            \
  cudaError_t e = cudaGetLastError();                                               \
  if(e != cudaSuccess) {                                                            \
      printf("cuda Failure %s, %d, '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));\
      exit(EXIT_FAILURE);                                                           \
  }                                                                                 \
}                                                                                   

__global__ void histogram(unsigned int * input, unsigned int * bins,
			  unsigned int num_elements, unsigned int num_bins)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while(i < num_elements)
    {
      atomicAdd(&bins[input[i]], 1);
      i += stride;
    }
}


int main(int argc, char * arv[])
{
  FILE *f;
  float x;
  unsigned long long inputLength = 1000000;
  unsigned int numBins = 4096;
  unsigned long long i = 0;
  unsigned int fileInput[inputLength]   = {0};
  unsigned int * deviceInput;
  unsigned int histogramOutput[numBins]       = {0};
  unsigned int * deviceOutput;
  f = fopen("float.dat", "r");
  while(!feof(f) && i < inputLength)
    {
      fscanf(f, "%f", &x);
      fileInput[i++] = ceil(x);
    }
  dim3 dimGrid(numBins);
  dim3 dimBlock(1024);
  cudaMalloc((void **) &deviceInput, inputLength*sizeof(int));
  cudaMalloc((void **) &deviceOutput, numBins*sizeof(int));
  cudaMemcpy(deviceInput, fileInput,inputLength*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceOutput, histogramOutput, numBins*sizeof(int), cudaMemcpyHostToDevice);
  histogram<<<dimGrid,dimBlock>>>(deviceInput, deviceOutput, inputLength, numBins);
  cudaDeviceSynchronize();
  cudaCheckError();
  cudaMemcpy(histogramOutput, deviceOutput, numBins*sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  printf("H = [...\n");
  for(unsigned i = 0;  i < numBins; i++)
    {
      printf("%d, %d;...\n", i, histogramOutput[i]);
    }
  printf("];\n");
}
