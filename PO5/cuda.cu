#include <stdio.h>

#define NUM_BINS 4096

// Provides feedback to the user of any CUDA errors that occur
#define CUDA_E(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"Device Error: %s %s line: %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel to compute a histogram of the input data
__global__ void histogram_kernel(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Define the stride
    int stride = blockDim.x * gridDim.x;
    // Loop over the input data
    while (i < num_elements) {
        // Increment the bin corresponding to the input value
        atomicAdd(&bins[input[i]], 1);
        // Increment the index
        i += stride + 1;
    }
}

void histogram(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins) 
{
  unsigned int *d_input, *d_bins;
  unsigned int num_threads = 512;
  unsigned int num_blocks = 4096;

  CUDA_E(cudaMalloc((void **)&d_input, num_elements * sizeof(unsigned int)));
  CUDA_E(cudaMalloc((void **)&d_bins, num_bins * sizeof(unsigned int)));
  CUDA_E(cudaDeviceSynchronize());

  CUDA_E(cudaMemcpy(d_input, input, num_elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
  CUDA_E(cudaMemset(d_bins, 0, num_bins * sizeof(unsigned int)));

  histogram_kernel<<<num_blocks, num_threads>>>(d_input, d_bins, num_elements, num_bins);

  CUDA_E(cudaMemcpy(bins, d_bins, num_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

int main(int argc, char **argv) 
{
  unsigned int *input, *bins;
  unsigned int num_elements = 1000000;
  unsigned int num_bins = NUM_BINS;

  input = (unsigned int *)malloc(num_elements * sizeof(unsigned int));
  bins = (unsigned int *)malloc(num_bins * sizeof(unsigned int));

// Read the file
    FILE *fp;
    float x;
    int i = 0;
    fp = fopen("float.dat", "r");
    if (fp == NULL) {
        printf("Error opening file\n");
        exit(1);
    }
    
    while(!feof(fp) && i < num_elements) 
    {
        fscanf(fp, "%f", &x);
        // Using the ceil() for bin val >= x function
        input[i] = (unsigned int)(ceil(x));
        i++;
    }
    fclose(fp);

    histogram(input, bins, num_elements, num_bins);

    printf("h = [\n");
    for (unsigned int i = 0; i < num_bins; i++) {
        printf("%d, %d\n", i, bins[i]);
    }
    printf("];");

    free(input);
    free(bins);

    return 0;
}
    

