#include <stdio.h>
#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)

#define cudaCheckError()                                                                   \
   {                                                                                       \
      cudaError_t e = cudaGetLastError();                                                  \
      if (e != cudaSuccess)                                                                \
      {                                                                                    \
         printf("cuda Failure %s, %d, '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
         exit(EXIT_FAILURE);                                                               \
      }                                                                                    \
   }

__global__ void convolution(unsigned int *I, const int *__restrict__ M, int *P, int channels, int width, int height)
{
   __shared__ int N_ds[w][w]; // threads share this memory (padded square workspace or “surface area differential ds”)
   int k;
   for (k = 0; k < channels; k++)
   {
      int dest = threadIdx.y * TILE_WIDTH + threadIdx.x, // local block 16x16 tile offset
          destY = dest / w,                              // destination in outer workspace N_ds of dimension w x w enclosing tile
          destX = dest % w,
          srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius, // we are mapping an input 1080p image onto a 120 x 68 grid of 16x16 blocks
          srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius,
          src = (srcY * width + srcX) * channels + k; // 3D nested offset to input image I
      if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
      {
         N_ds[destY][destX] = I[src]; // copy source pixel value from global grid shared memory to local block N_ds workspace
      }
      else
      {
         N_ds[destY][destX] = 0;
      }
      __syncthreads(); // barrier
      int accum = 0;
      int y, x;
      for (y = 0; y < Mask_width; y++)
      {
         for (x = 0; x < Mask_width; x++)
         {
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * Mask_width + x]; // parallel inner product computation
         }
      }
      y = blockIdx.y * TILE_WIDTH + threadIdx.y;
      x = blockIdx.x * TILE_WIDTH + threadIdx.x;
      if (y < height && x < width)
         P[(y * width + x) * channels + k] = accum; // 3D offset
      __syncthreads();                              // barrier
   }
}

int main(int argc, char *argv[])
{
   unsigned int *hostInputImage;
   unsigned int *hostOutputImage;
   unsigned int inputLength = 589824; // 384 * 512 * 3 = 589824
   printf("%% Importing 3-channel image data and creating memory on host\n");
   hostInputImage = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
   hostOutputImage = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
   FILE *f;
   unsigned int pixelValue = 0;
   unsigned int i = 0;
   f = fopen("peppers.dat", "r");
   while (!feof(f) && i < inputLength)
   {
      fscanf(f, "%d", &pixelValue);
      hostInputImage[i++] = pixelValue;
   }
   fclose(f);
   printf("%% Finished Importing Data\n");

   int maskRows = MASK_WIDTH;
   int maskColumns = MASK_WIDTH;
   int imageChannels = 3;
   int imageWidth = 512;
   int imageHeight = 384;

   // Sobel 5x5 horizontal convolution kernel for edge detection
   int hostMask[MASK_WIDTH][MASK_WIDTH] = {
       {2, 2, 4, 2, 2},
       {1, 1, 2, 1, 1},
       {0, 0, 0, 0, 0},
       {-1, -1, -2, -1, -1},
       {-2, -2, -4, -2, -2}};

   unsigned int *deviceInputImage;
   unsigned int *deviceOutputImage;
   unsigned int *deviceMask;
   cudaMalloc((void **)&deviceInputImage,
              imageWidth * imageHeight * imageChannels * sizeof(int));
   cudaMalloc((void **)&deviceOutputImage,
              imageWidth * imageHeight * imageChannels * sizeof(int));
   cudaMalloc((void **)&deviceMask, maskRows * maskColumns * sizeof(int));
   cudaMemcpy(deviceInputImage,
              hostInputImage,
              imageWidth * imageHeight * imageChannels * sizeof(int),
              cudaMemcpyHostToDevice);
   cudaMemcpy(deviceMask,
              hostMask,
              maskRows * maskColumns * sizeof(int),
              cudaMemcpyHostToDevice);

   dim3 dimGrid(ceil((float)imageWidth / TILE_WIDTH),
                ceil((float)imageHeight / TILE_WIDTH));
   dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
   printf("%% Starting Convilution\n");

   convolution<<<dimGrid, dimBlock>>>(deviceInputImage,
                                      deviceMask,
                                      deviceOutputImage,
                                      imageChannels, imageWidth, imageHeight);

   printf("%% Finished Convolution\n");
   cudaCheckError();
   cudaDeviceSynchronize();

   cudaMemcpy(hostOutputImage,
              deviceOutputImage,
              imageWidth * imageHeight * imageChannels * sizeof(int),
              cudaMemcpyDeviceToHost);

   f = fopen("peppers.out", "w");
   for (int i = 0; i < inputLength; ++i)
   {
      printf("%d\n", hostOutputImage[i]);
      fprintf(f, "%d\n", hostOutputImage[i]);
   }
   fclose(f);

   cudaFree(deviceInputImage);
   cudaFree(deviceOutputImage);
   cudaFree(deviceMask);

   free(hostInputImage);
   free(hostOutputImage);
   return (0);
}