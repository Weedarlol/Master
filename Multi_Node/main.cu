#include <stdio.h>
#include <math.h>
#include <time.h>


#include "jacobi.h"
#include <nvtx3/nvToolsExt.h>

// https://ori-cohen.medium.com/real-life-cuda-programming-part-4-error-checking-e66dcbad6b55
#define cudaErrorHandle(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) 
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

void fillValues(float *mat, float dx, float dy, int width, int height){
    float x, y;

    memset(mat, 0, height*width*sizeof(float));

    for(int i = 1; i < height - 1; i++) {
        y = i * dy; // y coordinate
        for(int j = 1; j < width - 1; j++) {
            x = j * dx; // x coordinate
            mat[j + i*width] = sin(M_PI*y)*sin(M_PI*x);
        }
    }
}


void start(int width, int height, int iter, float eps, float dx, float dy, dim3 blockDim, dim3 gridDim){
    /*
    Variables   | Type  | Description
    gpus        | int   | Number of available gpus
    total       | int   | Total number of elements in the matrix
    dataPerGpu  | int   | Number of elements given to each GPU
    maxThreads  | int   | Total number of available threads within the grid_g group
    jacobiSize  | int   | Number of elements in the matrix which is to be calculated each iteration
    amountPerThread|int | Number of elements to be calculated by each thread each iteration
    leftover    | int   | Number of threads which is required to compute one more element to calculate all the elements

    start       |clock_t| Start timer of area of interest
    end         |clock_t| End timer of area of interest

    mat         |*float | Pointer to the matrix allocated in the CPU
    mat_gpu     |**float| An array of pointers, where each pointer points at an device, specifically a matrix within that device
    mat_gpu_tmp |**float| An array of pointers, where each pointer points at an device, specifically a matrix within that device
    maxEps      |**int  | An array of pointers, where each pointer points at an device, specifically an array within that device that checks if the elements is in an acceptable state
    maxEps_print|*int   | Variable used to for the CPU to check if the GPUs are finished
    device_nr   |*int   | An array used to send the GPU number to each device when computing
    */
    
    int gpus;
    cudaErrorHandle(cudaGetDeviceCount(&gpus));

    // GPUDirect
    for(int g = 0; g < gpus; g++){
        cudaSetDevice(g);
        for(int j = 0; j < gpus; j++){
            cudaDeviceEnablePeerAccess(g, j);
        }
    }

    float* dataOnGPU0;
    float* dataOnGPU1;
    cudaErrorHandle(cudaMalloc((void**)&dataOnGPU0, sizeof(float) * width * height));
    cudaErrorHandle(cudaMalloc((void**)&dataOnGPU1, sizeof(float) * width * height));

    // Fill the data on GPU 0 with some values.

    // Transfer data from GPU 0 to GPU 1 using cudaMemcpyPeer.
    cudaErrorHandle(cudaMemcpyPeer(dataOnGPU1, 1, dataOnGPU0, 0, sizeof(float) * width * height));

    // Cleanup and deallocate memory if needed.

    // Disable P2P access if you no longer need it.
    cudaErrorHandle(cudaDeviceDisablePeerAccess(1));
    cudaErrorHandle(cudaFree(dataOnGPU0));

    
}



int main() {
    /*
    Functions   | Type           | Input
    start       | void           | int width, int height, int iter, float eps,
                                   float dx, float dy, dim3 blockDim,
                                   dim3 gridDim

    fillValues  | void           | float *mat, float dx, float dy, int width,
                                   int height

    jacobi      |__global__ void | float *mat_gpu, float *mat_tmp, float eps,
                                   int width, int height, int iter

    ____________________________________________________________________________
    Variables   | Type  | Description
    width       | int   | The width of the matrix
    height      | int   | The height of the matrix
    iter        | int   | Number of max iterations for the jacobian algorithm

    eps         | float | The limit for accepting the state of the matrix during jacobian algorithm
    dx          | float | Distance between each element in the matrix in x direction
    dy          | float | Distance between each element in the matrix in y direction

    blockDim    | dim3  | Number of threads in 3 directions for each block
    gridDim     | dim3  | Number of blocks in 3 directions for the whole grid
    */
    int width = 512;
    int height = 512;
    int iter = 500000;

    float eps = 1.0e-14;
    float dx = 2.0 / (width - 1);
    float dy = 2.0 / (height - 1);

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(16, 1, 1);

    start(width, height, iter, eps, dx, dy, blockDim, gridDim);

    return 0;
}
