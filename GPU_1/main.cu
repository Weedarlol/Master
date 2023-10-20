#include <stdio.h>
#include <math.h>
#include <nvtx3/nvToolsExt.h>
#include <time.h>
#include "jacobi.h"

#include <cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;


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
    total       | int   | Total number of elements in the matrix
    dataPerGpu  | int   | Number of elements per available gpu

    start       |clock_t| Start timer of program
    end         |clock_t| End timer of program

    mat         |*float | Pointer to the allocated matrix in the CPU
    mat_gpu     |**float| Pointer to an allocated matrix in the GPU
    mat_gpu_tmp |**float| Pointer to an allocated matrix in the GPU
    maxEps      |*int   | Pointer to an allocated vector in the GPU used for checking if the matrix is in an acceptable state
    comp_suc    |*int   | Checks if the computation is successfull or not
    */

    int total = width*height;
    int print_iter = iter;
    clock_t start, end;


    float *mat, *mat_gpu, *mat_gpu_tmp;
    cudaErrorHandle(cudaMallocHost(&mat, total*sizeof(float*)));
    cudaErrorHandle(cudaMalloc(&mat_gpu, total*sizeof(float*)));
    cudaErrorHandle(cudaMalloc(&mat_gpu_tmp, total*sizeof(float*)));
    

    int *maxEps, *comp_suc;;
    cudaErrorHandle(cudaMalloc(&maxEps, blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y*gridDim.z*sizeof(int*)));
    cudaErrorHandle(cudaMallocHost(&comp_suc, sizeof(int*)));


    /* initialization */
    fillValues(mat, dx, dy, width, height);
    





    // Here we are done with the allocation, and start with the compution
    nvtxRangePushA("Area of Interest");
    start = clock();

    // Copies elemts over from CPU to the device.
    cudaErrorHandle(cudaMemcpyAsync(mat_gpu, mat, total*sizeof(float), cudaMemcpyHostToDevice));
    cudaErrorHandle(cudaMemsetAsync(mat_gpu_tmp, 0, total*sizeof(float)));

    /* // Check if device can run Cooperative groups
    int dev = 0;
    int supportsCoopLaunch = 0;
    cudaErrorHandle(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev));
    if(supportsCoopLaunch){
        printf("Device support CudaCooperativeLaunch\n");
    } */

    // Creates an array where its elements are features in cudaLaunchCooperativeKernel
    void *kernelArgs[] = {&mat_gpu, &mat_gpu_tmp, &eps, &width, &height, &iter, &maxEps};


    // Runs device
    // jacobi<<<gridDim, blockDim>>>(mat_gpu, mat_gpu_tmp, eps, width, height, iter, maxEps);
    cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobi, gridDim, blockDim, kernelArgs));

    cudaErrorHandle(cudaDeviceSynchronize());

    // Copies back value from device i to CPU
    cudaErrorHandle(cudaMemcpy(mat, mat_gpu, total*sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaErrorHandle(cudaMemcpy(comp_suc, maxEps, sizeof(int*), cudaMemcpyDeviceToHost));

    cudaErrorHandle(cudaDeviceSynchronize());

    end = clock();
    nvtxRangePop();







    if(*comp_suc != 0){
        printf("The computation found a solution. It computed it within %i iterations (%i - %i) and %.3f seconds.\nWidth = %i, Height = %i\nthreadBlock = (%d, %d, %d), gridDim = (%d, %d, %d)\n\n", 
        print_iter - *comp_suc, print_iter, *comp_suc, ((double) (end - start)) / CLOCKS_PER_SEC, width, height, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    }
    else{
        printf("The computation did not find a solution after all its iterations, it ran = %i iterations (%i - %i). It completed it in %.3f seconds.\nWidth = %i, Height = %i\nthreadBlock = (%d, %d, %d), gridDim = (%d, %d, %d)\n\n", 
        print_iter - *comp_suc, print_iter, *comp_suc, ((double) (end - start)) / CLOCKS_PER_SEC, width, height, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    }

    cudaErrorHandle(cudaFreeHost(mat));
    cudaErrorHandle(cudaFree(mat_gpu));
    cudaErrorHandle(cudaFree(mat_gpu_tmp));
}



int main() {
    /*
    Functions   | Type           | Input
    start       | void           | int width, int height, int iter, float eps, float dx, float dy, dim3 blockDim, dim3 gridDim

    fillValues  | void           | float *mat, float dx, float dy, int width, int height

    jacobi      |__global__ void | float *mat_gpu, float *mat_gpu_tmp, float eps, int width, int height, int iter

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
    int iter = 10000000;
    
    // Burde stoppe etter 409020

    float eps = 1.0e-14;
    float dx = 2.0 / (width - 1);
    float dy = 2.0 / (height - 1);

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(16, 1, 1);


    start(width, height, iter, eps, dx, dy, blockDim, gridDim);

    return 0;
}
