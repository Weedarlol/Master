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

    int total = width*height;
    int print_iter = iter;
    int dataPerGpu = (width*height)/gpus;
    int dataLeftover = (width*height)%gpus;
    int maxThreads = blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y*gridDim.z;
    int jacobiSize = ((width - 2) * (height - 2))/gpus;
    int amountPerThread = jacobiSize / maxThreads;
    int leftover = jacobiSize % maxThreads;

    clock_t start, end;

    float *mat;
    float **mat_gpu, **mat_gpu_tmp;

    int *device_nr;
    int **maxEps, *maxEps_print;

    cudaErrorHandle(cudaMallocHost(&mat, total*sizeof(float*)));

    cudaErrorHandle(cudaMallocHost(&mat_gpu,      gpus*sizeof(float*)));
    cudaErrorHandle(cudaMallocHost(&mat_gpu_tmp,  gpus*sizeof(float*)));
    cudaErrorHandle(cudaMallocHost(&device_nr,    gpus*sizeof(int*)));
    cudaErrorHandle(cudaMallocHost(&maxEps,       gpus*sizeof(int*)));
    cudaErrorHandle(cudaMallocHost(&maxEps_print, gpus*sizeof(int)));
    

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMalloc(&mat_gpu[g],     total*sizeof(float)));
        cudaErrorHandle(cudaMalloc(&mat_gpu_tmp[g], total*sizeof(float)));
        cudaErrorHandle(cudaMalloc(&maxEps[g],      maxThreads*sizeof(int*)));
        
        maxEps_print[g] = 1;
        device_nr[g] = g;
    }

    cudaStream_t streams[gpus];
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaStreamCreate(&streams[g]));
    }

    fillValues(mat, dx, dy, width, height);

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMemcpyAsync(mat_gpu[g], mat, total*sizeof(float), cudaMemcpyHostToDevice, streams[g]));
    }

    void ***kernelColl;
    cudaErrorHandle(cudaMallocHost(&kernelColl, gpus * sizeof(void**)));
    // Allocates the elements in the kernelColl, used for cudaLaunchCooperativeKernel as functon variables.
    for (int g = 0; g < gpus; g++) {
        void **kernelArgs = new void*[12];
        kernelArgs[0] = &mat_gpu[g];
        kernelArgs[1] = &mat_gpu_tmp[g];
        kernelArgs[2] = &maxEps[g];
        kernelArgs[3] = &device_nr[g];
        kernelArgs[4] = &dataLeftover;
        kernelArgs[5] = &eps;
        kernelArgs[6] = &width;
        kernelArgs[7] = &height;
        kernelArgs[8] = &iter;
        kernelArgs[9] = &jacobiSize;
        kernelArgs[10] = &amountPerThread;
        kernelArgs[11] = &leftover;
        
        kernelColl[g] = kernelArgs;
    }

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaStreamSynchronize(streams[g]));
    }


    start = clock();
    // ________________________________________________________
    while(iter > 0 && maxEps_print[0] != 0){
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobi, gridDim, blockDim, kernelColl[g], 0, streams[g]));
            cudaErrorHandle(cudaMemcpyAsync(&maxEps_print[g], maxEps[g], sizeof(int), cudaMemcpyDeviceToHost, streams[g]));
        }

        if(gpus > 1){
            for(int g = 0; g < dataLeftover; g++){
                cudaErrorHandle(cudaSetDevice(g)); // Unnecessary?
                if(g == 0){
                    // Transfers data device 0 -> device 1
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[1] + dataPerGpu+1-width, 1, mat_gpu_tmp[0] + dataPerGpu+1-width, 0, width*sizeof(float), streams[g]));
                }
                else if(g < gpus-1){
                    // Transfers data device g -> device g+1
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g+1] + dataPerGpu*g+1-width, g+1, mat_gpu_tmp[g] + dataPerGpu*g+1-width, g, width*sizeof(float), streams[g]));
                    // Transfers data device g-1 -> device g
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + dataPerGpu*g+1 , g-1, mat_gpu_tmp[g] + dataPerGpu*g+1, g, width*sizeof(float), streams[g]));
                }
            }
            for(int g = dataLeftover; g < gpus; g++){
                cudaErrorHandle(cudaSetDevice(g)); // Unnecessary?
                if(g < gpus-1){
                    // Transfers data device g -> device g+1
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g+1] + dataPerGpu*g-width, g+1, mat_gpu_tmp[g] + dataPerGpu*g-width, g, width*sizeof(float), streams[g]));
                    // Transfers data device g-1 -> device g
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + dataPerGpu*g , g-1, mat_gpu_tmp[g] + dataPerGpu*g, g, width*sizeof(float), streams[g]));
                }
                else{
                    // Transfers data device -1 -> device -2
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + dataPerGpu*g, g-1, mat_gpu_tmp[g] + dataPerGpu*g, g, width*sizeof(float), streams[g]));
                }
            }
        }

        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaStreamSynchronize(streams[g]));
        }

        for(int g = 1; g < gpus; g++){
            maxEps_print[0] += maxEps_print[g];
        }

        for(int g = 0; g < gpus; g++){
            float *mat_change = mat_gpu[g];
            mat_gpu[g] = mat_gpu_tmp[g];
            mat_gpu_tmp[g] = mat_change;
        }

        iter--;

        
    }


    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMemcpyAsync(mat + (total/gpus)*g, mat_gpu[g] + (total/gpus)*g, (total/gpus)*sizeof(float), cudaMemcpyDeviceToHost, streams[g]));
    }

    end = clock();

    

    cudaErrorHandle(cudaDeviceSynchronize());
    
    printf("\nout %i\n", maxEps_print[0]);

    if(iter != 0){
        printf("The computation found a solution with %i gpus. It computed it within %i iterations (%i - %i) and %.3f seconds.\nWidth = %i, Height = %i\nthreadBlock = (%d, %d, %d), gridDim = (%d, %d, %d)\n\n", 
        gpus, print_iter - iter, print_iter, iter, ((double) (end - start)) / CLOCKS_PER_SEC, width, height, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    }
    else{
        printf("The computation did not find a solution with %i gpus after all its iterations, it ran = %i iterations (%i - %i). It completed it in %.3f seconds.\nWidth = %i, Height = %i\nthreadBlock = (%d, %d, %d), gridDim = (%d, %d, %d)\n\n", 
        gpus, print_iter - iter, print_iter, iter, ((double) (end - start)) / CLOCKS_PER_SEC, width, height, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    }

    for(int i = 0; i < 20; i++){
        for(int j = 0; j < 20; j++){
            printf("%.3f, ", mat[j + i*width]);
        }
        printf("\n");
    }

    cudaErrorHandle(cudaFreeHost(mat));
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaFree(mat_gpu[g]));
        cudaErrorHandle(cudaFree(mat_gpu_tmp[g]));
        cudaErrorHandle(cudaFree(maxEps[g]));
    }
    cudaErrorHandle(cudaFreeHost(mat_gpu));
    cudaErrorHandle(cudaFreeHost(mat_gpu_tmp));
    cudaErrorHandle(cudaFreeHost(maxEps));
    cudaErrorHandle(cudaFreeHost(device_nr));
    cudaErrorHandle(cudaFreeHost(maxEps_print));
    cudaErrorHandle(cudaFreeHost(kernelColl));
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
