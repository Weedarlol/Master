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
            mat[j + i*width] = i*width+j;
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
    int *device_nr;
    cudaErrorHandle(cudaGetDeviceCount(&gpus));
    cudaErrorHandle(cudaMallocHost(&device_nr, gpus*sizeof(float*)));

    int total = width*height;
    int jacobiSize = (width-2)*(height-2);




    // Ignores first and last row
    int row_extra = (height-2)%gpus;
    int *rows;
    cudaErrorHandle(cudaMallocHost(&rows, gpus*sizeof(float*)));
    // Finds the number of rows that each device should compute
     for(int g = 0; g < gpus; g++){
        // Adds number of rodes to each based on height and width
        if(g < row_extra){
            rows[g] = (height-2)/gpus+1;
        }
        else{
            rows[g] = (height-2)/gpus;
        }

        // Adds number of roder to each based on neighboring dervices
        if(g == 0 || g == gpus-1){
            rows[g] += 1;
        }
        else{
            rows[g] += 2;
        }

        // Adds device_nr for each GPU so it knows which device number it is
        device_nr[g] = g;
    }
    
    


    int *threadInformation;
    cudaErrorHandle(cudaMallocHost(&threadInformation, 4*sizeof(int*)));
    threadInformation[0] = blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y*gridDim.z; // Threads per device
    threadInformation[1] = row_extra; // How many rows there are extra
    threadInformation[2] = rows[gpus-1]-1; // How many rows per device, without adding ghost point, rounded down






    float *mat;
    float **mat_gpu, **mat_gpu_tmp;
    cudaErrorHandle(cudaMallocHost(&mat, total*sizeof(float*)));
    cudaErrorHandle(cudaMallocHost(&mat_gpu,      gpus*sizeof(float*)));
    cudaErrorHandle(cudaMallocHost(&mat_gpu_tmp,  gpus*sizeof(float*)));
    // Allocates memory on devices based on number of rows for each device
    for(int g = 0; g < gpus; g++){
        if(g == 0 || g == gpus-1){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaMalloc(&mat_gpu[g],     width*rows[g]*sizeof(float)));
            cudaErrorHandle(cudaMalloc(&mat_gpu_tmp[g], width*rows[g]*sizeof(float)));
        }
        else{
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaMalloc(&mat_gpu[g],     width*rows[g]*sizeof(float)));
            cudaErrorHandle(cudaMalloc(&mat_gpu_tmp[g], width*rows[g]*sizeof(float)));
        }
    }

    cudaStream_t streams[gpus];
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaStreamCreate(&streams[g]));
    }

    fillValues(mat, dx, dy, width, height);


    int sum_of_row = 0;
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMemcpyAsync(mat_gpu[g], mat+width+sum_of_row, rows[g]*width*sizeof(float), cudaMemcpyHostToDevice, streams[g]));
        sum_of_row += rows[g];
    }

    void ***kernelColl;
    cudaErrorHandle(cudaMallocHost(&kernelColl, gpus * sizeof(void**)));
    // Allocates the elements in the kernelColl, used for cudaLaunchCooperativeKernel as functon variables.
    for (int g = 0; g < gpus; g++) {
        void **kernelArgs = new void*[5];
        kernelArgs[0] = &mat_gpu[g];
        kernelArgs[1] = &mat_gpu_tmp[g];
        kernelArgs[2] = &rows[g];
        kernelArgs[3] = &device_nr[g];
        kernelArgs[4] = &gpus;
        
        kernelColl[g] = kernelArgs;
    }


    while(iter > 0){
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobi, gridDim, blockDim, kernelColl[g], 0, streams[g]))
        }

        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaStreamSynchronize(streams[g]));
        }

        iter--;
    }








    printf("FUNGERER!\n");



    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaStreamSynchronize(streams[g]));
    }

    cudaErrorHandle(cudaFreeHost(mat));
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaFree(mat_gpu[g]));
        cudaErrorHandle(cudaFree(mat_gpu_tmp[g]));
    }
    cudaErrorHandle(cudaFreeHost(mat_gpu));
    cudaErrorHandle(cudaFreeHost(mat_gpu_tmp));

    cudaErrorHandle(cudaFreeHost(rows));
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
    int width = 100;
    int height = 100;
    int iter = 100;

    float eps = 1.0e-14;
    float dx = 2.0 / (width - 1);
    float dy = 2.0 / (height - 1);

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(16, 1, 1);

    start(width, height, iter, eps, dx, dy, blockDim, gridDim);

    return 0;
}
