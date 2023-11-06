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

    int total = width*height;
    int jacobiSize = (width-2)*(height-2);
    int print_iter = iter;

    clock_t start, end;

    int gpus;
    int *device_nr;
    cudaErrorHandle(cudaGetDeviceCount(&gpus));
    cudaErrorHandle(cudaMallocHost(&device_nr, gpus*sizeof(int*)));
    for(int g = 0; g < gpus; g++){
        device_nr[g] = g;
    }


    int **maxEps, *maxEps_print;
    cudaErrorHandle(cudaMallocHost(&maxEps,       gpus*sizeof(int*)));
    cudaErrorHandle(cudaMallocHost(&maxEps_print, gpus*sizeof(int)));


    // Ignores first and last row
    int rows_total = height-2;
    int rows_leftover = rows_total%gpus;
    int rows_per_device = rows_total/gpus;
    int *rows_device, *rows_index, *rows_compute;
    cudaErrorHandle(cudaMallocHost(&rows_device, gpus*sizeof(int*)));
    cudaErrorHandle(cudaMallocHost(&rows_index, gpus*sizeof(int*)));
    cudaErrorHandle(cudaMallocHost(&rows_compute, gpus*sizeof(int*)));
    // Calculate the number of rows for each device
    for (int g = 0; g < gpus; g++) {
        int extra_row = (g < rows_leftover) ? 1 : 0;
  
        rows_device[g] = rows_per_device + extra_row + 2;

        rows_compute[g] = rows_per_device + extra_row;

        rows_index[g] = g * rows_per_device + min(g, rows_leftover);
    }



    


    int *threadInformation;
    int threadSize = blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y*gridDim.z;
    cudaErrorHandle(cudaMallocHost(&threadInformation, 4*sizeof(int*)));
    threadInformation[0] = (rows_compute[0]     *(width-2))/threadSize; // Find number of elements to compute for each thread, ignoring border elements.
    threadInformation[1] = (rows_compute[0]     *(width-2))%threadSize; // Finding which threads require 1 more element
    threadInformation[2] = (rows_compute[gpus-1]*(width-2))/threadSize; // Find number of elements to compute for each thread, ignoring border elements. -1 because of ghost row
    threadInformation[3] = (rows_compute[gpus-1]*(width-2))%threadSize; // Finding which threads require 1 more element

    for(int g = 0; g < gpus; g++){
        printf("%i -> index = %i, device = %i, compute = %i, leftover = %i, element_per_thread = %i, extra_element = %i\n", g, rows_index[g], rows_device[g], rows_compute[g], rows_leftover, threadInformation[2], threadInformation[3]);
    }



    float *mat;
    float **mat_gpu, **mat_gpu_tmp;
    cudaErrorHandle(cudaMallocHost(&mat,          total*sizeof(float*)));
    cudaErrorHandle(cudaMallocHost(&mat_gpu,      gpus*sizeof(float*)));
    cudaErrorHandle(cudaMallocHost(&mat_gpu_tmp,  gpus*sizeof(float*)));
    // Allocates memory on devices based on number of rows for each device
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMalloc(&mat_gpu[g],     width*rows_device[g]*sizeof(float)));
        cudaErrorHandle(cudaMalloc(&mat_gpu_tmp[g], width*rows_device[g]*sizeof(float)));
        cudaErrorHandle(cudaMalloc(&maxEps[g],      threadSize*sizeof(int*)));
        maxEps_print[g] = 1;
    }


    cudaStream_t streams[gpus];
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaStreamCreate(&streams[g]));
    }
    

    fillValues(mat, dx, dy, width, height);


    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMemcpyAsync(mat_gpu[g], mat+(rows_index[g])*width, rows_device[g]*width*sizeof(float), cudaMemcpyHostToDevice, streams[g]));
    }

    void ***kernelColl;
    cudaErrorHandle(cudaMallocHost(&kernelColl, gpus * sizeof(void**)));
    // Allocates the elements in the kernelColl, used for cudaLaunchCooperativeKernel as functon variables.
    for (int g = 0; g < gpus; g++) {
        void **kernelArgs = new void*[13];
        kernelArgs[0] = &mat_gpu[g];
        kernelArgs[1] = &mat_gpu_tmp[g];
        kernelArgs[2] = &rows_device[g]; // How many rows for each device
        kernelArgs[3] = &width;
        kernelArgs[4] = &height;
        kernelArgs[5] = &rows_leftover; // Tells how many of the devices will have 1 extra row
        kernelArgs[6] = &device_nr[g];
        kernelArgs[7] = &rows_compute[g];
        kernelArgs[8] = &threadInformation[0];
        kernelArgs[9] = &threadInformation[1];
        kernelArgs[10] = &threadInformation[2];
        kernelArgs[11] = &threadInformation[3];
        kernelArgs[12] = &maxEps[g];
        kernelArgs[13] = &eps;

        kernelColl[g] = kernelArgs;
    }





    // FØR __________________________________________________________


    start = clock();
    while(iter > 0 && maxEps_print[0] != 0){
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobi, gridDim, blockDim, kernelColl[g], 0, streams[g]));
            cudaErrorHandle(cudaMemcpyAsync(&maxEps_print[g], maxEps[g], sizeof(int), cudaMemcpyDeviceToHost, streams[g]));
        }

        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaStreamSynchronize(streams[g]));
        }

        if(gpus > 1){
            for(int g = 0; g < gpus; g++){
                cudaErrorHandle(cudaSetDevice(g));
                if(g == 0){
                    // Transfers data device 0 -> device 1
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[1],     1, mat_gpu_tmp[0] + (rows_compute[0])*width, 0, width*sizeof(float), streams[g]));
                }
                else if(g < gpus-1){
                    // Transfers data device g -> device g+1
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g+1], g+1, mat_gpu_tmp[g] + (rows_compute[g])*width, g, width*sizeof(float), streams[g]));
                    // Transfers data device g -> device g-1
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + (rows_compute[g-1]+1)*width, g-1, mat_gpu_tmp[g] + width, g, width*sizeof(float), streams[g]));
                }
                else{
                    // Transfers data device -1 -> device -2
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + (rows_compute[g-1]+1)*width, g-1, mat_gpu_tmp[g] + width, g, width*sizeof(float), streams[g]));
                }  
            }
        }

        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaStreamSynchronize(streams[g]));
        }

        for(int g = 1; g < gpus-1; g++){
            maxEps_print[0] += maxEps_print[g];
        }


        for(int g = 0; g < gpus; g++){
            float *mat_change = mat_gpu[g];
            mat_gpu[g] = mat_gpu_tmp[g];
            mat_gpu_tmp[g] = mat_change;
        }


        
        iter--;
    }
    end = clock();


    

    // ETTER __________________________________________________________




    float *mat_tmp;
    cudaErrorHandle(cudaMallocHost(&mat_tmp, total*sizeof(float)));
    cudaErrorHandle(cudaMemset(mat_tmp, 1, total*sizeof(float)));


    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMemcpyAsync(mat_tmp + (rows_index[g]+1)*width, mat_gpu[g] + width, rows_compute[g]*width*sizeof(float), cudaMemcpyDeviceToHost, streams[g]));
    }

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaStreamSynchronize(streams[g]));
    }

    printf("\nout %i\n", maxEps_print[0]);

    if(iter != 0){
        printf("The computation found a solution with %i gpus. It computed it within %i iterations (%i - %i) and %.3f seconds.\nWidth = %i, Height = %i\nthreadBlock = (%d, %d, %d), gridDim = (%d, %d, %d)\n\n", 
        gpus, print_iter - iter, print_iter, iter, ((double) (end - start)) / CLOCKS_PER_SEC, width, height, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    }
    else{
        printf("The computation did not find a solution with %i gpus after all its iterations, it ran = %i iterations (%i - %i). It completed it in %.3f seconds.\nWidth = %i, Height = %i\nthreadBlock = (%d, %d, %d), gridDim = (%d, %d, %d)\n\n", 
        gpus, print_iter - iter, print_iter, iter, ((double) (end - start)) / CLOCKS_PER_SEC, width, height, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    }


    /* printf("MaxEps \n");
    for(int g = 0; g < gpus; g++){
        printf("%i, ", maxEps_print[g]);
    }
    printf("\niter \n%i\n", iter); */




    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaStreamSynchronize(streams[g]));
    }

    /* cudaErrorHandle(cudaFreeHost(mat));
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaFree(mat_gpu[g]));
        cudaErrorHandle(cudaFree(mat_gpu_tmp[g]));
    }
    cudaErrorHandle(cudaFreeHost(mat_gpu));
    cudaErrorHandle(cudaFreeHost(mat_gpu_tmp));

    cudaErrorHandle(cudaFreeHost(rows)); */
    printf("FUNGERER!\n");
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
