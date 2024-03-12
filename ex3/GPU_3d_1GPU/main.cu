#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <nvtx3/nvToolsExt.h>

#include "programs/jacobi.h"
#include "../../cuda_functions.h"
#include "../../global_functions.h"
#include <cooperative_groups.h>


namespace cg = cooperative_groups;




void start(int width, int height, int depth, int iter, double dx, double dy, double dz, int compare, dim3 blockDim, dim3 gridDim){
    /*
    Variables   | Type  | Description
    total       | int   | Total number of elements in the matrix
    dataPerGpu  | int   | Number of elements per available gpu

    start       |clock_t| Start timer of program
    end         |clock_t| End timer of program

    data         |*double | Pointer to the allocated matrix in the CPU
    data_gpu     |**double| Pointer to an allocated matrix in the GPU
    data_gpu_tmp |**double| Pointer to an allocated matrix in the GPU

    maxEps      |*int   | Pointer to an allocated vector in the GPU used for checking if the matrix is in an acceptable state
    
    comp_suc    |*int   | Checks if the computation is successfull or not
    */

    int total = width*height*depth;
    int gpus = 1;
    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

    double *data, *data_gpu, *data_gpu_tmp;
    cudaErrorHandle(cudaMallocHost(&data, total*sizeof(double)));
    cudaErrorHandle(cudaMalloc(&data_gpu, total*sizeof(double*)));
    cudaErrorHandle(cudaMalloc(&data_gpu_tmp, total*sizeof(double*)));

    /* initialization */
    fillValues3D(data, width, height, depth, dx, dy, dz, 0);


    // Here we are done with the allocation, and start with the compution
    cudaErrorHandle(cudaEventRecord(startevent));

    // Copies elemts over from CPU to the device.
    cudaErrorHandle(cudaMemcpyAsync(data_gpu, data, total*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrorHandle(cudaMemsetAsync(data_gpu_tmp, 0, total*sizeof(double)));

    // Creates an array where its elements are features in cudaLaunchCooperativeKernel
    void *kernelArgs[] = {&data_gpu, &data_gpu_tmp, &width, &height, &depth, &iter};

    // Runs device
    cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobi, gridDim, blockDim, kernelArgs));

    cudaErrorHandle(cudaDeviceSynchronize());
    // Copies back value from device i to CPU
    cudaErrorHandle(cudaMemcpy(data, data_gpu, total*sizeof(double), cudaMemcpyDeviceToHost));
    cudaErrorHandle(cudaDeviceSynchronize());




    cudaErrorHandle(cudaEventRecord(stopevent));
    cudaErrorHandle(cudaEventSynchronize(stopevent));
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.5f s\n", milliseconds/1000);


    // Used to compare the matrix to the matrix which only the CPU created
    if(compare == 1){
        double* data_compare = (double*)malloc(width * height * depth* sizeof(double));
        FILE *fptr;
        char filename[100];
        sprintf(filename, "../CPU_3d/grids/CPUGrid%d_%d_%d.txt", width, height, depth);

        printf("Comparing the grids\n");

        fptr = fopen(filename, "r");
        if (fptr == NULL) {
            printf("Error opening file.\n");
            exit(EXIT_FAILURE);
        }

        // Read grid values from the file
        for(int i = 0; i < depth; i++){
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    if (fscanf(fptr, "%lf", &data_compare[k + j * width + i * width * height]) != 1) {
                        printf("Error reading from file.\n");
                        fclose(fptr);
                        free(data_compare);
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
        
        fclose(fptr);

        for(int i = 0; i < depth; i++){
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    if (fabs(data[k + j * width + i * width * height] - data_compare[k + j * width + i * width * height]) > 1e-15)  {
                        printf("Mismatch found at position (width = %d, height = %d, depth = %d) (data = %.16f, data_compare = %.16f)\n", k, j, i, data[k + j * width + i * width * height], data_compare[k + j * width + i * width * height]);
                        free(data_compare);
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }

        printf("All elements match!\n");
        
        // Free allocated memory
        free(data_compare);
    }
    

    cudaErrorHandle(cudaFreeHost(data));
    cudaErrorHandle(cudaFree(data_gpu));
    cudaErrorHandle(cudaFree(data_gpu_tmp));
}



int main(int argc, char *argv[]) {
    /*
    Functions   | Type           | Input
    start       | void           | int width, int height, int iter, double eps,
                                   double dx, double dy, dim3 blockDim,
                                   dim3 gridDim

    fillValues  | void           | double *data, double dx, double dy, int width,
                                   int height

    jacobi      |__global__ void | double *data_gpu, double *data_tmp, double eps,
                                   int width, int height, int iter

    ____________________________________________________________________________
    Variables   | Type  | Description
    width       | int   | The width of the matrix
    height      | int   | The height of the matrix
    iter        | int   | Number of max iterations for the jacobian algorithm

    eps         | double | The limit for accepting the state of the matrix during jacobian algorithm
    dx          | double | Distance between each element in the matrix in x direction
    dy          | double | Distance between each element in the matrix in y direction

    blockDim    | dim3  | Number of threads in 3 directions for each block
    gridDim     | dim3  | Number of blocks in 3 directions for the whole grid
    */
    if (argc != 6) {
        printf("Usage: %s <Width> <Height> <Depth> <Iterations>", argv[0]); // Programname
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int depth = atoi(argv[3]);
    int iter = atoi(argv[4]);
    int compare = atoi(argv[5]);

    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);
    double dz = 2.0 / (depth - 1);

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(16, 1, 1);

    start(width, height, depth, iter, dx, dy, dz, compare, blockDim, gridDim);

    return 0;
}
