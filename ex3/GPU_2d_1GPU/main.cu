#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <nvtx3/nvToolsExt.h>
#include "programs/jacobi.h"
#include "programs/cuda_functions.h"
#include <cooperative_groups.h>


namespace cg = cooperative_groups;

void fillValues(double *mat, double dx, double dy, int width, int height) {
    double x, y;

    memset(mat, 0, height * width * sizeof(double));

    for (int i = 1; i < height - 1; i++) {
        y = i * dy; // y coordinate
        for (int j = 1; j < width - 1; j++) {
            x = j * dx; // x coordinate
            mat[j + i * width] = sin(M_PI * y) * sin(M_PI * x);
        }
    }
}


void start(int width, int height, int iter, double dx, double dy, int compare, dim3 blockDim, dim3 gridDim){
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

    int total = width*height;
    int gpus = 1;
    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

    double *data, *data_gpu, *data_gpu_tmp;
    cudaErrorHandle(cudaMallocHost(&data, total*sizeof(double)));
    cudaErrorHandle(cudaMalloc(&data_gpu, total*sizeof(double*)));
    cudaErrorHandle(cudaMalloc(&data_gpu_tmp, total*sizeof(double*)));

    /* initialization */
    fillValues(data, dx, dy, width, height);


    // Here we are done with the allocation, and start with the compution
    cudaErrorHandle(cudaEventRecord(startevent));

    // Copies elemts over from CPU to the device.
    cudaErrorHandle(cudaMemcpyAsync(data_gpu, data, total*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrorHandle(cudaMemsetAsync(data_gpu_tmp, 0, total*sizeof(double)));

    // Creates an array where its elements are features in cudaLaunchCooperativeKernel
    void *kernelArgs[] = {&data_gpu, &data_gpu_tmp, &width, &height, &iter};

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
        double* data_compare = (double*)malloc(width * height * sizeof(double));
        FILE *fptr;
        char filename[100];
        sprintf(filename, "../CPU_2d/matrices/CPUMatrix%i_%i.txt", width, height);

        printf("Comparing the matrixes\n");

        fptr = fopen(filename, "r");
        if (fptr == NULL) {
            printf("Error opening file.\n");
            exit(EXIT_FAILURE);
        }

        // Read matrix values from the file
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (fscanf(fptr, "%lf", &data_compare[j + i * width]) != 1) {
                    printf("Error reading from file.\n");
                    fclose(fptr);
                    free(data_compare);
                    exit(EXIT_FAILURE);
                }
            }
        }

        fclose(fptr);


        // Comparing the elements
        for (int i = 1; i < height-1; i++) {
            for (int j = 1; j < width-1; j++) {
                if (fabs(data[j + i * width] - data_compare[j + i * width]) > 1e-15)  {
                    printf("Mismatch found at position (%d, %d) (%.16f, %.16f)\n", i, j, data[j + i * width], data_compare[j + i * width]);
                    free(data_compare);
                    exit(EXIT_FAILURE);
                    cudaErrorHandle(cudaDeviceSynchronize());
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
    if (argc != 5) {
        printf("Usage: %s <Width> <Height> <Iterations>", argv[0]); // Programname
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int iter = atoi(argv[3]);
    int compare = atoi(argv[4]);

    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(16, 1, 1);

    start(width, height, iter, dx, dy, compare, blockDim, gridDim);

    return 0;
}
