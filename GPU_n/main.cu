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

void fillValues(double *mat, double dx, double dy, int width, int height){
    double x, y;

    memset(mat, 0, height*width*sizeof(double));

    for(int i = 1; i < height - 1; i++) {
        y = i * dy; // y coordinate
        for(int j = 1; j < width - 1; j++) {
            x = j * dx; // x coordinate
            mat[j + i*width] = sin(M_PI*y)*sin(M_PI*x);
        }
    }
}

void start(int width, int height, int iter, double dx, double dy, int gpu_nr, int compare, int overlap, dim3 blockDim, dim3 gridDim){
    /*
    Variables            | Type        | Description
    total                | int         | The total number of elements within the matrix
    tmp_iter             | int         | Used to remeber how many iterations we want run
    overlap_calc         | int         | Used to find how many elements less the kernelCollMid has to compute when we have overlap
    threadSize           | int         | Finds the total amount of threads in use
    gpus                 | int         | Number of GPUs in use
    device_nr            | int*        | Allows the GPU to know its GPU index

    rows_total           | int         | Total number of rows to be computed on
    rows_per_device      | int         | Number of rows per device, rounded down
    rows_leftover        | int         | Number of rows leftover when rounded down

    rows_device          | int*        | Rows to allocate on the GPU
    rows_compute_device  | int*        | Rows the GPU will compute on
    rows_staring_index   | int*        | Index on the CPU matrix that the first element of the GPU matrix belongs

    threadInformation[0] | int         | Number of computations per thread on GPU 0, rounded down
    threadInformation[1] | int         | Number of computations left over when rounded down
    threadInformation[2] | int         | Number of computations per thread on GPU n-1, rounded down, is used as if there are an unequal amount of rows between device, 
                                         the first and last GPU will certainly be in each group
    threadInformation[3] | int         | Number of computations left over when rounded down
    threadInformation[4] | int         | Number of computations per thread for 1 row, rounded down
    threadInformation[5] | int         | Number of computations left over for 1 row when rounded down

    mat                  | double*     | The matrix allocated on the CPU
    mat_gpu              | double**    | One of the matrices allocated on the GPU
    mat_gpu_tmp          | double**    | The other matrix allocated on the GPU

    kernelCollEdge       | void***     | The inputfeatures to the jacobiEdge GPU kernel
    kernelCollMid        | void***     | The inputfeatures to the jacobiMid GPU kernel

    streams              | cudaStream_t| The streams which is utilized when computing on the GPU
    events               | cudaEvent_t | The events used to synchronize the streams
    startevent           | cudaEvent_t | The event used to start the timer for the computation
    stopevent            | cudaEvent_t | The event used to stop the timer for the computation

    */


    int total = width*height;
    int tmp_iter = iter;
    int overlap_calc = (width-2)*overlap;
    int threadSize = blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y*gridDim.z;

    int gpus;
    int *device_nr;
    cudaErrorHandle(cudaGetDeviceCount(&gpus));
    cudaErrorHandle(cudaMallocHost(&device_nr, gpus*sizeof(int*)));
    for(int g = 0; g < gpus; g++){
        device_nr[g] = g;
    }

    // Ignores first and last row
    int rows_total = height-2;
    int rows_per_device = rows_total/gpus;
    int rows_leftover = rows_total%gpus;
    int *rows_device, *rows_compute_device, *rows_starting_index;
    cudaErrorHandle(cudaMallocHost(&rows_device, gpus*sizeof(int*)));
    cudaErrorHandle(cudaMallocHost(&rows_starting_index, gpus*sizeof(int*)));
    cudaErrorHandle(cudaMallocHost(&rows_compute_device, gpus*sizeof(int*)));
    // Calculate the number of rows for each device
    for (int g = 0; g < gpus; g++) {
        int extra_row = (g < rows_leftover) ? 1 : 0;
  
        rows_device[g] = rows_per_device + extra_row + 2;

        rows_compute_device[g] = rows_per_device + extra_row - (2*overlap); // -2 as we are computing in 2 parts, 1 with point dependent on ghostpoints,and one without

        rows_starting_index[g] = g * rows_per_device + min(g, rows_leftover);
    }

    int *threadInformation;
    cudaErrorHandle(cudaMallocHost(&threadInformation, 6*sizeof(int)));
    threadInformation[0] = ((rows_compute_device[0])     *(width-2))/threadSize; // Find number of elements to compute for each thread, ignoring border elements.
    threadInformation[1] = ((rows_compute_device[0])     *(width-2))%threadSize; // Finding which threads require 1 more element
    threadInformation[2] = ((rows_compute_device[gpus-1])*(width-2))/threadSize; // Find number of elements to compute for each thread, ignoring border elements.
    threadInformation[3] = ((rows_compute_device[gpus-1])*(width-2))%threadSize; // Finding which threads require 1 more element
    threadInformation[4] = (1                            *(width-2))/threadSize; // Find number of elements for each thread for a row, if 0 it means there are more threads than elements in row
    threadInformation[5] = (1                            *(width-2))%threadSize; // Finding which threads require 1 more element

    

    double *mat;
    double **mat_gpu, **mat_gpu_tmp;
    cudaErrorHandle(cudaMallocHost(&mat,          total*sizeof(double)));
    cudaErrorHandle(cudaMallocHost(&mat_gpu,      gpus*sizeof(double*)));
    cudaErrorHandle(cudaMallocHost(&mat_gpu_tmp,  gpus*sizeof(double*)));


    // Allocates memory on devices based on number of rows for each device
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMalloc(&mat_gpu[g],     width*rows_device[g]*sizeof(double)));
        cudaErrorHandle(cudaMalloc(&mat_gpu_tmp[g], width*rows_device[g]*sizeof(double)));
    }

    void ***kernelCollEdge;
    cudaErrorHandle(cudaMallocHost(&kernelCollEdge, gpus * sizeof(void**)));
    // Allocates the elements in the kernelCollEdge, used for cudaLaunchCooperativeKernel as functon variables.
    for (int g = 0; g < gpus; g++) {
        void **kernelArgs = new void*[7];
        kernelArgs[0] = &mat_gpu[g];
        kernelArgs[1] = &mat_gpu_tmp[g];
        kernelArgs[2] = &width;
        kernelArgs[3] = &height;
        kernelArgs[4] = &rows_compute_device[g];
        kernelArgs[5] = &threadInformation[4];
        kernelArgs[6] = &threadInformation[5];

        kernelCollEdge[g] = kernelArgs;
    }

    void ***kernelCollMid;
    cudaErrorHandle(cudaMallocHost(&kernelCollMid, gpus * sizeof(void**)));
    // Allocates the elements in the kernelCollMid, used for cudaLaunchCooperativeKernel as functon variables.
    for (int g = 0; g < gpus; g++) {
        void **kernelArgs = new void*[12];
        kernelArgs[0] = &mat_gpu[g];     
        kernelArgs[1] = &mat_gpu_tmp[g];
        kernelArgs[2] = &width;
        kernelArgs[3] = &height;
        kernelArgs[4] = &rows_leftover;
        kernelArgs[5] = &device_nr[g];
        kernelArgs[6] = &rows_compute_device[g];
        kernelArgs[7] = &threadInformation[0];
        kernelArgs[8] = &threadInformation[1];
        kernelArgs[9] = &threadInformation[2];
        kernelArgs[10] = &threadInformation[3];
        kernelArgs[11] = &overlap_calc;

        kernelCollMid[g] = kernelArgs;
    }

    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaStreamCreate(&streams[g][0]));
        cudaErrorHandle(cudaStreamCreate(&streams[g][1]));
        cudaErrorHandle(cudaEventCreate(&events[g][0]));
        cudaErrorHandle(cudaEventCreate(&events[g][1]));
        cudaErrorHandle(cudaEventCreate(&events[g][2]));
        cudaErrorHandle(cudaEventCreate(&events[g][3]));
    }
    cudaErrorHandle(cudaEventCreate(&startevent));
    cudaErrorHandle(cudaEventCreate(&stopevent));


    fillValues(mat, dx, dy, width, height);


    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMemcpyAsync(mat_gpu[g], mat+rows_starting_index[g]*width, rows_device[g]*width*sizeof(double), cudaMemcpyHostToDevice, streams[g][0]));
        cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][0]));
    }

    
    for (int g = 0; g < gpus; g++) {
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaStreamSynchronize(streams[g][0]));
        cudaErrorHandle(cudaStreamSynchronize(streams[g][1]));
        cudaErrorHandle(cudaEventSynchronize(events[g][0]));
        cudaErrorHandle(cudaEventSynchronize(events[g][1]));
        cudaErrorHandle(cudaEventSynchronize(events[g][2]));
        cudaErrorHandle(cudaEventSynchronize(events[g][3]));
    }

    

    nvtxRangePushA("Area of Interest");
    cudaErrorHandle(cudaEventRecord(startevent));
    
    if(overlap == 1){
        if(gpus > 1){
            while(iter > 0){
                // Step 1
                for(int g = 0; g < gpus; g++){
                    cudaErrorHandle(cudaSetDevice(g));
                    // Computes the 2 rows
                    cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiEdge, gridDim, blockDim, kernelCollEdge[g], 0, streams[g][1]));
                    cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][1]));
                    // Computes the rest of the rows
                    cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiMid, gridDim, blockDim, kernelCollMid[g], 0, streams[g][0]));
                    cudaErrorHandle(cudaEventRecord(events[g][1], streams[g][0]));
                }



                // Step 2
                // Transfer 2 row of the matrix
                for(int g = 1; g < gpus; g++){
                    cudaErrorHandle(cudaSetDevice(g));
                    cudaErrorHandle(cudaStreamWaitEvent(streams[g][1], events[g][0]));
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + (rows_device[g-1]-1)*width + 1, g-1, mat_gpu_tmp[g] + width + 1, g, (width-2)*sizeof(double), streams[g][1]));
                    cudaErrorHandle(cudaEventRecord(events[g][2], streams[g][1]));
                }
                // Transfers n-2 row of the matrix
                for(int g = 0; g < gpus-1; g++){
                    cudaErrorHandle(cudaSetDevice(g));
                    cudaErrorHandle(cudaStreamWaitEvent(streams[g][1], events[g][0]));
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g+1] + 1, g+1, mat_gpu_tmp[g] + (rows_device[g]-2)*width + 1, g, (width-2)*sizeof(double), streams[g][1]));
                    cudaErrorHandle(cudaEventRecord(events[g][3], streams[g][1]));
                }


                // Step 3
                for (int g = 0; g < gpus; g++) {
                    cudaErrorHandle(cudaSetDevice(g));
                    cudaErrorHandle(cudaEventSynchronize(events[g][1]));
                    cudaErrorHandle(cudaEventSynchronize(events[g][2]));
                    cudaErrorHandle(cudaEventSynchronize(events[g][3]));
                }
                
                // Step 4
                for(int g = 0; g < gpus; g++){
                    double *mat_change = mat_gpu[g];
                    mat_gpu[g] = mat_gpu_tmp[g];
                    mat_gpu_tmp[g] = mat_change;
                }
                iter--;
            }
            
        }
    }
    else{
        if(gpus > 1){
            while(iter > 0){
                // Step 1
                // Computes the 2 row
                for(int g = 0; g < gpus; g++){
                    cudaErrorHandle(cudaSetDevice(g));
                    cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiMid, gridDim, blockDim, kernelCollMid[g], 0, streams[g][0]));
                    cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][0]));
                }


                // Step 2
                // Transfers 2 row of the matrix
                for(int g = 1; g < gpus; g++){
                    cudaErrorHandle(cudaSetDevice(g));
                    cudaErrorHandle(cudaStreamWaitEvent(streams[g][1], events[g][0]));
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + (rows_device[g-1]-1)*width + 1, g-1, mat_gpu_tmp[g] + width + 1, g, (width-2)*sizeof(double), streams[g][1]));
                    cudaErrorHandle(cudaEventRecord(events[g][1], streams[g][1]));
                }
                // Transfers n-2 row of the matrix
                for(int g = 0; g < gpus-1; g++){
                    cudaErrorHandle(cudaSetDevice(g));
                    cudaErrorHandle(cudaStreamWaitEvent(streams[g][1], events[g][0]));
                    cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g+1] + 1, g+1, mat_gpu_tmp[g] + (rows_device[g]-2)*width + 1, g, (width-2)*sizeof(double), streams[g][1]));
                    cudaErrorHandle(cudaEventRecord(events[g][2], streams[g][1]));
                }


                // Step 3
                for(int g = 0; g < gpus; g++){
                    cudaErrorHandle(cudaSetDevice(g));
                    cudaErrorHandle(cudaEventSynchronize(events[g][0]));
                    cudaErrorHandle(cudaEventSynchronize(events[g][1]));
                    cudaErrorHandle(cudaEventSynchronize(events[g][2]));
                }
                
                // Step 5
                for(int g = 0; g < gpus; g++){
                    double *mat_change = mat_gpu[g];
                    mat_gpu[g] = mat_gpu_tmp[g];
                    mat_gpu_tmp[g] = mat_change;
                }
                iter--;
            }
            
        }
    }
    
    cudaErrorHandle(cudaEventRecord(stopevent));

    cudaErrorHandle(cudaEventSynchronize(stopevent));
    
    nvtxRangePop();


    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", tmp_iter - iter);



    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMemcpyAsync(mat + (rows_starting_index[g]+1)*width, mat_gpu[g] + width, (rows_compute_device[g]+2*overlap)*width*sizeof(double), cudaMemcpyDeviceToHost));
    }
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }


    // Used to compare the matrix to the matrix which only the CPU created
    if(compare == 1){
        double* mat_compare = (double*)malloc(width * height * sizeof(double));
        FILE *fptr;
        char filename[30];
        sprintf(filename, "../CPU/CPUMatrix%i_%i.txt", width, height);

        printf("Comparing the matrixes\n");

        fptr = fopen(filename, "r");
        if (fptr == NULL) {
            printf("Error opening file.\n");
            exit(EXIT_FAILURE);
        }

        // Read matrix values from the file
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (fscanf(fptr, "%lf", &mat_compare[j + i * width]) != 1) {
                    printf("Error reading from file.\n");
                    fclose(fptr);
                    free(mat_compare);
                    exit(EXIT_FAILURE);
                }
            }
        }

        fclose(fptr);


        // Comparing the elements
        for (int i = 1; i < height-1; i++) {
            for (int j = 1; j < width-1; j++) {
                if (fabs(mat[j + i * width] - mat_compare[j + i * width]) > 1e-16)  {
                    printf("Mismatch found at position (%d, %d) (%.16f, %.16f)\n", i, j, mat[j + i * width], mat_compare[j + i * width]);
                    free(mat_compare);
                    exit(EXIT_FAILURE);
                    cudaErrorHandle(cudaDeviceSynchronize());
                }
            }
        }


        printf("All elements match!\n");
        

        // Free allocated memory
        free(mat_compare);
    }



    // Frees up memory as we are finished with the program
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaFree(mat_gpu[g]));
        cudaErrorHandle(cudaFree(mat_gpu_tmp[g]));
        cudaErrorHandle(cudaStreamDestroy(streams[g][0]));
        cudaErrorHandle(cudaStreamDestroy(streams[g][1]));
    }
    cudaErrorHandle(cudaFreeHost(mat));
    cudaErrorHandle(cudaFreeHost(mat_gpu));
    cudaErrorHandle(cudaFreeHost(mat_gpu_tmp));
    cudaErrorHandle(cudaFreeHost(threadInformation));
    cudaErrorHandle(cudaFreeHost(device_nr));
    cudaErrorHandle(cudaFreeHost(rows_device));
    cudaErrorHandle(cudaFreeHost(rows_starting_index));
    cudaErrorHandle(cudaFreeHost(rows_compute_device));
    // kernelCollMid and kernelCollEdge?
}



int main(int argc, char *argv[]) {
    /*
    Functions   | Type           | Input
    start       | void           | int width, int height, int iter,
                                   double dx, double dy, dim3 blockDim,
                                   dim3 gridDim

    ____________________________________________________________________________
    Variables   | Type  | Description
    width       | int   | The width of the matrix
    height      | int   | The height of the matrix
    iter        | int   | Amount of iterations
    gpu_nr      | int   | Number of GPUs in use
    compare     | int   | If one wants to compare the output with a previously CPU computed matrix
    overlap     | int   | If one want to overlap or not
    dx          | float | Used to give value to the elements of the matrix
    dy          | float | Used to give value to the elements of the matrix
    blockDim    | dim3  | Size of the threadblock
    gridDim     | dim3  | Size of the blockgrid

    For all true/false integers, 0 = false, 1 = true
    */
    // Checks if the correct amount of inputs is used
    if (argc != 7) {
        printf("Usage: %s <Width> <Height> <Iterations>", argv[0]);
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int iter = atoi(argv[3]);
    int gpu_nr = atoi(argv[4]);
    int compare = atoi(argv[5]);
    int overlap = atoi(argv[6]);

    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(16, 1, 1);

    start(width, height, iter, dx, dy, gpu_nr, compare, overlap, blockDim, gridDim);

    return 0;
}
