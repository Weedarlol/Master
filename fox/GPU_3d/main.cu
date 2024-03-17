#include <stdio.h>
#include <math.h>
#include <time.h>

#include "programs/scenarios.h"
#include "programs/cuda_functions.h"
#include <nvtx3/nvToolsExt.h>

void fillValues3D(double *mat, int width, int height, int depth, double dx, double dy, double dz, int rank) {
    double x, y, z;

    // Assuming the data in the matrix is stored contiguously in memory
    memset(mat, 0, height * width * depth * sizeof(double));

    for (int i = 1; i < depth-1; i++) {
        z = (i + rank * (depth - 2)) * dz; // z coordinate
        for (int j = 1; j < height - 1; j++) {
            y = j * dy; // z coordinate
            for (int k = 1; k < width - 1; k++) {
                x = k * dx; // x coordinate
                mat[k +  j*width + i*width*height] = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
            }
        }
    }
}

void initialization(int width, int height, int depth, int iter, double dx, double dy, double dz, int gpus, int compare, int overlap, int test, dim3 blockDim, dim3 gridDim){
    /*
    Variables            | Type        | Description
    total            |    | int         | The total number of elements within the grid
    tmp_iter             | int         | Used to remeber how many iterations we want run
    overlap_calc         | int         | Used to find how many elements less the kernelCollMid has to compute when we have overlap
    threadSize           | int         | Finds the total amount of threads in use
    gpus                 | int         | Number of gpus in use
    device_nr            | int*        | Allows the GPU to know its GPU index

    slices_total           | int         | Total number of slices to be computed on
    slices_per_device      | int         | Number of slices per device, rounded down
    slices_leftover        | int         | Number of slices leftover when rounded down

    slices_device          | int*        | slices to allocate on the GPU
    slices_compute_device  | int*        | slices the GPU will compute on
    slices_staring_index   | int*        | Index on the CPU grid that the first element of the GPU grid belongs

    threadInformation[0] | int         | Number of computations per thread on GPU 0, rounded down
    threadInformation[1] | int         | Number of computations left over when rounded down
    threadInformation[2] | int         | Number of computations per thread on GPU n-1, rounded down, is used as if there are an unequal amount of slices between device, 
                                         the first and last GPU will certainly be in each group
    threadInformation[3] | int         | Number of computations left over when rounded down
    threadInformation[4] | int         | Number of computations per thread for 1 slice, rounded down
    threadInformation[5] | int         | Number of computations left over for 1 slice when rounded down

    data                  | double*     | The grid allocated on the CPU
    data_gpu              | double**    | One of the matrices allocated on the GPU
    data_gpu_tmp          | double**    | The other grid allocated on the GPU

    kernelCollEdge       | void***     | The inputfeatures to the jacobiEdge GPU kernel
    kernelCollMid        | void***     | The inputfeatures to the jacobiMid GPU kernel

    streams              | cudaStream_t| The streams which is utilized when computing on the GPU
    events               | cudaEvent_t | The events used to synchronize the streams
    startevent           | cudaEvent_t | The event used to start the timer for the computation
    stopevent            | cudaEvent_t | The event used to stop the timer for the computation

    */


    int total = width*height*depth;
    int overlap_calc = overlap*width*height;
    int threadSize = blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y*gridDim.z;

    int *device_nr;
    cudaErrorHandle(cudaMallocHost(&device_nr, gpus*sizeof(int*)));
    for(int g = 0; g < gpus; g++){
        device_nr[g] = g;
    }

    // Ignores first and last slice
    int slices_total = depth-2;
    int slices_per_device = slices_total/gpus;
    int slices_leftover = slices_total%gpus;
    int *slices_device, *slices_compute_device, *slices_starting_index;
    cudaErrorHandle(cudaMallocHost(&slices_device, gpus*sizeof(int*)));
    cudaErrorHandle(cudaMallocHost(&slices_starting_index, gpus*sizeof(int*)));
    cudaErrorHandle(cudaMallocHost(&slices_compute_device, gpus*sizeof(int*)));
    // Calculate the number of slices for each device
    for (int g = 0; g < gpus; g++) {
        int extra_slice = (g < slices_leftover) ? 1 : 0;
  
        slices_device[g] = slices_per_device + extra_slice + 2;

        slices_compute_device[g] = slices_per_device + extra_slice - (2*overlap); // -2 as we are computing in 2 parts, 1 with point dependent on ghostpoints,and one without

        slices_starting_index[g] = g * slices_per_device + min(g, slices_leftover);
    }

    // Initialiserer og allokerer Matrise på CPU
    double *data;
    double **data_gpu, **data_gpu_tmp;
    cudaErrorHandle(cudaMallocHost(&data,          total*sizeof(double)));
    cudaErrorHandle(cudaMallocHost(&data_gpu,      gpus*sizeof(double*)));
    cudaErrorHandle(cudaMallocHost(&data_gpu_tmp,  gpus*sizeof(double*)));

    fillValues3D(data, width, height, depth, dx, dy, dz, 0);

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMalloc(&data_gpu[g],     width*height*slices_device[g]*sizeof(double)));
        cudaErrorHandle(cudaMalloc(&data_gpu_tmp[g], width*height*slices_device[g]*sizeof(double)));
    }

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMemcpy(data_gpu[g], data+slices_starting_index[g]*width*height, slices_device[g]*width*height*sizeof(double), cudaMemcpyHostToDevice));
    }

    int *threadInformation;
    cudaErrorHandle(cudaMallocHost(&threadInformation, 6*sizeof(int)));
    threadInformation[0] = ((slices_compute_device[0])     *(width-2)*(height-2))/threadSize; // Find number of elements to compute for each thread, ignoring border elements.
    threadInformation[1] = ((slices_compute_device[0])     *(width-2)*(height-2))%threadSize; // Finding which threads require 1 more element
    threadInformation[2] = ((slices_compute_device[gpus-1])*(width-2)*(height-2))/threadSize; // Find number of elements to compute for each thread, ignoring border elements.
    threadInformation[3] = ((slices_compute_device[gpus-1])*(width-2)*(height-2))%threadSize; // Finding which threads require 1 more element
    threadInformation[4] = (1                              *(width-2)*(height-2))/threadSize; // Find number of elements for each thread for a slice, if 0 it means there are more threads than elements in slice
    threadInformation[5] = (1                              *(width-2)*(height-2))%threadSize; // Finding which threads require 1 more element


    void ***kernelCollEdge;
    cudaErrorHandle(cudaMallocHost(&kernelCollEdge, gpus * sizeof(void**)));
    // Allocates the elements in the kernelCollEdge, used for cudaLaunchCooperativeKernel as functon variables.
    for (int g = 0; g < gpus; g++) {
        void **kernelArgs = new void*[8];
        kernelArgs[0] = &data_gpu[g];
        kernelArgs[1] = &data_gpu_tmp[g];
        kernelArgs[2] = &width;
        kernelArgs[3] = &height;
        kernelArgs[4] = &slices_compute_device[g];
        kernelArgs[5] = &threadInformation[4];
        kernelArgs[6] = &threadInformation[5];

        kernelCollEdge[g] = kernelArgs;
    }

    void ***kernelCollMid;
    cudaErrorHandle(cudaMallocHost(&kernelCollMid, gpus * sizeof(void**)));
    // Allocates the elements in the kernelCollMid, used for cudaLaunchCooperativeKernel as functon variables.
    for (int g = 0; g < gpus; g++) {
        void **kernelArgs = new void*[12];
        kernelArgs[0] = &data_gpu[g];     
        kernelArgs[1] = &data_gpu_tmp[g];
        kernelArgs[2] = &width;
        kernelArgs[3] = &height;
        kernelArgs[5] = &slices_leftover;
        kernelArgs[6] = &device_nr[g];
        kernelArgs[7] = &slices_compute_device[g];
        kernelArgs[8] = &threadInformation[0];
        kernelArgs[9] = &threadInformation[1];
        kernelArgs[10] = &threadInformation[2];
        kernelArgs[11] = &threadInformation[3];

        kernelCollMid[g] = kernelArgs;
    }






    full_calculation_nooverlap(data_gpu, data_gpu_tmp, width, height, depth, iter, gpus, slices_device, gridDim, blockDim, kernelCollMid);







    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMemcpyAsync(data + (slices_starting_index[g]+1)*width*height, data_gpu[g] + width*height, (slices_compute_device[g]+2*overlap)*width*height*sizeof(double), cudaMemcpyDeviceToHost));
    }

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }


    // Used to compare the grid to the grid which only the CPU created
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



    // Frees up memory as we are finished with the program
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaFree(data_gpu[g]));
        cudaErrorHandle(cudaFree(data_gpu_tmp[g]));
    }
    cudaErrorHandle(cudaFreeHost(data));
    cudaErrorHandle(cudaFreeHost(data_gpu));
    cudaErrorHandle(cudaFreeHost(data_gpu_tmp));
    cudaErrorHandle(cudaFreeHost(threadInformation));
    cudaErrorHandle(cudaFreeHost(device_nr));
    cudaErrorHandle(cudaFreeHost(slices_device));
    cudaErrorHandle(cudaFreeHost(slices_starting_index));
    cudaErrorHandle(cudaFreeHost(slices_compute_device));

}



int main(int argc, char *argv[]) {
    /*
    Functions   | Type           | Input
    start       | void           | int width, int height, int iter,
                                   double dx, double dy, dim3 blockDim,
                                   dim3 gridDim

    ____________________________________________________________________________
    Variables   | Type  | Description
    width       | int   | The width of the grid
    height      | int   | The height of the grid

    iter        | int   | Amount of iterations
    gpus      | int   | Number of gpus in use
    compare     | int   | If one wants to compare the output with a previously CPU computed grid
    overlap     | int   | If one want to overlap or not

    dx          | float | Used to give value to the elements of the grid
    dy          | float | Used to give value to the elements of the grid

    blockDim    | dim3  | Size of the threadblock
    gridDim     | dim3  | Size of the blockgrid

    For all true/false integers, 0 = false, 1 = true
    */
   
    // Checks if the correct amount of inputs is used
    if (argc != 9) {
        printf("Wrong amount of inputs: %s <width> <height> <depth> <iter> <gpus> <compare> <overlap> <test>", argv[0]);
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int depth = atoi(argv[3]);
    int iter = atoi(argv[4]);
    int gpus = atoi(argv[5]);
    int compare = atoi(argv[6]);
    int overlap = atoi(argv[7]);
    int test = atoi(argv[8]);

    if(width < 1){
        printf("Width is to small\n");
    }
    else if(height < 1){
        printf("Heigth is to small\n");
    }
    else if(depth < 1){
        printf("Depth is to small\n");
    }
    else if(iter < 1){
        printf("To few selected iterations\n");
    }
    else if(gpus < 1){
        printf("Selected to few GPUs\n");
    }
    else if(compare > 1 || compare < 0){
        printf("Compare variable can only be\n"
                "0 - Do not compare the output grid with previously created grid\n"
                "1 - Compare with previously created grid\n");
    }
    else if(overlap > 1 || overlap < 0){
        printf("You can only select the values\n"
                "0 - Do not overlap communication and computation\n"
                "1 - Overlap communication and computation\n");
    }
    else if(test > 4 || test < 0){
        printf("There is no test with the current value, please select one of the following.\n"
                "0 - Full Computation\n"
                "1 - No kernel\n"
                "2 - No Communication\n"
                "3 - Only computation\n"
                "4 - Only communication\n");
    }
 
    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);
    double dz = 2.0 / (depth - 1);

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(16, 1, 1);

    initialization(width, height, depth, iter, dx, dy, dz, gpus, compare, overlap, test, blockDim, gridDim);

    return 0;
}
