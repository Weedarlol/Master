#include <stdio.h>
#include <math.h>
#include <time.h>

#include "programs/errorHandle.h"
#include "programs/scenarios.h"
#include <nvtx3/nvToolsExt.h>

void fillValues3D(double *mat, double dx, double dy, double dz, int width, int height, int depth) {
    double x, y, z;

    // Assuming the data in the matrix is stored contiguously in memory
    memset(mat, 0, height * width * depth * sizeof(double));

    for (int i = 1; i < height - 1; i++) {
        y = i * dy; // y coordinate
        for (int j = 1; j < width - 1; j++) {
            x = j * dx; // x coordinate
            for (int k = 1; k < depth - 1; k++) {
                z = k * dz; // z coordinate
                mat[k +  j*depth + i*depth*width] = sin(M_PI * y) * sin(M_PI * x) * sin(M_PI * z);
            }
        }
    }
}

void initialization(int width, int height, int depth, int iter, double dx, double dy, double dz, int gpus, int compare, int overlap, int test, dim3 blockDim, dim3 gridDim){
    /*
    Variables            | Type        | Description
    total            |    | int         | The total number of elements within the matrix
    tmp_iter             | int         | Used to remeber how many iterations we want run
    overlap_calc         | int         | Used to find how many elements less the kernelCollMid has to compute when we have overlap
    threadSize           | int         | Finds the total amount of threads in use
    gpus                 | int         | Number of gpus in use
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


    int total = width*height*depth;
    int overlap_calc = (width-2)*overlap;
    int threadSize = blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y*gridDim.z;
    int warp_size = 32;

    int *device_nr;
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



    // Initialiserer og allokerer Matrise på CPU

    int *threadInformation;
    cudaErrorHandle(cudaMallocHost(&threadInformation, 7*sizeof(int)));
    threadInformation[0] = ((rows_compute_device[0])     *(width-2))/threadSize; // Find number of elements to compute for each thread, ignoring border elements.
    threadInformation[1] = ((rows_compute_device[0])     *(width-2))%threadSize; // Finding which threads require 1 more element
    threadInformation[2] = ((rows_compute_device[gpus-1])*(width-2))/threadSize; // Find number of elements to compute for each thread, ignoring border elements.
    threadInformation[3] = ((rows_compute_device[gpus-1])*(width-2))%threadSize; // Finding which threads require 1 more element
    threadInformation[4] = (1                            *(width-2))/threadSize; // Find number of elements for each thread for a row, if 0 it means there are more threads than elements in row
    threadInformation[5] = (1                            *(width-2))%threadSize; // Finding which threads require 1 more element
    threadInformation[6] = (width - 2) % warp_size != 0 ? ((width-2)/warp_size)*warp_size+warp_size : ((width-2)/warp_size)*warp_size;

    for(int g = 0; g < gpus; g++){
        printf("rows_compute_device = %i, threads per row %i, threads leftover %i\n", rows_compute_device[g], (width-2)/threadSize, (width-2)%threadSize);
    }

    double *mat;
    double **mat_gpu, **mat_gpu_tmp;
    cudaErrorHandle(cudaMallocHost(&mat,          total*sizeof(double)));
    cudaErrorHandle(cudaMallocHost(&mat_gpu,      gpus*sizeof(double*)));
    cudaErrorHandle(cudaMallocHost(&mat_gpu_tmp,  gpus*sizeof(double*)));


    fillValues3D(mat, dx, dy, dz, width, height, depth);

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
        void **kernelArgs = new void*[8];
        kernelArgs[0] = &mat_gpu[g];
        kernelArgs[1] = &mat_gpu_tmp[g];
        kernelArgs[2] = &width;
        kernelArgs[3] = &height;
        kernelArgs[4] = &rows_compute_device[g];
        kernelArgs[5] = &threadInformation[4];
        kernelArgs[6] = &threadInformation[5];
        kernelArgs[7] = &threadInformation[6];

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


    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMemcpy(mat_gpu[g], mat+rows_starting_index[g]*width, rows_device[g]*width*sizeof(double), cudaMemcpyHostToDevice));
    }

    cudaErrorHandle(cudaDeviceSynchronize());


    full_calculation_overlap(mat_gpu, mat_gpu_tmp, height, width, iter, gpus, rows_device, gridDim, blockDim, kernelCollEdge, kernelCollMid);


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

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaFree(mat_gpu[g]));
        cudaErrorHandle(cudaFree(mat_gpu_tmp[g]));
    }
    cudaErrorHandle(cudaFreeHost(mat));
    cudaErrorHandle(cudaFreeHost(mat_gpu));
    cudaErrorHandle(cudaFreeHost(mat_gpu_tmp));
    cudaErrorHandle(cudaFreeHost(threadInformation));
    cudaErrorHandle(cudaFreeHost(device_nr));
    cudaErrorHandle(cudaFreeHost(rows_device));
    cudaErrorHandle(cudaFreeHost(rows_starting_index));
    cudaErrorHandle(cudaFreeHost(rows_compute_device));




    for(int i = 0 ; i < depth; i++){
        for(int j = 0; j < height; j++){
            for(int k = 0; k < width; k++){
                printf("%.2f ", mat[k + j*height + i*height*depth]);
            }
            printf("\n");
        }
    }
    printf("Dette fungerer fint\n");

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
    gpus      | int   | Number of gpus in use
    compare     | int   | If one wants to compare the output with a previously CPU computed matrix
    overlap     | int   | If one want to overlap or not

    dx          | float | Used to give value to the elements of the matrix
    dy          | float | Used to give value to the elements of the matrix

    blockDim    | dim3  | Size of the threadblock
    gridDim     | dim3  | Size of the blockgrid

    For all true/false integers, 0 = false, 1 = true
    */
   
    // Checks if the correct amount of inputs is used
    if (argc != 9) {
        printf("Wrong amount of inputs: %s <width> <height> <iter> <gpus> <compare> <overlap> <test>", argv[0]);
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
                "0 - Do not compare the output matrix with previously created matrix\n"
                "1 - Compare with previously created matrix\n");
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
