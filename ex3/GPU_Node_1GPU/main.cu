#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#include "programs/cuda_functions.h"
#include "programs/scenarios.h"
#include <nvtx3/nvToolsExt.h>


void fillValues3D(double *mat, int width, int height, int depth, double dx, double dy, double dz, int rank, int overlap) {
    double x, y, z;
    int depth_overlap = 0;

    if(rank < overlap){
        depth_overlap = rank*(depth-2);
    }
    else{
        depth_overlap = (overlap)*(depth-1);
        depth_overlap += (rank-overlap)*(depth-2);
    }

    // Assuming the data in the matrix is stored contiguously in memory
    memset(mat, 0, width * height * depth * sizeof(double));

    for (int i = 1; i < depth-1; i++) {
        z = (i + depth_overlap) * dz; // z coordinate
        for (int j = 1; j < height - 1; j++) {
            y = j * dy; // z coordinate
            for (int k = 1; k < width - 1; k++) {
                x = k * dx; // x coordinate
                mat[k +  j*width + i*width*height] = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
            }
        }
    }
}

void initialization(int width, int height, int depth, int iter, double dx, double dy, double dz, int gpus, int nodes, int compare, int overlap, int test, int rank, int size, dim3 blockDim, dim3 gridDim){
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

   // Creates MPI requests
    MPI_Request myRequest[4];
    MPI_Status myStatus[4];

    // Finds number of slices per node
    int depth_node = (depth-2)/size;
    int depth_overlap = (depth-2)%size;
    if(depth_overlap > rank){
        depth_node += 3;
    }
    else{
        depth_node += 2;
    }

    // Finds number of elements and threads per node
    int total = width*height*depth_node;
    int threadSize = blockDim.x*blockDim.y*blockDim.z*gridDim.x*gridDim.y*gridDim.z;

    // Initialising CPU and GPU grids
    double *data, *data_gpu, *data_gpu_tmp;
    cudaErrorHandle(cudaMallocHost(&data, total*sizeof(double)));
    cudaErrorHandle(cudaMalloc(&data_gpu, total*sizeof(double)));
    cudaErrorHandle(cudaMalloc(&data_gpu_tmp, total*sizeof(double)));

    // Fills grids for each node depending on rank
    fillValues3D(data, width, height, depth_node, dx, dy, dz, rank, depth_overlap);

    // Sends border slices to neighboring nodes,
    if(rank == 0){
        MPI_Isend(&data[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[0]);
        MPI_Irecv(&data[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[1]); 
    }
    else if(rank == size-1){
        MPI_Irecv(&data[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]); 
        MPI_Isend(&data[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 
    }
    else{
        MPI_Irecv(&data[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]);
        MPI_Isend(&data[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 

        MPI_Isend(&data[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[2]);
        MPI_Irecv(&data[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[3]);
    }
    MPI_Waitall(rank == 0 || rank == size - 1 ? 2 : 4, myRequest, myStatus);

    cudaErrorHandle(cudaDeviceSynchronize());
    // Sends the grid from the CPU memory to GPU memory
    cudaErrorHandle(cudaMemcpy(data_gpu, data, total*sizeof(double), cudaMemcpyHostToDevice));
    cudaErrorHandle(cudaMemset(data_gpu_tmp, 0, total*sizeof(double)));


    int jacobiSize = (width - 2) * (height - 2) * (depth_node - 2);
    int elementsPerThread = jacobiSize / threadSize;
    int leftover = jacobiSize % threadSize;


    // Creates an array where its elements are features in cudaLaunchCooperativeKernel
    void *kernelArgs[] = {&data_gpu, &data_gpu_tmp, &width, &height, &depth_node, &iter, &threadSize, &jacobiSize, &elementsPerThread, &leftover};

    full_calculation(data_gpu, data_gpu_tmp, width, height, depth_node, iter, rank, size, gridDim, blockDim, kernelArgs);




    
    cudaErrorHandle(cudaDeviceSynchronize());

    // Copies data back from GPU memory into CPU memory
    cudaErrorHandle(cudaMemcpyAsync(data, data_gpu, total*sizeof(double), cudaMemcpyDeviceToHost));

    cudaErrorHandle(cudaDeviceSynchronize());

    // Combines the different node grid data all into Node 1 memory, so we can compare to original grid
    double *data_combined;
    cudaErrorHandle(cudaMallocHost(&data_combined, width*height*depth*sizeof(double)));
    int data_transfer = width*height*(depth_node-2);
    int displacement[size];
    int counts[size];
    if(rank < depth_overlap){
        for(int i = 0; i < size; i++){
            if(i < depth_overlap){
                displacement[i] = i*width*height*(depth_node-2);
                counts[i] = width*height*(depth_node-2);
            }
            else if(i == depth_overlap){
                displacement[i] = i*width*height*(depth_node-2);
                counts[i] = width*height*(depth_node-3);
            }
            else{
                displacement[i] = depth_overlap*width*height*(depth_node-2) + (i - depth_overlap)*width*height*(depth_node-3);
                counts[i] = width*height*(depth_node-3);
            }
        }
    }
    else{
        for(int i = 0; i < size; i++){
            if(i < depth_overlap){
                displacement[i] = i*width*height*(depth_node-1);
                counts[i] = width*height*(depth_node-1);
            }
            else if(i == depth_overlap){
                displacement[i] = i*width*height*(depth_node-1);
                counts[i] = width*height*(depth_node-2);
            }
            else{
                displacement[i] = depth_overlap*width*height*(depth_node-1) + (i - depth_overlap)*width*height*(depth_node-2);
                counts[i] = width*height*(depth_node-2);
            }
        }
    }
    

    if (rank == 1) {
        data_combined = (double*)malloc(width*height*depth * sizeof(double));
        /* printf("displacement = [");
        for(int i = 0; i < size; i++){
            printf("%i, ", displacement[i]);
        }
        printf("]\ncounts = [");
        for(int i = 0; i < size; i++){
            printf("%i, ", counts[i]);
        }
        printf("]\n"); */
    }

    MPI_Allgatherv(&data[width*height], width*height*(depth_node-2), MPI_DOUBLE, 
                data_combined + width * height, counts, displacement, MPI_DOUBLE, MPI_COMM_WORLD);


    /* if(rank == 2){
        for(int i = 0; i < 5; i++){
            for(int j = 0; j < 128; j++){
                for(int k = 0; k < 20; k++){
                    printf("%.8f, ", data[k + j*width + i*width*height]);
                }
                printf("\n");
            }
            printf("\n");
        }
    } */

    // Used to compare the grid to the grid which only the CPU created
    if(rank == 0){
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
                        if (fabs(data_combined[k + j * width + i * width * height] - data_compare[k + j * width + i * width * height]) > 1e-15)  {
                            printf("Mismatch found at position (width = %d, height = %d, depth = %d) (data = %.16f, data_compare = %.16f)\n", k, j, i, data_combined[k + j * width + i * width * height], data_compare[k + j * width + i * width * height]);
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
    }
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
    if (argc != 10) {
        printf("Wrong amount of inputs: %s <width> <height> <depth> <iter> <gpus> <compare> <overlap> <test>", argv[0]);
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int depth = atoi(argv[3]);
    int iter = atoi(argv[4]);
    int gpus = atoi(argv[5]);
    int nodes = atoi(argv[6]);
    int compare = atoi(argv[7]);
    int overlap = atoi(argv[8]);
    int test = atoi(argv[9]);

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

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    initialization(width, height, depth, iter, dx, dy, dz, gpus, nodes, compare, overlap, test, rank, size, blockDim, gridDim);

    MPI_Finalize();

    return 0;
}
