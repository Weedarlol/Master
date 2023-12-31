#include <stdio.h>
#include <math.h>
#include <time.h>

#include <mpi.h>

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

void start(int width, int height, int iter, double eps, double dx, double dy, int gpu_nr, int compare, int overlap, dim3 blockDim, dim3 gridDim){
    /*
    Variables   | Type  | Description
    total          | int    | The total number of elements within the matrix

    print_iter     | int    | Placeholder, used to find the number of iterations which has been run

    start          | clock_t| The clock when we reach a certain part of the code
    end            | clock_t| The clock when we are finished with that part of the code

    gpus           | int    | Number of GPUs used when running the program
    device_nr      |*int    | An array which will allow the GPU to see which GPU number it has

    maxEps         |**double| Used by the GPU to calculate how many elements which exceed the allowed limit of change for an element between two iterations
    maxEps_print   |*double | Used as a destination for the number of elements which the GPU has calculated exceeding the limit

    rows_total     | int    | Is the total number of rows which will be computed on by the devices
    rows_per_device| int    | The number of rows per 
    rows_leftover  | int    | The number of GPUs which is required to have an additional row when rows_total cannot be split equally between the devices
    rows_device    | int    | Number of rows each gpus will recieve, including ghostrows
    rows_index     | int    | Contains the index of the row which will be the first row to be transferred to the designated GPU
    rows_compute   | int    | Will tell the GPU how many rows it will compute on

    threadSize     | int    | Total number of threads in work in a GPU
    threadInformatin| int   | Is used to give the GPU information of how many elements each thread has to compute on

    mat            |*double | The matrix which is to be used as base for computations
    mat_gpu        |**double| The matrix which will be a part of the mat matrix, will be the part which is given to one GPU
    mat_gpu_tmp    |**double| Is used so that the GPU can change between the two matrixes every iteration

    streams        |cudaStream_t| Contains the streams each GPU will use to allow for asynchronous computing

    kernelColl     |***void | Is a collection of all the functions each GPU will use to run the CUDA function on in a kernel

    filename       | char   | The name of the textdocument which will be created to compare resulting matrixes
    fptr           |*FILE   | Used to create the file
    */


    // MPI INITIALISATION BEGINS HERE
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Rank of node
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Total number of nodes

    int total = width*height;
    int tmp_iter = iter;

    clock_t start;

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
    int rows_per_device = rows_total/gpus;
    int rows_leftover = rows_total%gpus;
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
    cudaErrorHandle(cudaMallocHost(&threadInformation, 4*sizeof(int)));
    threadInformation[0] = (rows_compute[0]     *(width-2))/threadSize; // Find number of elements to compute for each thread, ignoring border elements.
    threadInformation[1] = (rows_compute[0]     *(width-2))%threadSize; // Finding which threads require 1 more element
    threadInformation[2] = (rows_compute[gpus-1]*(width-2))/threadSize; // Find number of elements to compute for each thread, ignoring border elements. -1 because of ghost row
    threadInformation[3] = (rows_compute[gpus-1]*(width-2))%threadSize; // Finding which threads require 1 more element

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
        cudaErrorHandle(cudaMemcpyAsync(mat_gpu[g], mat+rows_index[g]*width, rows_device[g]*width*sizeof(double), cudaMemcpyHostToDevice, streams[g]));
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


    nvtxRangePushA("Area of Interest");
    start = clock();
    if(overlap == 1){
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
                        cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[1],                                   1, mat_gpu_tmp[0] + (rows_compute[0])*width, 0, width*sizeof(double), streams[g]));
                    }
                    else if(g < gpus-1){
                        // Transfers data device g -> device g+1
                        cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g+1],                               g+1, mat_gpu_tmp[g] + (rows_compute[g])*width, g, width*sizeof(double), streams[g]));
                        // Transfers data device g -> device g-1
                        cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + (rows_compute[g-1]+1)*width, g-1, mat_gpu_tmp[g] + width,                   g, width*sizeof(double), streams[g]));
                    }
                    else{
                        // Transfers data device -1 -> device -2
                        cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + (rows_compute[g-1]+1)*width, g-1, mat_gpu_tmp[g] + width,                   g, width*sizeof(double), streams[g]));
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
                double *mat_change = mat_gpu[g];
                mat_gpu[g] = mat_gpu_tmp[g];
                mat_gpu_tmp[g] = mat_change;
            }
            iter--;
        }
    }
    
    nvtxRangePop();

    printf("Time - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            ((double) (clock() - start)) / CLOCKS_PER_SEC, (iter == 0) ? "No" : "Yes", tmp_iter - iter);

    




    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaStreamSynchronize(streams[g]));
    }
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMemcpyAsync(mat + (rows_index[g]+1)*width, mat_gpu[g] + width, rows_compute[g]*width*sizeof(double), cudaMemcpyDeviceToHost, streams[g]));
    }
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaStreamSynchronize(streams[g]));
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
        cudaErrorHandle(cudaFree(maxEps[g]));
        cudaErrorHandle(cudaStreamDestroy(streams[g]));
    }
    cudaErrorHandle(cudaFreeHost(mat));
    cudaErrorHandle(cudaFreeHost(mat_gpu));
    cudaErrorHandle(cudaFreeHost(mat_gpu_tmp));
    cudaErrorHandle(cudaFreeHost(threadInformation));
    cudaErrorHandle(cudaFreeHost(device_nr));
    cudaErrorHandle(cudaFreeHost(maxEps));
    cudaErrorHandle(cudaFreeHost(maxEps_print));
    cudaErrorHandle(cudaFreeHost(rows_device));
    cudaErrorHandle(cudaFreeHost(rows_index));
    cudaErrorHandle(cudaFreeHost(rows_compute));
}



int main(int argc, char *argv[]) {
    /*
    Functions   | Type           | Input
    start       | void           | int width, int height, int iter, double eps,
                                   double dx, double dy, dim3 blockDim,
                                   dim3 gridDim

    fillValues  | void           | double *mat, double dx, double dy, int width,
                                   int height

    jacobi      |__global__ void | double *mat_gpu, double *mat_tmp, double eps,
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
    if (argc != 7) {
        printf("Usage: %s <Width> <Height> <Iterations>", argv[0]); // Programname
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int iter = atoi(argv[3]);
    int gpu_nr = atoi(argv[4]);
    int compare = atoi(argv[5]);
    int overlap = atoi(argv[6]);

    double eps = 1.0e-14;
    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(16, 1, 1);

    //start(width, height, iter, eps, dx, dy, gpu_nr, compare, overlap, blockDim, gridDim);

    return 0;
}
