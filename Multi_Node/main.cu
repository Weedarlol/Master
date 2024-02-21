#include <stdio.h>
#include <math.h>
#include <time.h>

#include <mpi.h>
#include "programs/errorHandle.h"

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

void initialization(int argc, char *argv[], int width, int height, int iter, double dx, double dy, int gpus, int compare, int overlap, int test, dim3 blockDim, dim3 gridDim){
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

    

    fillValues(mat, dx, dy, width, height);


    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaMemcpy(mat_gpu[g], mat+rows_starting_index[g]*width, rows_device[g]*width*sizeof(double), cudaMemcpyHostToDevice));
    }

    cudaErrorHandle(cudaDeviceSynchronize());


    



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


    MPI_Finalize();
}



int main(int argc, char *argv[]) {


    if (argc != 8) {
        printf("Wrong amount of inputs: %s <width> <height> <iter> <gpus> <compare> <overlap> <test>", argv[0]);
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int iter = atoi(argv[3]);
    int gpus = atoi(argv[4]);
    int compare = atoi(argv[5]);
    int overlap = atoi(argv[6]);
    int test = atoi(argv[7]);

    if(width < 1){
        printf("Width is to small\n");
    }
    else if(height < 1){
        printf("Heigth is to small\n");
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

    dim3 blockDim(32, 32, 1);
    dim3 gridDim(16, 1, 1);

    initialization(argc, argv, width, height, iter, dx, dy, gpus, compare, overlap, test, blockDim, gridDim);

    return 0;
}
