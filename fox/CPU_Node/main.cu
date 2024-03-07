#include <stdio.h>
#include <math.h>
#include <time.h>

#include "mpi.h"
#include "../../global_functions.h"

int main(int argc, char *argv[]) {
    /*
    width       | int   | The width of the grid
    height      | int   | The height of the grid
    iter        | int   | Number of max iterations for the jacobian algorithm

    eps         | double | The limit for accepting the state of the grid during jacobian algorithm
    maxdelta    | double | The largest difference in the grid between an iteration
    dx          | double | Distance between each element in the grid in x direction
    dy          | double | Distance between each element in the grid in y direction

    data         |*double | Pointer to the grid
    data_tmp     |*double | Pointer to the grid
    */

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 6) {
        printf("Wrong number of inputs\n Required inputs: %s <Width> <Height> <Depth> <Iterations> <Node> <Compare>", argv[0]); // Programname
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int depth = atoi(argv[3]);
    int iter = atoi(argv[4]);
    int compare = atoi(argv[5]);
    int depth_node = depth/size + 1;

    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);
    double dz = 2.0 / (depth - 1);

    double *data;
    double *data_tmp;

    clock_t start, end;

    data = (double*)malloc(width*height*depth_node*sizeof(double));
    data_tmp = (double*)malloc(width*height*depth_node*sizeof(double));

    /* initialization */
    fillValues3D(data, width, height, depth_node, dx, dy, dz, rank);

    start = clock();
    double division = 1/6.0;

    if(rank == 0){
        MPI_Send(&data[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
        MPI_Recv(&data[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else{
        MPI_Recv(&data[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&data[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
    }

    /* Performing Jacobian grid Calculation */
    // Performing a number of iterations while statement is not satisfied
    while (iter > 0) {
        for(int i = 1; i < depth_node - 1; i++){
            for(int j = 1; j < height - 1; j++){
                for(int k = 1; k < width - 1; k++) {
                    int index = k + j * width + i * width * height;
                    data_tmp[index] = division * (
                    data[index + 1]            + data[index - 1] +
                    data[index + width]        + data[index - width] +
                    data[index + width*height] + data[index - width*height]);
                }
            }
        }

        if(rank == 0){
            MPI_Send(&data_tmp[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
            MPI_Recv(&data_tmp[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else{
            MPI_Recv(&data_tmp[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&data_tmp[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
        }

        iter--;

        double *data_tmp_swap = data_tmp;
        data_tmp = data;
        data = data_tmp_swap;
    }


    end = clock();
    printf("Time(event) - %.5f s\n", ((double) (end - start)) / CLOCKS_PER_SEC);


    double *data_combined = NULL;
    if(rank == 0){
        data_combined = (double*)malloc(width * height * depth * sizeof(double));

        MPI_Gather(&data[0], width*height*(depth_node-1), MPI_DOUBLE, data_combined, width*height*(depth_node-1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else{

        MPI_Gather(&data[width*height], width*height*(depth_node-1), MPI_DOUBLE, data_combined, width*height*(depth_node-1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }



    if(compare == 1){
        if(rank == 0){
            double* data_compare = (double*)malloc(width * height * depth* sizeof(double));
            FILE *fptr;
            char filename[30];
            sprintf(filename, "../CPU_3d/grids/CPUGrid%i_%i_%i.txt", width, height, depth);

            printf("Comparing the grids\n");

            fptr = fopen(filename, "r");
            if (fptr == NULL) {
                printf("Error opening file.\n");
                MPI_Finalize();
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
                            MPI_Finalize();
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
                            printf("Mismatch found at position (width = %d, height = %d, depth = %d) (data_combined = %.16f, data_compare = %.16f)\n", k, j, i, data_combined[k + j * width + i * width * height], data_compare[k + j * width + i * width * height]);
                            free(data_compare);
                            MPI_Finalize();
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

    


    free(data);
    free(data_tmp);


    MPI_Finalize();

    return 0;
}
