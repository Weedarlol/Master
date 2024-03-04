#include <stdio.h>
#include <math.h>
#include <time.h>

#include "mpi.h"


void fillValues3D(double *mat, int width, int height, int depth_node, double dx, double dy, double dz, int rank) {
    double x, y, z;

    // Assuming the data in the matrix is stored contiguously in memory
    memset(mat, 0, height * width * depth_node * sizeof(double));

    for (int i = 1; i < depth_node-1; i++) {
        z = (i + (depth_node - 2)*rank) * dz; // z coordinate
        for (int j = 1; j < height - 1; j++) {
            y = j * dy; // z coordinate
            for (int k = 1; k < width - 1; k++) {
                x = k * dx; // x coordinate
                mat[k +  j*width + i*width*height] = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    /*
    width       | int   | The width of the matrix
    height      | int   | The height of the matrix
    iter        | int   | Number of max iterations for the jacobian algorithm

    eps         | double | The limit for accepting the state of the matrix during jacobian algorithm
    maxdelta    | double | The largest difference in the matrix between an iteration
    dx          | double | Distance between each element in the matrix in x direction
    dy          | double | Distance between each element in the matrix in y direction

    mat         |*double | Pointer to the matrix
    mat_tmp     |*double | Pointer to the matrix
    */

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        printf("Wrong number of inputs\n Required inputs: %s <Width> <Height> <Depth> <Iterations> <Node>", argv[0]); // Programname
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int depth = atoi(argv[3]);
    int iter = atoi(argv[4]);
    int print_iter = iter;
    int depth_node = depth/size + 1;

    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);
    double dz = 2.0 / (depth - 1);

    double *mat;
    double *mat_tmp;

    clock_t start, end;

    mat = (double*)malloc(width*height*depth_node*sizeof(double));
    mat_tmp = (double*)malloc(width*height*depth_node*sizeof(double));

    /* initialization */
    fillValues3D(mat, width, height, depth_node, dx, dy, dz, rank);

    start = clock();
    double division = 1/6.0;

    if(rank == 0){
        MPI_Send(&mat[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
        MPI_Recv(&mat[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else{
        MPI_Recv(&mat[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&mat[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
    }

    /* Performing Jacobian Matrix Calculation */
    // Performing a number of iterations while statement is not satisfied
    while (iter > 0) {
        for(int i = 1; i < depth_node - 1; i++){
            for(int j = 1; j < height - 1; j++){
                for(int k = 1; k < width - 1; k++) {
                    int index = k + j * width + i * width * height;
                    mat_tmp[index] = division * (
                    mat[index + 1]            + mat[index - 1] +
                    mat[index + width]        + mat[index - width] +
                    mat[index + width*height] + mat[index - width*height]);
                }
            }
        }

        if(rank == 0){
            MPI_Send(&mat_tmp[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
            MPI_Recv(&mat_tmp[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else{
            MPI_Recv(&mat_tmp[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&mat_tmp[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
        }

        iter--;

        double *mat_tmp_swap = mat_tmp;
        mat_tmp = mat;
        mat = mat_tmp_swap;
    }


    end = clock();
    double *mat_combined = NULL;

    printf("It computed through the whole %i iteration(%i - %i) in %.3f seconds \nWidth = %i, Height = %i, Depth = %i\n", 
    print_iter - iter, print_iter, iter, ((double) (end - start)) / CLOCKS_PER_SEC, width, height, depth);

    if(rank == 0){
        mat_combined = (double*)malloc(width * height * depth * sizeof(double));

        MPI_Gather(&mat[0], width*height*(depth_node-1), MPI_DOUBLE, mat_combined, width*height*(depth_node-1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else{

        MPI_Gather(&mat[width*height], width*height*(depth_node-1), MPI_DOUBLE, mat_combined, width*height*(depth_node-1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }



    int compare = 1;
    if(compare == 1){
        if(rank == 0){
            double* mat_compare = (double*)malloc(width * height * depth* sizeof(double));
            FILE *fptr;
            char filename[30];
            sprintf(filename, "../CPU_3d/matrices/CPUMatrix%i_%i_%i.txt", width, height, depth);

            printf("Comparing the matrixes\n");

            fptr = fopen(filename, "r");
            if (fptr == NULL) {
                printf("Error opening file.\n");
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }

            // Read matrix values from the file
            for(int i = 0; i < depth; i++){
                for (int j = 0; j < height; j++) {
                    for (int k = 0; k < width; k++) {
                        if (fscanf(fptr, "%lf", &mat_compare[k + j * width + i * width * height]) != 1) {
                            printf("Error reading from file.\n");
                            fclose(fptr);
                            free(mat_compare);
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
                        if (fabs(mat_combined[k + j * width + i * width * height] - mat_compare[k + j * width + i * width * height]) > 1e-15)  {
                            printf("Mismatch found at position (width = %d, height = %d, depth = %d) (mat_combined = %.16f, mat_compare = %.16f)\n", k, j, i, mat_combined[k + j * width + i * width * height], mat_compare[k + j * width + i * width * height]);
                            free(mat_compare);
                            MPI_Finalize();
                            exit(EXIT_FAILURE);
                        }
                    }
                }
            }


            printf("All elements match!\n");
            

            // Free allocated memory
            free(mat_compare);
        } 
    }

    


    free(mat);
    free(mat_tmp);


    MPI_Finalize();

    return 0;
}
