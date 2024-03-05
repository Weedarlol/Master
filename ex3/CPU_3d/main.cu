#include <stdio.h>
#include <math.h>
#include <time.h>



#include "../../global_functions.h"

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

    if (argc != 5) {
        printf("Usage: %s <Width> <Height> <Depth> <Iterations>", argv[0]); // Programname
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int depth = atoi(argv[3]);
    int iter = atoi(argv[4]);
    int print_iter = iter;

    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);
    double dz = 2.0 / (depth - 1);

    double *mat;
    double *mat_tmp;

    clock_t start, end;

    mat = (double*)malloc(width*height*depth*sizeof(double));
    mat_tmp = (double*)malloc(width*height*depth*sizeof(double));

    /* initialization */
    fillValues3D(mat, width, height, depth, dx, dy, dz);

    start = clock();
    double division = 1/6.0;

    /* Performing Jacobian Matrix Calculation */
    // Performing a number of iterations while statement is not satisfied
    while (iter > 0) {
        for(int i = 1; i < depth - 1; i++){
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
        

        iter--;

        double *mat_tmp_swap = mat_tmp;
        mat_tmp = mat;
        mat = mat_tmp_swap;
    }

    end = clock();

    printf("Time(event) - %.5f s\n", ((double) (end - start)) / CLOCKS_PER_SEC);


    // Creates an output which can be used to compare the different resulting matrixes
    FILE *fptr;
    char filename[30];
    sprintf(filename, "matrices/CPUMatrix%i_%i_%i.txt", width, height, depth);
    fptr = fopen(filename, "w");
    for(int i = 0; i < depth; i++){
        for(int j = 0; j < height; j++){
            for(int k = 0; k < width; k++){
                fprintf(fptr, "%.16f ", mat[k + j*width + i*width*height]);
            }
        fprintf(fptr, "\n");
        }
    }
    fclose(fptr);


    free(mat);
    free(mat_tmp);

    return 0;
}
