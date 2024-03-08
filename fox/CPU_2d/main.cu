#include <stdio.h>
#include <math.h>
#include <time.h>


#include "../../global_functions.h"

int main(int argc, char *argv[]) {
    /*
    width       | int   | The width of the matrix
    height      | int   | The height of the matrix
    iter        | int   | Number of max iterations for the jacobian algorithm

    dx          | double | Distance between each element in the matrix in x direction
    dy          | double | Distance between each element in the matrix in y direction

    data         |*double | Pointer to the matrix
    data_tmp     |*double | Pointer to the matrix
    */

    if (argc != 5) {
        printf("Usage: %s <Width> <Height> <Iterations> <CreateMatrix>", argv[0]); // Programname
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int iter = atoi(argv[3]);
    int createMatrix = atoi(argv[4]);

    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);

    double *data;
    double *data_tmp;

    clock_t start, end;

    data = (double*)malloc(width*height*sizeof(double));
    data_tmp = (double*)malloc(width*height*sizeof(double));

    /* initialization */
    fillValues(data, dx, dy, width, height);

    start = clock();

    /* Performing Jacobian Matrix Calculation */
    // Performing a number of iterations while statement is not satisfied
    while (iter > 0) {
        // Loops through the matrix from element 1 to -2
        for(int i = 1; i < height - 1; i++){
            // Calculates the element value from row
            int i_nr = i*width;
            // Loops through the matrix from element 1 to -2
            for(int j = 1; j < width - 1; j++) {
                // Calculates each element in the matrix from itself and neightbor values.
                data_tmp[i_nr + j] = 0.25 * (
                    data[i_nr + j + 1]     + data[i_nr + j - 1] +
                    data[i_nr + j + width] + data[i_nr + j - width]);
            }
        }

        iter--;
    }
    end = clock();

    printf("Time(event) - %.5f s\n", ((double) (end - start)) / CLOCKS_PER_SEC);

    if(createMatrix == 1){
        // Creates an output which can be used to compare the different resulting matrixes
        FILE *fptr;
        char filename[30];
        sprintf(filename, "matrices/CPUMatrix%i_%i.txt", width, height);
        fptr = fopen(filename, "w");
        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                fprintf(fptr, "%.16f ", data[j + i*width]);
            }
            fprintf(fptr, "\n");
        }
        fclose(fptr);
    }
    

    free(data);
    free(data_tmp);

    return 0;
}
