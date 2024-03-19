#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

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

    if (argc != 6) {
        printf("Usage: %s <Width> <Height> <Depth> <Iterations> <CreateGrid>", argv[0]); // Programname
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int depth = atoi(argv[3]);
    int iter = atoi(argv[4]);
    int createGrid = atoi(argv[5]);

    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);
    double dz = 2.0 / (depth - 1);

    double *data;
    double *data_tmp;

    clock_t start, end;

    data = (double*)malloc(width*height*depth*sizeof(double));
    data_tmp = (double*)malloc(width*height*depth*sizeof(double));

    /* initialization */
    fillValues3D(data, width, height, depth, dx, dy, dz, 0);

    start = clock();
    double division = 1/6.0;

    /* Performing Jacobian grid Calculation */
    // Performing a number of iterations while statement is not satisfied
    while (iter > 0) {
        for(int i = 1; i < depth - 1; i++){
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

        double *data_tmp_swap = data_tmp;
        data_tmp = data;
        data = data_tmp_swap;
        
        iter--;
    }

    end = clock();

    printf("Time(event) - %.5f s\n", ((double) (end - start)) / CLOCKS_PER_SEC);


    // Creates an output which can be used to compare the different resulting grides
    if(createGrid == 1){
        FILE *fptr;
        char filename[30];
        sprintf(filename, "grids/CPUGrid%d_%d_%d.txt", width, height, depth);
        fptr = fopen(filename, "w");
        for(int i = 0; i < depth; i++){
            for(int j = 0; j < height; j++){
                for(int k = 0; k < width; k++){
                    fprintf(fptr, "%.16f ", data[k + j*width + i*width*height]);
                }
            fprintf(fptr, "\n");
            }
        }
        fclose(fptr);
    }
    


    free(data);
    free(data_tmp);

    return 0;
}
