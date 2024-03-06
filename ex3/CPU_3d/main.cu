#include <stdio.h>
#include <math.h>
#include <time.h>

#include "../../global_functions.h"

int main(int argc, char *argv[]) {
    /*
    width       | int    | The width of the grid
    height      | int    | The height of the 
    depth       | int    | The depth of the grid
    iter        | int    | Number of max iterations for the jacobian algorithm
    create_grid | int    | Boolean for if one prints out the output grid into a file or not. 1 = yes, 0 = no

    dx          | double | Distance between each element in the grid in x direction
    dy          | double | Distance between each element in the grid in y direction
    dz          | double | Distance between each element in the grid in z direction

    grid         |*double | Pointer to the grid
    grid_tmp     |*double | Pointer to the grid

    start       | clock_t| Starttime of time estimation
    end         | clock_t| Endtime of time estimation

    division    | double | Made a variable to not have to calculate it for each element
    */

    if (argc != 5) {
        printf("Usage: %s <Width> <Height> <Depth> <Iterations>", argv[0]);
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int depth = atoi(argv[3]);
    int iter = atoi(argv[4]);
    int create_grid = atoi(argv[5]);

    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);
    double dz = 2.0 / (depth - 1);

    double *grid;
    double *grid_tmp;

    clock_t start, end;

    grid = (double*)malloc(width*height*depth*sizeof(double));
    grid_tmp = (double*)malloc(width*height*depth*sizeof(double));

    // Fills up the grid grid with starting values
    fillValues3D(grid, width, height, depth, dx, dy, dz);

    double division = 1/6.0;
    start = clock();

    /* Performing Jacobian grid Calculation */
    // Performing a number of iterations while statement is not satisfied
    while (iter > 0) {
        for(int i = 1; i < depth - 1; i++){
            for(int j = 1; j < height - 1; j++){
                for(int k = 1; k < width - 1; k++) {
                    int index = k + j * width + i * width * height;
                    grid_tmp[index] = division * (
                    grid[index + 1]            + grid[index - 1] +
                    grid[index + width]        + grid[index - width] +
                    grid[index + width*height] + grid[index - width*height]);
                }
            }
        }

        double *grid_tmp_swap = grid_tmp;
        grid_tmp = grid;
        grid = grid_tmp_swap;

        iter--;
    }

    end = clock();

    printf("Time(event) - %.5f s\n", ((double) (end - start)) / CLOCKS_PER_SEC);

    // Creates an output which can be used to compare the different resulting grids
    if(create_grid == 1){
        FILE *fptr;
        char filename[30];
        sprintf(filename, "grids/CPUGrid%i_%i_%i.txt", width, height, depth);
        fptr = fopen(filename, "w");
        for(int i = 0; i < depth; i++){
            for(int j = 0; j < height; j++){
                for(int k = 0; k < width; k++){
                    fprintf(fptr, "%.16f ", grid[k + j*width + i*width*height]);
                }
            fprintf(fptr, "\n");
            }
        }
        fclose(fptr);
    }

    free(grid);
    free(grid_tmp);

    return 0;
}
