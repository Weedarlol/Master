#include <stdio.h>
#include <math.h>
#include <time.h>


void fillValues(float *mat, float dx, float dy, int width, int height){
    float x, y;

    memset(mat, 0, height*width*sizeof(float));

    for(int i = 1; i < height - 1; i++) {
        y = i * dy; // y coordinate
        for(int j = 1; j < width - 1; j++) {
            x = j * dx; // x coordinate
            mat[j + i*width] = sin(M_PI*y)*sin(M_PI*x);
        }
    }
}

int main() {
    /*
    width       | int   | The width of the matrix
    height      | int   | The height of the matrix
    iter        | int   | Number of max iterations for the jacobian algorithm

    eps         | float | The limit for accepting the state of the matrix during jacobian algorithm
    maxdelta    | float | The largest difference in the matrix between an iteration
    dx          | float | Distance between each element in the matrix in x direction
    dy          | float | Distance between each element in the matrix in y direction

    mat         |*float | Pointer to the matrix
    mat_tmp     |*float | Pointer to the matrix
    */

    int width = 1024;
    int height = 1024;
    int iter = 10000000;
    int print_iter = iter;

    float dx = 2.0 / (width - 1);
    float dy = 2.0 / (height - 1);
    float maxdelta = 1.0;
    float eps = 1.0e-14;

    float *mat;
    float *mat_tmp;

    clock_t start, end;

    mat = (float*)malloc(width*height*sizeof(float));
    mat_tmp = (float*)malloc(width*height*sizeof(float));

    /* initialization */
    fillValues(mat, dx, dy, width, height);

    start = clock();

    /* Performing Jacobian Matrix Calculation */
    // Performing a number of iterations while statement is not satisfied
    while (iter > 0 && maxdelta > eps) {
        // Maxdelta is the highest delta value found in the matrix
        maxdelta = 0.0;
        // Loops through the matrix from element 1 to -2
        for(int i = 1; i < height - 1; i++){
            // Calculates the element value from row
            int i_nr = i*width;
            // Loops through the matrix from element 1 to -2
            for(int j = 1; j < width - 1; j++) {
                // Calculates each element in the matrix from itself and neightbor values.
                mat_tmp[i_nr + j] = 0.25 * (
                    mat[i_nr + j + 1]     + mat[i_nr + j - 1] +
                    mat[i_nr + j + width] + mat[i_nr + j - width]);

                // Finds the highest difference for an element over two iterations.
                maxdelta = max(maxdelta, abs(*(mat + j + i*width)
                                            - *(mat_tmp + j + i*width)));
            }
        }

        iter--;

        /* pointer swapping */
        float *mat_tmp_cha = mat_tmp;
        mat_tmp = mat;
        mat = mat_tmp_cha;
    }

    end = clock();

    /* for(int i = 0; i < 20; i++){
        for(int j = 0; j < 20; j++){
            printf("%.5f ", mat[i*width+j]);
        }
        printf("\n");
    } */

    FILE *fptr;

    fptr = fopen("CPUMatrix.txt", "w");
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            fprintf(fptr, "%.14f ", mat[j + i*width]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);


    free(mat);
    free(mat_tmp);

    /* if(maxdelta <= eps){
        printf("The computation found a solution. It computed it within %i iterations(%i - %i) in %.3f seconds.\nWidth = %i, Height = %i\n", 
        print_iter - iter, print_iter, iter, ((double) (end - start)) / CLOCKS_PER_SEC, width, height);
    }
    else{
        printf("The computation did not find a solution. It computed through the whole %i iteration(%i - %i) in %.3f seconds \nWidth = %i, Height = %i\n", 
        print_iter - iter, print_iter, iter, ((double) (end - start)) / CLOCKS_PER_SEC, width, height);
    } */

    return 0;
}
