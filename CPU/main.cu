#include <stdio.h>
#include <math.h>
#include <time.h>


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

    if (argc != 4) {
        printf("Usage: %s <Width> <Height> <Iterations>", argv[0]); // Programname
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int iter = atoi(argv[3]);
    int print_iter = iter;

    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);
    double maxdelta = 1.0;
    double eps = 1.0e-14;

    double *mat;
    double *mat_tmp;

    clock_t start, end;

    mat = (double*)malloc(width*height*sizeof(double));
    mat_tmp = (double*)malloc(width*height*sizeof(double));

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
        double *mat_tmp_cha = mat_tmp;
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

    // Creates an output which can be used to compare the different resulting matrixes
    FILE *fptr;
    char filename[30];
    sprintf(filename, "CPUMatrix%i_%i.txt", width, height);
    fptr = fopen(filename, "w");
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            fprintf(fptr, "%.16f ", mat[j + i*width]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);


    free(mat);
    free(mat_tmp);

    if(maxdelta <= eps){
        printf("The computation found a solution. It computed it within %i iterations(%i - %i) in %.3f seconds.\nWidth = %i, Height = %i\n", 
        print_iter - iter, print_iter, iter, ((double) (end - start)) / CLOCKS_PER_SEC, width, height);
    }
    else{
        printf("The computation did not find a solution. It computed through the whole %i iteration(%i - %i) in %.3f seconds \nWidth = %i, Height = %i\n", 
        print_iter - iter, print_iter, iter, ((double) (end - start)) / CLOCKS_PER_SEC, width, height);
    } 

    return 0;
}
