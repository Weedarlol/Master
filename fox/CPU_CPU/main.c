#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>

void fillValues3D(double *mat, int width, int height, int depth, double dx, double dy, double dz, int rank, int overlap) {
    double x, y, z;
    int depth_overlap = 0;

    if(rank < overlap){
        depth_overlap = rank*(depth-2);
    }
    else{
        depth_overlap = (overlap)*(depth-1);
        depth_overlap += (rank-overlap)*(depth-2);
    }

    // Assuming the data in the matrix is stored contiguously in memory
    memset(mat, 0, width * height * depth * sizeof(double));

    for (int i = 1; i < depth-1; i++) {
        z = (i + depth_overlap) * dz; // z coordinate
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

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Request myRequest[4];
    MPI_Status myStatus[4];

    if (argc != 7) {
        printf("Wrong number of inputs\n Required inputs: %s <Width> <Height> <Depth> <Iterations> <Node> <Compare> <Overlap>", argv[0]); // Programname
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int depth = atoi(argv[3]);
    int iter = atoi(argv[4]);
    int compare = atoi(argv[5]);
    int overlap = atoi(argv[6]);

    int depth_node = (depth-2)/size;
    int depth_overlap = (depth-2)%size;
    if(depth_overlap > rank){
        depth_node += 3;
    }
    else{
        depth_node += 2;
    }

    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);
    double dz = 2.0 / (depth - 1);

    double *data;
    double *data_tmp;

    clock_t start, end;

    data = (double*)malloc(width*height*depth_node*sizeof(double));
    data_tmp = (double*)malloc(width*height*depth_node*sizeof(double));

    /* initialization */
    fillValues3D(data, width, height, depth_node, dx, dy, dz, rank, depth_overlap);



    if(rank == 0){
        MPI_Isend(&data[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[0]);
        MPI_Irecv(&data[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[1]); 
    }
    else if(rank == size-1){
        MPI_Irecv(&data[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]); 
        MPI_Isend(&data[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 
    }
    else{
        MPI_Irecv(&data[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]);
        MPI_Isend(&data[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 

        MPI_Isend(&data[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[2]);
        MPI_Irecv(&data[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[3]);
    }
    MPI_Waitall(rank == 0 || rank == size - 1 ? 2 : 4, myRequest, myStatus);


    start = clock();
    double division = 1/6.0;

    /* Performing Jacobian grid Calculation */
    // Performing a number of iterations while statement is not satisfied
    if(overlap == 0){
        if(rank == 0){
            while(iter > 0){
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
                MPI_Isend(&data_tmp[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[0]);
                MPI_Irecv(&data_tmp[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[1]); 
                MPI_Waitall(2, myRequest, myStatus);

                double *data_tmp_swap = data_tmp;
                data_tmp = data;
                data = data_tmp_swap;

                iter--;
            }
        }
        else if(rank == size-1){
            while(iter > 0){
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
                MPI_Irecv(&data_tmp[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]); 
                MPI_Isend(&data_tmp[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 
                MPI_Waitall(2, myRequest, myStatus);

                double *data_tmp_swap = data_tmp;
                data_tmp = data;
                data = data_tmp_swap;

                iter--;
            }
        }
        else{
            while(iter > 0){
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
                MPI_Irecv(&data_tmp[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]);
                MPI_Isend(&data_tmp[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 
                MPI_Isend(&data_tmp[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[2]);
                MPI_Irecv(&data_tmp[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[3]);
                MPI_Waitall(4, myRequest, myStatus);

                double *data_tmp_swap = data_tmp;
                data_tmp = data;
                data = data_tmp_swap;

                iter--;
            }
        }
    }
    else if(overlap == 1){
        if(rank == 0){
            while(iter > 0){
                for(int j = 1; j < height - 1; j++){
                    for(int k = 1; k < width - 1; k++) {
                        int index = k + j * width + (depth_node-2) * width * height;
                        data_tmp[index] = division * (
                        data[index + 1]            + data[index - 1] +
                        data[index + width]        + data[index - width] +
                        data[index + width*height] + data[index - width*height]);
                    }
                }

                MPI_Isend(&data_tmp[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[0]);
                MPI_Irecv(&data_tmp[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[1]); 


                for(int i = 1; i < depth_node - 2; i++){
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
                MPI_Waitall(2, myRequest, myStatus);
            }
        }
        else if(rank == size-1){
            while(iter > 0){
                for(int j = 1; j < height - 1; j++){
                    for(int k = 1; k < width - 1; k++) {
                        int index = k + j * width + width * height;
                        data_tmp[index] = division * (
                        data[index + 1]            + data[index - 1] +
                        data[index + width]        + data[index - width] +
                        data[index + width*height] + data[index - width*height]);
                    }
                }
                MPI_Irecv(&data_tmp[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]); 
                MPI_Isend(&data_tmp[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 

                for(int i = 2; i < depth_node - 1; i++){
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
                MPI_Waitall(2, myRequest, myStatus);
            }
        }
        else{
            while(iter > 0){
                for(int j = 1; j < height - 1; j++){
                    for(int k = 1; k < width - 1; k++) {
                        int index = k + j * width + (depth_node-2) * width * height;
                        data_tmp[index] = division * (
                        data[index + 1]            + data[index - 1] +
                        data[index + width]        + data[index - width] +
                        data[index + width*height] + data[index - width*height]);
                    }
                }
                MPI_Isend(&data_tmp[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[0]);
                MPI_Irecv(&data_tmp[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[1]);
                for(int j = 1; j < height - 1; j++){
                    for(int k = 1; k < width - 1; k++) {
                        int index = k + j * width + width * height;
                        data_tmp[index] = division * (
                        data[index + 1]            + data[index - 1] +
                        data[index + width]        + data[index - width] +
                        data[index + width*height] + data[index - width*height]);
                    }
                }
                MPI_Irecv(&data_tmp[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[2]); 
                MPI_Isend(&data_tmp[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[3]);
                for(int i = 2; i < depth_node - 2; i++){
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
                MPI_Waitall(4, myRequest, myStatus);
            }
        }
    }



    end = clock();
    printf("Time(event) - %.5f s\n", ((double) (end - start)) / CLOCKS_PER_SEC);


    double *data_combined = (double*)malloc(width*height*depth*sizeof(double));
    int displacement[size];
    int counts[size];
    if(rank < depth_overlap){
        for(int i = 0; i < size; i++){
            if(i < depth_overlap){
                displacement[i] = i*width*height*(depth_node-2);
                counts[i] = width*height*(depth_node-2);
            }
            else if(i == depth_overlap){
                displacement[i] = i*width*height*(depth_node-2);
                counts[i] = width*height*(depth_node-3);
            }
            else{
                displacement[i] = depth_overlap*width*height*(depth_node-2) + (i - depth_overlap)*width*height*(depth_node-3);
                counts[i] = width*height*(depth_node-3);
            }
        }
    }
    else{
        for(int i = 0; i < size; i++){
            if(i < depth_overlap){
                displacement[i] = i*width*height*(depth_node-1);
                counts[i] = width*height*(depth_node-1);
            }
            else if(i == depth_overlap){
                displacement[i] = i*width*height*(depth_node-1);
                counts[i] = width*height*(depth_node-2);
            }
            else{
                displacement[i] = depth_overlap*width*height*(depth_node-1) + (i - depth_overlap)*width*height*(depth_node-2);
                counts[i] = width*height*(depth_node-2);
            }
        }
    }


    if(compare == 1){
        MPI_Gatherv(&data[width*height], width*height*(depth_node-2), MPI_DOUBLE, 
                   &data_combined[width*height], counts, displacement, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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

            for(int i = 0; i < depth_node; i++){
                for (int j = 0; j < height; j++) {
                    for (int k = 0; k < width; k++) {
                        if (fabs(data_combined[k + j * width + i * width * height] - data_compare[k + j * width + i * width * height]) > 1e-15)  {
                            printf("Mismatch found at position (width = %d, height = %d, depth = %d) (data_Node = %.16f, data_compare = %.16f)\n", k, j, i, data_combined[k + j * width + i * width * height], data_compare[k + j * width + i * width * height]);
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
