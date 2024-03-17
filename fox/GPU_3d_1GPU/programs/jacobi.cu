#include <stdio.h>
#include <math.h>
#include <nvtx3/nvToolsExt.h>

#include "cuda_functions.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;


__device__ void calc(double *data_gpu, double *data_gpu_tmp, int iter, int thread_size,
    int amountPerThread, int index_start, int width, int height, int depth,
    cg::grid_group grid_g){

    double division = 1.0/6;

    // Calculating Jacobian Matrix
    while(iter > 0){
        // Calculates each element except the border
        for(int i = 0; i < amountPerThread; i++){
            int index = index_start + i*thread_size;
            int x = index % (width - 2) + 1;
            int y = (index / (width - 2)) % (height - 2) + 1;
            int z = index / ((width - 2) * (height - 2)) + 1;
            index = x + y*width + z*width*height;

            data_gpu_tmp[index] = division * (
                data_gpu[index + 1]            + data_gpu[index - 1] +
                data_gpu[index + width]        + data_gpu[index - width] +
                data_gpu[index + width*height] + data_gpu[index - width*height]);
        } 

        grid_g.sync();
    }
}

__global__ void jacobi(double *data_gpu, double *data_gpu_tmp, int width, int height, int depth, int iter){
    /*
    Variables      | Type      | Description
    grid_g         | grid_group| Creates a group compromising of all the threads
    thread_size     | int       | Total number of available threads within the grid_g group
    jacobiSize     | int       | Number of elements in the matrix which is to be calculated each iteration
    amountPerThread| int       | Number of elements to be calculated by each thread each iteration
    leftover       | int       | Number of threads which is required to compute one more element to be calculate all the elements
    thread         | int       | The index of each thread
    index_start    | int       | Element index the thread will start computing on, unique for each thread in grid_g group
    */

    cg::grid_group grid_g = cg::this_grid();
    int thread_size = grid_g.num_threads();
    int thread = grid_g.thread_rank();
    int jacobiSize = (width - 2) * (height - 2) * (depth - 2);
    int amountPerThread = jacobiSize / thread_size;
    int leftover = jacobiSize % thread_size;
    int index_start = thread * amountPerThread + min(thread, leftover); //- (thread < leftover ? thread : 0);

    if(thread < leftover){
        amountPerThread++;
    }

    calc(data_gpu, data_gpu_tmp, iter, amountPerThread, index_start, thread_size,
        width, height, depth, grid_g);
}