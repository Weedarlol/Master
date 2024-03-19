#include <stdio.h>
#include <math.h>
#include <nvtx3/nvToolsExt.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;


__device__ void calc(double *data_gpu, double *data_gpu_tmp, int iter, int index_start, int amountPerThread, 
    int thread_size, int width, int height, int depth, cg::grid_group grid_g){

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

        // Changes pointers
        double *data_tmp_cha = data_gpu_tmp;
        data_gpu_tmp = data_gpu;
        data_gpu = data_tmp_cha;

        iter--;

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
    */

    cg::grid_group grid_g = cg::this_grid();
    int thread_size = grid_g.num_threads();
    int thread = grid_g.thread_rank();
    int jacobiSize = (width - 2) * (height - 2) * (depth - 2);
    int amountPerThread = jacobiSize / thread_size;
    int leftover = jacobiSize % thread_size;

    if(thread < leftover){
        amountPerThread++;
    }

    calc(data_gpu, data_gpu_tmp, iter, thread, amountPerThread, thread_size,
        width, height, depth, grid_g);
}