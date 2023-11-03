#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void calc(float *mat_gpu, float *mat_gpu_tmp, int amountPerThread, int index_start, int jacobiSize, int width, int height, float eps, int *maxEps, int thread, cg::grid_group grid_g){
    int local_var = 0;

    for(int i = 0; i < amountPerThread; i++){
        int index = index_start + i;
        int x = index % (width-2) +1;
        int y = index / (width-2) + 1;
        index = x+y*width;
        mat_gpu_tmp[index] = 0.25 * (
            mat_gpu[index + 1]     + mat_gpu[index - 1] +
            mat_gpu[index + width] + mat_gpu[index - width]);

        if(abs(mat_gpu[index] - mat_gpu_tmp[index]) > eps){
            local_var++;
        }

     // https://developer.nvidia.com/blog/cooperative-groups/
    for (int i = grid_g.num_threads() / 2; i > 0; i /= 2)
    {
        maxEps[thread] = local_var;
        grid_g.sync(); // wait for all threads to store
        if(thread<i) local_var += maxEps[thread + i];
        grid_g.sync(); // wait for all threads to load
    }

    }
}

__global__ void jacobi(float *mat_gpu, float *mat_gpu_tmp, int number_rows, int width, int height, 
                        int rows_leftover, int device_nr, int rows_compute, int amountPerThreadExtra, int extraLeftover,
                        int amountPerThread, int leftover, int *maxEps, float eps){
    /*
    Variables      | Type      | Description
    grid_g         | grid_group| Creates a group compromising of all the threads
    thread         | int       | The index of each thread
    index_start    | int       | Element index the thread will start computing on, unique for each thread in grid_g group. It is also taking into consideration what device is computing
    */

   // number_rows -> Totalt antall allokert på GPU
   // rows_leftover -> Antall 

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int index_start = width;
    
    index_start = thread*amountPerThread + min(thread, leftover);
    if(thread < leftover){
        amountPerThread++;
    }
    
    calc(mat_gpu, mat_gpu_tmp, amountPerThread, index_start, rows_compute, width, height, eps, maxEps, thread, grid_g);

}