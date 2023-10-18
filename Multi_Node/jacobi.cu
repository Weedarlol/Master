#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ int shared_var = 1;

__device__ void calc(float *mat_gpu, float *mat_gpu_tmp, int device_nr, int thread, int iter,
    int amountPerThread, int index_start, int jacobiSize, int width, int height,
    float eps, cg::grid_group grid_g, int *maxEps){

	int local_var = 0;

    // Calculates each element except the border
    for(int i = 0; i < amountPerThread; i++){
        int index = index_start + i;
        if(index < jacobiSize){
            int x = index % (width - 2) + 1;
            int y = index / (height - 2) + 1;
            int ind = x + y * width;

            mat_gpu_tmp[ind] = 0.25 * (
                mat_gpu[ind + 1]     + mat_gpu[ind - 1] +
                mat_gpu[ind + width] + mat_gpu[ind - width]);


            if(abs(mat_gpu[ind] - mat_gpu_tmp[ind]) > eps){
                local_var++;
            }
        }
    }

    // https://developer.nvidia.com/blog/cooperative-groups/
    for (int i = grid_g.num_threads() / 2; i > 0; i /= 2)
    {
        maxEps[thread] = local_var;
        grid_g.sync(); // wait for all threads to store
        if(thread<i) local_var += maxExp[thread + i];
        grid_g.sync(); // wait for all threads to load
    }

}

__global__ void jacobi(float *mat_gpu, float *mat_gpu_tmp, int *maxEps, int device_nr, int dataPerGpu, float eps, int width, int height, int iter, int jacobiSize, int amountPerThread, int leftover){
    /*
    Variables      | Type      | Description
    grid_g         | grid_group| Creates a group compromising of all the threads
    thread         | int       | The index of each thread
    index_start    | int       | Element index the thread will start computing on, unique for each thread in grid_g group. It is also taking into consideration what device is computing
    */

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int index_start = (thread * amountPerThread + min(thread, leftover)) + dataPerGpu*device_nr;

    if(thread < leftover){
        amountPerThread++;
    }

    calc(mat_gpu, mat_gpu_tmp, device_nr, thread, iter, amountPerThread, index_start, jacobiSize, width, height, eps, grid_g, maxEps);

}