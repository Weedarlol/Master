#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

__device__ int shared_var = 1;

__device__ void calc(float *mat_gpu, float *mat_gpu_tmp, int device_nr, int thread, int iter,
    int amountPerThread, int index_start, int jacobiSize, int width, int height,
    float eps, grid_group grid_g, int *maxEps){

	int local_var = 0;

    grid_g.sync();

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

    // If any element in the matrix is 1, the jacobian matrix is not finished, and we therefore continue
    maxEps[thread] = local_var;

    grid_g.sync();

    for(int i = 2; i <= grid_g.num_threads(); i*=2){
        if(thread < grid_g.num_threads()/i){
            maxEps[thread] =  maxEps[thread] + maxEps[thread + grid_g.num_threads()/i];
        }
    }

    grid_g.sync();

}

__global__ void jacobi(float *mat_gpu, float *mat_gpu_tmp, int *maxEps, int device_nr, int dataPerGpu, float eps, int width, int height, int iter){
    /*
    Variables      | Type      | Description
    grid_g         | grid_group| Creates a group compromising of all the threads
    maxThreads     | int       | Total number of available threads within the grid_g group
    jacobiSize     | int       | Number of elements in the matrix which is to be calculated each iteration
    amountPerThread| int       | Number of elements to be calculated by each thread each iteration
    leftover       | int       | Number of threads which is required to compute one more element to be calculate all the elements
    thread         | int       | The index of each thread
    index_start    | int       | Element index the thread will start computing on, unique for each thread in grid_g group
    */

    grid_group grid_g = this_grid();
    int maxThreads = grid_g.num_threads();
    int jacobiSize = (width - 2) * (height - 2);
    int amountPerThread = jacobiSize / maxThreads;
    int leftover = jacobiSize % maxThreads;
    int thread = grid_g.thread_rank();
    int index_start = thread * amountPerThread + min(thread, leftover);

    if(thread < leftover){
        amountPerThread++;
    }

    calc(mat_gpu, mat_gpu_tmp, device_nr, thread, iter, amountPerThread, index_start, jacobiSize, width, height, eps, grid_g, maxEps);

}