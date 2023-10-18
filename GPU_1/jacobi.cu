#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ int shared_var = 0;

__device__ void calc(float *mat_gpu, float *mat_gpu_tmp, int thread, int iter,
    int amountPerThread, int index_start, int jacobiSize, int width, int height,
    float eps, cg::grid_group grid_g, int *maxEps){
    /*
    Variables  | Type | Description
    local_var  | int  | Calculates how many elements a thread has that is accepted between two iterations
    shared_var | int  | Is a while condition that decides when the jacobian iteration is complete or not.
    */

	int local_var;

    grid_g.sync();

    // Calculating Jacobian Matrix
    while(iter > 0 && shared_var == 0){
        // Resets the local_var value
        local_var = 0;

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
        for (int i = grid_g.num_threads() / 2; i > 0; i /= 2){
            maxEps[thread] = local_var;
            grid_g.sync(); // wait for all threads to store
            if(thread<i) local_var += maxExp[thread + i];
            grid_g.sync(); // wait for all threads to load
        }

        // If the combined value is larger than 0, it means that there is at least one element which could be reduced further.
        if(thread == 0){
            if(maxEps[0] == 0){
                shared_var = iter;
            }
        }

        // Changes pointers
        float *mat_tmp_cha = mat_gpu_tmp;
        mat_gpu_tmp = mat_gpu;
        mat_gpu = mat_tmp_cha;

        iter--;

        grid_g.sync();
    
    }

    // Sets the maxEps value to be equal to the iter value, which will be 0 if all the iterations are run and a solution was not found.
    if(thread == 0){
        maxEps[0] = iter;
    }

}

__global__ void jacobi(float *mat_gpu, float *mat_gpu_tmp, float eps, int width, int height, int iter, int *maxEps){
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

    cg::grid_group grid_g = cg::this_grid();
    int maxThreads = grid_g.num_threads();
    int jacobiSize = (width - 2) * (height - 2);
    int amountPerThread = jacobiSize / maxThreads;
    int leftover = jacobiSize % maxThreads;
    int thread = grid_g.thread_rank();
    int index_start = thread * amountPerThread + min(thread, leftover); //- (thread < leftover ? thread : 0);

    if(thread < leftover){
        amountPerThread++;
    }

    calc(mat_gpu, mat_gpu_tmp, thread, iter, amountPerThread, index_start, jacobiSize,
        width, height, eps, grid_g, maxEps);

}