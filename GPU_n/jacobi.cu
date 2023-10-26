#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

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
        if(thread<i) local_var += maxEps[thread + i];
        grid_g.sync(); // wait for all threads to load
    }
}

__global__ void jacobi(float *mat_gpu, float *mat_gpu_tmp, int *maxEps, int device_nr, int dataLeftover, float eps, int width, int height, int iter, int jacobiSize, int amountPerThread, int leftover){
    /*
    Variables      | Type      | Description
    grid_g         | grid_group| Creates a group compromising of all the threads
    thread         | int       | The index of each thread
    index_start    | int       | Element index the thread will start computing on, unique for each thread in grid_g group. It is also taking into consideration what device is computing
    */

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int index_start;


    // jacobiSize = antall elementer per device
    // amountPerThread = antall elementer per thread
    // dataLeftover = antal GPUer som krever 1 ekstra element
    // leftover = antall elementer som krever 1 ekstra element


    // If there are an unequal amount of elements to spread to the devices, compute as below
    if(dataLeftover > device_nr){
        // Add another element to compute for the device
        jacobiSize++;
        // Find the starting location for the thread based on device and elements
        index_start = thread * amountPerThread + min(thread, leftover+1) + jacobiSize*device_nr;
        // Find the last element to compute on
        jacobiSize += jacobiSize*device_nr;

        // If elements are unequal within the device, all threads with id less than "leftover" is granted one more element
        if(thread <= leftover){
            amountPerThread++;
        }
    }
    // If there are an equal number of elements spread to the devices, do this instead
    else{
        index_start = thread * amountPerThread + min(thread, leftover) + jacobiSize*device_nr + dataLeftover;
        jacobiSize += jacobiSize*device_nr + dataLeftover;

        if(thread < leftover){
            amountPerThread++;
        }
    }

/* 


    // If number of elements per device is unequal, if true, then add push index of element start per thread backwards
    if(dataLeftover > device_nr){
        index_start = thread * amountPerThread + min(thread, leftover) + jacobiSize*device_nr + device_nr;
        jacobiSize += jacobiSize*device_nr + device_nr;
    }
    else{
        index_start = thread * amountPerThread + min(thread, leftover) + jacobiSize*device_nr + dataLeftover;
        jacobiSize += jacobiSize*device_nr + dataLeftover;
    }



    // If number of elements per thread is unequal
    if(thread < leftover){
        amountPerThread++;
    }
    else if(thread = leftover && dataLeftover > device_nr){
        amountPerThread++;
    } */

    calc(mat_gpu, mat_gpu_tmp, device_nr, thread, iter, amountPerThread, index_start, jacobiSize, width, height, eps, grid_g, maxEps);

}