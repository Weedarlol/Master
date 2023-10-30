#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void calc(float *mat_gpu, float *mat_gpu_tmp, int amountPerThread, int index_start, int jacobiSize, int width, int height, float eps){

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
}

__global__ void jacobi(float *mat_gpu, float *mat_gpu_tmp, int rows, int device_nr, int gpus){
    /*
    Variables      | Type      | Description
    grid_g         | grid_group| Creates a group compromising of all the threads
    thread         | int       | The index of each thread
    index_start    | int       | Element index the thread will start computing on, unique for each thread in grid_g group. It is also taking into consideration what device is computing
    */

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int index_start;

    

}