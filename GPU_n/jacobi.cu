#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void calc(double *mat_gpu, double *mat_gpu_tmp, int amountPerThread, int index_start, int width, int height, double eps, int *maxEps, int thread, cg::grid_group grid_g){
    int local_var = 0;

    for(int i = 0; i < amountPerThread; i++){
        int index = index_start + i;
        int x = index % (width-2) + 1;
        int y = index / (width-2) + 1;
        index = x + y*width;
        mat_gpu_tmp[index] = 0.25 * (
            mat_gpu[index + 1]     + mat_gpu[index - 1] +
            mat_gpu[index + width] + mat_gpu[index - width]);

        if(abs(mat_gpu[index] - mat_gpu_tmp[index]) > eps){
            local_var++;
        }
    }

     // https://developer.nvidia.com/blog/cooperative-groups/
    if((width-2)*2 > grid_g.num_threads()){
        for (int i = grid_g.num_threads() / 2; i > 0; i /= 2){
            maxEps[thread] = local_var;
            grid_g.sync(); // wait for all threads to store
            if(thread<i) local_var += maxEps[thread + i];
            grid_g.sync(); // wait for all threads to load
        }
    }
    
}

__global__ void jacobiTop(double *mat_gpu, double *mat_gpu_tmp, int number_rows, int width, int height, 
                        int rows_leftover, int device_nr, int rows_compute, int amountPerThread, int leftover, 
                        int *maxEps, double eps){

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();

    if(amountPerThread == 0){
        if(thread < leftover){
            amountPerThread++;
            calc(mat_gpu, mat_gpu_tmp, amountPerThread, thread, width, height, eps, maxEps, thread, grid_g);
        }
    }
    else{
        int index_start = thread*amountPerThread + min(thread,leftover);
        if(thread < leftover){
            amountPerThread++;
        }
        calc(mat_gpu, mat_gpu_tmp, amountPerThread, index_start, width, height, eps, maxEps, thread, grid_g);
    }
}

__global__ void jacobiBot(double *mat_gpu, double *mat_gpu_tmp, int number_rows, int width, int height, 
                        int rows_leftover, int device_nr, int rows_compute, int amountPerThread, int leftover, 
                        int *maxEps, double eps){

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int move_index = (rows_compute+1)*(width-2);

    if(amountPerThread == 0){
        int index_start = thread+move_index;
        if(thread < leftover){
            amountPerThread++;
            calc(mat_gpu, mat_gpu_tmp, amountPerThread, index_start, width, height, eps, maxEps, thread, grid_g);
        }
    }
    else{
        int index_start = thread*amountPerThread + min(thread,leftover) + move_index;
        if(thread < leftover){
            amountPerThread++;
        }
        calc(mat_gpu, mat_gpu_tmp, amountPerThread, index_start, width, height, eps, maxEps, thread, grid_g);
    }
}

__global__ void jacobiMid(double *mat_gpu, double *mat_gpu_tmp, int number_rows, int width, int height, 
                        int rows_leftover, int device_nr, int rows_compute, int amountPerThreadExtra, int leftoverExtra,
                        int amountPerThread, int leftover, int *maxEps, double eps, int overlap){


    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int index_start;

    
    if(device_nr < rows_leftover){
        index_start = thread*amountPerThreadExtra + min(thread, leftoverExtra) + (width-2)*overlap;
        if(thread < leftoverExtra){
            amountPerThreadExtra++;
        }
        calc(mat_gpu, mat_gpu_tmp, amountPerThreadExtra, index_start, width, height, eps, maxEps, thread, grid_g);
    }
    else{
        index_start = thread*amountPerThread + min(thread, leftover) + (width-2)*overlap;
        if(thread < leftover){
            amountPerThread++;
        }
        calc(mat_gpu, mat_gpu_tmp, amountPerThread, index_start, width, height, eps, maxEps, thread, grid_g);
    }
    
    
    

}