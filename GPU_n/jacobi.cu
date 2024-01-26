#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void calc(double *mat_gpu, double *mat_gpu_tmp, int amountPerThread, int index_start, int width, int height, int thread, cg::grid_group grid_g){
    int local_var = 0;

    for(int i = 0; i < amountPerThread; i++){
        int index = index_start + i;
        int x = index % (width-2) + 1;
        int y = index / (width-2) + 1;
        index = x + y*width;
        /* mat_gpu_tmp[index] = 0.25 * (mat_gpu[index + 1] + mat_gpu[index - 1]); */ // 2 loads, 1 store, 1 write allocates
        mat_gpu_tmp[index] = 0.25 * (
            mat_gpu[index + 1]     + mat_gpu[index - 1] +
            mat_gpu[index + width] + mat_gpu[index - width]);
    }
    
}

__global__ void jacobiTop(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int rows_compute, int amountPerThread, int leftover){

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();

    if(amountPerThread == 0){
        if(thread < leftover){
            amountPerThread++;
            calc(mat_gpu, mat_gpu_tmp, amountPerThread, thread, width, height, thread, grid_g);
        }
    }
    else{
        int index_start = thread*amountPerThread + min(thread,leftover);
        if(thread < leftover){
            amountPerThread++;
        }
        calc(mat_gpu, mat_gpu_tmp, amountPerThread, index_start, width, height, thread, grid_g);
    }
}

__global__ void jacobiBot(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int rows_compute, int amountPerThread, int leftover){

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();

    if(amountPerThread == 0){
        int index_start = thread + (rows_compute+1) * (width-2);
        if(thread < leftover){
            amountPerThread++;
            calc(mat_gpu, mat_gpu_tmp, amountPerThread, index_start, width, height, thread, grid_g);
        }
    }
    else{
        int index_start = thread*amountPerThread + min(thread,leftover) + (rows_compute+1) * (width-2);
        if(thread < leftover){
            amountPerThread++;
        }
        calc(mat_gpu, mat_gpu_tmp, amountPerThread, index_start, width, height, thread, grid_g);
    }
}

__global__ void jacobiMid(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int rows_leftover, int device_nr, int rows_compute, int amountPerThreadExtra, int leftoverExtra,
                        int amountPerThread, int leftover, int overlap_calc){


    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();



    // Overlap
    // 0, 1 < 2
    if(device_nr < rows_leftover){
        // start    =   x   *        340          + min(   x  ,      9562    ) +   4094
        int index_start = thread*amountPerThreadExtra + min(thread, leftoverExtra) + overlap_calc;
        //   x    <       9562
        if(thread < leftoverExtra){
            amountPerThreadExtra++;
        }
        calc(mat_gpu, mat_gpu_tmp, amountPerThreadExtra, index_start, width, height, thread, grid_g);
    }
    else{
        //  start   =   x   *     340        + min(  x   ,  5468   ) + 4094
        int index_start = thread*amountPerThread + min(thread, leftover) + overlap_calc;
        //    x   <   5468
        if(thread < leftover){
            amountPerThread++;
        }
        calc(mat_gpu, mat_gpu_tmp, amountPerThread, index_start, width, height, thread, grid_g);
    }
    
    
    

}