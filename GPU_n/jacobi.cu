#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void calc(double *mat_gpu, double *mat_gpu_tmp, int amountPerThread, int index_start, int width, int height, int thread, cg::grid_group grid_g, int thread_size){
    for(int i = 0; i < amountPerThread; i++){
        int index = index_start + i*thread_size;
        int x = index % (width-2) + 1;
        int y = index / (width-2) + 1;
        index = x + y*width;
        mat_gpu_tmp[index] = 0.25 * (mat_gpu[index + 1] + mat_gpu[index - 1]); // 2 loads, 1 store, 1 write allocates
        /* mat_gpu_tmp[index] = 0.25 * (
            mat_gpu[index + 1]     + mat_gpu[index - 1] +
            mat_gpu[index + width] + mat_gpu[index - width]); */
    }   
}

__global__ void jacobiEdge(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int rows_compute, int amountPerThread, int leftover){

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int thread_size = grid_g.size();

    // More threads than elements in 2 rows
    if(thread_size > leftover*2){
        amountPerThread++;
        // Selects all threads with index less than width
        if(thread < leftover){
            calc(mat_gpu, mat_gpu_tmp, amountPerThread, thread, width, height, thread, grid_g, thread_size);
        }
        // Selects all threads with index between width and width*2
        else if(thread < leftover*2){
            calc(mat_gpu, mat_gpu_tmp, amountPerThread, thread+rows_compute*(width-2), width, height, thread, grid_g, thread_size);
        }
    }
    // Thread quantity between 1 and 2 rows
    else if(thread_size > leftover){
        amountPerThread += 2;
        if(thread < leftover){
            // The same threads will compute both rows
            calc(mat_gpu, mat_gpu_tmp, amountPerThread, thread, width, height, thread, grid_g, thread_size);
            calc(mat_gpu, mat_gpu_tmp, amountPerThread+rows_compute*(width-2), thread, width, height, thread, grid_g, thread_size);
        }
    }
    // There are less threads than elements in 1 row
    else{
        calc(mat_gpu, mat_gpu_tmp, amountPerThread, thread, width, height, thread, grid_g, thread_size);
        calc(mat_gpu, mat_gpu_tmp, amountPerThread+rows_compute*(width-2), thread, width, height, thread, grid_g, thread_size);
    }
}


__global__ void jacobiEdgeTwo(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int rows_compute, int amountPerThread, int leftover){

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int thread_size = grid_g.size();

    if(amountPerThread == 0){
        int index_start = thread + (rows_compute+1) * (width-2);

        if(thread < leftover){
            amountPerThread++;
            calc(mat_gpu, mat_gpu_tmp, amountPerThread, thread, width, height, thread, grid_g, thread_size);
            calc(mat_gpu, mat_gpu_tmp, amountPerThread, index_start, width, height, thread, grid_g, thread_size);
        }
    }
    else{
        int index_start = thread*amountPerThread + min(thread,leftover);
        if(thread < leftover){
            amountPerThread++;
        }
        calc(mat_gpu, mat_gpu_tmp, amountPerThread, index_start, width, height, thread, grid_g, thread_size);
        index_start += (rows_compute+1) * (width-2);
        calc(mat_gpu, mat_gpu_tmp, amountPerThread, index_start, width, height, thread, grid_g, thread_size);
    }
}




__global__ void jacobiMid(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int rows_leftover, int device_nr, int rows_compute, int amountPerThreadExtra, int leftoverExtra,
                        int amountPerThread, int leftover, int overlap_calc){


    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int thread_size = grid_g.size();


    if(device_nr < rows_leftover){
        if(thread < leftoverExtra){
            amountPerThreadExtra++;
        }
        calc(mat_gpu, mat_gpu_tmp, amountPerThreadExtra, thread+overlap_calc, width, height, thread, grid_g, thread_size);
    }
    else{
        if(thread < leftover){
            amountPerThread++;
        }
        calc(mat_gpu, mat_gpu_tmp, amountPerThread, thread+overlap_calc, width, height, thread, grid_g, thread_size);
    }
}