#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void calc(double *mat_gpu, double *mat_gpu_tmp, int elementsPerThread, int index_start, int width, int height, int thread, cg::grid_group grid_g, int threadSize){
    for(int i = 0; i < elementsPerThread; i++){
        int index = index_start + i*threadSize;
        int x = index % (width-2) + 1;
        int y = index / (width-2) + 1;
        index = x + y*width;
        mat_gpu_tmp[index] = 0.25 * (
            mat_gpu[index + 1]     + mat_gpu[index - 1] +
            mat_gpu[index + width] + mat_gpu[index - width]);
    }   
}

__global__ void jacobiEdge(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int rows_compute, int elementsPerThread, int elementsLeftover,
                        int warpAmount){

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int threadSize = grid_g.size();

    // More threads than elements in 2 rows
    if(threadSize > elementsLeftover*2){
        elementsPerThread++;
        // Selects all threads with index less than width
        if(thread < elementsLeftover){
            calc(mat_gpu, mat_gpu_tmp, elementsPerThread, thread, width, height, thread, grid_g, threadSize);
        }
        // Selects all threads with index between width and width*2
        else if(thread > elementsLeftover && thread < elementsLeftover+elementsLeftover){
            calc(mat_gpu, mat_gpu_tmp, elementsPerThread, thread+rows_compute*(width-2), width, height, thread, grid_g, threadSize);
        }
    }
    else if(threadSize > elementsLeftover){
        elementsPerThread++;
        if(thread < elementsLeftover){
            // The same threads will compute both rows
            calc(mat_gpu, mat_gpu_tmp, elementsPerThread, thread, width, height, thread, grid_g, threadSize);
            calc(mat_gpu, mat_gpu_tmp, elementsPerThread, thread+rows_compute*(width-2), width, height, thread, grid_g, threadSize);
        }
    }
    // There are less threads than elements in 1 row
    else{
        calc(mat_gpu, mat_gpu_tmp, elementsPerThread, thread, width, height, thread, grid_g, threadSize);
        calc(mat_gpu, mat_gpu_tmp, elementsPerThread, thread+rows_compute*(width-2), width, height, thread, grid_g, threadSize);
    }
}



__global__ void jacobiMid(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int rows_elementsLeftover, int device_nr, int rows_compute, int elementsPerThreadExtra, int elementsLeftoverExtra,
                        int elementsPerThread, int elementsLeftover, int overlap_calc){


    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int threadSize = grid_g.size();


    if(device_nr < rows_elementsLeftover){
        if(thread < elementsLeftoverExtra){
            elementsPerThreadExtra++;
        }
        calc(mat_gpu, mat_gpu_tmp, elementsPerThreadExtra, thread+overlap_calc, width, height, thread, grid_g, threadSize);
    }
    else{
        if(thread < elementsLeftover){
            elementsPerThread++;
        }
        calc(mat_gpu, mat_gpu_tmp, elementsPerThread, thread+overlap_calc, width, height, thread, grid_g, threadSize);
    }
}