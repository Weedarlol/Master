#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void calc(double *data_gpu, double *data_gpu_tmp, int elementsPerThread, int index_start, int width, int height, int thread, int threadSize){
    for(int i = 0; i < elementsPerThread; i++){
        int index = index_start + i*threadSize;
        int x = index % (width-2) + 1;
        int y = index / (width-2) + 1;
        index = x + y*width;
        data_gpu_tmp[index] = 0.25 * (
            data_gpu[index + 1]     + data_gpu[index - 1] +
            data_gpu[index + width] + data_gpu[index - width]);
    }   
}

__global__ void jacobiEdge(double *data_gpu, double *data_gpu_tmp, int width, int height, 
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
            calc(data_gpu, data_gpu_tmp, elementsPerThread, thread, width, height, thread, threadSize);
        }
        // Selects all threads with index between width and width*2
        else if(thread > elementsLeftover && thread < elementsLeftover+elementsLeftover){
            calc(data_gpu, data_gpu_tmp, elementsPerThread, thread+rows_compute*(width-2), width, height, thread, threadSize);
        }
    }
    else if(threadSize > elementsLeftover){
        elementsPerThread++;
        if(thread < elementsLeftover){
            // The same threads will compute both rows
            calc(data_gpu, data_gpu_tmp, elementsPerThread, thread, width, height, thread, threadSize);
            calc(data_gpu, data_gpu_tmp, elementsPerThread, thread+rows_compute*(width-2), width, height, thread, threadSize);
        }
    }
    // There are less threads than elements in 1 row
    else{
        calc(data_gpu, data_gpu_tmp, elementsPerThread, thread, width, height, thread, threadSize);
        calc(data_gpu, data_gpu_tmp, elementsPerThread, thread+rows_compute*(width-2), width, height, thread, threadSize);
    }
}



__global__ void jacobiMid(double *data_gpu, double *data_gpu_tmp, int width, int height, 
                        int rows_elementsLeftover, int device_nr, int rows_compute, int elementsPerThreadExtra, int elementsLeftoverExtra,
                        int elementsPerThread, int elementsLeftover, int overlap_calc){


    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int threadSize = grid_g.size();


    if(device_nr < rows_elementsLeftover){
        if(thread < elementsLeftoverExtra){
            elementsPerThreadExtra++;
        }
        calc(data_gpu, data_gpu_tmp, elementsPerThreadExtra, thread+overlap_calc, width, height, thread, threadSize);
    }
    else{
        if(thread < elementsLeftover){
            elementsPerThread++;
        }
        calc(data_gpu, data_gpu_tmp, elementsPerThread, thread+overlap_calc, width, height, thread, threadSize);
    }
}