#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void calc(double *grid_gpu, double *grid_gpu_tmp, int amountPerThread, int index_start, int width, int height, int depth, int thread, int thread_size){
    double division = 1.0/6;
    for(int i = 0; i < amountPerThread; i++){
        int index = index_start + i*thread_size;
        int x = index % (width - 2) + 1;
        int y = (index / (width - 2)) % (height - 2) + 1;
        int z = index / ((width - 2) * (height - 2)) + 1;
        index = x + y*width + z*width*height;

        grid_gpu_tmp[index] = division * (
            grid_gpu[index + 1]            + grid_gpu[index - 1] +
            grid_gpu[index + width]        + grid_gpu[index - width] +
            grid_gpu[index + width*height] + grid_gpu[index - width*height]);
    }   
}

__global__ void jacobiEdge(double *grid_gpu, double *grid_gpu_tmp, int width, int height, 
                        int slices_compute, int amountPerThread, int leftover){

    cg::grid_group thread_grid = cg::this_grid();
    int thread = thread_grid.thread_rank();
    int thread_size = thread_grid.size();

    // More threads than elements in 2 slices
    if(thread_size > leftover*2){
        amountPerThread++;
        // Selects all threads with index less than width
        if(thread < leftover){
            calc(grid_gpu, grid_gpu_tmp, amountPerThread, thread, width, height, depth, thread, thread_size);
        }
        // Selects all threads with index between width and width*2
        else if(thread > leftover && thread < leftover+leftover){
            calc(grid_gpu, grid_gpu_tmp, amountPerThread, thread+slices_compute*(width-2), width, height, depth, thread, thread_size);
        }
    }
    else if(thread_size > leftover){
        amountPerThread++;
        if(thread < leftover){
            // The same threads will compute both slices
            calc(grid_gpu, grid_gpu_tmp, amountPerThread, thread, width, height, depth, thread, thread_size);
            calc(grid_gpu, grid_gpu_tmp, amountPerThread+slices_compute*(width-2), thread, width, height, depth, thread, thread_size);
        }
    }
    // There are less threads than elements in 1 slice
    else{
        calc(grid_gpu, grid_gpu_tmp, amountPerThread, thread, width, height, thread, thread_size);
        calc(grid_gpu, grid_gpu_tmp, amountPerThread+slices_compute*(width-2), thread, width, height, thread, thread_size);
    }
}



__global__ void jacobiMid(double *grid_gpu, double *grid_gpu_tmp, int width, int height, int depth,
                        int slices_elementsLeftover, int device_nr, int slices_compute, int elementsPerThreadExtra, int elementsLeftoverExtra,
                        int elementsPerThread, int elementsLeftover){


    cg::grid_group thread_grid = cg::this_grid();
    int thread = thread_grid.thread_rank(); 
    int threadSize = thread_grid.size();


    if(device_nr < slices_elementsLeftover){
        if(thread < elementsLeftoverExtra){
            elementsPerThreadExtra++;
        }
        calc(grid_gpu, grid_gpu_tmp, elementsPerThreadExtra, thread, width, height, depth, thread, threadSize);
    }
    else{
        if(thread < elementsLeftover){
            elementsPerThread++;
        }
        calc(grid_gpu, grid_gpu_tmp, elementsPerThread, thread, width, height, depth, thread, threadSize);
    }
}