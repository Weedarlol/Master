#include <stdio.h>
#include <math.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void calc(double *mat_gpu, double *mat_gpu_tmp, int amountPerThread, int index_start, int width, int height, int thread, int thread_size){
    double division = 1.0/6;
    for(int i = 0; i < amountPerThread; i++){
        int index = index_start + i*thread_size;
        int x = index % (width - 2) + 1;
        int y = (index / (width - 2)) % (height - 2) + 1;
        int z = index / ((width - 2) * (height - 2)) + 1;
        index = x + y*width + z*width*height;

        mat_gpu_tmp[index] = division * (
            mat_gpu[index + 1]            + mat_gpu[index - 1] +
            mat_gpu[index + width]        + mat_gpu[index - width] +
            mat_gpu[index + width*height] + mat_gpu[index - width*height]);
    }   
}

__global__ void jacobiEdge(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int slices_compute, int amountPerThread, int leftover){

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int thread_size = grid_g.size();

    // More threads than elements in 2 slices
    if(thread_size > leftover*2){
        amountPerThread++;
        // Selects all threads with index less than width
        if(thread < leftover){
            calc(mat_gpu, mat_gpu_tmp, amountPerThread, thread, width, height, thread, thread_size);
        }
        // Selects all threads with index between width and width*2
        else if(thread > leftover && thread < leftover+leftover){
            calc(mat_gpu, mat_gpu_tmp, amountPerThread, thread+slices_compute*(width-2), width, height, thread, thread_size);
        }
    }
    else if(thread_size > leftover){
        amountPerThread++;
        if(thread < leftover){
            // The same threads will compute both slices
            calc(mat_gpu, mat_gpu_tmp, amountPerThread, thread, width, height, thread, thread_size);
            calc(mat_gpu, mat_gpu_tmp, amountPerThread+slices_compute*(width-2), thread, width, height, thread, thread_size);
        }
    }
    // There are less threads than elements in 1 slice
    else{
        calc(mat_gpu, mat_gpu_tmp, amountPerThread, thread, width, height, thread, thread_size);
        calc(mat_gpu, mat_gpu_tmp, amountPerThread+slices_compute*(width-2), thread, width, height, thread, thread_size);
    }
}



__global__ void jacobiMid(double *mat_gpu, double *mat_gpu_tmp, int width, int height, int depth,
                        int slices_elementsLeftover, int device_nr, int slices_compute, int elementsPerThreadExtra, int elementsLeftoverExtra,
                        int elementsPerThread, int elementsLeftover, int overlap_calc){


    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank(); 
    int threadSize = grid_g.size();


    if(device_nr < slices_elementsLeftover){
        if(thread < elementsLeftoverExtra){
            elementsPerThreadExtra++;
        }
        calc(mat_gpu, mat_gpu_tmp, elementsPerThreadExtra, thread+overlap_calc, width, height, thread, threadSize);
    }
    else{
        if(thread < elementsLeftover){
            elementsPerThread++;
        }
        calc(mat_gpu, mat_gpu_tmp, elementsPerThread, thread+overlap_calc, width, height, thread, threadSize);
    }
}