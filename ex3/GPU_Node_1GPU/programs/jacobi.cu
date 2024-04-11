#include <stdio.h>
#include <math.h>
#include <nvtx3/nvToolsExt.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;


__device__ void calc(double *data_gpu, double *data_gpu_tmp, int index_start, int elementsPerThread, 
    int threadSize, int width, int height){
    double division = 1.0/6;
    for(int i = 0; i < elementsPerThread; i++){
        int index = index_start + i*threadSize;
        int x = index % (width - 2) + 1;
        int y = (index / (width - 2)) % (height - 2) + 1;
        int z = index / ((width - 2) * (height - 2)) + 1;
        index = x + y*width + z*width*height;

        data_gpu_tmp[index] = division * (
            data_gpu[index + 1]            + data_gpu[index - 1] +
            data_gpu[index + width]        + data_gpu[index - width] +
            data_gpu[index + width*height] + data_gpu[index - width*height]);
    } 
}

__global__ void jacobiMid(double *data_gpu, double *data_gpu_tmp, int width, int height, int threadSize, int elementsPerThread, int leftover, int overlap_calc){
    /*
    Variables      | Type      | Description
    grid_g         | grid_group| Creates a group compromising of all the threads
    threadSize    | int       | Total number of available threads within the grid_g group
    jacobiSize     | int       | Number of elements in the matrix which is to be calculated each iteration
    elementsPerThread| int       | Number of elements to be calculated by each thread each iteration
    leftover       | int       | Number of threads which is required to compute one more element to be calculate all the elements
    thread         | int       | The index of each thread
    */

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();

    if(thread < leftover){
        elementsPerThread++;
    }

    calc(data_gpu, data_gpu_tmp, thread + overlap_calc, elementsPerThread, 
        threadSize, width, height);
}

__global__ void jacobiEdge(double *data_gpu, double *data_gpu_tmp, int width, int height, int depth_node, int threadSize, int elementsPerThreadOverlap, int leftoverOverlap){
    /*
    Variables      | Type      | Description
    grid_g         | grid_group| Creates a group compromising of all the threads
    threadSize    | int       | Total number of available threads within the grid_g group
    jacobiSize     | int       | Number of elements in the matrix which is to be calculated each iteration
    elementsPerThread| int       | Number of elements to be calculated by each thread each iteration
    leftover       | int       | Number of threads which is required to compute one more element to be calculate all the elements
    thread         | int       | The index of each thread
    */

    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();

    if(elementsPerThreadOverlap > 0){
        if(thread < leftoverOverlap){
            elementsPerThreadOverlap++;
        }
        calc(data_gpu, data_gpu_tmp, thread, elementsPerThreadOverlap, threadSize, width, height);
        calc(data_gpu, data_gpu_tmp, thread + (depth_node-3)*(width-2)*(height-2), elementsPerThreadOverlap, threadSize, width, height);
    }
    // There are less threads than elements in 1 slice
    else{
        if(threadSize >= leftoverOverlap*2){
            elementsPerThreadOverlap++;
            // Selects all threads with index less than width
            if(thread < leftoverOverlap){
                calc(data_gpu, data_gpu_tmp, thread, elementsPerThreadOverlap, threadSize, width, height);
            }
            // Selects all threads with index between width and width*2
            else if(thread < leftoverOverlap+leftoverOverlap){
                calc(data_gpu, data_gpu_tmp, thread + (depth_node-3)*(width-2)*(height-2), elementsPerThreadOverlap, threadSize, width, height);
            }
        }
        else{
            elementsPerThreadOverlap++;
            if(thread < leftoverOverlap){
                // The same threads will compute both slices
                calc(data_gpu, data_gpu_tmp, thread, elementsPerThreadOverlap, threadSize, width, height);
                calc(data_gpu, data_gpu_tmp, thread + (depth_node-3)*(width-2)*(height-2), elementsPerThreadOverlap, threadSize, width, height);
            }
        }
    }




}