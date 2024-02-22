#include <stdio.h>
#include <math.h>


#include "errorHandle.h"
#include "jacobi.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

void initializeStreamsAndEvents(int gpus, cudaStream_t streams[][2], cudaEvent_t events[][4], cudaEvent_t *startevent, cudaEvent_t *stopevent){
    for (int g = 0; g < gpus; g++) {
        cudaSetDevice(g);
        cudaErrorHandle(cudaStreamCreate(&streams[g][0]));
        cudaErrorHandle(cudaStreamCreate(&streams[g][1]));
        cudaErrorHandle(cudaEventCreate(&events[g][0]));
        cudaErrorHandle(cudaEventCreate(&events[g][1]));
        cudaErrorHandle(cudaEventCreate(&events[g][2]));
        cudaErrorHandle(cudaEventCreate(&events[g][3]));
    }
    cudaErrorHandle(cudaEventCreate(startevent));
    cudaErrorHandle(cudaEventCreate(stopevent));
}

void freeStreamsAndEvents(int gpus, cudaStream_t streams[][2], cudaEvent_t events[][4], cudaEvent_t *startevent, cudaEvent_t *stopevent) {
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaStreamDestroy(streams[g][0]));
        cudaErrorHandle(cudaStreamDestroy(streams[g][1]));
        cudaErrorHandle(cudaEventDestroy(events[g][0]));
        cudaErrorHandle(cudaEventDestroy(events[g][1]));
        cudaErrorHandle(cudaEventDestroy(events[g][2]));
        cudaErrorHandle(cudaEventDestroy(events[g][3]));
    }
}


__global__ void jacobiEdge(cudaPitchedPtr mat_gpu, int width, int height, int depth){
    cg::grid_group grid_g = cg::this_grid();
    int thread = grid_g.thread_rank();
    int thread_size = grid_g.size();

    char* d_ptr = static_cast<char*>(mat_gpu.ptr);
    size_t pitch = mat_gpu.pitch;

    int * element  = (int *)(thread);
    element[thread] = thread;
    
}


void full_calculation_overlap(cudaPitchedPtr mat_gpu, int width, int height, int depth, int gpus, int iter, dim3 blockDim, dim3 gridDim){
    
    void ***kernelCollMid;
    cudaErrorHandle(cudaMallocHost(&kernelCollMid, gpus * sizeof(void**)));
    // Allocates the elements in the kernelCollMid, used for cudaLaunchCooperativeKernel as functon variables.
    for (int g = 0; g < gpus; g++) {
        void **kernelArgs = new void*[4];
        kernelArgs[0] = &mat_gpu;     
        kernelArgs[1] = &width;    
        kernelArgs[2] = &height;
        kernelArgs[3] = &depth;

        kernelCollMid[g] = kernelArgs;
    }

    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiMid, gridDim, blockDim, kernelCollMid[0], 0, streams[0][1]));

    cudaErrorHandle(cudaEventRecord(stopevent));
    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", iter - iter);

    freeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);
}

