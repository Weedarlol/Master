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





void full_calculation_overlap(cudaPitchedPtr mat_gpu, int width, int height, int depth){

}

