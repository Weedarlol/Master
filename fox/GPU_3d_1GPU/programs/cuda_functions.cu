#include "cuda_functions.h"

void initializeStreamsAndEvents(cudaStream_t streams[2], cudaEvent_t events[4], cudaEvent_t *startevent, cudaEvent_t *stopevent) {
    cudaErrorHandle(cudaStreamCreate(&streams[0]));
    cudaErrorHandle(cudaStreamCreate(&streams[1]));
    cudaErrorHandle(cudaEventCreate(&events[0]));
    cudaErrorHandle(cudaEventCreate(&events[1]));
    cudaErrorHandle(cudaEventCreate(&events[2]));
    cudaErrorHandle(cudaEventCreate(&events[3]));
    cudaErrorHandle(cudaEventCreate(startevent));
    cudaErrorHandle(cudaEventCreate(stopevent));
}

void freeStreamsAndEvents(cudaStream_t streams[2], cudaEvent_t events[4], cudaEvent_t *startevent, cudaEvent_t *stopevent) {
    cudaErrorHandle(cudaStreamDestroy(streams[0]));
    cudaErrorHandle(cudaStreamDestroy(streams[1]));
    cudaErrorHandle(cudaEventDestroy(events[0]));
    cudaErrorHandle(cudaEventDestroy(events[1]));
    cudaErrorHandle(cudaEventDestroy(events[2]));
    cudaErrorHandle(cudaEventDestroy(events[3]));

}
