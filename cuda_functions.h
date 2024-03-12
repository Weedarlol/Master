#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define cudaErrorHandle(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) 
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

void initializeStreamsAndEvents(int gpus, cudaStream_t streams[][2], cudaEvent_t events[][4], cudaEvent_t *startevent, cudaEvent_t *stopevent);

void freeStreamsAndEvents(int gpus, cudaStream_t streams[][2], cudaEvent_t events[][4], cudaEvent_t *startevent, cudaEvent_t *stopevent);

#endif // CUDA_UTILS_H
