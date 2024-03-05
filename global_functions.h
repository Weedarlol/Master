// cuda_utils.h

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

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

void fillValues(double *mat, double dx, double dy, int width, int height);

void fillValues3D(double *mat, int width, int height, int depth_node, double dx, double dy, double dz);

void initializeStreamsAndEvents(int gpus, cudaStream_t streams[][2], cudaEvent_t events[][4], cudaEvent_t *startevent, cudaEvent_t *stopevent);

void freeStreamsAndEvents(int gpus, cudaStream_t streams[][2], cudaEvent_t events[][4], cudaEvent_t *startevent, cudaEvent_t *stopevent);

#endif // CUDA_UTILS_H
