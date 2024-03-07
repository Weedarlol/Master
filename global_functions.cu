// cuda_utils.cu

#include "global_functions.h"

void fillValues(double *mat, double dx, double dy, int width, int height) {
    double x, y;

    memset(mat, 0, height * width * sizeof(double));

    for (int i = 1; i < height - 1; i++) {
        y = i * dy; // y coordinate
        for (int j = 1; j < width - 1; j++) {
            x = j * dx; // x coordinate
            mat[j + i * width] = sin(M_PI * y) * sin(M_PI * x);
        }
    }
}

void fillValues3D(double *mat, int width, int height, int depth_node, double dx, double dy, double dz, int rank) {
    double x, y, z;

    // Assuming the data in the matrix is stored contiguously in memory
    memset(mat, 0, height * width * depth_node * sizeof(double));

    for (int i = 1; i < depth_node-1; i++) {
        z = i * dz; // z coordinate
        for (int j = 1; j < height - 1; j++) {
            y = j * dy; // z coordinate
            for (int k = 1; k < width - 1; k++) {
                x = k * dx; // x coordinate
                mat[k +  j*width + i*width*height] = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
            }
        }
    }
}

void initializeStreamsAndEvents(int gpus, cudaStream_t streams[][2], cudaEvent_t events[][4], cudaEvent_t *startevent, cudaEvent_t *stopevent) {
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
    for (int g = 0; g < gpus; g++) {
        cudaErrorHandle(cudaStreamDestroy(streams[g][0]));
        cudaErrorHandle(cudaStreamDestroy(streams[g][1]));
        cudaErrorHandle(cudaEventDestroy(events[g][0]));
        cudaErrorHandle(cudaEventDestroy(events[g][1]));
        cudaErrorHandle(cudaEventDestroy(events[g][2]));
        cudaErrorHandle(cudaEventDestroy(events[g][3]));
    }
}
