#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// https://ori-cohen.medium.com/real-life-cuda-programming-part-4-error-checking-e66dcbad6b55
#define cudaErrorHandle(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) 
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

#define N 1000000

int main() {
    int *h_a, *h_b;
    int *d_a, *d_b;

    float milliseconds = 0;

    cudaEvent_t start, stop;

    // Allocate memory on the host (CPU)
    cudaErrorHandle(cudaMallocHost(&h_a, N*sizeof(int)));
    cudaErrorHandle(cudaMallocHost(&h_a, N*sizeof(int)));

    // Allocate memory on the device (GPU)
    cudaErrorHandle(cudaMalloc((void**)&d_a, N * sizeof(int)));
    cudaErrorHandle(cudaMalloc((void**)&d_b, N * sizeof(int)));

    // Initialize data on the host
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
    }

    // Record start time
    cudaErrorHandle(cudaEventCreate(&start));
    cudaErrorHandle(cudaEventCreate(&stop));
    cudaErrorHandle(cudaEventRecord(start));

    // Transfer data from host to device
    cudaErrorHandle(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));

    // Transfer data from device to host
    cudaErrorHandle(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Record stop time
    cudaErrorHandle(cudaEventRecord(stop));
    cudaErrorHandle(cudaEventSynchronize(stop));
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Latency: %f milliseconds\n", milliseconds);

    // Clean up
    cudaErrorHandle(cudaFree(d_a));
    cudaErrorHandle(cudaFree(d_b));
    cudaErrorHandle(cudaFreeHost(h_a));
    cudaErrorHandle(cudaFreeHost(h_b));

    return 0;
}

