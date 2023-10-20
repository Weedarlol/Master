#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1000

__global__ void kernel(int *d_a, int *d_b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        d_b[tid] = d_a[tid];
    }
}

int main() {
    int *h_a, *h_b;
    int *d_a, *d_b;
    cudaEvent_t start, stop;
    float milliseconds = 0;

    // Allocate memory on the host (CPU)
    h_a = (int *)malloc(N * sizeof(int));
    h_b = (int *)malloc(N * sizeof(int));

    // Allocate memory on the device (GPU)
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));

    // Initialize data on the host
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
    }

    // Record start time
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Transfer data from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    kernel<<<numBlocks, blockSize>>>(d_a, d_b);

    // Transfer data from device to host
    cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Latency: %f milliseconds\n", milliseconds);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    return 0;
}


