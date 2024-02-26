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

void fillValues(double *mat, int width, int height){
    double x, y;

    double dx = 2.0 / (width - 1);
    double dy = 2.0 / (height - 1);

    memset(mat, 0, height*width*sizeof(double));

    for(int i = 1; i < height - 1; i++) {
        y = i * dy; // y coordinate
        for(int j = 1; j < width - 1; j++) {
            x = j * dx; // x coordinate
            mat[j + i*width] = sin(M_PI*y)*sin(M_PI*x);
        }
    }
}






int main(int argc, char *argv[]) {
     if (argc != 3) {
        printf("Usage: %s <Width> <Height> <Iterations>", argv[0]); // Programname
        return 1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);

    clock_t beginning, ending;
    beginning = clock();

    cudaEvent_t start, stop;
    cudaErrorHandle(cudaEventCreate(&start));
    cudaErrorHandle(cudaEventCreate(&stop));

    double *mat_cpu, *mat_gpu;
    cudaErrorHandle(cudaMallocHost(&mat_cpu, width*height*sizeof(double)));
    cudaErrorHandle(cudaMalloc(&mat_gpu, width*height*sizeof(double)));

    fillValues(mat_cpu, width, height);

    cudaErrorHandle(cudaEventRecord(start));

    cudaErrorHandle(cudaMemcpy(mat_gpu, mat_cpu, width*height*sizeof(double), cudaMemcpyHostToDevice));

    cudaErrorHandle(cudaEventRecord(stop));
    cudaErrorHandle(cudaEventSynchronize(stop));

    float milliseconds;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, start, stop));

    ending = clock();

    

    printf("Transfer time - CPU->GPU = %.4f\nTotal time = %.4f", milliseconds, ((double) (ending - beginning)) / CLOCKS_PER_SEC);

    return 0;
}

