__global__ void jacobi(float *mat_gpu, float *mat_gpu_tmp, int number_rows, int width, int height, 
                        int rows_leftover, int device_nr, int rows_compute, int amountPerThreadExtra, int extraLeftover,
                        int amountPerThread, int leftover, int *maxEps, float eps);