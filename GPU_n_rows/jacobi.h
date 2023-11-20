__global__ void jacobi(double *mat_gpu, double *mat_gpu_tmp, int number_rows, int width, int height, 
                        int rows_leftover, int device_nr, int rows_compute, int amountPerThreadExtra, int extraLeftover,
                        int amountPerThread, int leftover, int *maxEps, double eps);