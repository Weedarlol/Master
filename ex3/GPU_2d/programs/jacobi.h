__global__ void jacobiEdge(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int rows_compute, int amountPerThread, int leftover);

__global__ void jacobiMid(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int rows_leftover, int device_nr, int rows_compute, int amountPerThreadExtra, int leftoverExtra,
                        int amountPerThread, int leftover, int overlap_calc);