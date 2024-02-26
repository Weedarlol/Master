__global__ void jacobiEdge(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int rows_compute, int amountPerThread, int leftover);

__global__ void jacobiMid(cudaPitchedPtr mat_gpu, int width, int height, int depth);