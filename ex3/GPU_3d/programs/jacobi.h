__global__ void jacobiEdge(double *mat_gpu, double *mat_gpu_tmp, int width, int height, 
                        int rows_compute, int amountPerThread, int leftover);

__global__ void jacobiMid(double *mat_gpu, double *mat_gpu_tmp, int width, int height, int depth,
                        int rows_elementsLeftover, int device_nr, int rows_compute, int elementsPerThreadExtra, int elementsLeftoverExtra,
                        int elementsPerThread, int elementsLeftover);