__global__ void jacobiEdge(double *data_gpu_tmp_gpu, double *data_gpu_tmp_gpu_tmp, int width, int height, 
                        int rows_compute, int amountPerThread, int leftover,
                        int warpAmount);

__global__ void jacobiMid(double *data_gpu_tmp_gpu, double *data_gpu_tmp_gpu_tmp, int width, int height, 
                        int rows_leftover, int device_nr, int rows_compute, int amountPerThreadExtra, int leftoverExtra,
                        int amountPerThread, int leftover, int overlap_calc);