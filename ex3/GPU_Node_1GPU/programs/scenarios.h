void full_calculation(double *data_gpu, double *data_gpu_tmp, int width, int height, int depth_node, int iter, int rank, int size, dim3 gridDim, dim3 blockDim, void** kernelCollMid);
