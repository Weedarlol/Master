void full_calculation_overlap(double **mat_gpu, double **mat_gpu_tmp, int height, int width, int iter, int gpus, int *rows_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid);