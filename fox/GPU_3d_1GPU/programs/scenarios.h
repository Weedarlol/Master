void full_calculation_overlap(double *data_gpu, double *data_gpu_tmp, int width, int height, int depth_node, int iter, int rank, int size, dim3 gridDim, dim3 blockDim, void** kernelMid, void** kernelEdge);
void full_calculation_nooverlap(double *data_gpu, double *data_gpu_tmp, int width, int height, int depth_node, int iter, int rank, int size, dim3 gridDim, dim3 blockDim, void** kernelMid);
