void full_calculation_overlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth, int iter, int gpus, int rank, int size, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelMid);
void full_calculation_nooverlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth, int iter, int gpus, int rank, int size, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelMid);
void only_computation_nooverlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth_node, int iter, int gpus, int rank, int size, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelMid);
void only_communication_nooverlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth_node, int iter, int gpus, int rank, int size, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelMid);
