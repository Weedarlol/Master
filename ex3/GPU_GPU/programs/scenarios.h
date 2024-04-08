void full_calculation_overlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid);
void full_calculation_nooverlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid);
void no_kernel_overlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid);
void no_kernel_nooverlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid);
void no_communication_overlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid);
void no_communication_nooverlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid);
void only_calculation_overlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid);
void only_calculation_nooverlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid);
void only_communication_overlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid);
void only_communication_nooverlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid);
