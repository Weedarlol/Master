/* void full_calculation_overlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *rows_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid); */
void full_calculation_nooverlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *rows_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid);
/* void no_kernel_overlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *rows_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid);
void no_kernel_nooverlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *rows_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid);
void no_communication_overlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *rows_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid);
void no_communication_nooverlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *rows_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid);
void only_calculation_overlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *rows_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid);
void only_calculation_nooverlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *rows_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid);
void only_communication_overlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *rows_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid);
void only_communication_nooverlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *rows_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid);
 */