__global__ void jacobiMid(double *data_gpu, double *data_gpu_tmp, int width, int height, int threadSize, int elementsPerThread, int leftover, int overlap_calc);
__global__ void jacobiEdge(double *data_gpu, double *data_gpu_tmp, int width, int height, int depth_node, int threadSize, int elementsPerThreadOverlap, int leftoverOverlap);
