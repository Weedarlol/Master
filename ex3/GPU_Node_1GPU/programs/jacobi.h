__global__ void jacobi(double *data_gpu, double *data_gpu_tmp, int width, int height, int depth_node, int iter, int threadSize, int jacobiSize, int elementsPerThread, int leftover);
