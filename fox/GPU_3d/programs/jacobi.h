__global__ void jacobiEdge(double *data_gpu, double *data_gpu_tmp, int width, int height, 
                        int slices_compute, int elementsPerThread, int leftover);

__global__ void jacobiMid(double *data_gpu, double *data_gpu_tmp, int width, int height,
                        int slices_Leftover, int device_nr, int slices_compute, int elementsPerThreadExtra, int elementsLeftoverExtra,
                        int elementsPerThread, int elementsLeftover, int overlap_calc);