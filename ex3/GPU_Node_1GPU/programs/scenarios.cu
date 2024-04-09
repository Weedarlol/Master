#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include "cuda_functions.h"
#include "jacobi.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

void full_calculation(double *data_gpu, double *data_gpu_tmp, int width, int height, int depth_node, int iter, int slices_device, int rank, dim3 gridDim, dim3 blockDim, void** kernelCollMid){
    cudaStream_t streams[2];
    cudaEvent_t events, startevent, stopevent;
    cudaErrorHandle(cudaStreamCreate(&streams[0]));
    cudaErrorHandle(cudaStreamCreate(&streams[1]));
    cudaErrorHandle(cudaEventCreate(&events));
    cudaErrorHandle(cudaEventCreate(&startevent));
    cudaErrorHandle(cudaEventCreate(&stopevent));
    MPI_Request myRequest[2];
    MPI_Status myStatus[2];
    double *data_cpu;
    cudaErrorHandle(cudaMallocHost(&data_cpu, 2*width*height*sizeof(double)));

    cudaErrorHandle(cudaDeviceSynchronize());
    cudaErrorHandle(cudaEventRecord(startevent));
    

    while(iter > 0){
        // Runs GPU Kernel
        cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobi, gridDim, blockDim, kernelCollMid, 0, streams[0]));
        cudaErrorHandle(cudaEventRecord(events, streams[0]));
        cudaErrorHandle(cudaEventSynchronize(events));

        // Copies data from GPU to CPU
        if(rank == 0){
            cudaErrorHandle(cudaMemcpy(data_cpu, data_gpu_tmp + (depth_node-2)*width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));
        }
        else if(rank == 1){
            cudaErrorHandle(cudaMemcpy(data_cpu, data_gpu_tmp + width*height,                width*height*sizeof(double), cudaMemcpyDeviceToHost));
        }

        // Sends data from node 0 CPU to node 1 CPU
        if(rank == 0){
            MPI_Isend(&data_cpu[0],            width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[0]);
            MPI_Irecv(&data_cpu[width*height], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[1]); 
        }
        else if(rank == 1){
            MPI_Irecv(&data_cpu[width*height], width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]); 
            MPI_Isend(&data_cpu[0],            width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 
        }
        MPI_Waitall(2, myRequest, myStatus);

        // Copies data from CPU to GPU
        if(rank == 0){
            cudaErrorHandle(cudaMemcpy(data_gpu_tmp + (depth_node-1)*width*height, data_cpu + width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));
        }
        else if(rank == 1){
            cudaErrorHandle(cudaMemcpy(data_gpu_tmp,                               data_cpu + width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));
        }

        double *data_change = data_gpu;
        data_gpu = data_gpu_tmp;
        data_gpu_tmp = data_change;
        
        void *temp = kernelCollMid[0];
        kernelCollMid[0] = kernelCollMid[1];
        kernelCollMid[1] = temp;

        iter--;
    }

    printf("Finished full computation loop\n");

    cudaErrorHandle(cudaDeviceSynchronize());

    cudaErrorHandle(cudaEventRecord(stopevent));

    cudaErrorHandle(cudaEventSynchronize(stopevent));

    cudaErrorHandle(cudaDeviceSynchronize());
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.5f s\n", milliseconds/1000);

    //freeStreamsAndEvents(1, streams, events, &startevent, &stopevent);
}