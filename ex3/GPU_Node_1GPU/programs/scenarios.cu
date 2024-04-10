#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include "cuda_functions.h"
#include "jacobi.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

void full_calculation(double *data_gpu, double *data_gpu_tmp, int width, int height, int depth_node, int iter, int rank, int size, dim3 gridDim, dim3 blockDim, void** kernelCollMid){
    cudaStream_t streams[2];
    cudaEvent_t events, startevent, stopevent;
    cudaErrorHandle(cudaStreamCreate(&streams[0]));
    cudaErrorHandle(cudaStreamCreate(&streams[1]));
    cudaErrorHandle(cudaEventCreate(&events));
    cudaErrorHandle(cudaEventCreate(&startevent));
    cudaErrorHandle(cudaEventCreate(&stopevent));
    MPI_Request myRequest[4];
    MPI_Status myStatus[4];
    double *data_cpu;
    cudaErrorHandle(cudaMallocHost(&data_cpu, 4*width*height*sizeof(double)));

    cudaErrorHandle(cudaDeviceSynchronize());
    cudaErrorHandle(cudaEventRecord(startevent));
    

    while(iter > 0){
        // Runs GPU Kernel
        cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobi, gridDim, blockDim, kernelCollMid, 0, streams[0]));
        cudaErrorHandle(cudaEventRecord(events, streams[0]));
        cudaErrorHandle(cudaEventSynchronize(events));

        // [...a] = second last row of GPU
        // [b...] = second first row of GPU
        /* data_cpu(rank) = [_, _, _, _], GPU transfer first two arrays to CPU
        rank0 
            [[...a], _     , _, _]
        rank-1
            [_     , [b...], _, _]
        rank between
            [[...a], [b...], _, _]
         */
        if(rank == 0){
            cudaErrorHandle(cudaMemcpy(data_cpu, data_gpu_tmp + (depth_node-2)*width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));
        }
        else if(rank == size-1){
            cudaErrorHandle(cudaMemcpy(data_cpu + width*height, data_gpu_tmp + width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));
        }
        else{
            cudaErrorHandle(cudaMemcpy(data_cpu, data_gpu_tmp + (depth_node-2)*width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));
            cudaErrorHandle(cudaMemcpy(data_cpu + width*height, data_gpu_tmp + width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));
        }

        /* data_cpu(rank)           -> data_cpu(rank-1)  + data_cpu(rank+1), CPU transfer to CPU
        rank0 
            [[...a], _, _, _]       ->                     [_, _, [...a], _]
        rank-1
            [_, [b...], _, _]       -> [_, _, _, [b...]]
        rank between
            [[...a], [b...], _, _]  -> [_, _, _, [b...]] + [_, _, [...a], _]
        */
        if(rank == 0){
            MPI_Isend(&data_cpu[0],              width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[0]);
            MPI_Irecv(&data_cpu[3*width*height], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[1]); 
        }
        else if(rank == size-1){
            MPI_Irecv(&data_cpu[2*width*height], width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]); 
            MPI_Isend(&data_cpu[width*height],   width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 
        }
        else{
            MPI_Irecv(&data_cpu[2*width*height], width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]); 
            MPI_Isend(&data_cpu[width*height],   width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 

            MPI_Isend(&data_cpu[0],              width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[2]);
            MPI_Irecv(&data_cpu[3*width*height], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[3]); 
        }

        MPI_Waitall(rank == 0 || rank == size - 1 ? 2 : 4, myRequest, myStatus);



        /* data_cpu(rank) = [_, _, a, b], CPU transfer last two arrays to GPU
        rank0 
            [_, _, [...a], _     ]
        rank-1
            [_, _, _     , [b...]]
        rank between
            [_, _, [...a], [b...]]
         */
        if(rank == 0){
            cudaErrorHandle(cudaMemcpy(data_gpu_tmp + (depth_node-1)*width*height, data_cpu + 3*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));
        }
        else if(rank == size-1){
            cudaErrorHandle(cudaMemcpy(data_gpu_tmp,                               data_cpu + 2*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));
        }
        else{
            cudaErrorHandle(cudaMemcpy(data_gpu_tmp + (depth_node-1)*width*height, data_cpu + 3*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));
            cudaErrorHandle(cudaMemcpy(data_gpu_tmp,                               data_cpu + 2*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));
        }

        double *data_change = data_gpu;
        data_gpu = data_gpu_tmp;
        data_gpu_tmp = data_change;
        
        void *temp = kernelCollMid[0];
        kernelCollMid[0] = kernelCollMid[1];
        kernelCollMid[1] = temp;

        iter--;
    }


    cudaErrorHandle(cudaDeviceSynchronize());

    cudaErrorHandle(cudaEventRecord(stopevent));

    cudaErrorHandle(cudaEventSynchronize(stopevent));

    cudaErrorHandle(cudaDeviceSynchronize());
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.5f s\n", milliseconds/1000);

    //freeStreamsAndEvents(1, streams, events, &startevent, &stopevent);
}