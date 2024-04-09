#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include "cuda_functions.h"
#include "jacobi.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

void full_calculation(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth_node, int iter, int *slices_device, int rank, dim3 gridDim, dim3 blockDim, void** kernelCollMid){
    cudaStream_t streams[2];
    cudaEvent_t events, startevent, stopevent;
    cudaErrorHandle(cudaStreamCreate(&streams[0]));
    cudaErrorHandle(cudaStreamCreate(&streams[1]));
    cudaErrorHandle(cudaEventCreate(&events));
    cudaErrorHandle(cudaEventCreate(&startevent));
    cudaErrorHandle(cudaEventCreate(&stopevent));
    MPI_Request myRequest[2];
    MPI_Status myStatus[2];

    cudaErrorHandle(cudaDeviceSynchronize());
    cudaErrorHandle(cudaEventRecord(startevent));
    

    while(iter > 0){
        cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobi, gridDim, blockDim, kernelCollMid, 0, streams[0]));
        cudaErrorHandle(cudaEventRecord(events, streams[0]));
        
        cudaErrorHandle(cudaEventSynchronize(events));

        /* if(rank == 0){
            MPI_Isend(&data_gpu_tmp[width*height*(depth_node-2)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[0]);
            MPI_Irecv(&data_gpu_tmp[width*height*(depth_node-1)], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[1]); 

            MPI_Waitall(2, myRequest, myStatus);
        }
        else if(rank == size-1){
            MPI_Irecv(&data_gpu_tmp[0],                           width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[rank*2-2]); 
            MPI_Isend(&data_gpu_tmp[width*height],                width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[rank*2-1]); 

            MPI_Waitall(2, &myRequest[rank*2-2], myStatus);
        }
        MPI_Barrier(MPI_COMM_WORLD); */
            
        double *data_change = data_gpu[0];
        data_gpu[0] = data_gpu_tmp[0];
        data_gpu_tmp[0] = data_change;
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