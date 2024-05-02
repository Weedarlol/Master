#include <stdio.h>
#include <math.h>

#include "cuda_functions.h"
#include "jacobi.h"
#include <mpi.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;





void full_calculation_overlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth_node, int iter, int gpus, int rank, int size, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelMid){
    cudaStream_t streams[gpus];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEventsOne(gpus, streams, events, &startevent, &stopevent);
    MPI_Request myRequest[4];
    MPI_Status myStatus[4];
    double *data_cpu;
    cudaErrorHandle(cudaMallocHost(&data_cpu, 4*width*height*sizeof(double)));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
        // Step 1
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            // Computes the upper and lower slice
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiEdge, gridDim, blockDim, kernelCollEdge[g], 0, streams[g]));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g]));
            // Computes the rest of the slices
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiMid, gridDim, blockDim, kernelMid[g], 0, streams[g]));
            cudaErrorHandle(cudaEventRecord(events[g][1], streams[g]));
        }


        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
            cudaErrorHandle(cudaStreamWaitEvent(streams[g], events[g][0]));
        }
        
        if(rank == 0){
            for(int g = 1; g < gpus; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g-1] + (slices_device[g-1]-1)*width*height, g-1, data_gpu_tmp[g] + width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][1], streams[g]));
            }
            // Transfers n-2 slice of the matrix
            for(int g = 0; g < gpus-1; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g+1], g+1, data_gpu_tmp[g] + (slices_device[g]-2)*width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][2], streams[g]));
            }
            
            cudaErrorHandle(cudaSetDevice(gpus-1));
            cudaErrorHandle(cudaMemcpy(data_cpu, data_gpu_tmp[gpus-1] + (slices_device[gpus-1]-2)*width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));
    

            MPI_Isend(&data_cpu[0],              width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[0]);
            MPI_Irecv(&data_cpu[3*width*height], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[1]); 

            MPI_Waitall(2, myRequest, myStatus);

            cudaErrorHandle(cudaMemcpy(data_gpu_tmp[gpus-1] + (slices_device[gpus-1]-1)*width*height, data_cpu +  3*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));
        }
        else if(rank == size-1){
            for(int g = 1; g < gpus; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g-1] + (slices_device[g-1]-1)*width*height, g-1, data_gpu_tmp[g] + width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][1], streams[g]));
            }
            // Transfers n-2 slice of the matrix
            for(int g = 0; g < gpus-1; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g+1], g+1, data_gpu_tmp[g] + (slices_device[g]-2)*width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][2], streams[g]));
            }
            
            cudaErrorHandle(cudaSetDevice(0));
            cudaErrorHandle(cudaMemcpy(data_cpu + width*height, data_gpu_tmp[0] + width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));

            MPI_Isend(&data_cpu[width*height], width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]);
            MPI_Irecv(&data_cpu[2*width*height], width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 

            MPI_Waitall(2, myRequest, myStatus);
            

            cudaErrorHandle(cudaMemcpy(data_gpu_tmp[0], data_cpu +  2*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));
        }
        else{
            for(int g = 1; g < gpus; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g-1] + (slices_device[g-1]-1)*width*height, g-1, data_gpu_tmp[g] + width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][1], streams[g]));
            }
            // Transfers n-2 slice of the matrix
            for(int g = 0; g < gpus-1; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g+1], g+1, data_gpu_tmp[g] + (slices_device[g]-2)*width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][2], streams[g]));
            }

            cudaErrorHandle(cudaSetDevice(gpus-1));
            cudaErrorHandle(cudaMemcpyAsync(data_cpu, data_gpu_tmp[gpus-1] + (slices_device[gpus-1]-2)*width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost, streams[gpus-1]));
            cudaErrorHandle(cudaEventRecord(events[gpus-1][0], streams[gpus-1]));
            cudaErrorHandle(cudaSetDevice(0));
            cudaErrorHandle(cudaMemcpy(data_cpu + width*height, data_gpu_tmp[0] + width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));


            for (int g = 0; g < gpus; g++) {
                cudaErrorHandle(cudaStreamWaitEvent(streams[g], events[gpus-1][0]));
            }

            

            MPI_Irecv(&data_cpu[2*width*height], width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]); 
            MPI_Isend(&data_cpu[width*height],   width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 

            MPI_Isend(&data_cpu[0],              width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[2]);
            MPI_Irecv(&data_cpu[3*width*height], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[3]); 


            for(int g = 0; g < gpus; g++){
                cudaErrorHandle(cudaEventSynchronize(events[g][0]));
                cudaErrorHandle(cudaStreamWaitEvent(streams[g], events[g][0]));
            }
            

            MPI_Waitall(4, myRequest, myStatus);

            cudaErrorHandle(cudaSetDevice(gpus-1));
            cudaErrorHandle(cudaMemcpyAsync(data_gpu_tmp[gpus-1] + (slices_device[gpus-1]-1)*width*height, data_cpu +  3*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice, streams[gpus-1]));
            cudaErrorHandle(cudaEventRecord(events[gpus-1][0], streams[gpus-1]));
            cudaErrorHandle(cudaSetDevice(0));
            cudaErrorHandle(cudaMemcpy(data_gpu_tmp[gpus-1], data_cpu +  2*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));

            for (int g = 0; g < gpus; g++) {
                cudaErrorHandle(cudaStreamWaitEvent(streams[g], events[gpus-1][0]));
            }

        }

        // Step 3
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
            cudaErrorHandle(cudaEventSynchronize(events[g][1]));
            cudaErrorHandle(cudaEventSynchronize(events[g][2]));
        }
        

        for(int g = 0; g < gpus; g++){
            double *data_change = data_gpu[g];
            data_gpu[g] = data_gpu_tmp[g];
            data_gpu_tmp[g] = data_change;
        }


        iter--;
    }

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }

    cudaErrorHandle(cudaEventRecord(stopevent));
    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.5f s\n", milliseconds/1000);

    freeStreamsAndEventsOne(gpus, streams, events, &startevent, &stopevent);
}



void full_calculation_nooverlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth_node, int iter, int gpus, int rank, int size, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelMid){
    cudaStream_t streams[gpus];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEventsOne(gpus, streams, events, &startevent, &stopevent);
    MPI_Request myRequest[4];
    MPI_Status myStatus[4];
    double *data_cpu;
    cudaErrorHandle(cudaMallocHost(&data_cpu, 4*width*height*sizeof(double)));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
        // Step 1
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiMid, gridDim, blockDim, kernelMid[g], 0, streams[g]));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g]));
        }


        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
            cudaErrorHandle(cudaStreamWaitEvent(streams[g], events[g][0]));
        }
        
        if(rank == 0){
            for(int g = 1; g < gpus; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g-1] + (slices_device[g-1]-1)*width*height, g-1, data_gpu_tmp[g] + width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][1], streams[g]));
            }
            // Transfers n-2 slice of the matrix
            for(int g = 0; g < gpus-1; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g+1], g+1, data_gpu_tmp[g] + (slices_device[g]-2)*width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][2], streams[g]));
            }
            
            cudaErrorHandle(cudaSetDevice(gpus-1));
            cudaErrorHandle(cudaMemcpy(data_cpu, data_gpu_tmp[gpus-1] + (slices_device[gpus-1]-2)*width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));
    

            MPI_Isend(&data_cpu[0],              width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[0]);
            MPI_Irecv(&data_cpu[3*width*height], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[1]); 

            MPI_Waitall(2, myRequest, myStatus);

            cudaErrorHandle(cudaMemcpy(data_gpu_tmp[gpus-1] + (slices_device[gpus-1]-1)*width*height, data_cpu +  3*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));
        }
        else if(rank == size-1){
            for(int g = 1; g < gpus; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g-1] + (slices_device[g-1]-1)*width*height, g-1, data_gpu_tmp[g] + width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][1], streams[g]));
            }
            // Transfers n-2 slice of the matrix
            for(int g = 0; g < gpus-1; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g+1], g+1, data_gpu_tmp[g] + (slices_device[g]-2)*width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][2], streams[g]));
            }
            
            cudaErrorHandle(cudaSetDevice(0));
            cudaErrorHandle(cudaMemcpy(data_cpu + width*height, data_gpu_tmp[0] + width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));

            MPI_Isend(&data_cpu[width*height], width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]);
            MPI_Irecv(&data_cpu[2*width*height], width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 

            MPI_Waitall(2, myRequest, myStatus);
            

            cudaErrorHandle(cudaMemcpy(data_gpu_tmp[0], data_cpu +  2*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));
        }
        else{
            for(int g = 1; g < gpus; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g-1] + (slices_device[g-1]-1)*width*height, g-1, data_gpu_tmp[g] + width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][1], streams[g]));
            }
            // Transfers n-2 slice of the matrix
            for(int g = 0; g < gpus-1; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g+1], g+1, data_gpu_tmp[g] + (slices_device[g]-2)*width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][2], streams[g]));
            }

            cudaErrorHandle(cudaSetDevice(gpus-1));
            cudaErrorHandle(cudaMemcpyAsync(data_cpu, data_gpu_tmp[gpus-1] + (slices_device[gpus-1]-2)*width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost, streams[gpus-1]));
            cudaErrorHandle(cudaEventRecord(events[gpus-1][0], streams[gpus-1]));
            cudaErrorHandle(cudaSetDevice(0));
            cudaErrorHandle(cudaMemcpy(data_cpu + width*height, data_gpu_tmp[0] + width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));


            for (int g = 0; g < gpus; g++) {
                cudaErrorHandle(cudaStreamWaitEvent(streams[g], events[gpus-1][0]));
            }

            

            MPI_Irecv(&data_cpu[2*width*height], width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]); 
            MPI_Isend(&data_cpu[width*height],   width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 

            MPI_Isend(&data_cpu[0],              width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[2]);
            MPI_Irecv(&data_cpu[3*width*height], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[3]); 


            for(int g = 0; g < gpus; g++){
                cudaErrorHandle(cudaEventSynchronize(events[g][0]));
                cudaErrorHandle(cudaStreamWaitEvent(streams[g], events[g][0]));
            }
            

            MPI_Waitall(4, myRequest, myStatus);

            cudaErrorHandle(cudaSetDevice(gpus-1));
            cudaErrorHandle(cudaMemcpyAsync(data_gpu_tmp[gpus-1] + (slices_device[gpus-1]-1)*width*height, data_cpu +  3*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice, streams[gpus-1]));
            cudaErrorHandle(cudaEventRecord(events[gpus-1][0], streams[gpus-1]));
            cudaErrorHandle(cudaSetDevice(0));
            cudaErrorHandle(cudaMemcpy(data_gpu_tmp[gpus-1], data_cpu +  2*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));

            for (int g = 0; g < gpus; g++) {
                cudaErrorHandle(cudaStreamWaitEvent(streams[g], events[gpus-1][0]));
            }

        }

        // Step 3
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
            cudaErrorHandle(cudaEventSynchronize(events[g][1]));
            cudaErrorHandle(cudaEventSynchronize(events[g][2]));
        }
        

        for(int g = 0; g < gpus; g++){
            double *data_change = data_gpu[g];
            data_gpu[g] = data_gpu_tmp[g];
            data_gpu_tmp[g] = data_change;
            
            /* void *temp_mid = kernelMid[g][0];
            kernelMid[g][0] = kernelMid[g][1];
            kernelMid[g][1] = temp_mid; */
        }


        iter--;
    }

    
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(stopevent));

    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.5f s\n", milliseconds/1000);

    freeStreamsAndEventsOne(gpus, streams, events, &startevent, &stopevent);
}

void only_computation_nooverlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth_node, int iter, int gpus, int rank, int size, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelMid){
    cudaStream_t streams[gpus];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEventsOne(gpus, streams, events, &startevent, &stopevent);
    MPI_Request myRequest[4];
    MPI_Status myStatus[4];
    double *data_cpu;
    cudaErrorHandle(cudaMallocHost(&data_cpu, 4*width*height*sizeof(double)));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
        // Step 1
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiMid, gridDim, blockDim, kernelMid[g], 0, streams[g]));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g]));
        }


        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
            cudaErrorHandle(cudaStreamWaitEvent(streams[g], events[g][0]));
        }
        

        for(int g = 0; g < gpus; g++){
            double *data_change = data_gpu[g];
            data_gpu[g] = data_gpu_tmp[g];
            data_gpu_tmp[g] = data_change;
        }


        iter--;
    }

    
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(stopevent));

    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.5f s\n", milliseconds/1000);

    freeStreamsAndEventsOne(gpus, streams, events, &startevent, &stopevent);
}

void only_communication_nooverlap(double **data_gpu, double **data_gpu_tmp, int width, int height, int depth_node, int iter, int gpus, int rank, int size, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelMid){
    cudaStream_t streams[gpus];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEventsOne(gpus, streams, events, &startevent, &stopevent);
    MPI_Request myRequest[4];
    MPI_Status myStatus[4];
    double *data_cpu;
    cudaErrorHandle(cudaMallocHost(&data_cpu, 4*width*height*sizeof(double)));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
        if(rank == 0){
            for(int g = 1; g < gpus; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g-1] + (slices_device[g-1]-1)*width*height, g-1, data_gpu_tmp[g] + width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][1], streams[g]));
            }
            // Transfers n-2 slice of the matrix
            for(int g = 0; g < gpus-1; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g+1], g+1, data_gpu_tmp[g] + (slices_device[g]-2)*width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][2], streams[g]));
            }
            
            cudaErrorHandle(cudaSetDevice(gpus-1));
            cudaErrorHandle(cudaMemcpy(data_cpu, data_gpu_tmp[gpus-1] + (slices_device[gpus-1]-2)*width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));
    

            MPI_Isend(&data_cpu[0],              width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[0]);
            MPI_Irecv(&data_cpu[3*width*height], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[1]); 

            MPI_Waitall(2, myRequest, myStatus);

            cudaErrorHandle(cudaMemcpy(data_gpu_tmp[gpus-1] + (slices_device[gpus-1]-1)*width*height, data_cpu +  3*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));
        }
        else if(rank == size-1){
            for(int g = 1; g < gpus; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g-1] + (slices_device[g-1]-1)*width*height, g-1, data_gpu_tmp[g] + width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][1], streams[g]));
            }
            // Transfers n-2 slice of the matrix
            for(int g = 0; g < gpus-1; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g+1], g+1, data_gpu_tmp[g] + (slices_device[g]-2)*width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][2], streams[g]));
            }
            
            cudaErrorHandle(cudaSetDevice(0));
            cudaErrorHandle(cudaMemcpy(data_cpu + width*height, data_gpu_tmp[0] + width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));

            MPI_Isend(&data_cpu[width*height], width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]);
            MPI_Irecv(&data_cpu[2*width*height], width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 

            MPI_Waitall(2, myRequest, myStatus);
            

            cudaErrorHandle(cudaMemcpy(data_gpu_tmp[0], data_cpu +  2*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));
        }
        else{
            for(int g = 1; g < gpus; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g-1] + (slices_device[g-1]-1)*width*height, g-1, data_gpu_tmp[g] + width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][1], streams[g]));
            }
            // Transfers n-2 slice of the matrix
            for(int g = 0; g < gpus-1; g++){
                cudaErrorHandle(cudaSetDevice(g));
                cudaErrorHandle(cudaMemcpyPeerAsync(data_gpu_tmp[g+1], g+1, data_gpu_tmp[g] + (slices_device[g]-2)*width*height, g, width*height*sizeof(double), streams[g]));
                cudaErrorHandle(cudaEventRecord(events[g][2], streams[g]));
            }

            cudaErrorHandle(cudaMemcpyAsync(data_cpu, data_gpu_tmp[gpus-1] + (slices_device[gpus-1]-2)*width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost, streams[gpus-1]));
            cudaErrorHandle(cudaEventRecord(events[gpus-1][0], streams[gpus-1]));
            cudaErrorHandle(cudaMemcpy(data_cpu + width*height, data_gpu_tmp[0] + width*height, width*height*sizeof(double), cudaMemcpyDeviceToHost));


            for (int g = 0; g < gpus; g++) {
                cudaErrorHandle(cudaStreamWaitEvent(streams[g], events[gpus-1][0]));
            }

            

            MPI_Irecv(&data_cpu[2*width*height], width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[0]); 
            MPI_Isend(&data_cpu[width*height],   width*height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &myRequest[1]); 

            MPI_Isend(&data_cpu[0],              width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[2]);
            MPI_Irecv(&data_cpu[3*width*height], width*height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &myRequest[3]); 


            for(int g = 0; g < gpus; g++){
                cudaErrorHandle(cudaEventSynchronize(events[g][0]));
                cudaErrorHandle(cudaStreamWaitEvent(streams[g], events[g][0]));
            }
            

            MPI_Waitall(4, myRequest, myStatus);

            cudaErrorHandle(cudaSetDevice(gpus-1));
            cudaErrorHandle(cudaMemcpyAsync(data_gpu_tmp[gpus-1] + (slices_device[gpus-1]-1)*width*height, data_cpu +  3*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice, streams[gpus-1]));
            cudaErrorHandle(cudaEventRecord(events[gpus-1][0], streams[gpus-1]));
            cudaErrorHandle(cudaSetDevice(0));
            cudaErrorHandle(cudaMemcpy(data_gpu_tmp[gpus-1], data_cpu +  2*width*height, width*height*sizeof(double), cudaMemcpyHostToDevice));

            for (int g = 0; g < gpus; g++) {
                cudaErrorHandle(cudaStreamWaitEvent(streams[g], events[gpus-1][0]));
            }

        }

        // Step 3
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][1]));
            cudaErrorHandle(cudaEventSynchronize(events[g][2]));
        }
        

        for(int g = 0; g < gpus; g++){
            double *data_change = data_gpu[g];
            data_gpu[g] = data_gpu_tmp[g];
            data_gpu_tmp[g] = data_change;
        }


        iter--;
    }

    
    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(stopevent));

    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.5f s\n", milliseconds/1000);

    freeStreamsAndEventsOne(gpus, streams, events, &startevent, &stopevent);
}