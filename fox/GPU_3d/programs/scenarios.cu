#include <stdio.h>
#include <math.h>


#include "../../global_functions.h"
#include "jacobi.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;




/* 
void full_calculation_overlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid){
    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

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
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiEdge, gridDim, blockDim, kernelCollEdge[g], 0, streams[g][0]));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][0]));
            // Computes the rest of the slices
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiMid, gridDim, blockDim, kernelCollMid[g], 0, streams[g][1]));
            cudaErrorHandle(cudaEventRecord(events[g][1], streams[g][1]));
        }


        // Step 2
        // Transfer 2 slice of the matrix
        for(int g = 1; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaStreamWaitEvent(streams[g][0], events[g][0]));
            cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + (slices_device[g-1]-1)*width + 1, g-1, mat_gpu_tmp[g] + width + 1, g, (width-2)*(height-2)*sizeof(double), streams[g][0]));
            cudaErrorHandle(cudaEventRecord(events[g][2], streams[g][0]));
        }
        // Transfers n-2 slice of the matrix
        for(int g = 0; g < gpus-1; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaStreamWaitEvent(streams[g][0], events[g][0]));
            cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g+1] + 1, g+1, mat_gpu_tmp[g] + (slices_device[g]-2)*width + 1, g, (width-2)*(height-2)*sizeof(double), streams[g][0]));
            cudaErrorHandle(cudaEventRecord(events[g][3], streams[g][0]));
        }


        // Step 3
        for (int g = 0; g < gpus; g++) {
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][1]));
            cudaErrorHandle(cudaEventSynchronize(events[g][2]));
            cudaErrorHandle(cudaEventSynchronize(events[g][3]));
        }
        
        // Step 4
        for(int g = 0; g < gpus; g++){
            double *mat_change = mat_gpu[g];
            mat_gpu[g] = mat_gpu_tmp[g];
            mat_gpu_tmp[g] = mat_change;
        }
        iter--;
    }

    cudaErrorHandle(cudaEventRecord(stopevent));
    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", iter - iter);

    freeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);
}
 */


void full_calculation_nooverlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid){
    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
        // Step 1
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiMid, gridDim, blockDim, kernelCollMid[g], 0, streams[g][0]));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][0]));
        }

        // Step 2
        // Transfers 2 slice of the matrix
        for(int g = 1; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaStreamWaitEvent(streams[g][1], events[g][0]));
            cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + (slices_device[g-1]-1)*width + 1, g-1, mat_gpu_tmp[g] + width + 1, g, (width-2)*sizeof(double), streams[g][1]));
            cudaErrorHandle(cudaEventRecord(events[g][1], streams[g][1]));
        }
        // Transfers n-2 slice of the matrix
        for(int g = 0; g < gpus-1; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaStreamWaitEvent(streams[g][1], events[g][0]));
            cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g+1] + 1, g+1, mat_gpu_tmp[g] + (slices_device[g]-2)*width + 1, g, (width-2)*sizeof(double), streams[g][1]));
            cudaErrorHandle(cudaEventRecord(events[g][2], streams[g][1]));
        }

        // Step 3
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
            cudaErrorHandle(cudaEventSynchronize(events[g][1]));
            cudaErrorHandle(cudaEventSynchronize(events[g][2]));
        }
        
        // Step 4
        for(int g = 0; g < gpus; g++){
            double *mat_change = mat_gpu[g];
            mat_gpu[g] = mat_gpu_tmp[g];
            mat_gpu_tmp[g] = mat_change;
        }
        iter--;
    }
    cudaErrorHandle(cudaEventRecord(stopevent));

    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", iter - iter);

    freeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);
}


/* 
void no_kernel_overlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid){
    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
        // Transfer 2 slice of the matrix
        for(int g = 1; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + (slices_device[g-1]-1)*width + 1, g-1, mat_gpu_tmp[g] + width + 1, g, (width-2)*(height-2)*sizeof(double), streams[g][0]));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][0]));
        }
        // Transfers n-2 slice of the matrix
        for(int g = 0; g < gpus-1; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g+1] + 1, g+1, mat_gpu_tmp[g] + (slices_device[g]-2)*width + 1, g, (width-2)*(height-2)*sizeof(double), streams[g][1]));
            cudaErrorHandle(cudaEventRecord(events[g][1], streams[g][1]));
        }

        // Step 3
        for (int g = 0; g < gpus; g++) {
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
            cudaErrorHandle(cudaEventSynchronize(events[g][1]));
        }
        
        // Step 4
        for(int g = 0; g < gpus; g++){
            double *mat_change = mat_gpu[g];
            mat_gpu[g] = mat_gpu_tmp[g];
            mat_gpu_tmp[g] = mat_change;
        }
        iter--;
    }
    cudaErrorHandle(cudaEventRecord(stopevent));
    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", iter - iter);

    freeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);
}



void no_kernel_nooverlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid){
    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
        // Step 2
        // Transfers 2 slice of the matrix
        for(int g = 1; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + (slices_device[g-1]-1)*width + 1, g-1, mat_gpu_tmp[g] + width + 1, g, (width-2)*(height-2)*sizeof(double), streams[g][0]));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][0]));
        }
        // Transfers n-2 slice of the matrix
        for(int g = 0; g < gpus-1; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g+1] + 1, g+1, mat_gpu_tmp[g] + (slices_device[g]-2)*width + 1, g, (width-2)*(height-2)*sizeof(double), streams[g][1]));
            cudaErrorHandle(cudaEventRecord(events[g][1], streams[g][1]));
        }

        // Step 3
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
            cudaErrorHandle(cudaEventSynchronize(events[g][1]));
        }
        
        // Step 4
        for(int g = 0; g < gpus; g++){
            double *mat_change = mat_gpu[g];
            mat_gpu[g] = mat_gpu_tmp[g];
            mat_gpu_tmp[g] = mat_change;
        }
        iter--;
    }
    cudaErrorHandle(cudaEventRecord(stopevent));

    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", iter - iter);

    freeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);
}



void no_communication_overlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid){
    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
        // Step 1
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiEdge, gridDim, blockDim, kernelCollEdge[g], 0, streams[g][0]));
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiMid, gridDim, blockDim, kernelCollMid[g], 0, streams[g][0]));
            cudaErrorHandle(cudaEventRecord(events[g][1], streams[g][0]));
        }

        for (int g = 0; g < gpus; g++) {
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][1]));
        }
        
        for(int g = 0; g < gpus; g++){
            double *mat_change = mat_gpu[g];
            mat_gpu[g] = mat_gpu_tmp[g];
            mat_gpu_tmp[g] = mat_change;
        }
        iter--;
    }
    cudaErrorHandle(cudaEventRecord(stopevent));
    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", iter - iter);

    freeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);
}



void no_communication_nooverlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid){
    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
        // Step 1
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiMid, gridDim, blockDim, kernelCollMid[g], 0, streams[g][0]));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][0]));
        }

        // Step 3
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
        }
        
        // Step 4
        for(int g = 0; g < gpus; g++){
            double *mat_change = mat_gpu[g];
            mat_gpu[g] = mat_gpu_tmp[g];
            mat_gpu_tmp[g] = mat_change;
        }
        iter--;
    }
    cudaErrorHandle(cudaEventRecord(stopevent));

    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", iter - iter);

    freeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);
}



void only_events(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid){

    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
         for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][0]));
        }

        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
        }

        for(int g = 0; g < gpus; g++){
            double *mat_change = mat_gpu[g];
            mat_gpu[g] = mat_gpu_tmp[g];
            mat_gpu_tmp[g] = mat_change;
        }
        iter--;
    }
    cudaErrorHandle(cudaEventRecord(stopevent));

    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", iter - iter);

    freeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);
}



void only_calculation_overlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid){
    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

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
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiEdge, gridDim, blockDim, kernelCollEdge[g], 0, streams[g][0]));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][0]));
            // Computes the rest of the slices
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiMid, gridDim, blockDim, kernelCollMid[g], 0, streams[g][1]));
            cudaErrorHandle(cudaEventRecord(events[g][1], streams[g][1]));
        }

        // Step 3
        for (int g = 0; g < gpus; g++) {
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
            cudaErrorHandle(cudaEventSynchronize(events[g][1]));
        }
        
        iter--;
    }
    

    cudaErrorHandle(cudaEventRecord(stopevent));
    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", iter - iter);

    freeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);
}



void only_calculation_nooverlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid){
    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
        // Step 1
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaLaunchCooperativeKernel((void*)jacobiMid, gridDim, blockDim, kernelCollMid[g], 0, streams[g][0]));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][0]));
        }

        // Step 3
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
        }

        iter--;
    }
    cudaErrorHandle(cudaEventRecord(stopevent));

    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", iter - iter);

    freeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);
}



void only_communication_overlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollEdge, void*** kernelCollMid){
    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
        // Transfer 2 slice of the matrix
        for(int g = 1; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + (slices_device[g-1]-1)*width + 1, g-1, mat_gpu_tmp[g] + width + 1, g, (width-2)*(height-2)*sizeof(double), streams[g][0]));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][0]));
        }
        // Transfers n-2 slice of the matrix
        for(int g = 0; g < gpus-1; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g+1] + 1, g+1, mat_gpu_tmp[g] + (slices_device[g]-2)*width + 1, g, (width-2)*(height-2)*sizeof(double), streams[g][1]));
            cudaErrorHandle(cudaEventRecord(events[g][1], streams[g][1]));
        }

        // Step 3
        for (int g = 0; g < gpus; g++) {
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
            cudaErrorHandle(cudaEventSynchronize(events[g][1]));
        }
        iter--;
    }
    cudaErrorHandle(cudaEventRecord(stopevent));
    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", iter - iter);

    freeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);
}



void only_communication_nooverlap(double **mat_gpu, double **mat_gpu_tmp, int width, int height, int depth, int iter, int gpus, int *slices_device, dim3 gridDim, dim3 blockDim, void*** kernelCollMid){
    cudaStream_t streams[gpus][2];
    cudaEvent_t events[gpus][4], startevent, stopevent;
    initializeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    cudaErrorHandle(cudaEventRecord(startevent));

    while(iter > 0){
        // Step 2
        // Transfers 2 slice of the matrix
        for(int g = 1; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g-1] + (slices_device[g-1]-1)*width + 1, g-1, mat_gpu_tmp[g] + width + 1, g, (width-2)*(height-2)*sizeof(double), streams[g][0]));
            cudaErrorHandle(cudaEventRecord(events[g][0], streams[g][0]));
        }
        // Transfers n-2 slice of the matrix
        for(int g = 0; g < gpus-1; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaMemcpyPeerAsync(mat_gpu_tmp[g+1] + 1, g+1, mat_gpu_tmp[g] + (slices_device[g]-2)*width + 1, g, (width-2)*(height-2)*sizeof(double), streams[g][1]));
            cudaErrorHandle(cudaEventRecord(events[g][1], streams[g][1]));
        }

        // Step 3
        for(int g = 0; g < gpus; g++){
            cudaErrorHandle(cudaSetDevice(g));
            cudaErrorHandle(cudaEventSynchronize(events[g][0]));
            cudaErrorHandle(cudaEventSynchronize(events[g][1]));
        }
        iter--;
    }
    cudaErrorHandle(cudaEventRecord(stopevent));

    cudaErrorHandle(cudaEventSynchronize(stopevent));

    for(int g = 0; g < gpus; g++){
        cudaErrorHandle(cudaSetDevice(g));
        cudaErrorHandle(cudaDeviceSynchronize());
    }
    
    float milliseconds = 0.0f;
    cudaErrorHandle(cudaEventElapsedTime(&milliseconds, startevent, stopevent));
    printf("Time(event) - %.4f, SolutionFound - %s, IterationsComputed - %i\n",
            milliseconds, (iter == 0) ? "No" : "Yes", iter - iter);

    freeStreamsAndEvents(gpus, streams, events, &startevent, &stopevent);
}

 */