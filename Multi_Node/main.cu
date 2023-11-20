#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mpi.h>

void mpi_gpu_communication(int my_rank, int gpu_count) {
    int cuda_device;
    cudaGetDevice(&cuda_device);
    float data = 0.0f;

    if (my_rank == 0) {
        // On the first node (rank 0), initialize data
        data = 42.0f;
    }

    // Use MPI to send data from rank 0 to rank 1 (different nodes)
    if (my_rank == 0) {
        MPI_Send(&data, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else if (my_rank == 1) {
        MPI_Recv(&data, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // GPU-to-GPU communication between different nodes (node 0 to node 1)
    if (my_rank == 0) {
        cudaSetDevice(cuda_device); // Select the GPU
        cudaDeviceEnablePeerAccess(1, 0); // Enable GPU Direct for GPU 0 on node 0
        cudaMemcpyPeer(data, 1, &data, 0, sizeof(float)); // Copy data to GPU 1 on node 1
    } else if (my_rank == 1) {
        cudaSetDevice(cuda_device); // Select the GPU
        cudaDeviceEnablePeerAccess(0, 0); // Enable GPU Direct for GPU 0 on node 1
        cudaMemcpyPeer(&data, 0, &data, 1, sizeof(float)); // Copy data to GPU 0 on node 0
    }

    // Clean up and disable GPU Direct if needed
    if (my_rank == 0) {
        cudaDeviceDisablePeerAccess(1); // Disable GPU Direct for GPU 0 on node 0
    } else if (my_rank == 1) {
        cudaDeviceDisablePeerAccess(0); // Disable GPU Direct for GPU 0 on node 1
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int gpu_count;
    cudaGetDeviceCount(&gpu_count);

    if (size != 2) {
        printf("This example requires exactly 2 MPI processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (gpu_count < 2) {
        printf("This example requires at least 2 GPUs.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank < 2) {
        cudaSetDevice(rank % gpu_count);
        cudaDeviceEnablePeerAccess(0, 0); // Enable GPU Direct for GPU 0 on each node
    }

    mpi_gpu_communication(rank, gpu_count);

    if (rank < 2) {
        cudaDeviceDisablePeerAccess(0); // Disable GPU Direct for GPU 0 on each node
    }

    MPI_Finalize();
    return 0;
}