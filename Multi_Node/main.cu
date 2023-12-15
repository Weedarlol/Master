#include <stdio.h>
#include <mpi.h>

#define ARRAY_SIZE 10

__global__ void kernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < ARRAY_SIZE) {
        data[idx] *= 2; // Perform some computation (e.g., doubling the values)
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int data[ARRAY_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        printf("This program requires exactly 2 MPI processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        // Initialize data on Node 0
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            data[i] = i;
        }

        // Send data to Node 1
        MPI_Send(data, ARRAY_SIZE, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        // Receive data from Node 0
        MPI_Recv(data, ARRAY_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Perform GPU computation on Node 1
        int *dev_data;
        cudaMalloc(&dev_data, ARRAY_SIZE * sizeof(int));
        cudaMemcpy(dev_data, data, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (ARRAY_SIZE + threadsPerBlock - 1) / threadsPerBlock;
        kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_data);

        cudaMemcpy(data, dev_data, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(dev_data);

        // Print the processed data received from Node 0
        printf("Node 1 received processed data:\n");
        for (int i = 0; i < ARRAY_SIZE; ++i) {
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
