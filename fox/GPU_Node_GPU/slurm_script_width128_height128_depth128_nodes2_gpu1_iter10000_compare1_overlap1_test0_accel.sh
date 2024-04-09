#!/bin/bash

#SBATCH --job-name=width128_height128_depth128_nodes2_gpu1_iter10000_compare1_overlap1_test0_accel
#SBATCH --account=ec12
#SBATCH --mem-per-gpu=4G
#SBATCH -p accel
#SBATCH -N 2                                   # Antall Noder
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1                   # Antall ganger en vil kjøre programmet på hver node
#SBATCH --cpus-per-task=1                     # Antall CPUer en vil allokere, er 64 cores per CPU, så kan i teorien bare trenge å øke dene når -n > 64
#SBATCH --gres=gpu:2                            # Antall GPUer en allokerer per job, så totale antall GPU. NUM_GPUS*NODES
#SBATCH --gpus-per-task=1                           # Antall GPUer en allokerer per task, så totale antall GPU delt på antall noder. NUM_GPUS
#SBATCH --gpus=1                                    # Antall GPUer en allokerer per node. NUM_GPUS
#SBATCH --output=output/width128_height128_depth128_nodes2_gpu1_iter10000_compare1_overlap1_test0_accel.out
#SBATCH --error=error/width128_height128_depth128_nodes2_gpu1_iter10000_compare1_overlap1_test0_accel.err

module purge
module load CUDA/11.7.0
module load OpenMPI/4.1.4-GCC-11.3.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0



nvcc main.cu programs/jacobi.cu programs/cuda_functions.cu -o out_width128_height128_depth128_nodes2_gpu1_iter10000_compare1_overlap1_test0_accel I -L -lmpi -lnccl -O3

mpirun -n 2 out_width128_height128_depth128_nodes2_gpu1_iter10000_compare1_overlap1_test0_accel 128 128 128 10000 1 2 1 1 0




exit 0

