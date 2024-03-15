#!/bin/bash

#SBATCH --job-name=width256_height256_depth128_iter10000_compare1_accel_GPU
#SBATCH --account=ec12
#SBATCH -p accel
#SBATCH --mem-per-cpu=4G
#SBATCH -N 1                                  # Antall noder, i dgx2q så er det bare 1 node bygd opp av DualProcessor AMD EPYC Milan 7763 64-core w/ 8 qty Nvidia Volta A100/80GB
#SBATCH -n 1                                  # Antall CPU cores som vil bli allokert
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1                   # Antall ganger en vil kjøre programmet på hver node
#SBATCH --cpus-per-task=1                     # Antall CPUer en vil allokere, er 64 cores per CPU, så kan i teorien bare trenge å øke dene når -n > 64
#SBATCH --gres=gpu:                # Antall GPUer en allokerer per job, så totale antall GPU
#SBATCH --gpus-per-task=           # Antall GPUer en allokerer per task, så totale antall GPU delt på antall noder
#SBATCH --gpus=                    # Antall GPUer en allokerer per node, så totale antall GPU delt på antall noder
#SBATCH --output=output/width256_height256_depth128_iter10000_compare1_accel.out
#SBATCH --error=error/width256_height256_depth128_iter10000_compare1_accel.err

module purge
module load CUDA/11.7.0
module load OpenMPI/4.1.4-GCC-11.3.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0

nvcc -rdc=true main.cu programs/jacobi.cu ../../functions//global_functions_fox.o ../../functions//cuda_functions_fox.o -o out_width256_height256_depth128_iter10000_compare1_accel -L/../../usr/lib/x86_64-linux-gnu/libnccl.so -lnccl

./out_width256_height256_depth128_iter10000_compare1_accel 256 256 128 10000 1

exit 0

