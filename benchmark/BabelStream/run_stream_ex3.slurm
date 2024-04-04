#!/bin/bash

#SBATCH --job-name=stream_benchmark
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=Build-x86_64/dgx2q.out
#SBATCH --error=error/dgx2q.err


module purge
module load cuda12.3/toolkit/12.3.2
module load cuda12.3/blas/12.3.2
module load cuda12.3/fft/12.3.2
module load cuda12.3/profiler/12.3.2
module load cuda12.3/nsight/12.3.2
module load cmake/gcc/3.27.9

cmake -B Build-x86_64/ -DCMAKE_INSTALL_PREFIX=. -H. -DSTREAM_ARRAY_SIZE=2147483648 -DNTIMES=20 -DMODEL=cuda -DCMAKE_CUDA_COMPILER=/cm/shared/apps/cuda12.3/toolkit/12.3.2/bin/nvcc -DCUDA_ARCH=sm_70 

cmake --build Build-x86_64/

./Build-x86_64/cuda-stream