#!/bin/bash

#SBATCH --job-name=stream_benchmark
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=bandwidth/dgx2q.out
#SBATCH --error=error/dgx2q.err

# hgx2q(A100) -> 40 MB L2 cache = 40 * 2**20 = 41943040 bytes
# dgx2q(V100) -> 6 MB L2 cache = 6 * 2**20 = 6291456 bytes

module purge
module load cuda12.3/toolkit/12.3.2
module load cuda12.3/blas/12.3.2
module load cuda12.3/fft/12.3.2
module load cuda12.3/profiler/12.3.2
module load cuda12.3/nsight/12.3.2
module load cmake/gcc/3.27.9

cmake -B bandwidth/ -DCMAKE_INSTALL_PREFIX=. -H. \
     -DSTREAM_ARRAY_SIZE=$(expr 4 \* 6291456) \
     -DNTIMES=20 \
     -DMODEL=cuda \
     -DCMAKE_CUDA_COMPILER=/cm/shared/apps/cuda12.3/toolkit/12.3.2/bin/nvcc \
     -DCUDA_ARCH=sm_70 

cmake --build bandwidth/

./bandwidth/cuda-stream