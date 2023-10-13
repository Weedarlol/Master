#!/bin/bash

#SBATCH --job-name=CPU_Overlap
#SBATCH -p dgx2q
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH -o output/output.out
#SBATCH -e output/error.err

module purge
module load slurm/21.08.8

module load cuda11.8/blas/11.8.0
module load cuda11.8/fft/11.8.0
module load cuda11.8/nsight/11.8.0
module load cuda11.8/profiler/11.8.0
module load cuda11.8/toolkit/11.8.0


make nvprofiler

make clean
