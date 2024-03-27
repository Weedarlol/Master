#!/bin/bash

#SBATCH --job-name=stream_benchmark
#SBATCH --account=ec12
#SBATCH -p accel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --output=bandwidth/accel.out
#SBATCH --error=error/accel.err



module purge

make all

make stream.gcc

./stream_gcc