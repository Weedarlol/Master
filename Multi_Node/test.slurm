#!/bin/bash

#SBATCH --account=ec12 
#SBATCH --job-name=MyJob
#SBATCH --partition=accel 
#SBATCH --gpus-per-node=1
#SBATCH --time=12:0:0
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output/hei.out
#SBATCH --error=error/hei.err

module purge
module load CUDA/11.8.0