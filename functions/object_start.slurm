#!/bin/bash
set -e

# List of source files

NVCCFLAGS="-lm"

PARTITION="$1"
USER="$2"

#SBATCH --job-name=creating_objects_${PARTITION}
#SBATCH --account=${USER}
#SBATCH -p ${PARTITION}
#SBATCH --mem-per-cpu=4G
#SBATCH -N 1                                  # Antall noder, i dgx2q så er det bare 1 node bygd opp av DualProcessor AMD EPYC Milan 7763 64-core w/ 8 qty Nvidia Volta A100/80GB
#SBATCH -n 1                                  # Antall CPU cores som vil bli allokert
#SBATCH -t 00:01:00
#SBATCH --ntasks-per-node=1                   # Antall ganger en vil kjøre programmet på hver node
#SBATCH --cpus-per-task=1                     # Antall CPUer en vil allokere, er 64 cores per CPU, så kan i teorien bare trenge å øke dene når -n > 64
#SBATCH --gres=gpu:0                # Antall GPUer en allokerer per job, så totale antall GPU
#SBATCH --gpus-per-task=0           # Antall GPUer en allokerer per task, så totale antall GPU delt på antall noder
#SBATCH --gpus=0}                    # Antall GPUer en allokerer per node, så totale antall GPU delt på antall noder

module purge
if [[ "$PARTITION" == "dgx2q" ]]; then
    module load slurm/21.08.8
    module load cuda11.8/blas/11.8.0
    module load cuda11.8/fft/11.8.0
    module load cuda11.8/nsight/11.8.0
    module load cuda11.8/profiler/11.8.0
    module load cuda11.8/toolkit/11.8.0
    NAME="ex3"
elif [[ "$PARTITION" == "normal" ]]; then
    module load CUDA/11.7.0
    NAME="fox"
fi

gcc -Wall -O3 -c global_functions.c -o global_functions_${NAME}.o

nvcc $NVCCFLAGS -c cuda_functions.cu -o cuda_functions_${NAME}.o