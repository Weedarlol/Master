#!/bin/bash

PARTITION="$1"

if [ "$PARTITION" == "hgx2q" ]; then
    CUDA_ARCH="sm_80"
elif [ "$PARTITION" == "dgx2q" ]; then
    CUDA_ARCH="sm_70"  # or the appropriate value for hgx2q
else
    echo "Invalid PARTITION value. Please use 'dgx2q' or 'hgx2q'."
    exit 1
fi

#SBATCH --job-name=Babel_${PARTITION}
#SBATCH -p ${PARTITION}
#SBATCH -N 1                                  # Antall noder, i dgx2q så er det bare 1 node bygd opp av DualProcessor AMD EPYC Milan 7763 64-core w/ 8 qty Nvidia Volta A100/80GB
#SBATCH -n 1                                  # Antall CPU cores som vil bli allokert
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1                   # Antall ganger en vil kjøre programmet på hver node
#SBATCH --cpus-per-task=1                     # Antall CPUer en vil allokere, er 64 cores per CPU, så kan i teorien bare trenge å øke dene når -n > 64
#SBATCH --gres=gpu:1                          # Antall GPUer en allokerer per job, så totale antall GPU
#SBATCH --gpus-per-task=1                     # Antall GPUer en allokerer per task, så totale antall GPU delt på antall noder
#SBATCH --gpus=1                              # Antall GPUer en allokerer per node, så totale antall GPU delt på antall noder
#SBATCH --output=output/Babel_${PARTITION}.out
#SBATCH --error=error/Babel_${PARTITION}.err

module purge
module load slurm/21.08.8
module load cuda11.8/blas/11.8.0
module load cuda11.8/fft/11.8.0
module load cuda11.8/nsight/11.8.0
module load cuda11.8/profiler/11.8.0
module load cuda11.8/toolkit/11.8.0
module load cmake/gcc/3.27.9

cd ../../../BabelStream

cmake -B ../Master/Mas/GPU_n/Build-x86_64/ -DCMAKE_INSTALL_PREFIX=. -H. -DMODEL=cuda -DCMAKE_CUDA_COMPILER=/cm/shared/apps/cuda11.8/toolkit/11.8.0/bin/nvcc -DCUDA_ARCH="$CUDA_ARCH"

cd ../Master/Mas/GPU_n/Build-x86_64/

cmake --build 

./cuda-stream
