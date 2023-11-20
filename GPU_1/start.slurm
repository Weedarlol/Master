#!/bin/bash

# Common flags for nvcc
NVCCFLAGS="-rdc=true"

# Directory for your library (adjust this to the actual path)
LIBDIR="/../../usr/lib/x86_64-linux-gnu/libnccl.so"

# List of source files
SOURCES="main.cu jacobi.cu"

# Object files derived from sources
OBJECTS=$(echo "$SOURCES" | sed 's/\.cu/\.o/g')

for SCENARIO in "$@"; do
    # Extract matrix size and GPU count from the run identifier
    MATRIX_SIZE=$(echo $SCENARIO | awk -F'size' '{print $2}' | awk -F'_' '{print $1}')
    NUM_GPUS=$(echo $SCENARIO | awk -F'gpu' '{print $2}' | awk -F'_' '{print $1}')
    ITERATIONS=$(echo $SCENARIO | awk -F'iter' '{print $2}' | awk -F'_' '{print $1}')
    PARTITION=$(echo $SCENARIO | awk -F'iter' '{print $2}' | awk -F'_' '{print $2}')

    # Define the variable values
    VAR1=$MATRIX_SIZE # Width
    VAR2=$MATRIX_SIZE # Height
    VAR3=$ITERATIONS  # Iterations

    # Create a temporary Slurm script for this scenario
    TEMP_SCRIPT="slurm_script_${SCENARIO}.sh"

    cat << EOF > "${TEMP_SCRIPT}"
#!/bin/bash

#SBATCH --job-name=${SCENARIO}
#SBATCH -p ${PARTITION}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --output=output/${SCENARIO}.out
#SBATCH --error=error/${SCENARIO}.err

module purge
module load slurm/21.08.8
module load cuda11.8/blas/11.8.0
module load cuda11.8/fft/11.8.0
module load cuda11.8/nsight/11.8.0
module load cuda11.8/profiler/11.8.0
module load cuda11.8/toolkit/11.8.0

nvcc $NVCCFLAGS $SOURCES -o out_${SCENARIO}

./out_${SCENARIO} $VAR1 $VAR2 $VAR3

exit 0

EOF

    # Submit the temporary Slurm script as a job
    sbatch "${TEMP_SCRIPT}"

    # Remove the temporary script after submission
    rm "${TEMP_SCRIPT}"

done


