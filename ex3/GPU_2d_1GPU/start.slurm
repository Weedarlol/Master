#!/bin/bash
set -e

# Common flags for nvcc
NVCCFLAGS="-rdc=true"

# Directory for your library (adjust this to the actual path)
LIBDIR="/../../usr/lib/x86_64-linux-gnu/libnccl.so"

# List of source files
SOURCES="main.cu jacobi.cu"
OBJECTS_DIR="../../functions/"
GLOBAL_FUNCTIONS_OBJECT="global_functions_ex3.o"
CUDA_FUNCTIONS_OBJECT="cuda_functions_ex3.o"


# Object files derived from sources
OBJECTS=$(echo "$SOURCES" | sed 's/\.cu/\.o/g')
C_OBJECTS=$(echo "$C_SOURCES" | sed 's/\.c/\.o/g')

# Gets the first element which indicates the partition type
PARTITION="$1"
shift


# Creates a scenario for the rest of the elements
for SCENARIO in "$@"; do
    # Extract matrix size and GPU count from the run identifier
    WIDTH=$(echo $SCENARIO | awk -F'width' '{print $2}' | awk -F'_' '{print $1}')
    HEIGHT=$(echo $SCENARIO | awk -F'height' '{print $2}' | awk -F'_' '{print $1}')
    ITERATIONS=$(echo $SCENARIO | awk -F'iter' '{print $2}' | awk -F'_' '{print $1}')
    COMPARE=$(echo $SCENARIO | awk -F'compare' '{print $2}')

    # Define the variable values
    VAR1=$WIDTH # Width
    VAR2=$HEIGHT # Height
    VAR3=$ITERATIONS  # Iterations
    VAR4=$COMPARE

    # Create a temporary Slurm script for this scenario
    TEMP_SCRIPT="slurm_script_${SCENARIO}_${PARTITION}.sh"

    cat << EOF > "${TEMP_SCRIPT}"
#!/bin/bash

#SBATCH --job-name=${SCENARIO}_${PARTITION}
#SBATCH -p ${PARTITION}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=output/${SCENARIO}_${PARTITION}.out
#SBATCH --error=error/${SCENARIO}_${PARTITION}.err

module purge
module load slurm/21.08.8
module load cuda11.8/blas/11.8.0
module load cuda11.8/fft/11.8.0
module load cuda11.8/nsight/11.8.0
module load cuda11.8/profiler/11.8.0
module load cuda11.8/toolkit/11.8.0

nvcc $NVCCFLAGS $SOURCES $OBJECTS_DIR/$GLOBAL_FUNCTIONS_OBJECT $OBJECTS_DIR/$CUDA_FUNCTIONS_OBJECT -o out_${SCENARIO}_${PARTITION} -L$LIBDIR -lnccl -O3

./out_${SCENARIO}_${PARTITION} $VAR1 $VAR2 $VAR3 $VAR4

exit 0

EOF

    # Submit the temporary Slurm script as a job
    sbatch "${TEMP_SCRIPT}"

    # Remove the temporary script after submission
    rm "${TEMP_SCRIPT}"

done