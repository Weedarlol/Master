#!/bin/bash
set -e

# Common flags for nvcc
NVCCFLAGS="-rdc=true"

# Directory for your library (adjust this to the actual path)
LIBDIR="/../../usr/lib/x86_64-linux-gnu/libnccl.so"

# List of source files
SOURCES="main.cu ../../global_functions.cu"

# Object files derived from sources
OBJECTS=$(echo "$SOURCES" | sed 's/\.cu/\.o/g')

# Gets the first element which indicates the partition type
PARTITION="$1"
shift


# Creates a scenario for the rest of the elements
for SCENARIO in "$@"; do
    # Extract grid size and GPU count from the run identifier
    WIDTH=$(echo $SCENARIO | awk -F'width' '{print $2}' | awk -F'_' '{print $1}')
    HEIGHT=$(echo $SCENARIO | awk -F'height' '{print $2}' | awk -F'_' '{print $1}')
    DEPTH=$(echo $SCENARIO | awk -F'depth' '{print $2}' | awk -F'_' '{print $1}')
    ITERATIONS=$(echo $SCENARIO | awk -F'iter' '{print $2}' | awk -F'_' '{print $1}')
    CREATEGRID=$(echo $SCENARIO | awk -F'createGrid' '{print $2}')

    # Define the variable values
    VAR1=$WIDTH
    VAR2=$HEIGHT
    VAR3=$DEPTH
    VAR4=$ITERATIONS
    VAR5=$CREATEGRID

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
#SBATCH --gres=gpu:0
#SBATCH --output=output/${SCENARIO}_${PARTITION}.out
#SBATCH --error=error/${SCENARIO}_${PARTITION}.err

module purge
module load slurm/21.08.8
module load cuda11.8/blas/11.8.0
module load cuda11.8/fft/11.8.0
module load cuda11.8/nsight/11.8.0
module load cuda11.8/profiler/11.8.0
module load cuda11.8/toolkit/11.8.0

nvcc $NVCCFLAGS $SOURCES -o out_${SCENARIO}_${PARTITION} -L$LIBDIR -lnccl

./out_${SCENARIO}_${PARTITION} $VAR1 $VAR2 $VAR3 $VAR4 $VAR5

exit 0

EOF

    # Submit the temporary Slurm script as a job
    sbatch "${TEMP_SCRIPT}"

    # Remove the temporary script after submission
    rm "${TEMP_SCRIPT}"

done


