#!/bin/bash
set -e

# Common flags for nvcc
NVCCFLAGS="-rdc=true"

# Directory for your library (adjust this to the actual path)
LIBDIR="/../../cm/shared/apps/openmpi4-cuda11.8-ofed5-gcc11/4.1.4/lib"

# List of source files
SOURCES="main.cu jacobi.cu"

# Object files derived from sources
OBJECTS=$(echo "$SOURCES" | sed 's/\.cu/\.o/g')

# Gets the first element which indicates the partition type
PARTITION="$1"

shift


# Creates a scenario for the rest of the elements
for SCENARIO in "$@"; do
    # Extract matrix size and GPU count from the run identifier
    WIDTH=$(echo $SCENARIO | awk -F'width' '{print $2}' | awk -F'_' '{print $1}')
    HEIGHT=$(echo $SCENARIO | awk -F'height' '{print $2}' | awk -F'_' '{print $1}')
    NUM_GPUS=$(echo $SCENARIO | awk -F'gpu' '{print $2}' | awk -F'_' '{print $1}')
    NUM_NODES=$(echo $SCENARIO | awk -F'nodes' '{print $2}' | awk -F'_' '{print $1}')
    ITERATIONS=$(echo $SCENARIO | awk -F'iter' '{print $2}' | awk -F'_' '{print $1}')
    COMPARE=$(echo $SCENARIO | awk -F'compare' '{print $2}' | awk -F'_' '{print $1}')
    OVERLAP=$(echo $SCENARIO | awk -F'overlap' '{print $2}')

    # Define the variable values
    VAR1=$WIDTH # Width
    VAR2=$HEIGHT # Height
    VAR3=$ITERATIONS  # Iterations
    VAR4=$NUM_GPUS    # GPUs
    VAR5=$COMPARE
    VAR6=$OVERLAP

    # Create a temporary Slurm script for this scenario
    TEMP_SCRIPT="slurm_script_${SCENARIO}_${PARTITION}.sh"


    cat << EOF > "${TEMP_SCRIPT}"
#!/bin/bash

#SBATCH --account=ec12
#SBATCH --job-name=${SCENARIO}_${PARTITION}
#SBATCH -p ${PARTITION}
#SBATCH -N ${NUM_NODES}
#SBATCH -n ${NUM_NODES}
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --gpus-per-node=${NUM_GPUS}
#SBATCH --output=output/${SCENARIO}_${PARTITION}.out
#SBATCH --error=error/${SCENARIO}_${PARTITION}.err

set -o errexit
set -o nounset

if [ "`uname -m`" = "x86_64" ];
  then
        module purge
        module load slurm/21.08.8
        module load cuda11.8/blas/11.8.0
        module load cuda11.8/fft/11.8.0
        module load cuda11.8/nsight/11.8.0
        module load cuda11.8/profiler/11.8.0
        module load cuda11.8/toolkit/11.8.0

        nvcc $NVCCFLAGS $SOURCES -o out_${SCENARIO}_${PARTITION} -L$LIBDIR -lmpi

        mpirun -np ${NUM_NODES} ./out_${SCENARIO}_${PARTITION} $VAR1 $VAR2 $VAR3 $VAR4 $VAR5 $VAR6


elif [ "`uname -m`" = "aarch64" ];
    then
        echo "The current Aarch64 ISA enabled partition has not enabled CUDA libraries, and can therefore not run"
fi





exit 0

EOF

    # Submit the temporary Slurm script as a job
    sbatch "${TEMP_SCRIPT}"

    # Remove the temporary script after submission
    rm "${TEMP_SCRIPT}"

done


