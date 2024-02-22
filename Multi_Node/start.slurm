#!/bin/bash
set -e

# List of source files
SOURCES="main.cu programs/jacobi.cu programs/scenarios.cu"

# Gets the first element which indicates the partition type
PARTITION="$1"

shift


# Creates a scenario for the rest of the elements
for SCENARIO in "$@"; do
    # Extract matrix size and GPU count from the run identifier
    WIDTH=$(echo $SCENARIO | awk -F'width' '{print $2}' | awk -F'_' '{print $1}')
    HEIGHT=$(echo $SCENARIO | awk -F'height' '{print $2}' | awk -F'_' '{print $1}')
    NUM_GPUS=$(echo $SCENARIO | awk -F'gpu' '{print $2}' | awk -F'_' '{print $1}')
    ITERATIONS=$(echo $SCENARIO | awk -F'iter' '{print $2}' | awk -F'_' '{print $1}')
    COMPARE=$(echo $SCENARIO | awk -F'compare' '{print $2}' | awk -F'_' '{print $1}')
    OVERLAP=$(echo $SCENARIO | awk -F'overlap' '{print $2}' | awk -F'_' '{print $1}')
    TEST=$(echo $SCENARIO | awk -F'test' '{print $2}' | awk -F'_' '{print $1}')
    NODES=$(echo $SCENARIO | awk -F'nodes' '{print $2}')

    # Define the variable values
    VAR1=$WIDTH # Width
    VAR2=$HEIGHT # Height
    VAR3=$ITERATIONS  # Iterations
    VAR4=$NUM_GPUS    # GPUs
    VAR5=$COMPARE
    VAR6=$OVERLAP
    VAR7=$TEST
    VAR8=$NODES

    # Create a temporary Slurm script for this scenario
    TEMP_SCRIPT="slurm_script_${SCENARIO}_${PARTITION}.sh"

    cat << EOF > "${TEMP_SCRIPT}"
#!/bin/bash

#SBATCH --account=ec12 
#SBATCH --job-name=${SCENARIO}_${PARTITION}_GPU${NUM_GPUS}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${NUM_GPUS}
#SBATCH --time=12:0:0
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=output/${SCENARIO}_${PARTITION}.out
#SBATCH --error=error/${SCENARIO}_${PARTITION}.err


module purge
module load CUDA/11.7.0
module load OpenMPI/4.1.4-GCC-11.3.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0

nvcc $SOURCES -o ./out_${SCENARIO}_${PARTITION} -lmpi

mpirun -n ${NODES} ./out_${SCENARIO}_${PARTITION} $VAR1 $VAR2 $VAR3 $VAR4 $VAR5 $VAR6 $VAR7





exit 0

EOF

    # Submit the temporary Slurm script as a job
    sbatch "${TEMP_SCRIPT}"

    # Remove the temporary script after submission
    rm "${TEMP_SCRIPT}"

done


