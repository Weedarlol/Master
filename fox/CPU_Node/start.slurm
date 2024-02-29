#!/bin/bash
set -e

# List of source files
SOURCES="main.cu"

LIBDIR="/cluster/software/EL9/easybuild/software/OpenMPI/4.1.4-GCC-11.3.0/lib"
INCLUDE="/cluster/software/EL9/easybuild/software/OpenMPI/4.1.4-GCC-11.3.0/include"



# Gets the first element which indicates the partition type
PARTITION="$1"
shift


# Creates a scenario for the rest of the elements
for SCENARIO in "$@"; do
    # Extract matrix size and GPU count from the run identifier
    WIDTH=$(echo $SCENARIO | awk -F'width' '{print $2}' | awk -F'_' '{print $1}')
    HEIGHT=$(echo $SCENARIO | awk -F'height' '{print $2}' | awk -F'_' '{print $1}')
    DEPTH=$(echo $SCENARIO | awk -F'depth' '{print $2}' | awk -F'_' '{print $1}')
    ITERATIONS=$(echo $SCENARIO | awk -F'iter' '{print $2}' | awk -F'_' '{print $1}')
    NODES=$(echo $SCENARIO | awk -F'nodes' '{print $2}')

    # Define the variable values
    VAR1=$WIDTH # Width
    VAR2=$HEIGHT # Height
    VAR3=$DEPTH # Depth
    VAR4=$ITERATIONS  # Iterations

    # Create a temporary Slurm script for this scenario
    TEMP_SCRIPT="slurm_script_${SCENARIO}_${PARTITION}.sh"

    cat << EOF > "${TEMP_SCRIPT}"
#!/bin/bash

#SBATCH --job-name=${SCENARIO}_${PARTITION}
#SBATCH --account=ec12
#SBATCH --mem-per-cpu=4G
#SBATCH -p ${PARTITION}
#SBATCH -N ${NODES}
#SBATCH -n ${NODES}
#SBATCH -t 14:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=output/${SCENARIO}_${PARTITION}.out
#SBATCH --error=error/${SCENARIO}_${PARTITION}.err

module purge
module load CUDA/11.7.0
module load OpenMPI/4.1.4-GCC-11.3.0

nvcc $SOURCES -o out_${SCENARIO}_${PARTITION} -lcudart -lmpi -L$LIBDIR -I$INCLUDE

srun ./out_${SCENARIO}_${PARTITION} $VAR1 $VAR2 $VAR3 $VAR4

exit 0

EOF

    # Submit the temporary Slurm script as a job
    sbatch "${TEMP_SCRIPT}"

    # Remove the temporary script after submission
    rm "${TEMP_SCRIPT}"

done


