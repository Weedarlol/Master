#!/bin/bash
set -e

# Gets the first element which indicates the partition type
PARTITION="$1"
shift


# Creates a scenario for the rest of the elements
for SCENARIO in "$@"; do
    # Extract matrix size and GPU count from the run identifier
    WIDTH=$(echo $SCENARIO | awk -F'width' '{print $2}' | awk -F'_' '{print $1}')
    HEIGHT=$(echo $SCENARIO | awk -F'height' '{print $2}' | awk -F'_' '{print $1}')
    DEPTH=$(echo $SCENARIO | awk -F'depth' '{print $2}' | awk -F'_' '{print $1}')
    NODES=$(echo $SCENARIO | awk -F'nodes' '{print $2}' | awk -F'_' '{print $1}')
    ITERATIONS=$(echo $SCENARIO | awk -F'iter' '{print $2}' | awk -F'_' '{print $1}')
    COMPARE=$(echo $SCENARIO | awk -F'compare' '{print $2}' | awk -F'_' '{print $1}')
    OVERLAP=$(echo $SCENARIO | awk -F'overlap' '{print $2}' | awk -F'_' '{print $1}')

    # Define the variable values
    VAR1=$WIDTH # Width
    VAR2=$HEIGHT # Height
    VAR3=$DEPTH
    VAR4=$ITERATIONS  # Iterations
    VAR5=$COMPARE
    VAR6=$OVERLAP

    # Create a temporary Slurm script for this scenario
    TEMP_SCRIPT="slurm_script_${SCENARIO}_${PARTITION}.sh"

    cat << EOF > "${TEMP_SCRIPT}"
#!/bin/bash

#SBATCH --job-name=${SCENARIO}_${PARTITION}
#SBATCH -p ${PARTITION}
#SBATCH -N 1                                   # Antall Noder
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=${NODES}                   # Antall tasks en kjørere på hver node, 1 Task = 1 MPI
#SBATCH --cpus-per-task=1                     # Antall CPUer en vil allokere, er 64 cores per CPU, så kan i teorien bare trenge å øke dene når -n > 64
#SBATCH --output=output/${SCENARIO}_${PARTITION}.out
#SBATCH --error=error/${SCENARIO}_${PARTITION}.err

module purge
module load slurm/21.08.8
module load gcc11/11.3.0
module load cuda11.8/toolkit/11.8.0
module load openmpi4/gcc/4.1.5

nvcc main.c -o out_${SCENARIO}_${PARTITION} -I$CPATH -L$LD_LIBRARY_PATH /cm/shared/apps/openmpi4/gcc/4.1.5/lib/libmpi.so -lnccl -O3

mpirun -np ${NODES}  out_${SCENARIO}_${PARTITION} $VAR1 $VAR2 $VAR3 $VAR4 $VAR5 $VAR6

exit 0

EOF

    # Submit the temporary Slurm script as a job
    sbatch "${TEMP_SCRIPT}"

    # Remove the temporary script after submission
    rm "${TEMP_SCRIPT}"

done


