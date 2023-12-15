#!/bin/bash

# Common flags for nvcc
NVCCFLAGS="-rdc=true"
MPIFLAGS="-lmpi -lhwloc"

# Directory for your library (adjust this to the actual path)
LIBDIR="/../../usr/mpi/gcc/openmpi-4.1.5a1/lib"
INCDIR="/../../usr/mpi/gcc/openmpi-4.1.5a1/include"

# List of source files
SOURCES="main.cu"

# Object files derived from sources
OBJECTS=$(echo "$SOURCES" | sed 's/\.cu/\.o/g')

for SCENARIO in "$@"; do
    # Extract matrix size and GPU count from the run identifier
    MATRIX_SIZE=$(echo $SCENARIO | awk -F'size' '{print $2}' | awk -F'_' '{print $1}')
    NUM_GPUS=$(echo $SCENARIO | awk -F'gpu' '{print $2}' | awk -F'_' '{print $1}')
    ITERATIONS=$(echo $SCENARIO | awk -F'iter' '{print $2}' | awk -F'_' '{print $1}')
    NODES=$(echo $SCENARIO | awk -F'nodes' '{print $2}' | awk -F'_' '{print $1}')
    PARTITION=$(echo $SCENARIO | awk -F'nodes' '{print $2}' | awk -F'_' '{print $2}')

    # Define the variable values(UNCOMMENT WHEN WORKING ON MATRIX)
    # VAR1=$MATRIX_SIZE # Width
    # VAR2=$MATRIX_SIZE # Height
    # VAR3=$ITERATIONS  # Iterations

    # Create a temporary Slurm script for this scenario
    TEMP_SCRIPT="slurm_script_${SCENARIO}.sh"

    cat << EOF > "${TEMP_SCRIPT}"
#!/bin/bash

#SBATCH --job-name=${SCENARIO}
#SBATCH -p ${PARTITION}
#SBATCH -N ${NODES}
#SBATCH -n ${NODES}
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --output=output/${SCENARIO}.out
#SBATCH --error=error/${SCENARIO}.err

module purge
module load slurm/21.08.8
module load cuda11.6/blas/11.6.0
module load cuda11.6/fft/11.6.0
module load cuda11.6/nsight/11.6.0
module load cuda11.6/profiler/11.6.0
module load cuda11.6/toolkit/11.6.0





mpicc $SOURCES -o out_${SCENARIO} -L$LIBDIR -I$INCDIR $MPIFLAGS $NVCCFLAGS

mpirun -np ${NODES} ./out_${SCENARIO}

exit 0

EOF

    # Submit the temporary Slurm script as a job
    sbatch "${TEMP_SCRIPT}"

    # Remove the temporary script after submission
    rm "${TEMP_SCRIPT}"

done


