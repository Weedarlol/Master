#!/bin/bash

# List of programs to run
programs=("GPU" "n_GPU")

# Create a unique timestamp for each run
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Create a directory to store all program outputs
output_dir="/output/$timestamp"
mkdir -p $output_dir

# Loop through the programs and submit Slurm jobs
for program in "${programs[@]}"; do
    program_dir="$output_dir/$program"
    mkdir -p $program_dir

    sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$program
#SBATCH -p dgx2q
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 14:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -o $program_dir/output.out
#SBATCH -e $program_dir/error.err

module purge
module load slurm/21.08.8

module load cuda11.8/blas/11.8.0
module load cuda11.8/fft/11.8.0
module load cuda11.8/nsight/11.8.0
module load cuda11.8/profiler/11.8.0
module load cuda11.8/toolkit/11.8.0

make $program
make clean
EOT
done

