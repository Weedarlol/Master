# Example  = 2 3 4
GPU_COUNTS = 2

# Define different matrix sizes
WIDTH = 32
HEIGHT = 32

# Number of max iterations to run for the scenario
ITERATIONS = 1

# Chooses which partition to run at. Only one at a time
PARTITION = accel

# Enables or disables overlap, 1 = overlap, 0 = no overlap
OVERLAP = 0

# If you want to compare the final resulting matrix with the matrix created by the CPU. (CPU has to computed it first and created its own textfile containing its matrix called "CPUMatrix%i_%i.txt" where %i are the width and height)
COMPARE = 0

# 0 = full calculation, 1 = No kernel, 2 = No communication, 3 = only calculation, 4 = only communication
TEST = 0

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),$(foreach gpu,$(GPU_COUNTS),$(foreach iter,$(ITERATIONS),$(foreach overlap,$(OVERLAP),$(foreach test,$(TEST),width$(width)_height$(height)_gpu$(gpu)_iter$(ITERATIONS)_compare$(COMPARE)_overlap$(overlap)_test$(test)))))))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch -p $(partition) --account=ec12 --mem=3G --time=12:0:0 --gpus-per-node=$(GPU_COUNTS) ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)



JobName:
	sacct --format="JobID,JobName%60"



# Clean rules
clean:
	rm -rf slurm* out_*

cleannn:
	rm -rf output/* error/* nsight/*




cancel:
	scancel -u vidaro