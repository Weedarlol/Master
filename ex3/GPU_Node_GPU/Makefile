
# Creates a list of values for GPU allocation, max 3 integers when running with nsys nvprof
# Example  = 2 3 4
GPU_COUNTS = 2
NODES = 2


# Define different matrix sizes
WIDTH = 512
HEIGHT = 512
DEPTH = 512

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# Chooses which partition to run at. Only one at a time
PARTITION = a100q

# Enables or disables overlap, 0 = no overlap, 1 = overlap
OVERLAP = 0 1

# If you want to compare the final resulting matrix with the matrix created by the CPU. (CPU has to computed it first and created its own textfile containing its matrix called "CPUMatrix%i_%i.txt" where %i are the width and height)
COMPARE = 1

# 0 = full calculation, 1 = No kernel, 2 = No communication, 3 = only calculation, 4 = only communication
TEST = 0

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH), \
					$(foreach height,$(HEIGHT), \
					$(foreach depth,$(DEPTH), \
					$(foreach gpu,$(GPU_COUNTS), \
					$(foreach node,$(NODES),$ \
					$(foreach overlap,$(OVERLAP), \
					$(foreach test,$(TEST), \
					width$(width)_height$(height)_depth$(depth)_gpu$(gpu)_nodes$(NODES)_iter$(ITERATIONS)_compare$(COMPARE)_overlap$(overlap)_test$(test)_createGrid0)))))))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch -p $(partition) ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)



# View rules
JobName:
	sacct --format="JobID,JobName%60"

# Clean rules
clean:
	rm -rf slurm* out_*

clean_all:
	rm -rf slurm* out_* output/* error/*

cancel:
	scancel -u vidaro
