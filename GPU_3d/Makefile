
# Creates a list of values for GPU allocation, max 3 integers when running with nsys nvprof
# Example  = 2 3 4
GPU_COUNTS = 1

# Define different matrix sizes
WIDTH = 32
HEIGHT = 32
DEPTH = 32

# Number of max iterations to run for the scenario
ITERATIONS = 1

# Chooses which partition to run at. Only one at a time
PARTITION = dgx2q

# Enables or disables overlap, 1 = overlap, 0 = no overlap
OVERLAP = 0

# If you want to compare the final resulting matrix with the matrix created by the CPU. (CPU has to computed it first and created its own textfile containing its matrix called "CPUMatrix%i_%i.txt" where %i are the width and height)
COMPARE = 0

# 0 = full calculation, 1 = No kernel, 2 = No communication, 3 = only calculation
TEST = 0

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),$(foreach depth,$(DEPTH),$(foreach gpu,$(GPU_COUNTS),$(foreach iter,$(ITERATIONS),$(foreach overlap,$(OVERLAP),$(foreach test,$(TEST),width$(width)_height$(height)_depth$(depth)_gpu$(gpu)_iter$(ITERATIONS)_compare$(COMPARE)_overlap$(overlap)_test$(test))))))))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch -p $(partition) ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)


babel:
	sbatch -p $(PARTITION) ./babel.slurm $(PARTITION)

# View rules (On local computer)
plot:
	python3 plot_graphs.py
	

JobName:
	sacct --format="JobID,JobName%60"


gitpush:
	git push origin ex3


# Clean rules
clean:
	rm -rf slurm* out_*

cleannn:
	rm -rf output/* error/* nsight/*




cancel:
	scancel -u vidaro
