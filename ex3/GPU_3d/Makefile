
# Creates a list of values for GPU allocation, max 3 integers when running with nsys nvprof
# Example  = 2 3 4
GPU_COUNTS = 2

# Define different matrix sizes
WIDTH = 128
HEIGHT = 128
DEPTH = 128

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# Chooses which partition to run at. Only one at a time
PARTITION = dgx2q

# Enables or disables overlap, 0 = no overlap, 1 = overlap
OVERLAP = 0 1

# If you want to compare the final resulting matrix with the matrix created by the CPU. (CPU has to computed it first and created its own textfile containing its matrix called "CPUMatrix%i_%i.txt" where %i are the width and height)
COMPARE = 1

# 0 = full calculation, 1 = No kernel, 2 = No communication, 3 = only calculation
TEST = 0

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),$(foreach depth,$(DEPTH),$(foreach gpu,$(GPU_COUNTS),$(foreach iter,$(ITERATIONS),$(foreach overlap,$(OVERLAP),$(foreach test,$(TEST),width$(width)_height$(height)_depth$(depth)_gpu$(gpu)_iter$(ITERATIONS)_compare$(COMPARE)_overlap$(overlap)_test$(test))))))))









submit:
	$(foreach partition,$(PARTITION), \
		sbatch -p $(partition) ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)


babel:
	sbatch -p $(PARTITION) ./Babel/babel.slurm $(PARTITION)

# View rules (On local computer)
plot:
	python3 plot_graphs.py
	

JobName:
	sacct --format="JobID,JobName%100"



# Clean rules
clean:
	rm -rf slurm* out_*

cleann:
	rm -rf output/* error/*

cleannn:
	rm -rf slurm* out_* output/* error/*





cancel:
	scancel -u vidaro
