# Define different matrix sizes
WIDTH = 8192
HEIGHT = 8192

GPU_COUNT = 0

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# Enables or disables overlap, 1 = overlap, 0 = no overlap
OVERLAP = 0 1

# 0 = full calculation, 1 = No kernel, 2 = No communication, 3 = only calculation, 4 = only communication
TEST = 0

# If you want to create a file which stores the output matrix, can be used as an guidance if other 2d computations are correct
CREATEMATRIX = 0

# Chooses which partition to run at. (hgx2q and dgx2q for more than 1 GPU, a100q and a40q for more than 1 cluster)
PARTITION = dgx2q

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),$(foreach iter,$(ITERATIONS),width$(width)_height$(height)_gpu0_iter$(ITERATIONS)_compare0_overlap0_test0_createMatrix$(CREATEMATRIX))))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch -p $(partition) ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)



pythonplot:
	python3 2d_plot.py


# View rules
JobName:
	sacct --format="JobID,JobName%60"



# Clean rules
clean:
	rm -rf slurm* out_*

cleann:
	rm -rf output/* error/*

cleannn:
	rm -rf slurm* out_* output/* error/*



cancel:
	scancel -u vidaro