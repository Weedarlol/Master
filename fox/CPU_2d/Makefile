# Define different matrix sizes
WIDTH = 512
HEIGHT = 512

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# If you want to create a file which stores the output matrix, can be used as an guidance if other 2d computations are correct
CREATEMATRIX = 0

# Chooses which partition to run at. (hgx2q and dgx2q for more than 1 GPU, a100q and a40q for more than 1 cluster)
PARTITION = accel

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH), \
					$(foreach height,$(HEIGHT), \
					width$(width)_height$(height)_gpu0_iter$(ITERATIONS)_compare0_overlap0_test0_createMatrix$(CREATEMATRIX)))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch --account=ec12 -p $(partition) --time=00-00:01:00 --mem-per-cpu=3G --gpus-per-node=0 ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
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