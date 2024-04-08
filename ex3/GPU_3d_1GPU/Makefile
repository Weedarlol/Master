# Define different grid sizes
WIDTH = 1024
HEIGHT = 1024
DEPTH = 1024

# Number of max iterations to run for the scenario
ITERATIONS = 10000

COMPARE = 0

PARTITION = dgx2q

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH), \
					$(foreach height,$(HEIGHT), \
					$(foreach depth,$(DEPTH), \
					width$(width)_height$(height)_depth$(depth)_gpu1_iter$(ITERATIONS)_compare$(COMPARE)_overlap1_test0_createGrid0)))


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