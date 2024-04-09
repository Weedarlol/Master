# Define different grid sizes
WIDTH = 128
HEIGHT = 128
DEPTH = 128

# How many nodes to run the program on
NODES = 1

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# If you want to compare the final resulting grid with the grid created by the CPU.
COMPARE = 1

# Chooses which partition to run at. Only one at a time
PARTITION = hgx2q


# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH), \
					$(foreach height,$(HEIGHT), \
					$(foreach depth,$(DEPTH), \
					width$(width)_height$(height)_depth$(depth)_gpu0_nodes$(NODES)_iter$(ITERATIONS)_compare$(COMPARE)_overlap0_test0_createGrid0)))


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
	scancel -u vidarlol