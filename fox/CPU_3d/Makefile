# Define different grid sizes
WIDTH = 512
HEIGHT = 512
DEPTH = 512

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# Chooses which partition to run at.
PARTITION = accel

CREATEGRID = 0

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH), \
					$(foreach height,$(HEIGHT), \
					$(foreach depth,$(DEPTH), \
					width$(width)_height$(height)_depth$(depth)_gpu0_iter$(ITERATIONS)_compare0_overlap0_test0_createGrid$(CREATEGRID))))
	
submit:
	$(foreach partition,$(PARTITION), \
		sbatch --account=ec12 -p $(partition) --time=00-00:01:00 --mem-per-cpu=4G --gpus-per-node=0  ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
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
	scancel -u ec-vidarlol