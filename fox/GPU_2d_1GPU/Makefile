# Define different matrix sizes
WIDTH = 8192
HEIGHT = 8192

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# If you want to compare the final resulting matrix with the matrix created by the CPU. (CPU has to computed it first and created its own textfile containing its matrix called "CPUMatrix%i_%i.txt" where %i are the width and height)
COMPARE = 1

# Chooses which partition to run at. (hgx2q and dgx2q for more than 1 GPU, a100q and a40q for more than 1 cluster)
PARTITION = accel

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH), \
					$(foreach height,$(HEIGHT), \
					width$(width)_height$(height)_gpu1_iter$(ITERATIONS)_compare$(COMPARE)_overlap0_test0_createMatrix0))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch --account=ec12 -p $(partition) --time=00-00:01:00 --gpus-per-node=1 --mem-per-gpu=3G ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)


# Clean rules
clean:
	rm -rf slurm* out_*

clean_all:
	rm -rf output/* error/* nsight/*

cancel:
	scancel -u vidaro