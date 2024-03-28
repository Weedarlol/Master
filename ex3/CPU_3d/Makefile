# Define different grid sizes
WIDTH = 128
HEIGHT = 128
DEPTH = 128

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# Chooses which partition to run at. (hgx2q and dgx2q for more than 1 GPU, a100q and a40q for more than 1 cluster)
# Example = hgx2q, dgx2q, a100q
PARTITION = hgx2q


CREATEGRID = 0

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),$(foreach depth,$(DEPTH),width$(width)_height$(height)_depth$(depth)_gpu0_iter$(ITERATIONS)_compare0_overlap0_test0_createGrid$(CREATEGRID))))
																																
submit:
	$(foreach partition,$(PARTITION), \
		sbatch -p $(partition) ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)



pythonplot:
	python3 3d_plot.py




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