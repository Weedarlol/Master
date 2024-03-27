# Define different grid sizes
WIDTH = 512
HEIGHT = 512
DEPTH = 512

# Number of max iterations to run for the scenario
ITERATIONS = 10000

PARTITION = hgx2q

COMPARE = 1

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),$(foreach depth,$(DEPTH),width$(width)_height$(height)_depth$(depth)_gpu1_iter$(ITERATIONS)_compare$(COMPARE)_overlap0_test0_createGrid0)))


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