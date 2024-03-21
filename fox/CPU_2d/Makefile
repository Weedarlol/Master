# Define different matrix sizes
WIDTH = 8192
HEIGHT = 8192

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# Chooses which partition to run at. Only one at a time
PARTITION = accel

# 0 = full calculation, 1 = No kernel, 2 = No communication, 3 = only calculation, 4 = only communication
CREATEMATRIX = 0

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),width$(width)_height$(height)_gpu0_iter$(ITERATIONS)_compare0_overlap0_test0_createMatrix$(CREATEMATRIX)))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch --account=ec12 -p $(partition) --time=00-00:01:00 --mem-per-cpu=3G --gpus-per-node=0 ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)






# View rules
JobName:
	sacct --format="JobID,JobName%60"



pythonplot:
	python3 2d_plot.py


# Clean rules
clean:
	rm -rf slurm* out_*

clean_results:
	rm -rf output/* error/*

clean_all:
	rm -rf slurm* out_* output/* error/*




cancel:
	scancel -u ec-vidarlol