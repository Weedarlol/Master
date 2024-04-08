# Define different grid sizes
WIDTH = 1024
HEIGHT = 1024
DEPTH = 1024

# Number of max iterations to run for the scenario
ITERATIONS = 10000

COMPARE = 0

PARTITION = accel

submit:
	$(foreach partition,$(PARTITION), \
		sbatch --account=ec12 -p $(partition) --time=00-00:01:00 --gpus-per-node=1 --mem-per-gpu=3G ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
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