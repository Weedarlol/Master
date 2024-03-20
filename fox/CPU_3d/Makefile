# Define different grid sizes
WIDTH = 512
HEIGHT = 512
DEPTH = 512

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# Chooses which partition to run at. (hgx2q and dgx2q for more than 1 GPU, a100q and a40q for more than 1 cluster)
# Example = hgx2q, dgx2q, a100q
PARTITION = accel


CREATEGRID = 1

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),$(foreach depth,$(DEPTH),$(foreach iter,$(ITERATIONS),width$(width)_height$(height)_depth$(depth)_iter$(ITERATIONS)_createGrid$(CREATEGRID)))))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch --account=ec12 -p $(partition) --time=00-00:01:00 --mem-per-cpu=3G --gpus-per-node=0  ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)






# View rules
JobName:
	sacct --format="JobID,JobName%60"



# Clean rules
clean:
	rm -rf slurm* out_*

clean_results:
	rm -rf output/* error/*

clean_all:
	rm -rf slurm* out_* output/* error/*



cancel:
	scancel -u ec-vidarlol