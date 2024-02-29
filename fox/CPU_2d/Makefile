# Define different matrix sizes
WIDTH = 512
HEIGHT = 512

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# Chooses which partition to run at. (hgx2q and dgx2q for more than 1 GPU, a100q and a40q for more than 1 cluster)
# Example = hgx2q, dgx2q, a100q
PARTITION = normal

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),$(foreach iter,$(ITERATIONS),width$(width)_height$(height)_iter$(ITERATIONS))))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch --account=ec12 -p $(partition) --time=00-00:01:00 --mem-per-cpu=3G ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)






# View rules
JobName:
	sacct --format="JobID,JobName%60"



# Clean rules
clean:
	rm -rf slurm* out_*

cleannn:
	rm -rf output/* error/*




cancel:
	scancel -u ec-vidarlol