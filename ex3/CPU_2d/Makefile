# Define different matrix sizes
WIDTH = 8192
HEIGHT = 8192

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# Chooses which partition to run at. (hgx2q and dgx2q for more than 1 GPU, a100q and a40q for more than 1 cluster)
PARTITION = dgx2q

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),$(foreach iter,$(ITERATIONS),width$(width)_height$(height)_iter$(ITERATIONS))))


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

cleann:
	rm -rf output/* error/*

cleannn:
	rm -rf slurm* out_* output/* error/*



cancel:
	scancel -u vidaro