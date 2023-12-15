# Define different matrix sizes
WIDTH = 8192
HEIGHT = 8192

# Number of max iterations to run for the scenario
ITERATIONS = 1000

# Chooses which partition to run at. (hgx2q and dgx2q for more than 1 GPU, a100q and a40q for more than 1 cluster)
# Example = hgx2q, dgx2q, a100q
PARTITION = dgx2q

COMPARE = 1

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),$(foreach iter,$(ITERATIONS),width$(width)_height$(height)_iter$(ITERATIONS)_compare${COMPARE})))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch -p $(partition) ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)






# View rules
view:
	cat output/output.out

vieww:
	cat output/error.err



# Clean rules
clean:
	rm -rf slurm* out_*

cleannn:
	rm -rf output/* error/* nsight/*




cancel:
	scancel -u vidaro