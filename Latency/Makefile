# Define different matrix sizes (integers)
HEIGHT = 2048
WIDTH = 2048

# Decide partition, (dgx2q, hgx2q)
PARTITION = dgx2q

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach height,$(HEIGHT),$(foreach width,$(WIDTH),width$(width)_height$(height)_$(PARTITION)))

submit:
	sbatch -p $(PARTITION) ./start.slurm $(RUN_IDENTIFIERS)






# View rules
view:
	cat output/output.out

vieww:
	cat output/error.err

# Clean rules
clean:
	$(foreach identifier,$(RUN_IDENTIFIERS),$(RM) out_$(identifier)) rm -rf slurm*

cancel:
	scancel -u vidaro