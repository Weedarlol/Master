
# Define different matrix sizes
MATRIX_SIZES = 512 1024

ITERATIONS = 1000000

PARTITION = dgx2q hgx2q

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach size,$(MATRIX_SIZES),size$(size)_iter$(ITERATIONS))


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
	$(foreach partition,$(PARTITION), \
		$(foreach identifier,$(RUN_IDENTIFIERS),$(RM) out_$(identifier)_$(partition)); \
	) rm -rf slurm*

cleannn:
	$(foreach partition,$(PARTITION), \
		$(foreach identifier,$(RUN_IDENTIFIERS),$(RM) error/$(identifier)_$(partition).err output/$(identifier)_$(partition).out); \
	) rm -rf nsight/size*




cancel:
	scancel -u vidaro