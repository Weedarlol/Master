# Creates a list of values for GPU allocation
GPU_COUNTS = 1

# Decides the number of nodes to work on
NODES = 2

# Define different matrix sizes
MATRIX_SIZES = 512

# Number of iterations
ITERATIONS = 10000000

# Which partition/nodes
PARTITION = a40q











# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach size,$(MATRIX_SIZES),$(foreach gpu,$(GPU_COUNTS),size$(size)_gpu$(gpu)_iter$(ITERATIONS)_nodes$(NODES)_$(PARTITION)))

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