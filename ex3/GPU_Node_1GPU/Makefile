
# Define different matrix sizes
WIDTH = 128
HEIGHT = 128
DEPTH = 128

NODES = 2

# Number of max iterations to run for the scenario
ITERATIONS = 10000

# If you want to compare the final resulting matrix with the matrix created by the CPU. (CPU has to computed it first and created its own textfile containing its matrix called "CPUMatrix%i_%i.txt" where %i are the width and height)
COMPARE = 1

# Enables or disables overlap, 0 = no overlap, 1 = full overlap, 2 = CPU to CPU overlap. 3 = GPU to CPU overlap
OVERLAP = 1

# 0 = full calculation, 1 = No kernel, 2 = No communication, 3 = only calculation, 4 = only communication
TEST = 0

# Chooses which partition to run at. Only one at a time
PARTITION = dgx2q

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH), \
					$(foreach height,$(HEIGHT), \
					$(foreach depth,$(DEPTH),$ \
					$(foreach node,$(NODES),$ \
					$(foreach overlap,$(OVERLAP),$ \
					$(foreach test,$(TEST),$ \
					width$(width)_height$(height)_depth$(depth)_gpu1_nodes$(node)_iter$(ITERATIONS)_compare$(COMPARE)_overlap$(overlap)_test$(test)_createGrid0))))))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch -p $(partition) ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)




JobName:
	sacct --format="JobID,JobName%100"

# Clean rules
clean:
	rm -rf slurm* out_*

clean_all:
	rm -rf slurm* out_* output/* error/*

cancel:
	scancel -u vidaro
