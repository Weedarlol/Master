
# Creates a list of values for GPU allocation, max 3 integers when running with nsys nvprof
# Example  = 1 2 4
GPU_AMOUNTS = 1

# Define different matrix sizes
WIDTH = 8192
HEIGHT = 8192

# Number of max iterations to run for the scenario
ITERATIONS = 1000

# Chooses which partition to run at. (hgx2q and dgx2q for more than 1 GPU, a100q and a40q for more than 1 cluster)
# Example = hgx2q, dgx2q, a100q, a40q
PARTITION = a40q

# Enables or disables overlap, 1 = overlap, 0 = no overlap
OVERLAP = 1

# If you want to compare the final resulting matrix with the matrix created by the CPU. (CPU has to computed it first and created its own textfile containing its matrix called "CPUMatrix%i_%i.txt" where %i are the width and height)
COMPARE = 1


NODE_AMOUNTS = 2

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),$(foreach gpu,$(GPU_AMOUNTS),$(foreach iter,$(ITERATIONS),$(foreach overlap,$(OVERLAP),width$(width)_height$(height)_gpu$(gpu)_nodes$(NODE_AMOUNTS)_iter$(ITERATIONS)_compare$(COMPARE)_overlap$(overlap))))))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch -p $(partition) ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)






# View rules (On local computer)
plot:
	python3 plot_graphs.py
	

JobName:
	sacct --format="JobID,JobName%60"



# Clean rules
clean:
	rm -rf slurm* out_*

cleannn:
	rm -rf output/* error/* nsight/*




cancel:
	scancel -u vidaro