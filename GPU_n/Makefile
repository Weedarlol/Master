
# Creates a list of values for GPU allocation
GPU_COUNTS = 1

# Define different matrix sizes
WIDTH = 512 
HEIGHT = 512 

ITERATIONS = 100 1000 10000

PARTITION = dgx2q

# Enables or disables overlap, 1 = overlap, 0 = no overlap
OVERLAP = 1

# Create a list of run identifiers for all combinations
RUN_IDENTIFIERS = $(foreach width,$(WIDTH),$(foreach height,$(HEIGHT),$(foreach gpu,$(GPU_COUNTS),$(foreach iter,$(ITERATIONS),$(foreach overlap,$(OVERLAP),width$(width)_height$(height)_gpu$(gpu)_iter$(ITERATIONS)_overlap$(overlap))))))


submit:
	$(foreach partition,$(PARTITION), \
		sbatch -p $(partition) ./start.slurm $(partition) $(RUN_IDENTIFIERS); \
	)






# View rules (On local computer)
plot:
	python3 plot_graphs.py
	

JobName:
	sacct --format="JobID,JobName%40"





# Clean rules
clean:
	$(foreach partition,$(PARTITION), \
		$(foreach identifier,$(RUN_IDENTIFIERS),$(RM) out_$(identifier)_$(partition)); \
	) rm -rf slurm*

cleannn:
	$(foreach partition,$(PARTITION), \
		$(foreach identifier,$(RUN_IDENTIFIERS),$(RM) error/$(identifier)_$(partition).err output/$(identifier)_$(partition).out); \
	)




cancel:
	scancel -u vidaro