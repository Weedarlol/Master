# Common flags for nvcc
NVCCFLAGS = -rdc=true

# List of source files for n_GPU and GPU
n_GPU_SOURCES = n_GPU.cu
GPU_SOURCES = GPU.cu

# Object files derived from sources
n_GPU_OBJECTS = $(n_GPU_SOURCES:.cu=.o)
GPU_OBJECTS = $(GPU_SOURCES:.cu=.o)

# Starting the ex3 system
ex3:
	sbatch -p dgx2q ./start.slurm


# Checking latency between CPU and GPU
GPU_out: $(GPU_OBJECTS)
	nvcc $(NVCCFLAGS) $^ -o $@ 

# Rule for compiling CUDA source files into object files
%.o: %.cu
	nvcc $(NVCCFLAGS) -c $< -o $@



# To view the output
view:
	cat output/output.out
# To view the error
vieww:
	cat output/error.err


# Clean content
clean:
	$(RM) $(n_GPU_OBJECTS) $(GPU_OBJECTS) n_GPU_out GPU_out
# Clean all content
cleann:
	$(RM) main.o out output/output.out output/error.err  report1.nsys-rep report1.sqlite report2.nsys-rep report2.sqlite report3.nsys-rep report3.sqlite report4.nsys-rep report4.sqlite report5.nsys-rep report5.sqlite


cancel:
	scancel -u vidaro