# Common flags for nvcc
NVCCFLAGS = -rdc=true

# Directory for your library (adjust this to the actual path)
LIBDIR = /../../usr/lib/x86_64-linux-gnu/libnccl.so

# List of source files
SOURCES = main.cu jacobi.cu

# Object files derived from sources
OBJECTS = $(SOURCES:.cu=.o)

ex3:
	sbatch -p dgx2q ./start.slurm


# Targets
all: nvprofiler

nvprofiler: out
	./out

nvprofilerr: out
	nsys nvprof --kernels "jacobi" ./out

out: $(OBJECTS)
	nvcc $(NVCCFLAGS) $^ -o $@ -L$(LIBDIR)

# Rule for compiling CUDA source files into object files
%.o: %.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	$(RM) $(OBJECTS) out

# Additional targets for your convenience
view:
	cat output/output.out

vieww:
	cat output/error.err

cleann:
	$(RM) main.o out output/output.out output/error.err  report1.nsys-rep report1.sqlite report2.nsys-rep report2.sqlite report3.nsys-rep report3.sqlite report4.nsys-rep report4.sqlite report5.nsys-rep report5.sqlite

cancel:
	scancel -u vidaro