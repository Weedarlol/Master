ex3:
	sbatch -p a100q ./start.slurm



nvprofiler: out
	./out

nvprofilerr: out
	nsys nvprof --kernels "jacobi" ./out

out: main.o jacobi.o
	nvcc $^ -o $@ -L/../../usr/lib/x86_64-linux-gnu/libnccl.so

%.o: %.cu
	nvcc -rdc=true -c $< -o $@ 







view:
	cat output/output.out

vieww:
	cat output/error.err





clean:
	$(RM) main.o out jacobi.o

cleann:
	$(RM) main.o out output/output.out output/error.err  report1.nsys-rep report1.sqlite report2.nsys-rep report2.sqlite report3.nsys-rep report3.sqlite report4.nsys-rep report4.sqlite report5.nsys-rep report5.sqlite



cancel:
	scancel -u vidaro
