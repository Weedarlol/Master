ex3:
	sbatch -p dgx2q ./start.slurm



n_GPU: n_GPU_out
	nsys nvprof ./GPU_out

n_GPU_out: n_GPU.o
	nvcc $^ -o $@ -L/../../usr/lib/x86_64-linux-gnu/libnccl.so

%.o: n_GPU.cu
	nvcc -rdc=true -c $< -o $@ 




GPU: GPU_out
	nsys nvprof ./GPU_out

GPU_out: GPU.o
	nvcc $^ -o $@ -L/../../usr/lib/x86_64-linux-gnu/libnccl.so

%.o: GPU.cu
	nvcc -rdc=true -c $< -o $@ 





view:
	cat output/output.out

vieww:
	cat output/error.err





clean:
	$(RM) GPU.o n_GPU.o GPU_out n_GPU_out

cleann:
	$(RM) main.o out output/output.out output/error.err  report1.nsys-rep report1.sqlite report2.nsys-rep report2.sqlite report3.nsys-rep report3.sqlite report4.nsys-rep report4.sqlite report5.nsys-rep report5.sqlite



cancel:
	scancel -u vidaro

