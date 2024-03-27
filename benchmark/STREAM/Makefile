CC = gcc
CFLAGS = -O2 -fopenmp

FC = gfortran
FFLAGS = -O2 -fopenmp

PARTITION = dgx2q

all: stream_f.exe stream_c.exe

stream_f.exe: stream.f mysecond.o
	$(CC) $(CFLAGS) -c mysecond.c
	$(FC) $(FFLAGS) -c stream.f
	$(FC) $(FFLAGS) stream.o mysecond.o -o stream_f.exe

stream_c.exe: stream.c
	$(CC) $(CFLAGS) stream.c -o stream_c.exe

clean:
	rm -f stream_f.exe stream_c.exe *.o stream_gcc slurm-*

# an example of a more complex build line for the Intel icc compiler
stream.gcc: stream.c
	gcc -O3 -march=native -mcmodel=medium -DSTREAM_ARRAY_SIZE=2147483648 -DNTIMES=20 stream.c -o stream_gcc

run_stream_ex3:
	sbatch -p $(PARTITION) run_stream_ex3.slurm

run_stream_fox:
	sbatch --account=ec12 -p accel --time=00-00:10:00 --gpus-per-node=0 --mem-per-cpu=512G ./run_stream_fox.slurm