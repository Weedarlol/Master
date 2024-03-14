
# 0 = CPU_2d, 1 = CPU_3d, 2 = CPU_Node, 3 = GPU_2d, 4 = GPU_2d_1GPU, 5 = GPU_3d, 6 = GPU_3d_1GPU, 7 = GPU_Node
SCENARIO = 0

# 0 - width, height,        iterations, create
# 1 - width, height, depth, iterations,                       create
# 2 - width, height, depth, iterations, nodes,                compare
# 3 - width, height,        iterations,        gpus, compare,         overlap, test
# 4 - width, height,        iterations,              compare,         overlap, test
# 5 - width, height, depth, iterations,        gpus, compare,         overlap, test
# 6 - width, height, depth, iterations,              compare,         overlap, test
# 7 - width, height, depth, iterations, nodes, gpus, compare,         overlap, test

WIDTH = 512
HEIGHT = 512
DEPTH = 512
ITERATIONS = 10000
NODES = 1
GPUS = 0
COMPARE = 0
CREATE = 0
OVERLAP = 0
TEST = 0
PARTITION = accel

RUN_IDENTIFIERS = $(width_$(WIDTH)_height$(HEIGHT)_depth$(DEPTH)_nodes$(NODES)_gpus$(GPUS)_compare$(COMPARE)_create$(CREATE)_overlap$(OVERLAP)_test$(TEST)_partition$(PARTITION))

CC = cc
NVCC = nvcc

CFLAGS = -Wall -O3
NVCCFLAGS = -O3

SRC_C = ../global_functions.c
SRC_CUDA = ../CUDA_functions.cu

OBJ_C = $(SRC_C:.c=.o)
OBJ_CUDA = $(notdir $(SRC_CUDA:.cu=.o)) 

