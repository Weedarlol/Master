# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /cm/shared/apps/cmake/gcc/3.27.9/bin/cmake

# The command to remove a file.
RM = /cm/shared/apps/cmake/gcc/3.27.9/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vidaro/D1/Master/Mas/benchmark/BabelStream

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vidaro/D1/Master/Mas/benchmark/BabelStream/Build-x86_64

# Include any dependencies generated for this target.
include CMakeFiles/cuda-stream.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cuda-stream.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda-stream.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda-stream.dir/flags.make

CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.o: CMakeFiles/cuda-stream.dir/flags.make
CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.o: CMakeFiles/cuda-stream.dir/includes_CUDA.rsp
CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.o: /home/vidaro/D1/Master/Mas/benchmark/BabelStream/src/cuda/CUDAStream.cu
CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.o: CMakeFiles/cuda-stream.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vidaro/D1/Master/Mas/benchmark/BabelStream/Build-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.o"
	/cm/shared/apps/cuda12.3/toolkit/12.3.2/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.o -MF CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.o.d -x cu -c /home/vidaro/D1/Master/Mas/benchmark/BabelStream/src/cuda/CUDAStream.cu -o CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.o

CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cuda-stream.dir/src/main.cpp.o: CMakeFiles/cuda-stream.dir/flags.make
CMakeFiles/cuda-stream.dir/src/main.cpp.o: /home/vidaro/D1/Master/Mas/benchmark/BabelStream/src/main.cpp
CMakeFiles/cuda-stream.dir/src/main.cpp.o: CMakeFiles/cuda-stream.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vidaro/D1/Master/Mas/benchmark/BabelStream/Build-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/cuda-stream.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cuda-stream.dir/src/main.cpp.o -MF CMakeFiles/cuda-stream.dir/src/main.cpp.o.d -o CMakeFiles/cuda-stream.dir/src/main.cpp.o -c /home/vidaro/D1/Master/Mas/benchmark/BabelStream/src/main.cpp

CMakeFiles/cuda-stream.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/cuda-stream.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vidaro/D1/Master/Mas/benchmark/BabelStream/src/main.cpp > CMakeFiles/cuda-stream.dir/src/main.cpp.i

CMakeFiles/cuda-stream.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/cuda-stream.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vidaro/D1/Master/Mas/benchmark/BabelStream/src/main.cpp -o CMakeFiles/cuda-stream.dir/src/main.cpp.s

# Object files for target cuda-stream
cuda__stream_OBJECTS = \
"CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.o" \
"CMakeFiles/cuda-stream.dir/src/main.cpp.o"

# External object files for target cuda-stream
cuda__stream_EXTERNAL_OBJECTS =

cuda-stream: CMakeFiles/cuda-stream.dir/src/cuda/CUDAStream.cu.o
cuda-stream: CMakeFiles/cuda-stream.dir/src/main.cpp.o
cuda-stream: CMakeFiles/cuda-stream.dir/build.make
cuda-stream: CMakeFiles/cuda-stream.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/vidaro/D1/Master/Mas/benchmark/BabelStream/Build-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable cuda-stream"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda-stream.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda-stream.dir/build: cuda-stream
.PHONY : CMakeFiles/cuda-stream.dir/build

CMakeFiles/cuda-stream.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda-stream.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda-stream.dir/clean

CMakeFiles/cuda-stream.dir/depend:
	cd /home/vidaro/D1/Master/Mas/benchmark/BabelStream/Build-x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vidaro/D1/Master/Mas/benchmark/BabelStream /home/vidaro/D1/Master/Mas/benchmark/BabelStream /home/vidaro/D1/Master/Mas/benchmark/BabelStream/Build-x86_64 /home/vidaro/D1/Master/Mas/benchmark/BabelStream/Build-x86_64 /home/vidaro/D1/Master/Mas/benchmark/BabelStream/Build-x86_64/CMakeFiles/cuda-stream.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/cuda-stream.dir/depend

