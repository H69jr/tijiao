# 编译器设置
CC = g++
NVCC = /data/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-12.8.0-vphbzjdsorewbmnlbe74g6hqtz3guuig/bin/nvcc

# 路径设置
CUDA_HOME = /data/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/cuda-12.8.0-vphbzjdsorewbmnlbe74g6hqtz3guuig
CUDA_INCLUDE = -I$(CUDA_HOME)/include
CUDA_LIB = -L$(CUDA_HOME)/lib64
CUDA_LDFLAGS = $(CUDA_LIB) -lcudart -lcublas

# 编译选项
CFLAGS = -O3 -g -Wall -fopenmp -std=c++17 $(CUDA_INCLUDE) -DNDEBUG
CUDAFLAGS = -O3 -arch=sm_89 -Xcompiler="-fopenmp" $(CUDA_INCLUDE) -DNDEBUG

# 源文件（注意文件名一致性！）
SRCS = driver.cc winograd.cc winograd_kernels_fixed.cu
OBJS = driver.o winograd.o winograd_kernels_fixed.o

# 目标
all: winograd

winograd: $(OBJS)
	$(NVCC) $(CUDAFLAGS) -o $@ $(OBJS) $(CUDA_LDFLAGS)

# 编译规则
%.o: %.cc
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

clean:
	rm -f winograd *.o *.cu_o

.PHONY: all clean
