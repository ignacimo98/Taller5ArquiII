NVCC = nvcc

all: matrix_cuda

%.o : %.cu
	$(NVCC) -c $< -o $@

matrix_cuda : matrix_cuda.o
	$(NVCC) $^ -o $@

clean:
	rm -rf *.o *.a matrix_cuda
