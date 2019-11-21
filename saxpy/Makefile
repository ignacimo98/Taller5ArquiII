NVCC = nvcc

all: saxpy

%.o : %.cu
	$(NVCC) -c $< -o $@

saxpy : saxpy.o
	$(NVCC) $^ -o $@

clean:
	rm -rf *.o *.a saxpy
