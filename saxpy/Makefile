NVCC = nvcc

all: saxpy

%.o : %.cu
	$(NVCC) -c $< -o $@ -O0

saxpy : saxpy.o
	$(NVCC) $^ -o $@ -O0

clean:
	rm -rf *.o *.a saxpy
