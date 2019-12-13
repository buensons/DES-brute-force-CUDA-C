CC=gcc
NVCC=nvcc
LDLIBS = mt19937.o

all: des

des: *.cu *.h *.cuh *.o
	$(NVCC) -o des des_brute_force.cu

.PHONY: 
	clean all
clean:
	rm all

