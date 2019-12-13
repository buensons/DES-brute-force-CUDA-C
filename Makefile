CC=gcc
NVCC=nvcc

all: des

des: *.cu *.h *.cuh
	$(NVCC) -o des des_brute_force.cu

.PHONY: 
	clean all
clean:
	rm all

