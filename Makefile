CC=gcc
NVCC=nvcc

all: des

des: *.cu *.h *.cuh
	$(NVCC) -arch=sm_30 -o des des_brute_force.cu

.PHONY: 
	clean all
clean:
	rm all

