CC=gcc
NVCC=nvcc
CFLAGS= -std=gnu99 -Wall
LDLIBS = mt19937.o

all: des

des: *.cu *.h *.cuh
	$(NVCC) $(CFLAGS) -o des des_brute_force.cu $(LDLIBS)

.PHONY: 
	clean all
clean:
	rm all

