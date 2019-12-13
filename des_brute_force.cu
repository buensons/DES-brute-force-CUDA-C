#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <strings.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "des_cpu_functions.cuh"
#include "des_gpu_functions.cuh"

#define ERR(source) (perror(source), fprintf(stderr,"%s:%d\n",__FILE__,__LINE__), exit(EXIT_FAILURE))

int main(int argc, char ** argv) {

    // could-have: take input message and split into 64-bit blocks
    uint64 data = 0x0123456789ABCDEF;

    if(argc != 2) {
        printf("Usage: %s <key_size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int key_size = atoi(argv[1]);
    if(key_size > 64) {
        printf("Key size reduced to 64 bits.");
        key_size = 64;
    }
    uint64 key = generate_key(key_size);
    uint64 encrypted_message = encrypt_message(data, key);
    clock_t start, end;
    float time_elapsed;

    // --------- CPU -------------

    printf("CPU : Brute forcing DES...\n");
    
    start = clock();

    for(uint64 i = 0; i <= ~(0ULL); i++) {
        uint64 msg = encrypt_message(data, i);
        //printBits(i);
        if(msg == encrypted_message) {
            end = clock();
            time_elapsed = ((float) (end - start)) / CLOCKS_PER_SEC;
            printf("CPU : Key found!\n");
            printf("CPU : Found key: %llX\n", i);
            printf("CPU : Time elapsed - %f\n", time_elapsed);
            break;
        }
    }


    // --------- GPU ------------
    
    //cudaSetDevice(cutGetMaxGflopsDeviceId());

    int * has_key = NULL;
    int temp = 0;
    uint64 * cracked_key = NULL;
    uint64 found_key;

    cudaError_t error;

    if((error = cudaMalloc(&has_key, sizeof(int))) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }

    if((error = cudaMalloc(&cracked_key, sizeof(uint64))) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }

    if((error = cudaMemcpy(has_key, &temp, sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }

    printf("\nGPU : Brute forcing DES...\n");
    start = clock();

    brute_force<<<4096, 1024>>>(data, encrypted_message, cracked_key, has_key);

    end = clock();
    time_elapsed = ((float) (end - start)) / CLOCKS_PER_SEC;

    if((error = cudaDeviceSynchronize()) != cudaSuccess) ERR(cudaGetErrorString(error));
    
    if((error = cudaMemcpy(&found_key, cracked_key, sizeof(uint64), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }

    printf("GPU : Key found!\n");
    printf("GPU : Time elapsed - %f\n", time_elapsed);
    printf("GPU : Cracked key: %llX\n", found_key);

    cudaFree(has_key);
    cudaFree(cracked_key);

    return EXIT_SUCCESS;
}
