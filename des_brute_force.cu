#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <strings.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "des_cpu_functions.cuh"
#include "des_gpu_functions.cuh"

#define DES_KEY_SIZE 64

#define ERR(source) (perror(source), fprintf(stderr,"%s:%d\n",__FILE__,__LINE__), exit(EXIT_FAILURE))

int main() {

    // change it later and add padding with 0's if mod 64 != 0
    uint64 data = 0x0123456789ABCDEF;
    uint64 key = generate_key();
    uint64 encrypted_message = encrypt_message(data, key);

    // --------- CPU -------------

    printf("CPU : Brute forcing DES...\n");

    uint64 i = 0;

    for(i = ~(i); i >= 0; i--) {
        uint64 msg = encrypt_message(data, i);
        //printBits(i);
        if(msg == encrypted_message) {
            printf("CPU : Key found!\n");
            printf("CPU : Original Key: %llX\n", key);
            printf("CPU : Found key: %llX\n", i);
            break;
        }
    }

    // --------- CUDA ------------
    
    //cudaSetDevice(cutGetMaxGflopsDeviceId());

    bool * has_key = NULL;
    bool temp = false;
    uint64 * cracked_key = NULL;
    uint64 found_key;

    cudaError_t error;

    if((error = cudaMalloc(has_key, sizeof(bool))) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }

    if((error = cudaMalloc(cracked_key, sizeof(uint64))) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }

    if((error = cudaMemcpy(has_key, &temp, sizeof(bool), cudaMemcpyHostToDevice)) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }

    printf("GPU : Brute forcing DES...\n");
    brute_force<<<4096, 1024>>>(data, encrypted_message, cracked_key, has_key);

    if((error = cudaDeviceSynchronize()) != cudaSuccess) ERR(cudaGetErrorString(error));
    
    if((error = cudaMemcpy(&found_key, cracked_key, sizeof(uint64), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        ERR(cudaGetErrorString(error));
    }

    printf("GPU : Key found!\n");
    printf("GPU : Time elapsed - ");
    printf("GPU : Cracked key: %llX\n", found_key);

    cudaFree(has_key);
    cudaFree(cracked_key);

    return 0;
}
