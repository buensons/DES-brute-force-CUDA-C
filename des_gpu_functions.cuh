#ifndef DES_GPU_FUNCTIONS
#define DES_GPU_FUNCTIONS

typedef unsigned long long uint64;

__device__ generate_subkeys_gpu(uint64_t key, uint64_t * subkeys);

__device__ unsigned char get_S_value_gpu(unsigned char B, int s_idx);

__device__ uint32_t f_gpu(uint32_t R, uint64_t K);

__device__ uint64_t encrypt_message_gpu(uint64_t message, uint64_t key);

#endif