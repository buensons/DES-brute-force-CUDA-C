#ifndef DES_GPU_FUNCTIONS
#define DES_GPU_FUNCTIONS

typedef unsigned long long uint64;
typedef unsigned long uint32;

__global__ void brute_force(uint64 message, uint64 encrypted_message, uint64 * original_key, bool * has_key);

__device__ void generate_subkeys_gpu(uint64 key, uint64 * subkeys);

__device__ unsigned char get_S_value_gpu(unsigned char B, int s_idx);

__device__ uint32 f_gpu(uint32 R, uint64 K);

__device__ uint64 encrypt_message_gpu(uint64 message, uint64 key);

__device__ __host__ void printBits(uint64 n);

__device__ __host__ uint64 permute(uint64 key, int * table, int size);

#endif