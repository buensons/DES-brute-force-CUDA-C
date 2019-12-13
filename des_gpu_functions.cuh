#ifndef DES_GPU_FUNCTIONS
#define DES_GPU_FUNCTIONS

typedef unsigned long long uint64;
typedef unsigned long uint32;

__device__ generate_subkeys_gpu(uint64 key, uint64 * subkeys);

__device__ unsigned char get_S_value_gpu(unsigned char B, int s_idx);

__device__ uint32 f_gpu(uint32 R, uint64 K);

__device__ uint64 encrypt_message_gpu(uint64 message, uint64 key);

#endif