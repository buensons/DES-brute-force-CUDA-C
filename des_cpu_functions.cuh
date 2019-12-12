#ifndef DES_CPU_FUNCTIONS
#define DES_CPU_FUNCTIONS

#include <stdlib.h>
#include <time.h>
#include <strings.h>
#include <math.h>

typedef unsigned long long uint64;

__host__ uint64 generate_key();

__host__ void generate_subkeys(uint64 key, uint64 * subkeys);

__host__ unsigned char get_S_value(unsigned char B, int s_idx);

__host__ uint32_t f(uint32_t R, uint64 K);

__host__ uint64 encrypt_message(uint64 message, uint64 key);

#endif