#include "des_cpu_functions.cuh"
#include "des_constants.cuh"
#include "mt64.h"

__host__ uint64 generate_key() {

    init_genrand64(time(NULL));
    uint64 key = 0;

    for(int i = 0; i < DES_KEY_SIZE; i++) {
        const uint64 bit = (uint64)round(genrand64_real2());
        key = (key & ~(1ULL << i)) | (bit << i);
    }
    return key;
}

__host__ void generate_subkeys(uint64 key, uint64 * subkeys) {
    int size_PC1 = sizeof(PC_1)/sizeof(PC_1[0]);
    int size_PC2 = sizeof(PC_2)/sizeof(PC_2[0]);

    uint64 permuted_key = permute(key, PC_1, size_PC1);

    uint32_t C[17], D[17];

    C[0]  = (uint32_t) (permuted_key >> 28  & 0xFFFFFFF);
    D[0]  = (uint32_t) (permuted_key >> 0 & 0xFFFFFFF);

    // apply left shifts
    for(int i = 1; i <= 16; i++) {

        C[i] = C[i-1] << SHIFTS[i-1];
        D[i] = D[i-1] << SHIFTS[i-1];

        C[i] |= C[i] >> 28;
        D[i] |= D[i] >> 28;

        C[i] &= ~(3UL << 28);
        D[i] &= ~(3UL << 28);

        uint64 merged_subkey = ((uint64)C[i] << 28) | D[i];
        subkeys[i-1] = permute(merged_subkey, PC_2, size_PC2);
    }
}

__host__ uint64 encrypt_message(uint64 message, uint64 key) {
    uint64 K[16];
    uint32_t L[17], R[17];

    generate_subkeys(key, K);

    int size_IP = sizeof(IP)/sizeof(IP[0]);
    uint64 IP_message = permute(message, IP, size_IP);

    L[0]  = (uint32_t) (IP_message >> 32 & 0xFFFFFFFF);
    R[0]  = (uint32_t) (IP_message >> 0 & 0xFFFFFFFF);

    for(int i = 1; i <= 16; i++) {
        L[i] = R[i-1];
        R[i] = L[i-1] ^ f(R[i-1], K[i-1]);
    }

    uint64 RL = ((uint64) R[16] << 32) | L[16];
    uint64 encrypted_message = permute(RL, IP_REV, 64);

    return encrypted_message;
}

__host__ uint32_t f(uint32_t R, uint64 K) {
    int size_E = sizeof(E_BIT)/sizeof(E_BIT[0]);
    unsigned char S[8];
    uint32_t s_string = 0;
    uint64 expanded_R = permute(R, E_BIT, size_E);

    uint64 R_xor_K = expanded_R ^ K;

    for(int i = 0; i < 8; i++) {
        S[i] = get_S_value((unsigned char) (R_xor_K >> 6*(7 - i)) & 0x3F, i);
        s_string |= S[i];
        s_string <<= (i != 7) ? 4 : 0;
    }
    return (uint32_t) permute(s_string, P, 32);
}

__host__ unsigned char get_S_value(unsigned char B, int s_idx) {
    unsigned int i = (((B >> 5) & 1U) << 1) | ((B >> 0) & 1U);
    unsigned int j = 0;

    for(int k = 4; k > 0; k--) {
        j |= ((B >> k) & 1U);
        j <<= (k != 1) ? 1 : 0;
    }

    return (unsigned char) S_POINTER[s_idx][16 * i + j];
}