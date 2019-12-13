#include "des_cuda_constants.cuh"
#include "des_gpu_functions.cuh"

__device__ void generate_subkeys_gpu(uint64 key, uint64 * subkeys) {
    int size_PC1 = sizeof(PC_1_CUDA)/sizeof(PC_1_CUDA[0]);
    int size_PC2 = sizeof(PC_2_CUDA)/sizeof(PC_2_CUDA[0]);

    uint64 permuted_key = permute(key, PC_1_CUDA, size_PC1);

    uint32 C[17], D[17];

    C[0]  = (uint32) (permuted_key >> 28  & 0xFFFFFFF);
    D[0]  = (uint32) (permuted_key >> 0 & 0xFFFFFFF);

    // apply left shifts
    for(int i = 1; i <= 16; i++) {

        C[i] = C[i-1] << SHIFTS_CUDA[i-1];
        D[i] = D[i-1] << SHIFTS_CUDA[i-1];

        C[i] |= C[i] >> (29 - SHIFTS_CUDA[i-1]);
        D[i] |= D[i] >> (29 - SHIFTS_CUDA[i-1]);

        C[i] &= ~(3UL << 28);
        D[i] &= ~(3UL << 28);

        uint64 merged_subkey = ((uint64)C[i] << 28) | D[i];
        subkeys[i-1] = permute(merged_subkey, PC_2_CUDA, size_PC2);
    }
}

__device__ uint64 encrypt_message_gpu(uint64 message, uint64 key) {
    uint64 K[16];
    uint32 L[17], R[17];

    generate_subkeys_gpu(key, K);

    int size_IP = sizeof(IP_CUDA)/sizeof(IP_CUDA[0]);
    uint64 IP_message = permute(message, IP_CUDA, size_IP);

    L[0]  = (uint32) (IP_message >> 32 & 0xFFFFFFFF);
    R[0]  = (uint32) (IP_message >> 0 & 0xFFFFFFFF);

    for(int i = 1; i <= 16; i++) {
        L[i] = R[i-1];
        R[i] = L[i-1] ^ f(R[i-1], K[i-1]);
    }

    uint64 RL = ((uint64) R[16] << 32) | L[16];
    uint64 encrypted_message = permute(RL, IP_REV_CUDA, 64);

    return encrypted_message;
}

__device__ uint32 f_gpu(uint32 R, uint64 K) {
    int size_E = sizeof(E_BIT_CUDA)/sizeof(E_BIT_CUDA[0]);
    unsigned char S[8];
    uint32 s_string = 0;
    uint64 expanded_R = permute(R, E_BIT_CUDA, size_E);

    uint64 R_xor_K = expanded_R ^ K;

    for(int i = 0; i < 8; i++) {
        S[i] = get_S_value_gpu((unsigned char) (R_xor_K >> 6*(7 - i)) & 0x3F, i);
        s_string |= S[i];
        s_string <<= (i != 7) ? 4 : 0;
    }
    return (uint32) permute(s_string, P_CUDA, 32);
}

__device__ unsigned char get_S_value_gpu(unsigned char B, int s_idx) {
    unsigned int i = (((B >> 5) & 1U) << 1) | ((B >> 0) & 1U);
    unsigned int j = 0;

    for(int k = 4; k > 0; k--) {
        j |= ((B >> k) & 1U);
        j <<= (k != 1) ? 1 : 0;
    }

    return (unsigned char) S_POINTER_CUDA[s_idx][16 * i + j];
}