#include "modexp.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static __device__ void modexp_limbs(
    const uint64_t* base,
    const uint64_t* exp,
    const uint64_t* mod,
    uint64_t* result,
    int digits
) {
    uint64_t temp[MAX_DIGITS];
    uint64_t temp2[MAX_DIGITS];
    uint64_t base_copy[MAX_DIGITS];
    
    for (int i = 0; i < digits; i++) {
        result[i] = (i == 0) ? 1 : 0;
        base_copy[i] = base[i];
    }

    for (int i = digits - 1; i >= 0; i--) {
        uint64_t e = exp[i];
        for (int bit = 63; bit >= 0; bit--) {
            mul_limbs(result, result, temp, digits);
            sub_limbs(temp, mod, temp2, digits);
            while (cmp_limbs(temp, mod, digits) >= 0) {
                for (int j = 0; j < digits; j++) {
                    temp[j] = temp2[j];
                }
                sub_limbs(temp, mod, temp2, digits);
            }
            for (int j = 0; j < digits; j++) {
                result[j] = temp[j];
            }
            
            if ((e >> bit) & 1) {
                mul_limbs(result, base_copy, temp, digits);
                sub_limbs(temp, mod, temp2, digits);
                while (cmp_limbs(temp, mod, digits) >= 0) {
                    for (int j = 0; j < digits; j++) {
                        temp[j] = temp2[j];
                    }
                    sub_limbs(temp, mod, temp2, digits);
                }
                for (int j = 0; j < digits; j++) {
                    result[j] = temp[j];
                }
            }
        }
    }
}

extern "C" {
    __global__ void rsa_encrypt_kernel(
        const uint64_t* message,
        const uint64_t* public_key,
        const uint64_t* modulus,
        uint64_t* cipher,
        int digits
    ) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            modexp_limbs(message, public_key, modulus, cipher, digits);
        }
    }

    __global__ void rsa_decrypt_kernel(
        const uint64_t* cipher,
        const uint64_t* private_key,
        const uint64_t* modulus,
        uint64_t* decrypted,
        int digits
    ) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            modexp_limbs(cipher, private_key, modulus, decrypted, digits);
        }
    }
}

void string_to_limbs(const char* str, uint64_t* limbs, int max_digits) {
    memset(limbs, 0, max_digits * sizeof(uint64_t));
    int len = strlen(str);
    int pos = 0;
    
    for (int i = 0; i < len && pos < max_digits; i += 8) {
        uint64_t val = 0;
        int chunk_size = (len - i < 8) ? len - i : 8;
        memcpy(&val, str + i, chunk_size);
        limbs[pos++] = val;
    }
}

void limbs_to_string(const uint64_t* limbs, char* str, int max_digits) {
    memset(str, 0, max_digits * 8 + 1);
    int pos = 0;
    
    for (int i = 0; i < max_digits && limbs[i] != 0; i++) {
        int chunk_size = 8;
        memcpy(str + pos, &limbs[i], chunk_size);
        pos += chunk_size;
    }
    str[pos] = '\0';
}

// Example usage
int main() {
    uint64_t message[MAX_DIGITS] = {
        0x724250656a6f7250ULL,
        0x5f7570635f636574ULL
    };
    uint64_t cipher[MAX_DIGITS] = {0};
    uint64_t decrypted[MAX_DIGITS] = {0};
    
    uint64_t public_key[MAX_DIGITS] = {65537}; // e = 65537
    
    // n = 0xE103ABD94892E3E74AFD724BF28E78366D9676BCCC70118BD0AA1968DBB143D1
    uint64_t modulus[MAX_DIGITS] = {
        0xD0AA1968DBB143D1ULL,
        0x6D9676BCCC70118BULL,
        0x4AFD724BF28E7836ULL,
        0xE103ABD94892E3E7ULL
    };
    
    // d = 0xC5148B1AD7C6D85847C52EABB879F26CCCEE8EB70D2282DC6050309C18F85D85
    uint64_t private_key[MAX_DIGITS] = {
        0x6050309C18F85D85ULL,
        0xCCEE8EB70D2282DCULL,
        0x47C52EABB879F26CULL,
        0xC5148B1AD7C6D858ULL
    };
    
    // Allocate device memory
    uint64_t *d_message, *d_cipher, *d_decrypted;
    uint64_t *d_public_key, *d_private_key, *d_modulus;
    
    cudaMalloc(&d_message, MAX_DIGITS * sizeof(uint64_t));
    cudaMalloc(&d_cipher, MAX_DIGITS * sizeof(uint64_t));
    cudaMalloc(&d_decrypted, MAX_DIGITS * sizeof(uint64_t));
    cudaMalloc(&d_public_key, MAX_DIGITS * sizeof(uint64_t));
    cudaMalloc(&d_private_key, MAX_DIGITS * sizeof(uint64_t));
    cudaMalloc(&d_modulus, MAX_DIGITS * sizeof(uint64_t));
    
    // Copy data to device
    cudaMemcpy(d_message, message, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_public_key, public_key, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_private_key, private_key, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_modulus, modulus, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Encrypt
    rsa_encrypt_kernel<<<1,1>>>(d_message, d_public_key, d_modulus, d_cipher, MAX_DIGITS);
    cudaDeviceSynchronize();
    
    // Copy cipher back to host
    cudaMemcpy(cipher, d_cipher, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    printf("Encrypted message (hex):\n");
    for(int i = MAX_DIGITS-1; i >= 0; i--) {
        if(cipher[i] != 0) {
            printf("%016lx", cipher[i]);
        }
    }
    printf("\n");
    
    // Decrypt
    rsa_decrypt_kernel<<<1,1>>>(d_cipher, d_private_key, d_modulus, d_decrypted, MAX_DIGITS);
    cudaDeviceSynchronize();
    
    // Copy decrypted back to host
    cudaMemcpy(decrypted, d_decrypted, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    printf("Decrypted message (hex):\n");
    for(int i = MAX_DIGITS-1; i >= 0; i--) {
        if(decrypted[i] != 0) {
            printf("%016lx", decrypted[i]);
        }
    }
    printf("\n");
    
    // Print as ASCII
    printf("Decrypted message (ASCII): ");
    unsigned char* bytes = (unsigned char*)decrypted;
    for(int i = 0; i < MAX_DIGITS * 8; i++) {
        if(bytes[i] >= 32 && bytes[i] <= 126) { // printable ASCII
            printf("%c", bytes[i]);
        }
    }
    printf("\n");
    
    // Cleanup
    cudaFree(d_message);
    cudaFree(d_cipher);
    cudaFree(d_decrypted);
    cudaFree(d_public_key);
    cudaFree(d_private_key);
    cudaFree(d_modulus);
    
    return 0;
} 