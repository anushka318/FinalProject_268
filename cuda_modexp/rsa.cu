#include "modexp.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIGITS 4

static __device__ void modexp_limbs(
    const uint64_t* base,
    const uint64_t* exp,
    const uint64_t* mod,
    const uint64_t* mu,      // <-- Added mu
    uint64_t* result,
    int digits
) {
    uint64_t temp[2 * MAX_DIGITS];     // for mul result
    uint64_t base_copy[MAX_DIGITS];

    // Initialize result = 1, base_copy = base
    for (int i = 0; i < digits; i++) {
        result[i] = (i == 0) ? 1 : 0;
        base_copy[i] = base[i];
    }

    // Process each 64-bit word of exp
    for (int i = digits - 1; i >= 0; i--) {
        uint64_t e = exp[i];
        for (int bit = 63; bit >= 0; bit--) {
            // result = result * result mod mod
            mul_limbs(result, result, temp, digits);                   // 2k-limb product
            barrettReductionKernel(temp, mod, mu, result, digits);           // k-limb reduced result

            if ((e >> bit) & 1) {
                mul_limbs(result, base_copy, temp, digits);            // 2k-limb product
                barrettReductionKernel(temp, mod, mu, result, digits);       // k-limb reduced result
            }
        }
    }
}



    __global__ void rsa_encrypt_kernel(
        const uint64_t* message,
        const uint64_t* public_key,
        const uint64_t* modulus,
        const uint64_t* mu,           // <-- Added mu
        uint64_t* cipher,
        int digits
    ) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            modexp_limbs(message, public_key, modulus, mu, cipher, digits);
        }
    }

    __global__ void rsa_decrypt_kernel(
        const uint64_t* cipher,
        const uint64_t* private_key,
        const uint64_t* modulus,
        const uint64_t* mu,           // <-- Added mu
        uint64_t* decrypted,
        int digits
    ) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            modexp_limbs(cipher, private_key, modulus, mu, decrypted, digits);
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



// Barrett mu = floor(b^(2k) / m), precomputed for the modulus
// Precomputed mu (Barrett constant), 8 limbs (least significant limb first)
const uint64_t mu_host[8] = {
    0xb41c30eed2ed9b0cULL,
    0x83ba4ee0a5c9a2a3ULL,
    0xeffb4e84cc8fd8aaULL,
    0x340a6f1f3cd9c7b5ULL,
    0x0000000000000012ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL
};

// Device pointer for mu
uint64_t* d_mu;

// Pass d_mu as argument to your kernels/functions that require mu



int main() {
uint64_t message[MAX_DIGITS] = {

        0x724250656a6f7250ULL,
        0x5f7570635f636574ULL,
        0x724250656a6f7250ULL,
        0x5f7570635f636574ULL


//    0x5f7463656a6f7250ULL,  // bytes 0-7 reversed in little-endian order
 //   0x636e65622020757063ULL, // bytes 8-15 reversed
 //   0x00006b72616d6868ULL,   // bytes 16-21 + padding zeros
 //   0x0ULL                   // padding limb
};


    uint64_t cipher[MAX_DIGITS] = {0};
    uint64_t decrypted[MAX_DIGITS] = {0};

    uint64_t public_key[MAX_DIGITS] = {65537};

    uint64_t modulus[MAX_DIGITS] = {
        0xD0AA1968DBB143D1ULL,
        0x6D9676BCCC70118BULL,
        0x4AFD724BF28E7836ULL,
        0xE103ABD94892E3E7ULL
    };

    uint64_t private_key[MAX_DIGITS] = {
        0x6050309C18F85D85ULL,
        0xCCEE8EB70D2282DCULL,
        0x47C52EABB879F26CULL,
        0xC5148B1AD7C6D858ULL
    };

    // Device memory
    uint64_t *d_message, *d_cipher, *d_decrypted;
    uint64_t *d_public_key, *d_private_key, *d_modulus, *d_mu;

    cudaMalloc(&d_message,     MAX_DIGITS * sizeof(uint64_t));
    cudaMalloc(&d_cipher,      MAX_DIGITS * sizeof(uint64_t));
    cudaMalloc(&d_decrypted,   MAX_DIGITS * sizeof(uint64_t));
    cudaMalloc(&d_public_key,  MAX_DIGITS * sizeof(uint64_t));
    cudaMalloc(&d_private_key, MAX_DIGITS * sizeof(uint64_t));
    cudaMalloc(&d_modulus,     MAX_DIGITS * sizeof(uint64_t));
    cudaMalloc(&d_mu, 2 * MAX_DIGITS * sizeof(uint64_t));

    cudaMemcpy(d_message,     message,     MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_public_key,  public_key,  MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_private_key, private_key, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_modulus,     modulus,     MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu,          mu_host,          2 * MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Encrypt
    rsa_encrypt_kernel<<<1,1>>>(d_message, d_public_key, d_modulus, d_mu, d_cipher, MAX_DIGITS);
    cudaDeviceSynchronize();

    // Copy cipher back to host
    cudaMemcpy(cipher, d_cipher, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    printf("Encrypted message (hex):\n");
    for(int i = MAX_DIGITS - 1; i >= 0; i--) {
        printf("%016lx", cipher[i]);
    }
    printf("\n");

    // Decrypt
    rsa_decrypt_kernel<<<1,1>>>(d_cipher, d_private_key, d_modulus, d_mu, d_decrypted, MAX_DIGITS);
    cudaDeviceSynchronize();

    // Copy decrypted message back
    cudaMemcpy(decrypted, d_decrypted, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    printf("Decrypted message (hex):\n");
    for(int i = MAX_DIGITS - 1; i >= 0; i--) {
        printf("%016lx", decrypted[i]);
    }
    printf("\n");

    // Print as ASCII
    printf("Decrypted message (ASCII): ");
    unsigned char* bytes = (unsigned char*)decrypted;
    for(int i = 0; i < MAX_DIGITS * 8; i++) {
        if(bytes[i] >= 32 && bytes[i] <= 126) {
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
    cudaFree(d_mu);

    return 0;
}
