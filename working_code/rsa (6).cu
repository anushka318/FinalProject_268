#include "modexp.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_DIGITS 16
#define WARP_SIZE 32
#define MAX_MESSAGES_PER_BLOCK 8  // Reduced for better performance
#define BATCH_SIZE 64  // Smaller batch size for more frequent updates

// These will be set dynamically based on GPU properties
int BLOCK_SIZE = 256;
int NUM_BLOCKS = 32;

// Utility function to check CUDA errors
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Timer functions
void startTimer(cudaEvent_t* start, cudaEvent_t* stop) {
    CHECK_CUDA_ERROR(cudaEventCreate(start));
    CHECK_CUDA_ERROR(cudaEventCreate(stop));
    CHECK_CUDA_ERROR(cudaEventRecord(*start));
}

float stopTimer(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    return milliseconds;
}

// Function to optimize launch parameters based on GPU properties
void optimizeLaunchParameters() {
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0));
    
    // Calculate optimal block size (multiple of warp size)
    BLOCK_SIZE = 256;  // Fixed to a more conservative value
    
    // Calculate number of blocks based on multiprocessor count
    NUM_BLOCKS = prop.multiProcessorCount;  // One block per SM
    
    // Get available GPU memory
    size_t free_memory, total_memory;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_memory, &total_memory));
    
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Available GPU Memory: %.2f GB\n", free_memory / (1024.0 * 1024.0 * 1024.0));
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Selected Configuration:\n");
    printf("- Block Size: %d\n", BLOCK_SIZE);
    printf("- Number of Blocks: %d\n", NUM_BLOCKS);
}

// Verify host data
void print_host_data(const char* label, const uint64_t* data, int digits) {
    printf("%s:\n", label);
    for(int i = digits - 1; i >= 0; i--) {
        printf("%016lx", data[i]);
    }
    printf("\n");
}

// Debug print function for device
__device__ void debug_print_limbs(const char* label, const uint64_t* limbs, int digits) {
    printf("%s: ", label);
    for (int i = digits - 1; i >= 0; i--) {
        printf("%016llx", limbs[i]);
    }
    printf("\n");
}

// Verify kernel to check GPU data
__global__ void verify_data_kernel(
    const uint64_t* message,
    const uint64_t* public_key,
    const uint64_t* modulus,
    const uint64_t* mu,
    int digits
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        //printf("\nGPU Data Verification:\n");
        //debug_print_limbs("Message", message, digits);
        //debug_print_limbs("Public Key", public_key, digits);
        //debug_print_limbs("Modulus", modulus, digits);
        //debug_print_limbs("Mu", mu, MAX_DIGITS + 1);
    }
}

static __device__ void modexp_limbs(
    const uint64_t* base,
    const uint64_t* exp,
    const uint64_t* mod,
    const uint64_t* mu,
    uint64_t* result,
    int digits
) {
    uint64_t temp[MAX_DIGITS * 2];  // For multiplication result
    uint64_t base_copy[MAX_DIGITS];
    uint64_t squared[MAX_DIGITS * 2];
    
    // Debug prints
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        debug_print_limbs("ModExp base", base, digits);
        debug_print_limbs("ModExp exp", exp, digits);
        debug_print_limbs("ModExp mod", mod, digits);
    }
    
    // Initialize result = 1
    for (int i = 0; i < digits; i++) {
        result[i] = (i == 0) ? 1 : 0;
        base_copy[i] = base[i];
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        debug_print_limbs("Initial result", result, digits);
        debug_print_limbs("Base copy", base_copy, digits);
    }
    
    // Process each bit of the exponent
    for (int i = digits; i > 0; i--) {  // Changed from i = digits to i = digits - 1
        uint64_t e = exp[i];
        for (int bit = 63; bit >= 0; bit--) {
            // Square step
            mul_limbs(result, result, squared, digits);
            if (threadIdx.x == 0 && blockIdx.x == 0 && i == digits-1 && bit == 63) {
                debug_print_limbs("First square result", squared, 2 * digits);
            }
            
            barrettReductionKernel(squared, mod, mu, result, digits);
            if (threadIdx.x == 0 && blockIdx.x == 0 && i == digits-1 && bit == 63) {
              //  debug_print_limbs("First reduction result", result, digits);
            }
            
            // Multiply step (if current bit is 1)
            if ((e >> bit) & 1) {
                mul_limbs(result, base_copy, temp, digits);
                barrettReductionKernel(temp, mod, mu, result, digits);
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                  //  debug_print_limbs("After multiply and reduce", result, digits);
                }
            }
        }
        
        
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        debug_print_limbs("Final modexp result", result, digits);
    }
}

__global__ void rsa_encrypt_kernel(
    const uint64_t* message,
    const uint64_t* public_key,
    const uint64_t* modulus,
    const uint64_t* mu,
    uint64_t* cipher,
    int digits,
    int num_messages,
    volatile int* progress
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_messages) {
        int offset = idx * digits;
        
        if (idx == 0) {
            printf("\nEncrypting message %d:\n", idx);
            debug_print_limbs("Input message", message + offset, digits);
            debug_print_limbs("Public key", public_key, digits);
            debug_print_limbs("Modulus", modulus, digits);
        }
        
        modexp_limbs(
            message + offset,
            public_key,
            modulus,
            mu,
            cipher + offset,
            digits
        );
        
        if (idx == 0) {
            debug_print_limbs("Output cipher", cipher + offset, digits);
        }
        
        if (progress) {
            atomicAdd((int*)progress, 1);
        }
    }
}

__global__ void rsa_decrypt_kernel(
    const uint64_t* cipher,
    const uint64_t* private_key,
    const uint64_t* modulus,
    const uint64_t* mu,
    uint64_t* decrypted,
    int digits,
    int num_messages,
    volatile int* progress  // Add progress counter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_messages) {
        int offset = idx * digits;
        if (idx == 0) {
            printf("\nEncrypting message %d:\n", idx);
            debug_print_limbs("Modulus", modulus, digits);
        }

        modexp_limbs(
            cipher + offset,
            private_key,
            modulus,
            mu,
            decrypted + offset,
            digits
        );
        if (idx == 0) {
            debug_print_limbs("Output cipher", cipher + offset, digits);
        }
        
        if (progress) {
            atomicAdd((int*)progress, 1);
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

// Compute Barrett mu = floor(b^(2k) / m)
void compute_barrett_mu(const uint64_t* m, uint64_t* mu, int k) {
    // Clear mu array
    memset(mu, 0, (k + 1) * sizeof(uint64_t));
    
    // Find the highest non-zero limb in m
    int highest_limb = k - 1;
    while (highest_limb > 0 && m[highest_limb] == 0) {
        highest_limb--;
    }
    
    // Set the initial value in mu
    if (m[highest_limb] != 0) {
        // Calculate mu = 2^(128+64) / m[highest_limb]
        __uint128_t dividend = ((__uint128_t)1 << 127);  // Use 2^127 to avoid overflow
        __uint128_t quotient = dividend / m[highest_limb];
        
        // Store the result in mu
        mu[k] = (uint64_t)(quotient >> 64);
        mu[k-1] = (uint64_t)quotient;
        
        // Adjust for the full division
        for (int i = k-2; i >= 0; i--) {
            mu[i] = 0xFFFFFFFFFFFFFFFFULL;
        }
    }
    /*
    // Print the computed mu for debugging
    printf("\nComputed Barrett mu value:\n");
    for (int i = k; i >= 0; i--) {
        printf("%016lx", mu[i]);
    }
    printf("\n");
    */
}

int main() {
    // Initialize optimal launch parameters
    optimizeLaunchParameters();
    
    // Define test message
    uint64_t test_message[MAX_DIGITS] = {
        0x724250656a6f7250ULL,  // Explicitly initialize all values
        0x5f7570635f636574ULL,
        0x724250656a6f7250ULL,
        0x5f7570635f636574ULL,
        0, 0, 0, 0,            // Zero out remaining digits
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    uint64_t public_key[MAX_DIGITS] = {3, 0};  // Initialize with zeros
    for(int i = 2; i < MAX_DIGITS; i++) {
        public_key[i] = 0;
    }

    uint64_t modulus[MAX_DIGITS] = {
        0xD0AA1968DBB143D1ULL,
        0x6D9676BCCC70118BULL,
        0x4AFD724BF28E7836ULL,
        0xE103ABD94892E3E7ULL,
        0, 0, 0, 0,            // Zero out remaining digits
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    // Compute Barrett mu
    uint64_t mu[MAX_DIGITS + 1] = {0};  // Only needs k+1 digits
        // Print host data before any operations
    //printf("\nHost Data Verification:\n");
    //print_host_data("Test Message", test_message, MAX_DIGITS);
    //print_host_data("Public Key", public_key, MAX_DIGITS);
    //print_host_data("Modulus", modulus, MAX_DIGITS);
    compute_barrett_mu(modulus, mu, MAX_DIGITS);

    // Print host data before any operations
    printf("\nHost Data Verification:\n");
    //print_host_data("Test Message", test_message, MAX_DIGITS);
    //print_host_data("Public Key", public_key, MAX_DIGITS);
    //print_host_data("Modulus", modulus, MAX_DIGITS);
    //print_host_data("Computed Mu", mu, MAX_DIGITS + 1);

    // Allocate device memory
    uint64_t *d_message, *d_public_key, *d_modulus, *d_mu;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_message, MAX_DIGITS * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_public_key, MAX_DIGITS * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_modulus, MAX_DIGITS * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_mu, (MAX_DIGITS + 1) * sizeof(uint64_t)));  // Allocate k+1 digits

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_message, test_message, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_public_key, public_key, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_modulus, modulus, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_mu, mu, (MAX_DIGITS + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Verify data on GPU
    verify_data_kernel<<<1, 1>>>(d_message, d_public_key, d_modulus, d_mu, MAX_DIGITS);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Allocate space for result
    uint64_t *d_result;
    uint64_t host_result[MAX_DIGITS] = {0};
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, MAX_DIGITS * sizeof(uint64_t)));

    // Run single encryption
   // printf("\nRunning single encryption test...\n");
    rsa_encrypt_kernel<<<1, 1>>>(
        d_message,
        d_public_key,
        d_modulus,
        d_mu,
        d_result,
        MAX_DIGITS,
        1,  // Single message
        nullptr  // No progress counter needed
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA_ERROR(cudaMemcpy(host_result, d_result, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Print result
    printf("\nEncryption Result:\n");
    print_host_data("Encrypted Message", host_result, MAX_DIGITS);

    // Set up private key for decryption (this is just a test key - in practice, use proper RSA key generation)
    uint64_t private_key[MAX_DIGITS] = {
        0x0000000000000003ULL,  // Simple test private key matching public key
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    // Allocate device memory for private key and decrypted result
    uint64_t *d_private_key, *d_decrypted;
    CHECK_CUDA_ERROR(cudaMalloc(&d_private_key, MAX_DIGITS * sizeof(uint64_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_decrypted, MAX_DIGITS * sizeof(uint64_t)));

    // Initialize d_decrypted with zeros
    CHECK_CUDA_ERROR(cudaMemset(d_decrypted, 0, MAX_DIGITS * sizeof(uint64_t)));

    // Copy private key and encrypted result to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_private_key, private_key, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    printf("\nRunning decryption...\n");
    
    // Verify memory contents before decryption
    verify_data_kernel<<<1, 1>>>(d_result, d_private_key, d_modulus, d_mu, MAX_DIGITS);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Run decryption kernel
    rsa_decrypt_kernel<<<1, 1>>>(
        d_result,        // Encrypted data input
        d_private_key,   // Private key
        d_modulus,       // Same modulus as encryption
        d_mu,           // Same mu value as encryption
        d_decrypted,    // Output buffer
        MAX_DIGITS,     // Number of digits
        1,              // Single message
        nullptr         // No progress counter
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // Allocate space for result
    // Copy decrypted result back to host
    uint64_t decrypted_result[MAX_DIGITS] = {0};
    CHECK_CUDA_ERROR(cudaMemcpy(decrypted_result, d_decrypted, MAX_DIGITS * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Print decrypted result
    printf("\nDecryption Result:\n");
    print_host_data("Decrypted Message", decrypted_result, MAX_DIGITS);

    // Convert decrypted result to string
    char decrypted_str[MAX_DIGITS * 8 + 1];
    limbs_to_string(decrypted_result, decrypted_str, MAX_DIGITS);
    printf("\nDecrypted Text: %s\n", decrypted_str);

    // Additional cleanup for new allocations
    cudaFree(d_private_key);
    cudaFree(d_decrypted);

    // Cleanup
    cudaFree(d_message);
    cudaFree(d_public_key);
    cudaFree(d_modulus);
    cudaFree(d_mu);
    cudaFree(d_result);

    return 0;
}