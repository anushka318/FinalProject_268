/*

Use this command to execute in Google Collab
!nvcc -arch=sm_75 Barrett_GPU.cu -o Barrett_GPU
!./Barrett_GPU

*/

#include <stdio.h>    
#include <stdlib.h>   
#include <string.h>   
#include <cuda_runtime.h>
#include <stdint.h>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        exit(EXIT_FAILURE); \
    } \
}
//required to know if there are no kernel issues


// TIMERS FOR BENCHMARKING
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

// DEVICE INFORMATION
void deviceinfo() {
    cudaDeviceProp prop; 
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, 0)); 

    size_t free_memory, total_memory; 
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_memory, &total_memory)); 
    
    printf("GPU: %s\n", prop.name); 
  // printf("Compute : %d.%d\n", prop.major, prop.minor); 
   // printf(" GPU Mem: %.2f GB\n", free_memory / (1024.0 * 1024.0 * 1024.0));

}

// PRINT U64 
//prints needed to debug
void print_u64(const char* label, uint64_t value) {
    printf("%s: %llu (0x%llx)\n", label, value, value); 
}



// MU CALCULATION
uint64_t compute_barrett_mu(uint64_t n) {
    if (n == 0) { 
        fprintf(stderr, "Error: Moduluscannot be zero\n"); 
        exit(EXIT_FAILURE); 
    }
    if (n == 1) {
        return 0xFFFFFFFFFFFFFFFFULL; 
    }

    uint64_t mu_val = (uint64_t)((((__uint128_t)-1) / n) + 1);
    return mu_val; 
}

// BARRET REDUCTION FUNCTION
__device__ uint64_t barrett_reduce(uint64_t x_low, uint64_t x_high, uint64_t n, uint64_t mu) {

    __uint128_t X = ((__uint128_t)x_high << 64) | x_low;
    uint64_t tmp1_high_xl_mu = __umul64hi(x_low, mu);
    
    uint64_t tmp2_high_xh_mu = __umul64hi(x_high, mu);
    uint64_t tmp2_low_xh_mu = x_high * mu;

    uint64_t sum_mid_part = tmp2_low_xh_mu + tmp1_high_xl_mu;
    uint64_t carry_to_q = (sum_mid_part < tmp2_low_xh_mu); 

    uint64_t q_val = tmp2_high_xh_mu + carry_to_q;

    __uint128_t r = X - ((__uint128_t)q_val * n);

    while (r >= n) {
        r -= n;
    }

    return (uint64_t)r; 
}
__device__ uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t n, uint64_t mu) {
    __uint128_t product = (__uint128_t)a * b;
    return barrett_reduce((uint64_t)product, (uint64_t)(product >> 64), n, mu);
}
// BARRET MODULAR EXPONENTIATION
__device__ uint64_t modexp_barrett(uint64_t base, uint64_t exp, uint64_t n, uint64_t mu) {
    uint64_t result = 1; 
    base = base % n;     

    while (exp > 0) { 
        if (exp & 1) { 
            //result = barrett_reduce((uint64_t)prod, (uint64_t)(prod >> 64), n, mu);
            result = mod_mul(result, base, n, mu);

        }
        __uint128_t prod = (__uint128_t)base * base; 
        base = barrett_reduce((uint64_t)prod, (uint64_t)(prod >> 64), n, mu);
        exp >>= 1;
    }
    return result; 
}


// RSA LAUNCH FUNCTION
__global__ void rsa_barrett_kernel(uint64_t* out, uint64_t base, uint64_t exp, uint64_t n, uint64_t mu) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = modexp_barrett(base, exp, n, mu); 
    }
}

int main() {
    deviceinfo(); 

    uint64_t p = 61;
    uint64_t q = 53;
    uint64_t n = p * q;        
    uint64_t e = 17;           
    uint64_t d = 2753;         
    uint64_t msg = 123;        

    uint64_t mu = compute_barrett_mu(n); 

    printf("\nRSA Parameters:\n");
    print_u64("  Modulus (n)", n);
    print_u64("  Public Key (e)", e);
    print_u64("  Private Key (d)", d);
    print_u64("  Message", msg);
    print_u64("  Computed Mu", mu);

    uint64_t* d_result; 
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(uint64_t))); 

    cudaEvent_t start, stop;
    float total_encryption_time_ms = 0; 
    float total_decryption_time_ms = 0; 

    int num_iterations = 1000; 

    printf("\n === Running Benchmarking (%d Iterations) ===\n\n", num_iterations);

    for (int i = 0; i < num_iterations; ++i) { 
        startTimer(&start, &stop); 
        rsa_barrett_kernel<<<1, 1>>>(d_result, msg, e, n, mu); 
        uint64_t cipher;
        CHECK_CUDA_ERROR(cudaMemcpy(&cipher, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost)); 
        total_encryption_time_ms += stopTimer(start, stop); 

        startTimer(&start, &stop); 
        rsa_barrett_kernel<<<1, 1>>>(d_result, cipher, d, n, mu); 
        uint64_t decrypted;
        CHECK_CUDA_ERROR(cudaMemcpy(&decrypted, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        total_decryption_time_ms += stopTimer(start, stop);

        // ERROR VERIFICATION
        int err_cntr = 0;
        if (decrypted != msg) { 
            printf("\nVerification: FAILED! Decrypted message (%llu) does NOT match original (%llu).\n", decrypted, msg);
            ++err_cntr;
        }
        if(err_cntr > 0 ) {
            printf("\nVerification failed %d times.\n", err_cntr);
        }
        else if((err_cntr == 0) & (i == num_iterations - 1) ) {
            printf("\nAll %d iterations PASSED!\n", num_iterations);
        }


        // PRINT ONLY ON THE LAST ITERATION
        if (i == num_iterations - 1) {
            printf("\n--- Final Verification (from last iteration) ---\n");
            printf("Encrypted Result: %llu\n", cipher);
            printf("Decrypted Result: %llu\n", decrypted);
            if (decrypted == msg) { 
                printf("Verification: SUCCESS! Decrypted message matches original.\n");
            } else {
                printf("Verification: FAILED! Decrypted message (%llu) does NOT match original (%llu).\n", decrypted, msg);
            }
        }
    }

    printf("\n--- Benchmarking Summary ---\n");
    printf("Average Encryption Time: %.4f ms\n", total_encryption_time_ms / num_iterations);
    printf("Average Decryption Time: %.4f ms\n", total_decryption_time_ms / num_iterations);

    CHECK_CUDA_ERROR(cudaFree(d_result));
    return 0; 
}
