#ifndef MODEXP_CUH
#define MODEXP_CUH

#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_DIGITS 32

// Basic limb operations
static __device__ __forceinline__ void add_limbs(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* res,
    int digits
) {
    uint64_t carry = 0;
    for (int i = 0; i < digits; ++i) {
        uint64_t sum = a[i] + b[i] + carry;
        res[i] = sum;
        carry = (sum < a[i]) || (sum - carry < b[i]);
    }
}

static __device__ __forceinline__ void sub_limbs(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* res,
    int digits
) {
    uint64_t borrow = 0;
    for (int i = 0; i < digits; ++i) {
        uint64_t diff;
        if (borrow > a[i]) {
            diff = (UINT64_MAX - b[i] + a[i] + 1);
            borrow = 1;
        } else {
            diff = a[i] - b[i] - borrow;
            borrow = (a[i] - borrow < b[i]);
        }
        res[i] = diff;
    }
}

static __device__ __forceinline__ void mul_limbs(
    const uint64_t* a,
    const uint64_t* b,
    uint64_t* res,
    int digits
) {
    for (int i = 0; i < 2 * digits; ++i) {
        res[i] = 0;
    }

    for (int i = 0; i < digits; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < digits; ++j) {
            uint64_t a_val = a[i];
            uint64_t b_val = b[j];
            uint64_t res_val = res[i+j];
            
            uint64_t hi, lo;
            asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a_val), "l"(b_val));
            asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a_val), "l"(b_val));
            
            uint64_t sum1 = lo + res_val;
            uint64_t carry1 = (sum1 < lo) ? 1 : 0;
            uint64_t sum2 = sum1 + carry;
            uint64_t carry2 = (sum2 < sum1) ? 1 : 0;
            
            res[i+j] = sum2;
            carry = hi + carry1 + carry2;
        }
        if (i + digits < 2 * digits) {
            res[i + digits] = carry;
        }
    }
}

static __device__ __forceinline__ int cmp_limbs(
    const uint64_t* a,
    const uint64_t* b,
    int digits
) {
    for (int i = digits - 1; i >= 0; --i) {
        if (a[i] != b[i]) { 
            return a[i] < b[i] ? -1 : 1;
        }
    }
    return 0;
}

// Modular exponentiation
static __device__ void modexp_limbs(
    const uint64_t* base,
    const uint64_t* exp,
    const uint64_t* mod,
    uint64_t* result,
    int digits
);

// RSA kernels
extern "C" {
    __global__ void rsa_encrypt_kernel(
        const uint64_t* message,
        const uint64_t* public_key,
        const uint64_t* modulus,
        uint64_t* cipher,
        int digits
    );

    __global__ void rsa_decrypt_kernel(
        const uint64_t* cipher,
        const uint64_t* private_key,
        const uint64_t* modulus,
        uint64_t* decrypted,
        int digits
    );
}

// String conversion utilities
void string_to_limbs(const char* str, uint64_t* limbs, int max_digits);
void limbs_to_string(const uint64_t* limbs, char* str, int max_digits);

#endif // MODEXP_CUH 