#ifndef MODEXP_CUH
#define MODEXP_CUH

#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_DIGITS 16

// Debug print function for device
__device__ void debug_print_limbs(const char* label, const uint64_t* limbs, int digits);

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
    // Clear result array
    for (int i = 0; i < 2 * digits; ++i) {
        res[i] = 0;
    }

    // Multiply each limb
    for (int i = 0; i < digits; ++i) {
        uint64_t carry = 0;
        uint64_t a_val = a[i];
        
        for (int j = 0; j < digits; ++j) {
            uint64_t b_val = b[j];
            uint64_t res_val = res[i+j];
            
            // Compute high and low parts of multiplication
            uint64_t hi, lo;
            asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a_val), "l"(b_val));
            asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a_val), "l"(b_val));
            
            // Add to existing result with carry
            uint64_t sum = lo + res_val + carry;
            res[i+j] = sum;
            carry = hi + (sum < lo ? 1 : 0);
        }
        
        // Store final carry
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

static __device__ void barrettReductionKernel(
    const uint64_t* x,        // Input number (2k digits)
    const uint64_t* m,        // Modulus (k digits)
    const uint64_t* mu,       // Barrett constant (k+1 digits)
    uint64_t* result,         // Result (k digits)
    int digits                // k
) {
    uint64_t q1[MAX_DIGITS + 1] = {0};    // k+1 digits
    uint64_t q2[MAX_DIGITS * 3] = {0};    // 3k digits for full multiplication
    uint64_t q3[MAX_DIGITS] = {0};
    uint64_t r1[MAX_DIGITS + 1] = {0};    // k+1 digits for mod operation
    uint64_t r2[MAX_DIGITS * 2] = {0};    // 2k digits for multiplication
    uint64_t temp[MAX_DIGITS * 2] = {0};   // Temporary array

    // Debug prints
   // debug_print_limbs("Barrett input x", x, 2 * digits);
   // debug_print_limbs("Barrett modulus m", m, digits);
    //debug_print_limbs("Barrett mu", mu, digits + 1);

    // q1 = floor(x / b^(k-1))
    // Copy the high k+1 digits of x into q1
    for (int i = 0; i < digits + 1; i++) {
        if (i + digits - 1 < 2 * digits) {
            q1[i] = x[i + digits - 1];
        }
    }
   // debug_print_limbs("q1 after copy", q1, digits + 1);

    // q2 = q1 * mu
    mul_limbs(q1, mu, q2, digits + 1);
   // debug_print_limbs("q2 after multiplication", q2, 2 * (digits + 1));

    // q3 = floor(q2 / b^(k+1))
    // Copy the high k digits of q2 into q3
    for (int i = 0; i < digits; i++) {
        if (i + digits + 1 < 3 * digits) {
            q3[i] = q2[i + digits + 1];
        }
    }
   // debug_print_limbs("q3 after division", q3, digits);

    // r1 = x mod b^(k+1)
    // Copy the low k+1 digits of x into r1
    for (int i = 0; i < digits + 1; i++) {
        r1[i] = x[i];
    }
   // debug_print_limbs("r1 after mod", r1, digits + 1);

    // r2 = (q3 * m) mod b^(k+1)
    mul_limbs(q3, m, r2, digits);
    // Only keep k+1 digits
    for (int i = digits + 1; i < 2 * digits; i++) {
        r2[i] = 0;
    }
   // debug_print_limbs("r2 after multiplication", r2, digits + 1);

    // result = r1 - r2
    // If r1 < r2, we need to add b^(k+1) to r1 first
    if (cmp_limbs(r1, r2, digits + 1) < 0) {
        uint64_t carry = 1;
        for (int i = digits + 1; i < 2 * digits; i++) {
            r1[i] = carry;
            carry = 0;
        }
    }
    sub_limbs(r1, r2, result, digits + 1);
  //  debug_print_limbs("After subtraction", result, digits + 1);

    // Final reduction if needed
    while (cmp_limbs(result, m, digits) >= 0) {
        sub_limbs(result, m, result, digits);
      //  debug_print_limbs("After reduction", result, digits);
    }
}

// RSA kernels
extern "C" {
    __global__ void rsa_encrypt_kernel(
        const uint64_t* message,
        const uint64_t* public_key,
        const uint64_t* modulus,
        const uint64_t* mu,
        uint64_t* cipher,
        int digits,
        int num_messages,
        volatile int* progress
    );

    __global__ void rsa_decrypt_kernel(
        const uint64_t* cipher,
        const uint64_t* private_key,
        const uint64_t* modulus,
        const uint64_t* mu,
        uint64_t* decrypted,
        int digits,
        int num_messages,
        volatile int* progress
    );
}

// String conversion utilities
void string_to_limbs(const char* str, uint64_t* limbs, int max_digits);
void limbs_to_string(const uint64_t* limbs, char* str, int max_digits);

#endif // MODEXP_CUH 
