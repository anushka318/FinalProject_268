#include <stdio.h>
#include <openssl/bn.h>
#include <time.h>

// BARRET REDUCTION
void barrett_reduce(BIGNUM *r, BIGNUM *a, BIGNUM *m, BIGNUM *mu, int k2, BN_CTX *ctx) {
    BIGNUM *q = BN_new();
    BIGNUM *tmp = BN_new();
    BIGNUM *r_tmp = BN_new();

    BN_mul(tmp, a, mu, ctx);   
    BN_rshift(q, tmp, k2);     
    BN_mul(r_tmp, q, m, ctx); 
    BN_sub(r, a, r_tmp);      

    while (BN_cmp(r, m) >= 0)  
        BN_sub(r, r, m);

    BN_free(q);
    BN_free(tmp);
    BN_free(r_tmp);
}

// MODULAR MULTIPLICATION
void mod_mul_barrett(BIGNUM *r, BIGNUM *a, BIGNUM *b, BIGNUM *mod, BIGNUM *mu, int k2, BN_CTX *ctx) {
    BIGNUM *tmp = BN_new();
    BN_mul(tmp, a, b, ctx);
    barrett_reduce(r, tmp, mod, mu, k2, ctx);
    BN_free(tmp);
}


// BARRET MODULAR EXPONENTIATION
void barrett_mod_exp(BIGNUM *result, BIGNUM *base, BIGNUM *exp, BIGNUM *mod, BN_CTX *ctx) {
    BN_CTX_start(ctx);
    BIGNUM *res = BN_CTX_get(ctx);
    BIGNUM *tmp = BN_CTX_get(ctx);
    BIGNUM *mu  = BN_CTX_get(ctx);
    BIGNUM *two_pow_2k = BN_CTX_get(ctx);

    int k = BN_num_bits(mod);     
    int k2 = 2 * k;            

    BN_copy(tmp, base);
    BN_one(res);                   

    // MU CALCULATION
    BN_lshift(two_pow_2k, BN_value_one(), k2);
    BN_div(mu, NULL, two_pow_2k, mod, ctx);

    int bits = BN_num_bits(exp);
    for (int i = bits - 1; i >= 0; i--) {
        mod_mul_barrett(res, res, res, mod, mu, k2, ctx );

	// Multiply only if the bit is 1
        if (BN_is_bit_set(exp, i)) {
        mod_mul_barrett(res, res, tmp, mod, mu, k2, ctx );
        }
    }

    BN_copy(result, res);
    BN_CTX_end(ctx);
}


// BARRET BENCHMARK FUNCTION
void benchmark_barrett(int bit_size, int trials) {
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *base = BN_new();
    BIGNUM *exp  = BN_new();
    BIGNUM *mod  = BN_new();
    BIGNUM *res  = BN_new();

    double total_time = 0;

    for (int i = 0; i < trials; i++) {
        BN_rand(base, bit_size, 0, 0);
        BN_rand(exp, bit_size, 0, 0);
        BN_generate_prime_ex(mod, bit_size, 1, NULL, NULL, NULL);  // Generate safe prime numbers
        clock_t start = clock();
        barrett_mod_exp(res, base, exp, mod, ctx);
        clock_t end = clock();

        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    printf("BARRET_MOD_EXP\n");
    printf("Bit size      : %d bits\n", bit_size);
    printf("No of Trials  : %d\n", trials);
    printf("Average time  : %.6f sec\n", total_time / trials);
    printf("=====================================\n");

    BN_free(base); 
    BN_free(exp); 
    BN_free(mod); 
    BN_free(res);
    BN_CTX_free(ctx);
}

// BN BENCHMARK FUNCTION
void benchmark_BN_MOD(int bit_size, int trials) {
    BN_CTX *ctx = BN_CTX_new();
    BIGNUM *base = BN_new();
    BIGNUM *exp  = BN_new();
    BIGNUM *mod  = BN_new();
    BIGNUM *res  = BN_new();

    double total_time = 0;

    for (int i = 0; i < trials; i++) {
        BN_rand(base, bit_size, 0, 0);
        BN_rand(exp, bit_size, 0, 0);
        BN_generate_prime_ex(mod, bit_size, 1, NULL, NULL, NULL);  

        clock_t start = clock();
        BN_mod_exp(res, base, exp, mod, ctx);
        clock_t end = clock();

        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }
    
    printf("BN_MOD_EXP\n");
    printf("Bit size      : %d bits\n", bit_size);
    printf("No of Trials  : %d\n", trials);
    printf("Average time  : %.6f sec\n", total_time / trials);
    printf("=====================================\n");

    BN_free(base); 
    BN_free(exp);
    BN_free(mod); 
    BN_free(res);
    BN_CTX_free(ctx);
}



int main() {

    int trials = 10;

    benchmark_barrett(256, trials);
    benchmark_barrett(512, trials);
    benchmark_barrett(1024, trials);
    
    printf("\n\n\n");

    benchmark_BN_MOD(256, trials);
    benchmark_BN_MOD(512, trials);
    benchmark_BN_MOD(1024, trials);


    return 0;
}

