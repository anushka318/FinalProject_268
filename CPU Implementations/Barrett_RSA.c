#include <stdio.h>
#include <openssl/bn.h> 
#include <time.h>      
#include <stdlib.h>    
#include <string.h>    

// PRINT BIGNUM
void print_bignum(const char* label, const BIGNUM* bn) {
    char* bn_str = BN_bn2dec(bn);
    printf("%s: %s\n", label, bn_str);
    OPENSSL_free(bn_str);
}


// BARRET REDUCTION FUNCTION
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

// BARRET MOD EXP FUNCTION
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

    BN_lshift(two_pow_2k, BN_value_one(), k2);
    BN_div(mu, NULL, two_pow_2k, mod, ctx);

    int bits = BN_num_bits(exp);
    for (int i = bits - 1; i >= 0; i--) {
        mod_mul_barrett(res, res, res, mod, mu, k2, ctx );
        
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
        BN_generate_prime_ex(mod, bit_size, 1, NULL, NULL, NULL); 
        
        clock_t start = clock();
        barrett_mod_exp(res, base, exp, mod, ctx);
        clock_t end = clock();

        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    printf("BARRET_MOD_EXP (Custom)\n");
    printf("Bit size      : %d bits\n", bit_size);
    printf("No of Trials  : %d\n", trials);
    printf("Average time  : %.6f sec\n", total_time / CLOCKS_PER_SEC);
    printf("=====================================\n");

    BN_free(base);  
    BN_free(exp);   
    BN_free(mod);   
    BN_free(res);
    BN_CTX_free(ctx);
}

// BN MOD EXP BENCHMARK FUNCTION
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
    
    printf("BN_MOD_EXP (OpenSSL Built-in)\n");
    printf("Bit size      : %d bits\n", bit_size);
    printf("No of Trials  : %d\n", trials);
    printf("Average time  : %.6f sec\n", total_time / CLOCKS_PER_SEC);
    printf("=====================================\n");

    BN_free(base);  
    BN_free(exp);
    BN_free(mod);   
    BN_free(res);
    BN_CTX_free(ctx);
}



int main() {
    BN_CTX *ctx = BN_CTX_new();


    BIGNUM *p = BN_new();
    BIGNUM *q = BN_new();
    BIGNUM *n = BN_new();       
    BIGNUM *phi_n = BN_new();   
    BIGNUM *e = BN_new();       
    BIGNUM *d = BN_new();       
    BIGNUM *msg = BN_new();     
    BIGNUM *encrypted_msg = BN_new(); 
    BIGNUM *decrypted_msg = BN_new(); 

    int rsa_bit_size = 64; 

    printf("--- RSA Key Generation (%d bits) ---\n", rsa_bit_size);

    
    BN_generate_prime_ex(p, rsa_bit_size / 2, 0, NULL, NULL, NULL);
    BN_generate_prime_ex(q, rsa_bit_size / 2, 0, NULL, NULL, NULL);
    

    if (BN_cmp(p, q) == 0) {
        BN_generate_prime_ex(q, rsa_bit_size / 2, 0, NULL, NULL, NULL);
    }
    
    
    BN_mul(n, p, q, ctx);

    BIGNUM *p_minus_1 = BN_new();
    BIGNUM *q_minus_1 = BN_new();
    BN_sub(p_minus_1, p, BN_value_one());
    BN_sub(q_minus_1, q, BN_value_one());
    BN_mul(phi_n, p_minus_1, q_minus_1, ctx);

    // SET PUBLICK KEY EXPONENT
    BN_set_word(e, 65537);


    BN_mod_inverse(d, e, phi_n, ctx);
    
    printf("Key Generation Complete.\n");
    print_bignum("p", p);
    print_bignum("q", q);
    print_bignum("n", n);
    print_bignum("phi_n", phi_n);
    print_bignum("e", e);
    print_bignum("d", d);
    printf("----------------------------------\n\n");

    // BENCHMARKING 64 BIT RSA BARRET
    printf("--- RSA Demonstration (%d bits) ---\n", rsa_bit_size);
    BN_set_word(msg, 12345); 
    printf("Original Message: %s\n", BN_bn2dec(msg));

    
    clock_t start_enc_barrett = clock();
    barrett_mod_exp(encrypted_msg, msg, e, n, ctx);
    clock_t end_enc_barrett = clock();
    printf("Encrypted Message (Custom Barrett): %s\n", BN_bn2dec(encrypted_msg));
    printf("Encryption Time (Custom Barrett): %.6f sec\n", (double)(end_enc_barrett - start_enc_barrett) / CLOCKS_PER_SEC);

    clock_t start_dec_barrett = clock();
    barrett_mod_exp(decrypted_msg, encrypted_msg, d, n, ctx);
    clock_t end_dec_barrett = clock();
    printf("Decrypted Message (Custom Barrett): %s\n", BN_bn2dec(decrypted_msg));
    printf("Decryption Time (Custom Barrett): %.6f sec\n", (double)(end_dec_barrett - start_dec_barrett) / CLOCKS_PER_SEC);
    if (BN_cmp(msg, decrypted_msg) == 0) {
        printf("Verification (Custom Barrett): SUCCESS!\n");
    } else {
        printf("Verification (Custom Barrett): FAILED!\n");
    }
    printf("----------------------------------\n\n");

    // BENCHMARKING 64 BIT RSA OPENSSL BN
    printf("--- RSA Demonstration with OpenSSL Built-in (%d bits) ---\n", rsa_bit_size);
    
    clock_t start_enc_bn_mod = clock();
    BN_mod_exp(encrypted_msg, msg, e, n, ctx);
    clock_t end_enc_bn_mod = clock();
    printf("Encrypted Message (OpenSSL Built-in): %s\n", BN_bn2dec(encrypted_msg));
    printf("Encryption Time (OpenSSL Built-in): %.6f sec\n", (double)(end_enc_bn_mod - start_enc_bn_mod) / CLOCKS_PER_SEC);

    clock_t start_dec_bn_mod = clock();
    BN_mod_exp(decrypted_msg, encrypted_msg, d, n, ctx);
    clock_t end_dec_bn_mod = clock();
    printf("Decrypted Message (OpenSSL Built-in): %s\n", BN_bn2dec(decrypted_msg));
    printf("Decryption Time (OpenSSL Built-in): %.6f sec\n", (double)(end_dec_bn_mod - start_dec_bn_mod) / CLOCKS_PER_SEC);
    if (BN_cmp(msg, decrypted_msg) == 0) {
        printf("Verification (OpenSSL Built-in): SUCCESS!\n");
    } else {
        printf("Verification (OpenSSL Built-in): FAILED!\n");
    }
    printf("----------------------------------\n\n");

    BN_free(p);
    BN_free(q);
    BN_free(n);
    BN_free(phi_n);
    BN_free(e);
    BN_free(d);
    BN_free(msg);
    BN_free(encrypted_msg);
    BN_free(decrypted_msg);
    BN_free(p_minus_1); 
    BN_free(q_minus_1); 
    BN_CTX_free(ctx);

    return 0;
}
