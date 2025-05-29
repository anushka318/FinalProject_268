#include <stdio.h>
#include <openssl/bn.h>

#define NBITS 128
#include <time.h>
#include <stdio.h>

struct timespec start, end;


//////////////////////////////////////////////
//Barett ciode
void printBN(char *msg, BIGNUM * a)
{
   /* Use BN_bn2hex(a) for hex string
    * Use BN_bn2dec(a) for decimal string */
   char * number_str = BN_bn2hex(a);
   printf("%s %s\n", msg, number_str);
   OPENSSL_free(number_str);
}
void barrett_reduction(BIGNUM *result, const BIGNUM *x, const BIGNUM *m, BN_CTX *ctx) {


    int k = BN_num_bits(m); 
    BIGNUM *u = BN_new();       
    BIGNUM *q1 = BN_new();
    BIGNUM *q2 = BN_new();
    BIGNUM *r1 = BN_new();
    BIGNUM *r2 = BN_new();
    BIGNUM *tmp = BN_new();
    BIGNUM *power = BN_new();   // b^(2k)
    
    // b^(2k)
    BN_lshift(power, BN_value_one(), 2 * k);

    // u = floor(b^(2k) / m)
    BN_div(u, NULL, power, m, ctx);

    // q1 = floor(x / b^(k-1))
    BN_rshift(q1, x, k - 1);

    // q2 = q1 * u
    BN_mul(q2, q1, u, ctx);

    // q3 = floor(q2 / b^(k+1))
    BN_rshift(q2, q2, k + 1);

     BN_copy(r1, x);
       BN_mask_bits(r1, k + 1);

// r2 = (q2 * m) mod 2^(k+1)
     BN_mul(tmp, q2, m, ctx);
	BN_copy(r2, tmp);
    BN_mask_bits(r2, k + 1);    // r = r1 - r2
    BN_sub(result, r1, r2);

    if (BN_is_negative(result)) {
        BN_add(result, result, power);
    }

    while (BN_cmp(result, m) >= 0) {
        BN_sub(result, result, m);
    }

    BN_free(u);
    BN_free(q1);
    BN_free(q2);
    BN_free(r1);
    BN_free(r2);
    BN_free(tmp);
    BN_free(power);
}

int main() {
	clock_gettime(CLOCK_MONOTONIC, &start);
    BN_CTX *ctx = BN_CTX_new();

    BIGNUM *msg = BN_new();
    BIGNUM *e = BN_new();
    BIGNUM *d = BN_new();
    BIGNUM *n = BN_new();
    BIGNUM *cipher = BN_new();
    BIGNUM *decrypted = BN_new();
//message is Project_cpu benchmark
//
    BN_hex2bn(&msg, "50726f6a6563745f6370752062656e63686d61726b");                        
    BN_dec2bn(&e, "65537");                             
    BN_hex2bn(&n, "E103ABD94892E3E74AFD724BF28E78366D9676BCCC70118BD0AA1968DBB143D1");                        
    BN_hex2bn(&d, "C5148B1AD7C6D85847C52EABB879F26CCCEE8EB70D2282DC6050309C18F85D85");                        

    // Encrypt
    BN_mod_exp(cipher, msg, e, n, ctx);
    printBN("cipher",cipher);
    // Decrypt
    BN_mod_exp(decrypted, cipher, d, n, ctx);
    printBN("decrypt",decrypted);
    // Clean up
    BN_free(msg);
    BN_free(e);
    BN_free(d);
    BN_free(n);
    BN_free(cipher);
    BN_free(decrypted);
    BN_CTX_free(ctx);

clock_gettime(CLOCK_MONOTONIC, &end);
double time_taken = (end.tv_sec - start.tv_sec) * 1e3 +
                    (end.tv_nsec - start.tv_nsec) / 1e6; // in milliseconds

printf(" time: %.3f ms\n", time_taken);


}
////////////////////////////////////////////////////////////


