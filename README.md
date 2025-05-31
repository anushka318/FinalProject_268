# FinalProject_268




- `modexp.cuh`: Header file containing declarations and inline implementations for basic limb operations and RSA function interfaces
- `modexp.cu`: Core implementation of modular arithmetic operations
- `rsa.cu`: RSA encryption/decryption implementation with example usage

Large integers are represented as arrays of 64-bit limbs (uint64_t). The maximum number of limbs is defined by `MAX_DIGITS` (32), supporting numbers up to 2048 bits.

- `add_limbs`: Addition with carry propagation
- `sub_limbs`: Subtraction with borrow handling
- `mul_limbs`: Multiplication using CUDA PTX assembly
- `cmp_limbs`: Comparison of multi-limb numbers
- `modexp_limbs`: Modular exponentiation for RSA


- `rsa_encrypt_kernel`: Encrypts a message using public key (e, n)
- `rsa_decrypt_kernel`: Decrypts a ciphertext using private key (d, n)

Compile command for collab
!nvcc -o rsa_cuda rsa.cu -O3

