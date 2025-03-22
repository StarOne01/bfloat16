#ifndef LLVM_LIBC_BFLOAT16_T_H
#define LLVM_LIBC_BFLOAT16_T_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// First check if hardware/compiler has native support
#if defined(__BF16_MANT_DIG__) || defined(__SUPPORTS_BF16__)
  // Use compiler's built-in bfloat16 type if available
  typedef __bf16 bfloat16_t;
  #define BFLOAT16_NATIVE_SUPPORT 1
#elif defined(__ARM_FEATURE_BF16)
  // ARM BF16 intrinsic support
  typedef __bf16 bfloat16_t;
  #define BFLOAT16_NATIVE_SUPPORT 1
#else
  // Software implementation using uint16_t
  typedef struct {
    uint16_t bits;
  } bfloat16_t;
  #define BFLOAT16_NATIVE_SUPPORT 0
#endif

// Constants for bfloat16 (Brain Floating Point)
#define BFLOAT16_SIGN_MASK    0x8000U
#define BFLOAT16_EXP_MASK     0x7F80U
#define BFLOAT16_MANT_MASK    0x007FU
#define BFLOAT16_EXP_BIAS     127     // Same as float32
#define BFLOAT16_MANT_BITS    7
#define BFLOAT16_EXP_BITS     8

// Define special values
#define BFLOAT16_INFINITY      ((bfloat16_t){.bits = 0x7F80})
#define BFLOAT16_NEG_INFINITY  ((bfloat16_t){.bits = 0xFF80})
#define BFLOAT16_NAN           ((bfloat16_t){.bits = 0x7FC0})
#define BFLOAT16_ZERO          ((bfloat16_t){.bits = 0x0000})
#define BFLOAT16_NEG_ZERO      ((bfloat16_t){.bits = 0x8000})

// Function declarations
bfloat16_t float_to_bfloat16(float value);
float bfloat16_to_float(bfloat16_t value);
double bfloat16_to_double(bfloat16_t value);
bfloat16_t double_to_bfloat16(double value);

// Basic arithmetic operations
bfloat16_t bfloat16_add(bfloat16_t a, bfloat16_t b);
bfloat16_t bfloat16_subtract(bfloat16_t a, bfloat16_t b);
bfloat16_t bfloat16_multiply(bfloat16_t a, bfloat16_t b);
bfloat16_t bfloat16_divide(bfloat16_t a, bfloat16_t b);
bfloat16_t bfloat16_negate(bfloat16_t a);

// Comparison functions
int bfloat16_eq(bfloat16_t a, bfloat16_t b);
int bfloat16_lt(bfloat16_t a, bfloat16_t b);
int bfloat16_gt(bfloat16_t a, bfloat16_t b);
int bfloat16_isnan(bfloat16_t a);
int bfloat16_isinf(bfloat16_t a);

// Exponent bias utility functions
int16_t bfloat16_get_exponent(bfloat16_t value);
bfloat16_t bfloat16_set_exponent(bfloat16_t value, int16_t exp);

#ifdef __cplusplus
}
#endif

#endif // LLVM_LIBC_BFLOAT16_T_H