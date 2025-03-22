#include "bfloat16.h"

// Equality comparison
int bfloat16_eq(bfloat16_t a, bfloat16_t b) {
#if BFLOAT16_NATIVE_SUPPORT
  return (float)a == (float)b;
#else
  // Handle NaN cases
  if ((a.bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK && 
      (a.bits & BFLOAT16_MANT_MASK) != 0) return 0;
  if ((b.bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK && 
      (b.bits & BFLOAT16_MANT_MASK) != 0) return 0;
      
  // Handle signed zeros
  if (((a.bits & ~BFLOAT16_SIGN_MASK) == 0) && 
      ((b.bits & ~BFLOAT16_SIGN_MASK) == 0)) return 1;
      
  return a.bits == b.bits;
#endif
}

// Implementation of isnan check
int bfloat16_isnan(bfloat16_t a) {
#if BFLOAT16_NATIVE_SUPPORT
  return isnan((float)a);
#else
  return (a.bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK && 
         (a.bits & BFLOAT16_MANT_MASK) != 0;
#endif
}

// Less than comparison
int bfloat16_lt(bfloat16_t a, bfloat16_t b) {
#if BFLOAT16_NATIVE_SUPPORT
  return (float)a < (float)b;
#else
  // Handle NaN cases
  if ((a.bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK && 
      (a.bits & BFLOAT16_MANT_MASK) != 0) return 0;
  if ((b.bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK && 
      (b.bits & BFLOAT16_MANT_MASK) != 0) return 0;

  // Handle sign differences
  bool a_neg = (a.bits & BFLOAT16_SIGN_MASK);
  bool b_neg = (b.bits & BFLOAT16_SIGN_MASK);
  
  if (a_neg != b_neg)
    return a_neg && ((a.bits | b.bits) & ~BFLOAT16_SIGN_MASK); // Check if not both zero

  // Same sign, compare magnitude
  return a_neg ? (a.bits > b.bits) : (a.bits < b.bits);
#endif
}

// Greater than comparison
int bfloat16_gt(bfloat16_t a, bfloat16_t b) {
#if BFLOAT16_NATIVE_SUPPORT
  return (float)a > (float)b;
#else
  return bfloat16_lt(b, a);
#endif
}

// Check for infinity
int bfloat16_isinf(bfloat16_t a) {
#if BFLOAT16_NATIVE_SUPPORT
  return isinf((float)a);
#else
  return (a.bits & ~BFLOAT16_SIGN_MASK) == 0x7F80;
#endif
}