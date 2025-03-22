#include "bfloat16.h"
#include <stdint.h>

// Extract sign, exponent, and mantissa from bfloat16
static void bfloat16_unpack(bfloat16_t value, uint16_t* sign, int16_t* exp, uint16_t* mant) {
    *sign = (value.bits & BFLOAT16_SIGN_MASK) >> 15;
    *exp = ((value.bits & BFLOAT16_EXP_MASK) >> 7);
    *mant = (value.bits & BFLOAT16_MANT_MASK);
    
    // Add implied leading 1 for normalized numbers
    if (*exp > 0 && *exp < 255) {
        *mant |= 0x0080; // 1.mantissa (add implied bit)
    }
}

// Pack sign, exponent, and mantissa into bfloat16
static bfloat16_t bfloat16_pack(uint16_t sign, int16_t exp, uint16_t mant) {
    bfloat16_t result;
    
    // Handle special cases
    if (exp > 254) {
        // Infinity or NaN
        result.bits = sign << 15 | 0x7F80; // Exponent all 1's
        if (mant != 0) {
            result.bits |= 0x40; // Make it a quiet NaN
        }
        return result;
    }
    
    if (exp < 1) {
        // Zero or denormal
        if (mant == 0) {
            result.bits = sign << 15; // Zero with appropriate sign
            return result;
        }
        
        // Handle denormals: shift until leading 1 is in position
        while ((mant & 0x0080) == 0) {
            mant <<= 1;
            exp--;
        }
        
        // Remove implied 1
        mant &= 0x007F;
        
        // Check if denormal underflowed to zero
        if (exp < -126) {
            result.bits = sign << 15; // Underflow to signed zero
            return result;
        }
        
        exp++; // Adjust exponent for denormal
    } else {
        // Remove implied 1 for normal numbers
        mant &= 0x007F;
    }
    
    // Combine components
    result.bits = (sign << 15) | ((exp & 0xFF) << 7) | (mant & 0x007F);
    return result;
}

// Addition implementation without using float
bfloat16_t bfloat16_add(bfloat16_t a, bfloat16_t b) {
    // Special case handling
    if ((a.bits & BFLOAT16_EXP_MASK) == 0) return b; // a is zero
    if ((b.bits & BFLOAT16_EXP_MASK) == 0) return a; // b is zero
    
    // Check for NaN
    if ((a.bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK && 
        (a.bits & BFLOAT16_MANT_MASK) != 0) return a; // a is NaN
    if ((b.bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK && 
        (b.bits & BFLOAT16_MANT_MASK) != 0) return b; // b is NaN
    
    // Extract components
    uint16_t sign_a, sign_b, sign_r;
    int16_t exp_a, exp_b, exp_r;
    uint16_t mant_a, mant_b, mant_r;
    
    bfloat16_unpack(a, &sign_a, &exp_a, &mant_a);
    bfloat16_unpack(b, &sign_b, &exp_b, &mant_b);
    
    // Align exponents
    int16_t exp_diff = exp_a - exp_b;
    if (exp_diff > 0) {
        // Shift b's mantissa
        if (exp_diff > 8) {
            return a; // b is too small to matter
        }
        mant_b >>= exp_diff;
        exp_r = exp_a;
    } else if (exp_diff < 0) {
        // Shift a's mantissa
        if (exp_diff < -8) {
            return b; // a is too small to matter
        }
        mant_a >>= -exp_diff;
        exp_r = exp_b;
    } else {
        // Same exponent
        exp_r = exp_a;
    }
    
    // Perform addition or subtraction based on signs
    if (sign_a == sign_b) {
        // Same signs: add mantissas
        sign_r = sign_a;
        mant_r = mant_a + mant_b;
        
        // Handle overflow in mantissa
        if (mant_r & 0x0100) {
            mant_r >>= 1;
            exp_r++;
        }
    } else {
        // Different signs: subtract smaller from larger
        if (mant_a > mant_b) {
            sign_r = sign_a;
            mant_r = mant_a - mant_b;
        } else if (mant_b > mant_a) {
            sign_r = sign_b;
            mant_r = mant_b - mant_a;
        } else {
            // Equal magnitude, result is zero
            bfloat16_t zero = {.bits = 0};
            return zero;
        }
        
        // Normalize result (shift left until leading 1)
        if (mant_r != 0) {
            while ((mant_r & 0x0080) == 0) {
                mant_r <<= 1;
                exp_r--;
            }
        } else {
            // Result is zero
            bfloat16_t zero = {.bits = 0};
            return zero;
        }
    }
    
    // Pack result
    return bfloat16_pack(sign_r, exp_r, mant_r);
}

// Example of updating a multiplication function to explicitly handle bias

bfloat16_t bfloat16_multiply(bfloat16_t a, bfloat16_t b) {
#if BFLOAT16_NATIVE_SUPPORT
    return float_to_bfloat16((float)a * (float)b);
#else
    // Handle special cases like zero, infinity, NaN
    if (bfloat16_isnan(a) || bfloat16_isnan(b)) return BFLOAT16_NAN;
    
    // Extract components
    uint16_t sign_a = (a.bits & BFLOAT16_SIGN_MASK) ? 1 : 0;
    uint16_t sign_b = (b.bits & BFLOAT16_SIGN_MASK) ? 1 : 0;
    uint16_t sign_r = sign_a ^ sign_b;
    
    // Extract biased exponents
    int16_t biased_exp_a = (a.bits & BFLOAT16_EXP_MASK) >> 7;
    int16_t biased_exp_b = (b.bits & BFLOAT16_EXP_MASK) >> 7;
    
    // Get actual exponents by removing bias
    int16_t exp_a = biased_exp_a ? (biased_exp_a - BFLOAT16_EXP_BIAS) : -126;
    int16_t exp_b = biased_exp_b ? (biased_exp_b - BFLOAT16_EXP_BIAS) : -126;
    
    // Calculate result exponent
    int16_t exp_r = exp_a + exp_b;
    
    // Extract mantissas (add implied 1 for normalized numbers)
    uint16_t mant_a = biased_exp_a ? 
        ((a.bits & BFLOAT16_MANT_MASK) | 0x80) : (a.bits & BFLOAT16_MANT_MASK);
    uint16_t mant_b = biased_exp_b ? 
        ((b.bits & BFLOAT16_MANT_MASK) | 0x80) : (b.bits & BFLOAT16_MANT_MASK);
    
    // Multiply mantissas
    uint32_t mant_r = (uint32_t)mant_a * mant_b;
    
    // Normalize result
    if (mant_r != 0) {
        // Find position of leading 1
        if (mant_r & 0x00008000) {
            // Result has 15 significant bits, adjust exponent
            mant_r >>= 8;
            exp_r += 1;
        } else {
            mant_r >>= 7;
        }
        
        // Apply bias to get biased exponent
        int16_t biased_exp_r;
        if (exp_r <= -126) {
            // Denormal result
            biased_exp_r = 0;
            // Shift mantissa right to create denormal
            mant_r >>= (1 - (exp_r + 126));
        } else if (exp_r >= 128) {
            // Overflow to infinity
            return (bfloat16_t){.bits = static_cast<uint16_t>((sign_r << 15) | 0x7F80)};
        } else {
            // Normal number
            biased_exp_r = exp_r + BFLOAT16_EXP_BIAS;
        }
        
        // Assemble result
        bfloat16_t result;
        result.bits = (sign_r << 15) | (biased_exp_r << 7) | (mant_r & BFLOAT16_MANT_MASK);
        return result;
    } else {
        // Result is zero
        return (bfloat16_t){.bits = static_cast<uint16_t>(sign_r << 15)};
    }
#endif
}

// Division implementation without using float
bfloat16_t bfloat16_divide(bfloat16_t a, bfloat16_t b) {
    // Special case handling
    if ((a.bits & ~BFLOAT16_SIGN_MASK) == 0) {
        // 0 / x = 0 (with appropriate sign)
        if ((b.bits & ~BFLOAT16_SIGN_MASK) == 0) {
            // 0/0 = NaN
            return BFLOAT16_NAN;
        }
        return (bfloat16_t){.bits = static_cast<uint16_t>(((a.bits ^ b.bits) & BFLOAT16_SIGN_MASK))};
    }
    
    if ((b.bits & ~BFLOAT16_SIGN_MASK) == 0) {
        // x / 0 = Infinity (with appropriate sign)
        return (bfloat16_t){.bits = static_cast<uint16_t>(((a.bits ^ b.bits) & BFLOAT16_SIGN_MASK) | 0x7F80)};
    }
    
    // Extract components
    uint16_t sign_a, sign_b, sign_r;
    int16_t exp_a, exp_b, exp_r;
    uint16_t mant_a, mant_b;
    uint32_t mant_r;
    
    bfloat16_unpack(a, &sign_a, &exp_a, &mant_a);
    bfloat16_unpack(b, &sign_b, &exp_b, &mant_b);
    
    // Result sign is XOR of input signs
    sign_r = sign_a ^ sign_b;
    
    // Subtract exponents (add bias)
    exp_r = exp_a - exp_b + BFLOAT16_EXP_BIAS;
    
    // Division by shifting and subtraction
    mant_a <<= 8; // Shift for precision
    mant_r = ((uint32_t)mant_a / (uint32_t)mant_b);
    
    // Normalize result
    if (mant_r == 0) {
        // Result is zero
        return (bfloat16_t){.bits = static_cast<uint16_t>(sign_r << 15)};
    }
    
    // Find position of leading 1
    int shift = 0;
    uint32_t temp = mant_r;
    while (temp & 0xFF00) {
        temp >>= 1;
        shift++;
    }
    
    if (shift > 0) {
        mant_r >>= shift;
        exp_r += shift;
    } else {
        while ((mant_r & 0x0080) == 0) {
            mant_r <<= 1;
            exp_r--;
        }
    }
    
    // Pack result
    return bfloat16_pack(sign_r, exp_r, mant_r & 0x00FF);
}

// Subtraction implementation without using float
bfloat16_t bfloat16_subtract(bfloat16_t a, bfloat16_t b) {
    // Negate b and add
    bfloat16_t neg_b = {.bits = static_cast<uint16_t>(b.bits ^ BFLOAT16_SIGN_MASK)};
    return bfloat16_add(a, neg_b);
}