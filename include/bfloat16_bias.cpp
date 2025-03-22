#include "bfloat16.h"
#include <math.h>

// Extract the unbiased exponent (actual exponent, not the biased representation)
int16_t bfloat16_get_exponent(bfloat16_t value) {
#if BFLOAT16_NATIVE_SUPPORT
    float f = (float)value;
    int32_t bits;
    memcpy(&bits, &f, sizeof(float));
    
    // Extract biased exponent
    int16_t biased_exp = (bits >> 23) & 0xFF;
    
    // Handle special cases
    if (biased_exp == 0) {
        return (bits & 0x007FFFFF) ? -126 : -INT16_MAX; // Denormal or zero
    } else if (biased_exp == 0xFF) {
        return INT16_MAX; // Infinity or NaN
    }
    
    // Return unbiased exponent
    return biased_exp - BFLOAT16_EXP_BIAS;
#else
    // Extract biased exponent
    uint16_t biased_exp = (value.bits & BFLOAT16_EXP_MASK) >> 7;
    
    // Handle special cases
    if (biased_exp == 0) {
        return (value.bits & BFLOAT16_MANT_MASK) ? -126 : -INT16_MAX; // Denormal or zero
    } else if (biased_exp == 0xFF) {
        return INT16_MAX; // Infinity or NaN
    }
    
    // Return unbiased exponent
    return biased_exp - BFLOAT16_EXP_BIAS;
#endif
}

// Set exponent to a specified value (takes an actual exponent, not biased)
bfloat16_t bfloat16_set_exponent(bfloat16_t value, int16_t exp) {
#if BFLOAT16_NATIVE_SUPPORT
    // Convert to float, modify, and convert back
    float f = (float)value;
    int32_t bits;
    memcpy(&bits, &f, sizeof(float));
    
    // Clear exponent bits
    bits &= ~(0xFF << 23);
    
    // Handle special cases
    if (exp <= -126) {
        // Denormal or zero, keep mantissa and sign only
        return float_to_bfloat16(f * powf(2.0f, exp + 126));  // Use scaling to create denormal
    } else if (exp >= 128) {
        // Infinity
        uint16_t sign_bit = value.bits & BFLOAT16_SIGN_MASK;
        return (bfloat16_t){.bits = sign_bit | 0x7F80};
    }
    
    // Set new exponent
    bits |= ((exp + BFLOAT16_EXP_BIAS) & 0xFF) << 23;
    
    // Convert back to float and then to bfloat16
    memcpy(&f, &bits, sizeof(float));
    return float_to_bfloat16(f);
#else
    bfloat16_t result = value;
    
    // Clear exponent bits
    result.bits &= ~BFLOAT16_EXP_MASK;
    
    // Handle special cases
    if (exp <= -126) {
        // Try to create denormal by shifting mantissa
        uint16_t mant = value.bits & BFLOAT16_MANT_MASK;
        uint16_t shift = 1 - (exp + 126);
        
        if (shift >= BFLOAT16_MANT_BITS) {
            // Underflow to zero
            result.bits &= BFLOAT16_SIGN_MASK; // Keep sign only
        } else {
            // Denormal
            mant >>= shift;
            result.bits |= mant;
        }
        return result;
    } else if (exp >= 128) {
        // Infinity
        return (bfloat16_t){.bits = static_cast<uint16_t>((value.bits & BFLOAT16_SIGN_MASK) | 0x7F80)};
    }
    
    // Set new exponent (biased)
    result.bits |= ((exp + BFLOAT16_EXP_BIAS) & 0xFF) << 7;
    return result;
#endif
}

// Scale a bfloat16 value by a power of 2 (effectively adds to exponent)
bfloat16_t bfloat16_ldexp(bfloat16_t value, int exp) {
#if BFLOAT16_NATIVE_SUPPORT
    return float_to_bfloat16(ldexpf((float)value, exp));
#else
    int16_t current_exp = bfloat16_get_exponent(value);
    return bfloat16_set_exponent(value, current_exp + exp);
#endif
}

// Get 2^exp as a bfloat16 (inverse of log2)
bfloat16_t bfloat16_exp2(int16_t exp) {
#if BFLOAT16_NATIVE_SUPPORT
    return float_to_bfloat16(exp2f((float)exp));
#else
    if (exp < -126) {
        // Too small, return zero
        return BFLOAT16_ZERO;
    } else if (exp > 127) {
        // Too large, return infinity
        return BFLOAT16_INFINITY;
    }
    
    // 2^exp = 1.0 * 2^exp
    bfloat16_t one = {.bits = 0x3F80}; // 1.0 in bfloat16
    return bfloat16_set_exponent(one, exp);
#endif
}