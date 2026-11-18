#include "bfloat16.h"

#include <math.h>

/*
 * Exponent utilities.
 *
 * get_exponent: reports the *unbiased* exponent. Zero maps to INT16_MIN
 * (no meaningful exponent), Inf/NaN to INT16_MAX. Denormals report -126,
 * the effective exponent of the denormal range (the smallest denormal is
 * 2^-133, which is what ilogb would return).
 *
 * set_exponent: raw bit manipulation preserving sign + mantissa. NaN, Inf,
 * and zero are returned unchanged. Normal to normal is exact; normal to
 * denormal truncates shifted-out mantissa bits (documented, not rounded);
 * underflow past the denormal range yields signed zero; overflow yields
 * infinity.
 *
 * ldexp / exp2: true arithmetic (correctly rounded), implemented through
 * the double intermediate so edge cases behave like the rest of the lib.
 */

int16_t bfloat16_get_exponent(bfloat16_t value) {
    uint16_t biased = (uint16_t)((value.bits & BFLOAT16_EXP_MASK) >> 7);

    if (biased == 0U) {
        if ((value.bits & BFLOAT16_MANT_MASK) != 0U) {
            return -126; /* denormal */
        }
        return INT16_MIN; /* zero has no exponent */
    }
    if (biased == 0xFFU) {
        return INT16_MAX; /* Inf or NaN */
    }
    return (int16_t)((int)biased - BFLOAT16_EXP_BIAS);
}

bfloat16_t bfloat16_set_exponent(bfloat16_t value, int16_t exp) {
    uint16_t sign;
    uint16_t mant;

    /* NaN, Inf, and zero are left unchanged. */
    if (((value.bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK) ||
        ((value.bits & (uint16_t)~BFLOAT16_SIGN_MASK) == 0U)) {
        return value;
    }

    sign = (uint16_t)(value.bits & BFLOAT16_SIGN_MASK);
    mant = (uint16_t)(value.bits & BFLOAT16_MANT_MASK);

    if (exp >= 128) {
        return bfloat16_from_bits((uint16_t)(sign | BFLOAT16_EXP_MASK));
    }
    if (exp >= -126) {
        return bfloat16_from_bits(
            (uint16_t)(sign | (((uint16_t)(exp + BFLOAT16_EXP_BIAS)) << 7) |
                       mant));
    }

    /* Denormal target: the implicit leading 1 becomes explicit, then shifts
     * out. exp == -127 keeps bit 6, exp == -133 keeps bit 0, and exp == -134
     * shifts the whole significand out to signed zero. */
    {
        int shift = -126 - (int)exp;
        uint16_t full = (uint16_t)(mant | 0x0080U);
        uint16_t den =
            (shift >= 8) ? 0U : (uint16_t)((full >> shift) & BFLOAT16_MANT_MASK);
        return bfloat16_from_bits((uint16_t)(sign | den));
    }
}

bfloat16_t bfloat16_ldexp(bfloat16_t value, int exp) {
    /* Arithmetic scaling, correctly rounded through double. Handles zero,
     * denormals, overflow to Inf, and NaN/Inf propagation. */
    double scaled;
    if (bfloat16_isnan(value) || bfloat16_isinf(value) || bfloat16_iszero(value)) {
        return value;
    }
    scaled = ldexp(bfloat16_to_double(value), exp);
    return double_to_bfloat16(scaled);
}

bfloat16_t bfloat16_exp2(int16_t exp) {
    if (exp > 127) {
        return BFLOAT16_INFINITY;
    }
    if (exp < -133) {
        return BFLOAT16_ZERO; /* below half of the smallest denormal */
    }
    if (exp < -126) {
        /* Exact denormal power of two: single bit. */
        int shift = (int)exp + 133; /* 0..6 */
        return bfloat16_from_bits((uint16_t)(1U << (unsigned)shift));
    }
    return bfloat16_from_bits(
        (uint16_t)(((uint16_t)(exp + BFLOAT16_EXP_BIAS)) << 7));
}
