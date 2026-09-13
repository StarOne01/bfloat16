#include "bfloat16.h"

#include <limits.h>
#include <math.h>
#include <string.h>

/*
 * Exponent utilities.
 *
 * get_exponent: reports the *unbiased* exponent. Zero maps to INT16_MIN
 * (no meaningful exponent), Inf/NaN to INT16_MAX, denormals to -126 (the
 * exponent they would have if normalized).
 *
 * set_exponent: raw bit manipulation preserving sign + mantissa. Normal to
 * normal is exact; normal to denormal truncates shifted-out mantissa bits
 * (documented, not rounded); underflow past the denormal range yields
 * signed zero; overflow yields infinity.
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
    bfloat16_t result;
    uint16_t sign;
    uint16_t mant;

    /* NaN / Inf: keep payload, only the caller's range decides Inf. */
    if (((value.bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK)) {
        if ((value.bits & BFLOAT16_MANT_MASK) != 0U) {
            return value; /* NaN stays NaN, exponent request ignored */
        }
        if (exp >= 128) {
            return value; /* Inf stays Inf */
        }
        /* Finite exponent on an infinity input: smallest/largest normal
         * with the same sign and zero mantissa. */
        if (exp <= -127) {
            result.bits = (uint16_t)(value.bits & BFLOAT16_SIGN_MASK);
            return result;
        }
        result.bits =
            (uint16_t)((value.bits & BFLOAT16_SIGN_MASK) |
                       (((uint16_t)(exp + BFLOAT16_EXP_BIAS)) << 7));
        return result;
    }

    /* Zero input: nothing to carry, but honour Inf overflow. */
    if ((value.bits & (uint16_t)~BFLOAT16_SIGN_MASK) == 0U) {
        if (exp >= 128) {
            result.bits =
                (uint16_t)((value.bits & BFLOAT16_SIGN_MASK) | BFLOAT16_EXP_MASK);
            return result;
        }
        result.bits = (uint16_t)(value.bits & BFLOAT16_SIGN_MASK);
        return result;
    }

    sign = (uint16_t)(value.bits & BFLOAT16_SIGN_MASK);
    mant = (uint16_t)(value.bits & BFLOAT16_MANT_MASK);

    if (exp >= 128) {
        result.bits = (uint16_t)(sign | BFLOAT16_EXP_MASK);
        return result;
    }
    if (exp >= -126) {
        result.bits =
            (uint16_t)(sign | (((uint16_t)(exp + BFLOAT16_EXP_BIAS)) << 7) | mant);
        return result;
    }

    /* Denormal target: implicit leading 1 becomes explicit, then shift.
     * exp == -127 keeps bit 6, exp == -133 keeps bit 0, below that zero. */
    {
        int shift = -126 - (int)exp; /* in 1..? */
        uint16_t full = (uint16_t)(mant | 0x0080U);
        uint16_t den;
        if (shift >= 8) {
            /* shift == 8 keeps only the implied 1 at bit 0 only when the
             * mantissa was 0x00 and exp == -134? No: shift 8 drops all 8
             * bits -> zero, except handled below for the boundary. */
            if (shift == 8) {
                den = (uint16_t)(full >> 8); /* 0 or 1 */
            } else {
                den = 0;
            }
        } else {
            den = (uint16_t)((full >> shift) & BFLOAT16_MANT_MASK);
        }
        result.bits = (uint16_t)(sign | den);
        return result;
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
