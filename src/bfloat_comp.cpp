#include "bfloat16.h"

/*
 * Comparisons and classification, implemented directly on the bit pattern.
 *
 * bfloat16 shares float32's ordering trick: for same-sign finite values,
 * unsigned integer comparison of the raw bits matches value comparison.
 * NaN is the only unordered case and is checked first; signed zeros are
 * normalized so +0 == -0.
 *
 * All predicates return exactly 0 or 1. No <stdbool.h> / <math.h> needed,
 * so this TU stays freestanding-friendly.
 */

static int bf16_is_nan_bits(uint16_t bits) {
    return ((bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK) &&
           ((bits & BFLOAT16_MANT_MASK) != 0);
}

int bfloat16_isnan(bfloat16_t a) {
    return bf16_is_nan_bits(a.bits) ? 1 : 0;
}

int bfloat16_isinf(bfloat16_t a) {
    return ((a.bits & (uint16_t)~BFLOAT16_SIGN_MASK) == BFLOAT16_EXP_MASK) ? 1 : 0;
}

int bfloat16_isfinite(bfloat16_t a) {
    return ((a.bits & BFLOAT16_EXP_MASK) != BFLOAT16_EXP_MASK) ? 1 : 0;
}

int bfloat16_iszero(bfloat16_t a) {
    return ((a.bits & (uint16_t)~BFLOAT16_SIGN_MASK) == 0) ? 1 : 0;
}

int bfloat16_issubnormal(bfloat16_t a) {
    return (((a.bits & BFLOAT16_EXP_MASK) == 0) &&
            ((a.bits & BFLOAT16_MANT_MASK) != 0)) ? 1 : 0;
}

int bfloat16_signbit(bfloat16_t a) {
    return ((a.bits & BFLOAT16_SIGN_MASK) != 0) ? 1 : 0;
}

int bfloat16_eq(bfloat16_t a, bfloat16_t b) {
    if (bf16_is_nan_bits(a.bits) || bf16_is_nan_bits(b.bits)) {
        return 0;
    }
    if (bfloat16_iszero(a) && bfloat16_iszero(b)) {
        return 1; /* +0 == -0 */
    }
    return (a.bits == b.bits) ? 1 : 0;
}

int bfloat16_ne(bfloat16_t a, bfloat16_t b) {
    if (bf16_is_nan_bits(a.bits) || bf16_is_nan_bits(b.bits)) {
        return 1; /* NaN != anything, including itself */
    }
    return bfloat16_eq(a, b) ? 0 : 1;
}

int bfloat16_lt(bfloat16_t a, bfloat16_t b) {
    if (bf16_is_nan_bits(a.bits) || bf16_is_nan_bits(b.bits)) {
        return 0;
    }
    {
        int a_neg = (a.bits & BFLOAT16_SIGN_MASK) != 0;
        int b_neg = (b.bits & BFLOAT16_SIGN_MASK) != 0;
        if (a_neg != b_neg) {
            /* Different signs: negative < positive, unless both are zero. */
            if (bfloat16_iszero(a) && bfloat16_iszero(b)) {
                return 0;
            }
            return a_neg ? 1 : 0;
        }
        /* Same sign: integer order matches value order for positives and
         * reverses for negatives. Zeros are equal, covered by < below. */
        if (a_neg) {
            return (a.bits > b.bits) ? 1 : 0;
        }
        return (a.bits < b.bits) ? 1 : 0;
    }
}

int bfloat16_le(bfloat16_t a, bfloat16_t b) {
    if (bf16_is_nan_bits(a.bits) || bf16_is_nan_bits(b.bits)) {
        return 0;
    }
    return (bfloat16_lt(a, b) || bfloat16_eq(a, b)) ? 1 : 0;
}

int bfloat16_gt(bfloat16_t a, bfloat16_t b) {
    return bfloat16_lt(b, a);
}

int bfloat16_ge(bfloat16_t a, bfloat16_t b) {
    return bfloat16_le(b, a);
}
