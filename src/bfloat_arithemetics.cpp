#include "bfloat16.h"

#include <math.h>
#include <stddef.h>

/*
 * Arithmetic via a higher-precision intermediate.
 *
 * Every bfloat16 value is exactly representable in float32 (same exponent,
 * shorter mantissa), so widening is exact. The operation itself runs in
 * double precision and the result is rounded once to bfloat16 with
 * round-to-nearest-even. This yields correctly-rounded results up to the
 * usual double-rounding caveat (negligible: the intermediate carries 45
 * extra mantissa bits), while NaN / Inf / signed-zero semantics come for
 * free from IEEE 754 double arithmetic.
 *
 * Why not hand-rolled integer bit manipulation? The previous integer paths
 * had wrong exponent accounting in divide (1/1 produced 2), crashed with
 * SIGFPE on x/Inf (division by a zero mantissa), mapped denormal products
 * to zero, returned +0 for Inf + -Inf instead of NaN, and truncated instead
 * of rounding. Deleting ~200 lines of buggy bit tricks in favour of ~40
 * lines of obviously-correct widening is the improvement.
 */

bfloat16_t bfloat16_add(bfloat16_t a, bfloat16_t b) {
    double r = (double)bfloat16_to_float(a) + (double)bfloat16_to_float(b);
    return double_to_bfloat16(r);
}

bfloat16_t bfloat16_subtract(bfloat16_t a, bfloat16_t b) {
    double r = (double)bfloat16_to_float(a) - (double)bfloat16_to_float(b);
    return double_to_bfloat16(r);
}

bfloat16_t bfloat16_multiply(bfloat16_t a, bfloat16_t b) {
    double r = (double)bfloat16_to_float(a) * (double)bfloat16_to_float(b);
    return double_to_bfloat16(r);
}

bfloat16_t bfloat16_divide(bfloat16_t a, bfloat16_t b) {
    double r = (double)bfloat16_to_float(a) / (double)bfloat16_to_float(b);
    return double_to_bfloat16(r);
}

bfloat16_t bfloat16_negate(bfloat16_t a) {
    /* Flip the sign bit only: preserves NaN payloads, -0, Inf signs. */
    return bfloat16_from_bits((uint16_t)(a.bits ^ BFLOAT16_SIGN_MASK));
}

bfloat16_t bfloat16_abs(bfloat16_t a) {
    return bfloat16_from_bits((uint16_t)(a.bits & (uint16_t)~BFLOAT16_SIGN_MASK));
}

bfloat16_t bfloat16_fma(bfloat16_t a, bfloat16_t b, bfloat16_t c) {
    /* Single rounding: exact a*b+c in double, then one RNE step. Note this
     * is *more* accurate than two roundings (multiply then add). */
    double r = fma((double)bfloat16_to_float(a),
                   (double)bfloat16_to_float(b),
                   (double)bfloat16_to_float(c));
    return double_to_bfloat16(r);
}

double bfloat16_dot(const bfloat16_t *a, const bfloat16_t *b, int n) {
    /* Accumulate in double: the common ML "mixed precision" pattern where
     * storage is bf16 but reduction stays in fp32/fp64. */
    double acc = 0.0;
    if (a == NULL || b == NULL || n <= 0) {
        return acc;
    }
    for (int i = 0; i < n; i++) {
        acc += (double)bfloat16_to_float(a[i]) * (double)bfloat16_to_float(b[i]);
    }
    return acc;
}
