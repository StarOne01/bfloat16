#include "bfloat16.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/*
 * Arithmetic via a higher-precision intermediate.
 *
 * Every bfloat16 value is exactly representable in float32 (same exponent,
 * shorter mantissa), so widening is exact. The operation itself runs in
 * double precision and the result is rounded once to bfloat16 with
 * round-to-nearest-even.
 *
 * Add, subtract, multiply, and divide are correctly rounded: double carries
 * 53 significand bits, well above the 2p+2 = 18 bits needed for double
 * rounding to be harmless when the target has p = 8 bits. Fused multiply-add
 * is different: a*b+c can sit exactly on a bf16 midpoint while the tiny
 * residual is below a double ulp, so its intermediate is first perturbed to
 * odd (round-to-odd) before the final narrowing.
 *
 * NaN / Inf / signed-zero semantics come for free from IEEE 754 double
 * arithmetic.
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
    /* a*b is exact in double (at most 16 significand bits). TwoSum gives the
     * exact residual of x + y even when the sum rounds; forcing an inexact
     * result to odd makes the final bf16 rounding equivalent to rounding the
     * exact a*b+c once. */
    double x = (double)bfloat16_to_float(a) * (double)bfloat16_to_float(b);
    double y = (double)bfloat16_to_float(c);
    double s = x + y;
    if (isfinite(s)) {
        double bb = s - x;
        double err = (x - (s - bb)) + (y - bb);
        if (err != 0.0) {
            uint64_t u;
            memcpy(&u, &s, sizeof(u));
            if ((u & 1ULL) == 0ULL) {
                s = nextafter(s, err > 0.0 ? INFINITY : -INFINITY);
            }
        }
    }
    return double_to_bfloat16(s);
}

double bfloat16_dot(const bfloat16_t *a, const bfloat16_t *b, int n) {
    /* Mixed precision: storage is bf16, each product is exact in double, and
     * terms are accumulated left-to-right (i = 0..n-1) in one double
     * accumulator. Every addition rounds to double; there is no reordering. */
    double acc = 0.0;
    if (a == NULL || b == NULL || n <= 0) {
        return acc;
    }
    for (int i = 0; i < n; i++) {
        acc += (double)bfloat16_to_float(a[i]) * (double)bfloat16_to_float(b[i]);
    }
    return acc;
}
