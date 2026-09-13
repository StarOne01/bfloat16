// Comprehensive test suite for the bfloat16 library.
//
// Build (CMake, recommended):
//   cmake -S . -B build && cmake --build build && ctest --test-dir build
// Build (direct):
//   g++ -Iinclude -Wall -Wextra -std=c++17 bfloat_test.cpp src/*.cpp -o bfloat_test
//   ./bfloat_test

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "bfloat16.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (cond) {                                                            \
            g_pass++;                                                          \
        } else {                                                               \
            g_fail++;                                                          \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
        }                                                                      \
    } while (0)

#define CHECK_BITS(got, want)                                                  \
    do {                                                                       \
        uint16_t g_ = (got);                                                   \
        uint16_t w_ = (want);                                                  \
        if (g_ == w_) {                                                        \
            g_pass++;                                                          \
        } else {                                                               \
            g_fail++;                                                          \
            printf("FAIL %s:%d: bits got 0x%04X want 0x%04X (%s)\n", __FILE__,  \
                   __LINE__, g_, w_, #got);                                    \
        }                                                                      \
    } while (0)

static void test_float_conversions(void) {
    CHECK_BITS(float_to_bfloat16(0.0f).bits, 0x0000U);
    CHECK_BITS(float_to_bfloat16(-0.0f).bits, 0x8000U);
    CHECK_BITS(float_to_bfloat16(1.0f).bits, 0x3F80U);
    CHECK_BITS(float_to_bfloat16(-1.0f).bits, 0xBF80U);
    CHECK_BITS(float_to_bfloat16(2.0f).bits, 0x4000U);
    CHECK_BITS(float_to_bfloat16(0.5f).bits, 0x3F00U);
    CHECK_BITS(float_to_bfloat16(1.5f).bits, 0x3FC0U);
    CHECK_BITS(float_to_bfloat16(INFINITY).bits, 0x7F80U);
    CHECK_BITS(float_to_bfloat16(-INFINITY).bits, 0xFF80U);
    CHECK(bfloat16_isnan(float_to_bfloat16(NAN)) == 1);

    /* Round-trip of exactly-representable values. */
    {
        const uint16_t vals[] = {0x0000U, 0x8000U, 0x3F80U, 0xBF80U, 0x4000U,
                                 0x3F00U, 0x7F80U, 0xFF80U, 0x0001U, 0x7F7FU};
        for (size_t i = 0; i < sizeof(vals) / sizeof(vals[0]); i++) {
            bfloat16_t b = bfloat16_from_bits(vals[i]);
            if (bfloat16_isinf(b) || bfloat16_iszero(b) ||
                bfloat16_issubnormal(b) || bfloat16_isfinite(b)) {
                bfloat16_t rt = float_to_bfloat16(bfloat16_to_float(b));
                CHECK_BITS(rt.bits, vals[i]);
            }
        }
    }

    /* Round-to-nearest-even at exact halfway points.
     * Spacing of bf16 in [1,2) is 2^-7; halfway values are exact in f32. */
    CHECK_BITS(float_to_bfloat16(1.00390625f).bits,
               0x3F80U); /* halfway 1.0(even) / 1.0078125 -> even */
    CHECK_BITS(float_to_bfloat16(1.01171875f).bits,
               0x3F82U); /* halfway 1.0078125(odd) / 1.015625 -> even */

    /* NaN payload must survive (never collapses to zero or infinity). */
    {
        float big_nan;
        uint32_t u = 0x7FFFFFFFU;
        memcpy(&big_nan, &u, sizeof(big_nan));
        bfloat16_t b = float_to_bfloat16(big_nan);
        CHECK(bfloat16_isnan(b) == 1);
        CHECK(bfloat16_isinf(b) == 0);
    }
    {
        float qnan;
        uint32_t u = 0x7FC00000U;
        memcpy(&qnan, &u, sizeof(qnan));
        CHECK_BITS(float_to_bfloat16(qnan).bits, 0x7FC0U);
    }

    /* Denormal extremes. */
    CHECK(bfloat16_to_float(bfloat16_from_bits(0x0001U)) > 0.0f);
    CHECK_BITS(float_to_bfloat16(1e-45f).bits, 0x0000U); /* underflows */
}

static void test_double_conversions(void) {
    CHECK_BITS(double_to_bfloat16(0.0).bits, 0x0000U);
    CHECK_BITS(double_to_bfloat16(-0.0).bits, 0x8000U);
    CHECK_BITS(double_to_bfloat16(1.0).bits, 0x3F80U);
    CHECK_BITS(double_to_bfloat16(-2.5).bits, float_to_bfloat16(-2.5f).bits);
    CHECK_BITS(double_to_bfloat16(INFINITY).bits, 0x7F80U);
    CHECK_BITS(double_to_bfloat16(-INFINITY).bits, 0xFF80U);
    CHECK(bfloat16_isnan(double_to_bfloat16(NAN)) == 1);
    CHECK(bfloat16_to_double(bfloat16_from_bits(0x3F80U)) == 1.0);

    /* Overflow / underflow through the direct (single-rounding) path. */
    CHECK_BITS(double_to_bfloat16(1e39).bits, 0x7F80U);
    CHECK_BITS(double_to_bfloat16(-1e39).bits, 0xFF80U);
    CHECK_BITS(double_to_bfloat16(1e-45).bits, 0x0000U);

    /* Exact powers of two, incl. the denormal range. */
    CHECK_BITS(double_to_bfloat16(0.5).bits, 0x3F00U);
    CHECK_BITS(bfloat16_exp2(-133).bits, 0x0001U);
    CHECK_BITS(double_to_bfloat16(ldexp(1.0, -133)).bits, 0x0001U);
    CHECK_BITS(double_to_bfloat16(ldexp(1.0, -127)).bits, 0x0040U);

    /* Halfway rounding through the double path. */
    CHECK_BITS(double_to_bfloat16(1.00390625).bits, 0x3F80U);
    CHECK_BITS(double_to_bfloat16(1.01171875).bits, 0x3F82U);
}

static void test_arithmetic(void) {
    bfloat16_t one = float_to_bfloat16(1.0f);
    bfloat16_t two = float_to_bfloat16(2.0f);
    bfloat16_t three = float_to_bfloat16(3.0f);
    bfloat16_t half = float_to_bfloat16(0.5f);
    bfloat16_t inf = BFLOAT16_INFINITY;
    bfloat16_t ninf = BFLOAT16_NEG_INFINITY;
    bfloat16_t zero = BFLOAT16_ZERO;

    CHECK_BITS(bfloat16_add(one, one).bits, 0x4000U);          /* 2 */
    CHECK_BITS(bfloat16_add(float_to_bfloat16(1.5f),
                            float_to_bfloat16(1.5f)).bits, 0x4040U); /* 3 */
    CHECK_BITS(bfloat16_subtract(one, one).bits, 0x0000U);     /* +0 */
    CHECK_BITS(bfloat16_subtract(three, two).bits, 0x3F80U);   /* 1 */
    CHECK_BITS(bfloat16_multiply(float_to_bfloat16(1.5f),
                                 float_to_bfloat16(1.5f)).bits,
               0x4010U);                                       /* 2.25 */
    CHECK_BITS(bfloat16_multiply(two, two).bits, 0x4080U);     /* 4 */
    CHECK_BITS(bfloat16_divide(one, one).bits, 0x3F80U);       /* 1 */
    CHECK_BITS(bfloat16_divide(one, two).bits, 0x3F00U);       /* 0.5 */
    CHECK_BITS(bfloat16_divide(three, two).bits, 0x3FC0U);     /* 1.5 */

    /* IEEE special-value propagation. */
    CHECK(bfloat16_isnan(bfloat16_add(inf, ninf)) == 1);
    CHECK_BITS(bfloat16_add(inf, one).bits, 0x7F80U);
    CHECK(bfloat16_isnan(bfloat16_divide(zero, zero)) == 1);
    CHECK_BITS(bfloat16_divide(one, zero).bits, 0x7F80U);
    CHECK_BITS(bfloat16_divide(float_to_bfloat16(-1.0f), zero).bits, 0xFF80U);
    CHECK(bfloat16_isnan(bfloat16_multiply(zero, inf)) == 1);
    CHECK_BITS(bfloat16_divide(one, inf).bits, 0x0000U);
    CHECK_BITS(bfloat16_add(half, bfloat16_negate(half)).bits, 0x0000U);

    /* Previously-crashing divisor: x / Inf must not trap. */
    CHECK_BITS(bfloat16_divide(two, inf).bits, 0x0000U);
    CHECK(bfloat16_isnan(bfloat16_divide(inf, inf)) == 1);

    /* Overflow saturates to infinity; denormals survive scaling by one. */
    CHECK(bfloat16_isinf(bfloat16_multiply(bfloat16_from_bits(0x7F7FU),
                                           two)) == 1);
    CHECK_BITS(bfloat16_multiply(bfloat16_from_bits(0x0001U), one).bits,
               0x0001U);

    /* Missing-in-original-API functions. */
    CHECK_BITS(bfloat16_negate(one).bits, 0xBF80U);
    CHECK_BITS(bfloat16_negate(BFLOAT16_NEG_ZERO).bits, 0x0000U);
    CHECK_BITS(bfloat16_abs(bfloat16_from_bits(0xBF80U)).bits, 0x3F80U);
    CHECK_BITS(bfloat16_abs(BFLOAT16_NEG_ZERO).bits, 0x0000U);

    /* Fused multiply-add and dot product helpers. */
    CHECK_BITS(bfloat16_fma(two, three, float_to_bfloat16(4.0f)).bits,
               0x4120U); /* 10 */
    {
        bfloat16_t a[] = {float_to_bfloat16(1.0f), float_to_bfloat16(2.0f),
                          float_to_bfloat16(3.0f)};
        bfloat16_t b[] = {float_to_bfloat16(4.0f), float_to_bfloat16(5.0f),
                          float_to_bfloat16(6.0f)};
        CHECK(bfloat16_dot(a, b, 3) == 32.0);
    }

    /* C++ operator overloads. */
    CHECK_BITS((one + one).bits, 0x4000U);
    CHECK_BITS((two * two).bits, 0x4080U);
    CHECK_BITS((-one).bits, 0xBF80U);
    CHECK((one < two) && (two > one) && (one == one));
}

static void test_comparisons(void) {
    bfloat16_t p0 = BFLOAT16_ZERO, n0 = BFLOAT16_NEG_ZERO;
    bfloat16_t one = float_to_bfloat16(1.0f);
    bfloat16_t two = float_to_bfloat16(2.0f);
    bfloat16_t nan = BFLOAT16_NAN;
    bfloat16_t inf = BFLOAT16_INFINITY, ninf = BFLOAT16_NEG_INFINITY;

    CHECK(bfloat16_eq(p0, n0) == 1);
    CHECK(bfloat16_ne(p0, n0) == 0);
    CHECK(bfloat16_eq(one, one) == 1);
    CHECK(bfloat16_eq(one, two) == 0);
    CHECK(bfloat16_eq(nan, nan) == 0);
    CHECK(bfloat16_ne(nan, nan) == 1);
    CHECK(bfloat16_lt(one, two) == 1);
    CHECK(bfloat16_lt(two, one) == 0);
    CHECK(bfloat16_lt(ninf, inf) == 1);
    CHECK(bfloat16_le(one, one) == 1);
    CHECK(bfloat16_gt(two, one) == 1);
    CHECK(bfloat16_ge(two, two) == 1);
    CHECK(bfloat16_lt(p0, n0) == 0);
    CHECK(bfloat16_le(p0, n0) == 1);
    /* NaN is unordered: every ordering predicate is false. */
    CHECK(bfloat16_lt(nan, one) == 0);
    CHECK(bfloat16_lt(one, nan) == 0);
    CHECK(bfloat16_le(nan, nan) == 0);
    CHECK(bfloat16_gt(nan, one) == 0);
    CHECK(bfloat16_ge(one, nan) == 0);
    /* Predicates are normalized to exactly 0/1. */
    CHECK((bfloat16_lt(ninf, one) == 1) && (bfloat16_gt(inf, one) == 1));

    CHECK(bfloat16_isnan(nan) == 1 && bfloat16_isnan(one) == 0);
    CHECK(bfloat16_isinf(inf) == 1 && bfloat16_isinf(ninf) == 1 &&
          bfloat16_isinf(one) == 0);
    CHECK(bfloat16_isfinite(one) == 1 && bfloat16_isfinite(inf) == 0 &&
          bfloat16_isfinite(nan) == 0);
    CHECK(bfloat16_iszero(p0) == 1 && bfloat16_iszero(n0) == 1 &&
          bfloat16_iszero(one) == 0);
    CHECK(bfloat16_issubnormal(bfloat16_from_bits(0x0001U)) == 1 &&
          bfloat16_issubnormal(one) == 0);
    CHECK(bfloat16_signbit(n0) == 1 && bfloat16_signbit(p0) == 0 &&
          bfloat16_signbit(ninf) == 1);
}

static void test_exponent_utils(void) {
    CHECK(bfloat16_get_exponent(float_to_bfloat16(1.0f)) == 0);
    CHECK(bfloat16_get_exponent(float_to_bfloat16(2.0f)) == 1);
    CHECK(bfloat16_get_exponent(float_to_bfloat16(0.5f)) == -1);
    CHECK(bfloat16_get_exponent(BFLOAT16_ZERO) == INT16_MIN);
    CHECK(bfloat16_get_exponent(BFLOAT16_INFINITY) == INT16_MAX);
    CHECK(bfloat16_get_exponent(BFLOAT16_NAN) == INT16_MAX);
    CHECK(bfloat16_get_exponent(bfloat16_from_bits(0x0001U)) == -126);

    CHECK_BITS(bfloat16_set_exponent(float_to_bfloat16(1.0f), 1).bits,
               0x4000U); /* 2.0 */
    CHECK_BITS(bfloat16_set_exponent(float_to_bfloat16(1.0f), -127).bits,
               0x0040U); /* 2^-127 denormal */
    CHECK(bfloat16_isinf(bfloat16_set_exponent(float_to_bfloat16(1.0f), 128)) ==
          1);
    CHECK(bfloat16_isnan(bfloat16_set_exponent(BFLOAT16_NAN, 5)) == 1);

    CHECK_BITS(bfloat16_ldexp(float_to_bfloat16(1.0f), 1).bits, 0x4000U);
    CHECK_BITS(bfloat16_ldexp(float_to_bfloat16(1.5f), -1).bits,
               float_to_bfloat16(0.75f).bits);
    CHECK_BITS(bfloat16_exp2(0).bits, 0x3F80U);
    CHECK_BITS(bfloat16_exp2(1).bits, 0x4000U);
    CHECK(bfloat16_isinf(bfloat16_exp2(128)) == 1);
    CHECK_BITS(bfloat16_exp2(-134).bits, 0x0000U);
}

int main(void) {
    test_float_conversions();
    test_double_conversions();
    test_arithmetic();
    test_comparisons();
    test_exponent_utils();

    printf("bfloat16 tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
