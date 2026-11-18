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

#if defined(__BFLT16_MANT_DIG__) || defined(__BF16_MANT_DIG__)
#define BFLOAT16_TEST_HAVE_REF_BF16 1
#else
#define BFLOAT16_TEST_HAVE_REF_BF16 0
#endif

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

    /* Overflow rounds to infinity, as in IEEE round-to-nearest; denormals
     * survive scaling by one. */
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

    /* dot accumulates left-to-right in double without reordering:
     * (2^60 + 1) - 2^60 rounds to 0.0 because 1 is below the double ulp at
     * 2^60, whereas an exact/reordered sum would be 1.0. */
    {
        bfloat16_t big = double_to_bfloat16(ldexp(1.0, 60));
        bfloat16_t nbig = bfloat16_negate(big);
        bfloat16_t a[] = {big, one, nbig};
        bfloat16_t b[] = {one, one, one};
        CHECK(bfloat16_dot(a, b, 3) == 0.0);
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
    CHECK_BITS(bfloat16_set_exponent(float_to_bfloat16(1.0f), -133).bits,
               0x0001U); /* smallest denormal */
    CHECK_BITS(bfloat16_set_exponent(float_to_bfloat16(1.0f), -134).bits,
               0x0000U); /* shifted out completely */
    CHECK(bfloat16_isinf(bfloat16_set_exponent(float_to_bfloat16(1.0f), 128)) ==
          1);
    CHECK(bfloat16_isnan(bfloat16_set_exponent(BFLOAT16_NAN, 5)) == 1);

    /* NaN, Inf, and zero are never rewritten. */
    CHECK_BITS(bfloat16_set_exponent(BFLOAT16_NAN, -10).bits, 0x7FC0U);
    CHECK_BITS(bfloat16_set_exponent(BFLOAT16_INFINITY, 5).bits, 0x7F80U);
    CHECK_BITS(bfloat16_set_exponent(BFLOAT16_NEG_INFINITY, -10).bits, 0xFF80U);
    CHECK_BITS(bfloat16_set_exponent(BFLOAT16_ZERO, 128).bits, 0x0000U);
    CHECK_BITS(bfloat16_set_exponent(BFLOAT16_NEG_ZERO, -10).bits, 0x8000U);

    CHECK_BITS(bfloat16_ldexp(float_to_bfloat16(1.0f), 1).bits, 0x4000U);
    CHECK_BITS(bfloat16_ldexp(float_to_bfloat16(1.5f), -1).bits,
               float_to_bfloat16(0.75f).bits);
    CHECK_BITS(bfloat16_exp2(0).bits, 0x3F80U);
    CHECK_BITS(bfloat16_exp2(1).bits, 0x4000U);
    CHECK(bfloat16_isinf(bfloat16_exp2(128)) == 1);
    CHECK_BITS(bfloat16_exp2(-134).bits, 0x0000U);
}

static void test_fma_regressions(void) {
    bfloat16_t a = float_to_bfloat16(15.0f);
    bfloat16_t b = float_to_bfloat16(23.0f);
    bfloat16_t tiny = double_to_bfloat16(ldexp(1.0, -60));
    bfloat16_t na = bfloat16_negate(a);
    bfloat16_t ntiny = bfloat16_negate(tiny);

    /* Exact a*b+c = 345 + 2^-60, just above the bf16 midpoint 345, so RNE
     * must round up to 346. A plain double intermediate collapses to 345 and
     * would round down to 344 (ties-to-even). */
    CHECK_BITS(bfloat16_fma(a, b, tiny).bits, 0x43ADU);   /* 346 */
    CHECK_BITS(bfloat16_fma(a, b, ntiny).bits, 0x43ACU);  /* 344 */
    CHECK_BITS(bfloat16_fma(na, b, ntiny).bits, 0xC3ADU); /* -346 */
    CHECK_BITS(bfloat16_fma(na, b, tiny).bits, 0xC3ACU);  /* -344 */

    /* Special values still follow IEEE 754. */
    CHECK(bfloat16_isnan(bfloat16_fma(BFLOAT16_INFINITY, BFLOAT16_ZERO,
                                      BFLOAT16_ONE)) == 1);
    CHECK(bfloat16_isnan(bfloat16_fma(BFLOAT16_INFINITY,
                                      float_to_bfloat16(1.0f),
                                      BFLOAT16_NEG_INFINITY)) == 1);
    CHECK_BITS(bfloat16_fma(BFLOAT16_INFINITY,
                            float_to_bfloat16(1.0f), BFLOAT16_ONE).bits,
               0x7F80U);
    CHECK_BITS(bfloat16_fma(BFLOAT16_NEG_ZERO, BFLOAT16_ONE,
                            BFLOAT16_NEG_ZERO).bits,
               0x8000U);
}

#if BFLOAT16_TEST_HAVE_REF_BF16
static __bf16 ref_bf16_from_bits(uint16_t bits) {
    __bf16 value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t ref_bf16_to_bits(__bf16 value) {
    uint16_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int test_nan_bits(uint16_t bits) {
    return ((bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK) &&
           ((bits & BFLOAT16_MANT_MASK) != 0U);
}

/* NaN payloads are not part of the arithmetic contract; everything else
 * (including the sign of zero and infinity) must match bit-for-bit. */
static int test_bits_match(uint16_t got, uint16_t want) {
    if (test_nan_bits(got) || test_nan_bits(want)) {
        return test_nan_bits(got) && test_nan_bits(want);
    }
    return got == want;
}

static int exhaustive_float_conversions(void) {
    uint64_t failures = 0;
    uint32_t u = 0;
    do {
        float f;
        memcpy(&f, &u, sizeof(f));

        uint16_t got = float_to_bfloat16(f).bits;
        uint16_t want = ref_bf16_to_bits((__bf16)f);
        if (!test_bits_match(got, want)) {
            if (failures < 10) {
                printf("FAIL exhaustive float_to_bfloat16: input=0x%08X "
                       "got=0x%04X want=0x%04X\n",
                       (unsigned)u, (unsigned)got, (unsigned)want);
            }
            failures++;
        }
    } while (++u != 0U);

    printf("exhaustive float_to_bfloat16: 4294967296 inputs, %llu failures\n",
           (unsigned long long)failures);
    return failures == 0 ? 0 : 1;
}

static int exhaustive_arithmetic(void) {
    static bfloat16_t lib_values[65536];
    static __bf16 ref_values[65536];
    uint64_t failures[4] = {0, 0, 0, 0};
    const char *names[4] = {"add", "subtract", "multiply", "divide"};
    uint32_t i;

    for (i = 0; i < 65536U; i++) {
        lib_values[i] = bfloat16_from_bits((uint16_t)i);
        ref_values[i] = ref_bf16_from_bits((uint16_t)i);
    }

    for (i = 0; i < 65536U; i++) {
        bfloat16_t la = lib_values[i];
        __bf16 ra = ref_values[i];
        uint32_t j;

        for (j = 0; j < 65536U; j++) {
            bfloat16_t lb = lib_values[j];
            __bf16 rb = ref_values[j];
            uint16_t got;
            uint16_t want;

            got = bfloat16_add(la, lb).bits;
            want = ref_bf16_to_bits(ra + rb);
            if (!test_bits_match(got, want)) {
                if (failures[0] < 5) {
                    printf("FAIL exhaustive add: a=0x%04X b=0x%04X "
                           "got=0x%04X want=0x%04X\n",
                           (unsigned)la.bits, (unsigned)lb.bits,
                           (unsigned)got, (unsigned)want);
                }
                failures[0]++;
            }

            got = bfloat16_subtract(la, lb).bits;
            want = ref_bf16_to_bits(ra - rb);
            if (!test_bits_match(got, want)) {
                if (failures[1] < 5) {
                    printf("FAIL exhaustive subtract: a=0x%04X b=0x%04X "
                           "got=0x%04X want=0x%04X\n",
                           (unsigned)la.bits, (unsigned)lb.bits,
                           (unsigned)got, (unsigned)want);
                }
                failures[1]++;
            }

            got = bfloat16_multiply(la, lb).bits;
            want = ref_bf16_to_bits(ra * rb);
            if (!test_bits_match(got, want)) {
                if (failures[2] < 5) {
                    printf("FAIL exhaustive multiply: a=0x%04X b=0x%04X "
                           "got=0x%04X want=0x%04X\n",
                           (unsigned)la.bits, (unsigned)lb.bits,
                           (unsigned)got, (unsigned)want);
                }
                failures[2]++;
            }

            got = bfloat16_divide(la, lb).bits;
            want = ref_bf16_to_bits(ra / rb);
            if (!test_bits_match(got, want)) {
                if (failures[3] < 5) {
                    printf("FAIL exhaustive divide: a=0x%04X b=0x%04X "
                           "got=0x%04X want=0x%04X\n",
                           (unsigned)la.bits, (unsigned)lb.bits,
                           (unsigned)got, (unsigned)want);
                }
                failures[3]++;
            }
        }
    }

    for (i = 0; i < 4U; i++) {
        printf("exhaustive %s: 4294967296 pairs, %llu failures\n", names[i],
               (unsigned long long)failures[i]);
    }
    return (failures[0] | failures[1] | failures[2] | failures[3]) == 0 ? 0 : 1;
}
#else
static int exhaustive_float_conversions(void) {
    printf("exhaustive float_to_bfloat16: skipped (no __bf16 reference)\n");
    return 0;
}

static int exhaustive_arithmetic(void) {
    printf("exhaustive arithmetic: skipped (no __bf16 reference)\n");
    return 0;
}
#endif

int main(int argc, char **argv) {
    int exhaustive = 0;
    int exhaustive_failed = 0;
    int i;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--exhaustive") == 0) {
            exhaustive = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("usage: %s [--exhaustive]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            fprintf(stderr, "usage: %s [--exhaustive]\n", argv[0]);
            return 2;
        }
    }

    test_float_conversions();
    test_double_conversions();
    test_arithmetic();
    test_fma_regressions();
    test_comparisons();
    test_exponent_utils();

    if (exhaustive) {
        exhaustive_failed += exhaustive_float_conversions();
        exhaustive_failed += exhaustive_arithmetic();
    }

    printf("bfloat16 tests: %d passed, %d failed\n", g_pass, g_fail);
    if (exhaustive && exhaustive_failed != 0) {
        printf("exhaustive tests: FAILED\n");
    }
    return (g_fail == 0 && exhaustive_failed == 0) ? 0 : 1;
}
