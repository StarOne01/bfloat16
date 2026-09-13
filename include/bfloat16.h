#ifndef BFLOAT16_H
#define BFLOAT16_H

/*
 * bfloat16 - Brain Floating Point (16-bit) library.
 *
 * Layout (big picture):
 *   1 sign bit | 8 exponent bits (bias 127, same as float32) | 7 mantissa bits
 *
 * Design notes:
 * - The canonical storage is always a 16-bit struct (`bfloat16_t.bits`),
 *   even on compilers with a native `__bf16` type. This keeps bit
 *   manipulation portable. Native HW types may be used *internally* as a
 *   fast path in the future, but they never change the ABI.
 * - All conversions use round-to-nearest-even (RNE).
 * - Arithmetic is computed in higher precision (float operands promoted to
 *   double) and then rounded once to bfloat16, giving correctly-rounded
 *   results in all but adversarial double-rounding corner cases, with full
 *   IEEE 754 propagation of NaN / Inf / signed zero.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Informational: 1 if the compiler advertises a native bf16 type.
 * This never changes the `bfloat16_t` layout (always the 16-bit struct). */
#if defined(__BF16_MANT_DIG__) || defined(__BFLT16_MANT_DIG__) || \
    defined(__ARM_FEATURE_BF16) || defined(__SUPPORT_BF16__)
#define BFLOAT16_HAS_NATIVE 1
#else
#define BFLOAT16_HAS_NATIVE 0
#endif
/* Backwards-compat alias: historical name for "native storage". Storage is
 * always the software struct now, so this is always 0. New code should use
 * BFLOAT16_HAS_NATIVE only as a hint, not for layout decisions. */
#define BFLOAT16_NATIVE_SUPPORT 0

/* Canonical 16-bit storage. */
typedef struct {
    uint16_t bits;
} bfloat16_t;

/* Field masks / parameters. */
#define BFLOAT16_SIGN_MASK 0x8000U
#define BFLOAT16_EXP_MASK  0x7F80U
#define BFLOAT16_MANT_MASK 0x007FU
#define BFLOAT16_QNAN_MASK 0x0040U /* quiet bit within the 7-bit mantissa */
#define BFLOAT16_EXP_BIAS  127     /* same as float32 */
#define BFLOAT16_MANT_BITS 7
#define BFLOAT16_EXP_BITS  8
#define BFLOAT16_EXP_MAX   255

/* Raw bit patterns for common values (positional init: portable C99 + C++). */
#define BFLOAT16_POS_ZERO_BITS 0x0000U
#define BFLOAT16_NEG_ZERO_BITS 0x8000U
#define BFLOAT16_POS_INF_BITS  0x7F80U
#define BFLOAT16_NEG_INF_BITS  0xFF80U
#define BFLOAT16_QNAN_BITS     0x7FC0U
#define BFLOAT16_ONE_BITS      0x3F80U

#ifdef __cplusplus
#define BFLOAT16_ZERO         (bfloat16_t{0x0000})
#define BFLOAT16_NEG_ZERO     (bfloat16_t{0x8000})
#define BFLOAT16_INFINITY     (bfloat16_t{0x7F80})
#define BFLOAT16_NEG_INFINITY (bfloat16_t{0xFF80})
#define BFLOAT16_NAN          (bfloat16_t{0x7FC0})
#define BFLOAT16_ONE          (bfloat16_t{0x3F80})
#else
#define BFLOAT16_ZERO         ((bfloat16_t){0x0000})
#define BFLOAT16_NEG_ZERO     ((bfloat16_t){0x8000})
#define BFLOAT16_INFINITY     ((bfloat16_t){0x7F80})
#define BFLOAT16_NEG_INFINITY ((bfloat16_t){0xFF80})
#define BFLOAT16_NAN          ((bfloat16_t){0x7FC0})
#define BFLOAT16_ONE          ((bfloat16_t){0x3F80})
#endif

/* Bit-level constructors / accessors (always inline, no linkage issues). */
static inline bfloat16_t bfloat16_from_bits(uint16_t bits) {
    bfloat16_t v;
    v.bits = bits;
    return v;
}

static inline uint16_t bfloat16_to_bits(bfloat16_t v) {
    return v.bits;
}

/* Conversions (round-to-nearest-even). */
bfloat16_t float_to_bfloat16(float value);
float bfloat16_to_float(bfloat16_t value);
bfloat16_t double_to_bfloat16(double value);
double bfloat16_to_double(bfloat16_t value);

/* Basic arithmetic. NaN/Inf/signed-zero follow IEEE 754 (via the higher
 * precision intermediate). Overflow rounds to infinity. */
bfloat16_t bfloat16_add(bfloat16_t a, bfloat16_t b);
bfloat16_t bfloat16_subtract(bfloat16_t a, bfloat16_t b);
bfloat16_t bfloat16_multiply(bfloat16_t a, bfloat16_t b);
bfloat16_t bfloat16_divide(bfloat16_t a, bfloat16_t b);
bfloat16_t bfloat16_negate(bfloat16_t a);
bfloat16_t bfloat16_abs(bfloat16_t a);
bfloat16_t bfloat16_fma(bfloat16_t a, bfloat16_t b, bfloat16_t c); /* a*b+c, single rounding */
double bfloat16_dot(const bfloat16_t *a, const bfloat16_t *b, int n); /* sum(a[i]*b[i]) in double */

/* Comparisons. Return 1 (true) or 0 (false); any NaN operand makes
 * eq/lt/le/gt/ge false and ne true. Signed zeros compare equal. */
int bfloat16_eq(bfloat16_t a, bfloat16_t b);
int bfloat16_ne(bfloat16_t a, bfloat16_t b);
int bfloat16_lt(bfloat16_t a, bfloat16_t b);
int bfloat16_le(bfloat16_t a, bfloat16_t b);
int bfloat16_gt(bfloat16_t a, bfloat16_t b);
int bfloat16_ge(bfloat16_t a, bfloat16_t b);

/* Classification. Each returns 1 or 0. */
int bfloat16_isnan(bfloat16_t a);
int bfloat16_isinf(bfloat16_t a);
int bfloat16_isfinite(bfloat16_t a);
int bfloat16_iszero(bfloat16_t a);
int bfloat16_issubnormal(bfloat16_t a);
int bfloat16_signbit(bfloat16_t a); /* 1 if negative (incl. -0, -NaN), else 0 */

/* Exponent utilities. Exponents are *unbiased* (e.g. 1.0 -> 0, 2.0 -> 1).
 * get_exponent returns INT16_MAX for Inf/NaN and INT16_MIN for zero
 * (denormals report -126). set_exponent is raw bit manipulation: it keeps
 * sign+mantissa and rewrites the biased field; transitions into the
 * denormal range truncate the mantissa. ldexp/exp2 are arithmetic
 * (correctly rounded multiply-by-2^exp / 2^exp). */
int16_t bfloat16_get_exponent(bfloat16_t value);
bfloat16_t bfloat16_set_exponent(bfloat16_t value, int16_t exp);
bfloat16_t bfloat16_ldexp(bfloat16_t value, int exp);
bfloat16_t bfloat16_exp2(int16_t exp);

#ifdef __cplusplus
} /* extern "C" */

/* Lightweight C++ ergonomics (header-only, zero ABI change). */
inline bfloat16_t operator+(bfloat16_t a, bfloat16_t b) { return bfloat16_add(a, b); }
inline bfloat16_t operator-(bfloat16_t a, bfloat16_t b) { return bfloat16_subtract(a, b); }
inline bfloat16_t operator*(bfloat16_t a, bfloat16_t b) { return bfloat16_multiply(a, b); }
inline bfloat16_t operator/(bfloat16_t a, bfloat16_t b) { return bfloat16_divide(a, b); }
inline bfloat16_t operator-(bfloat16_t a) { return bfloat16_negate(a); }
inline bool operator==(bfloat16_t a, bfloat16_t b) { return bfloat16_eq(a, b) != 0; }
inline bool operator!=(bfloat16_t a, bfloat16_t b) { return bfloat16_ne(a, b) != 0; }
inline bool operator<(bfloat16_t a, bfloat16_t b) { return bfloat16_lt(a, b) != 0; }
inline bool operator<=(bfloat16_t a, bfloat16_t b) { return bfloat16_le(a, b) != 0; }
inline bool operator>(bfloat16_t a, bfloat16_t b) { return bfloat16_gt(a, b) != 0; }
inline bool operator>=(bfloat16_t a, bfloat16_t b) { return bfloat16_ge(a, b) != 0; }

#endif /* __cplusplus */

#endif /* BFLOAT16_H */
