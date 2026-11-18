/* Small C99 API smoke test: including bfloat16.h from C must not pull in
 * any C++ overloads, and all public calls must link through the C ABI. */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "bfloat16.h"

static int g_fail = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
            g_fail++;                                                          \
        }                                                                      \
    } while (0)

#define CHECK_BITS(got, want)                                                  \
    do {                                                                       \
        uint16_t g_ = (uint16_t)(got);                                         \
        uint16_t w_ = (uint16_t)(want);                                        \
        if (g_ != w_) {                                                        \
            printf("FAIL %s:%d: bits got 0x%04X want 0x%04X\n", __FILE__,      \
                   __LINE__, (unsigned)g_, (unsigned)w_);                      \
            g_fail++;                                                          \
        }                                                                      \
    } while (0)

int main(void) {
    bfloat16_t one = float_to_bfloat16(1.0f);
    bfloat16_t two = float_to_bfloat16(2.0f);
    bfloat16_t three = float_to_bfloat16(3.0f);
    bfloat16_t a = float_to_bfloat16(15.0f);
    bfloat16_t b = float_to_bfloat16(23.0f);
    bfloat16_t tiny = double_to_bfloat16(ldexp(1.0, -60));

    CHECK(bfloat16_to_float(one) == 1.0f);
    CHECK(bfloat16_to_double(two) == 2.0);
    CHECK_BITS(bfloat16_add(one, two).bits, 0x4040U);      /* 3 */
    CHECK_BITS(bfloat16_subtract(three, two).bits, 0x3F80U); /* 1 */
    CHECK_BITS(bfloat16_multiply(two, three).bits, 0x40C0U); /* 6 */
    CHECK_BITS(bfloat16_divide(three, two).bits, 0x3FC0U);   /* 1.5 */
    CHECK(bfloat16_isnan(bfloat16_divide(BFLOAT16_ZERO,
                                         BFLOAT16_ZERO)) == 1);
    CHECK_BITS(bfloat16_fma(a, b, tiny).bits, 0x43ADU); /* 346, not 344 */

    CHECK(bfloat16_eq(one, one) == 1);
    CHECK(bfloat16_lt(one, two) == 1);
    CHECK(bfloat16_isfinite(one) == 1);
    CHECK(bfloat16_isinf(BFLOAT16_INFINITY) == 1);
    CHECK(bfloat16_get_exponent(two) == 1);
    CHECK_BITS(bfloat16_set_exponent(one, 1).bits, 0x4000U);
    CHECK_BITS(bfloat16_ldexp(one, 1).bits, 0x4000U);
    CHECK_BITS(bfloat16_exp2(1).bits, 0x4000U);

    {
        bfloat16_t av[] = {one, two};
        bfloat16_t bv[] = {three, float_to_bfloat16(4.0f)};
        CHECK(bfloat16_dot(av, bv, 2) == 11.0);
    }

    if (g_fail != 0) {
        printf("bfloat16 C99 tests: %d failed\n", g_fail);
        return 1;
    }
    printf("bfloat16 C99 tests: passed\n");
    return 0;
}
