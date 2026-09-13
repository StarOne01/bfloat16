#include "bfloat16.h"

#include <stdint.h>
#include <string.h>

/*
 * Conversions between bfloat16 and float/double.
 *
 * bfloat16 is the upper 16 bits of float32, so bf16 -> float is an exact
 * left shift, and float -> bf16 is a right shift with round-to-nearest-even.
 *
 * NaN/Inf are handled explicitly so a large NaN payload can never round up
 * into infinity (or into signed zero), and payload top bits are preserved
 * with the quiet bit forced.
 */

bfloat16_t float_to_bfloat16(float value) {
    uint32_t u;
    memcpy(&u, &value, sizeof(u));

    /* Preserve NaN / Inf exactly (no rounding carry). */
    if ((u & 0x7F800000U) == 0x7F800000U) {
        uint16_t top = (uint16_t)(u >> 16);
        if ((u & 0x007FFFFFU) != 0U) {
            /* NaN: keep payload top bits, force quiet bit so truncation of a
             * low-only payload cannot turn the value into infinity. */
            top |= (uint16_t)BFLOAT16_QNAN_MASK;
        }
        return bfloat16_from_bits(top);
    }

    /* Round-to-nearest-even: bias = 0x7FFF + LSB of the kept half. */
    uint32_t bias = 0x7FFFU + ((u >> 16) & 1U);
    u += bias;
    return bfloat16_from_bits((uint16_t)(u >> 16));
}

float bfloat16_to_float(bfloat16_t value) {
    /* Exact: bf16 is the upper half of f32. Denormals stay bit-correct
     * because the shift preserves their scaled mantissa. */
    uint32_t u = ((uint32_t)value.bits) << 16;
    float out;
    memcpy(&out, &u, sizeof(out));
    return out;
}

bfloat16_t double_to_bfloat16(double value) {
    uint64_t u;
    memcpy(&u, &value, sizeof(u));

    const uint16_t sign = (uint16_t)((u >> 48) & 0x8000U);
    const int exp_d = (int)((u >> 52) & 0x7FFU);
    const uint64_t mant_d = u & 0xFFFFFFFFFFFFFULL; /* 52 bits */

    /* NaN / Inf */
    if (exp_d == 0x7FF) {
        if (mant_d != 0ULL) {
            /* Preserve the top 6 payload bits + quiet bit. */
            uint16_t payload = (uint16_t)((mant_d >> 46) & 0x003FU);
            uint16_t bits = (uint16_t)(0x7F80U | 0x0040U | payload | sign);
            return bfloat16_from_bits(bits);
        }
        return bfloat16_from_bits((uint16_t)(0x7F80U | sign));
    }

    /* Zero / double subnormal: every double subnormal is far below half of
     * the smallest bf16 subnormal (2^-134), so all round to signed zero. */
    if (exp_d == 0) {
        return bfloat16_from_bits(sign);
    }

    const int unbiased = exp_d - 1023;

    /* Overflow beyond bf16 max (unbiased > 127) -> signed infinity. */
    if (unbiased > 127) {
        return bfloat16_from_bits((uint16_t)(0x7F80U | sign));
    }

    if (unbiased >= -126) {
        /* Normal bf16 range: round the 52-bit mantissa down to 7 bits.
         * Full mantissa with implied 1 is 53 bits; keep top 8 (1 + 7),
         * round on the lower 45 bits with RNE. */
        const uint64_t full = (1ULL << 52) | mant_d;
        const uint64_t kept = full >> 45;              /* 8 bits */
        const uint64_t rem = full & ((1ULL << 45) - 1ULL); /* low 45 bits */
        const uint64_t half = 1ULL << 44;
        uint64_t rounded = kept;
        if (rem > half || (rem == half && (kept & 1ULL) != 0ULL)) {
            rounded += 1ULL; /* round up */
        }
        /* Mantissa carry (e.g. 1.111... rounds to 10.0) bumps exponent. */
        int exp_biased = unbiased + BFLOAT16_EXP_BIAS;
        uint16_t mant7;
        if (rounded == 0x100ULL) {
            exp_biased += 1;
            mant7 = 0;
        } else {
            mant7 = (uint16_t)(rounded & 0x7FULL);
        }
        if (exp_biased >= 0xFF) {
            return bfloat16_from_bits((uint16_t)(0x7F80U | sign));
        }
        return bfloat16_from_bits(
            (uint16_t)(sign | ((uint16_t)exp_biased << 7) | mant7));
    }

    /* Denormal bf16 range (or underflow to zero).
     * Value = full * 2^(unbiased - 52) with full in [2^52, 2^53).
     * Target denormal: m * 2^-133, so m_exact = full * 2^(unbiased + 81).
     * Here unbiased <= -127, hence rshift = -(unbiased + 81) >= 46 > 0. */
    {
        const uint64_t full = (1ULL << 52) | mant_d;
        const int rshift = -(unbiased + 81);
        if (rshift >= 54) {
            /* kept would be 0 and half = 2^(rshift-1) >= 2^53 > full,
             * so the value is below half of the smallest denormal. */
            return bfloat16_from_bits(sign);
        }
        uint64_t kept = full >> (unsigned)rshift;
        const uint64_t mask = (1ULL << (unsigned)rshift) - 1ULL;
        const uint64_t rem = full & mask;
        const uint64_t half = 1ULL << (unsigned)(rshift - 1);
        if (rem > half || (rem == half && (kept & 1ULL) != 0ULL)) {
            kept += 1ULL;
        }
        if (kept > 0x7FULL) {
            /* Rounded up into the normal range -> smallest normal. */
            return bfloat16_from_bits((uint16_t)(sign | (1U << 7)));
        }
        return bfloat16_from_bits((uint16_t)(sign | (uint16_t)kept));
    }
}

double bfloat16_to_double(bfloat16_t value) {
    /* Exact: every bf16 value is exactly representable in double. */
    return (double)bfloat16_to_float(value);
}
