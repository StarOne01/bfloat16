#include "bfloat16.h"
#include <stdint.h>

static void bfloat16_unpack(bfloat16_t value, uint16_t* sign, int16_t* exp, uint16_t* mant) {
    *sign = (value.bits & BFLOAT16_SIGN_MASK) >> 15;
    *exp = ((value.bits & BFLOAT16_EXP_MASK) >> 7);
    *mant = (value.bits & BFLOAT16_MANT_MASK);
    
    if (*exp > 0 && *exp < 255) {
        *mant |= 0x0080;
    }
}

static bfloat16_t bfloat16_pack(uint16_t sign, int16_t exp, uint16_t mant) {
    bfloat16_t result;
    
    if (exp > 254) {
        result.bits = sign << 15 | 0x7F80;
        if (mant != 0) {
            result.bits |= 0x40;
        }
        return result;
    }
    
    if (exp < 1) {
        if (mant == 0) {
            result.bits = sign << 15;
            return result;
        }
        
        while ((mant & 0x0080) == 0) {
            mant <<= 1;
            exp--;
        }
        
        mant &= 0x007F;
        
        if (exp < -126) {
            result.bits = sign << 15;
            return result;
        }
        
        exp++;
    } else {
        mant &= 0x007F;
    }
    
    result.bits = (sign << 15) | ((exp & 0xFF) << 7) | (mant & 0x007F);
    return result;
}



bfloat16_t bfloat16_add(bfloat16_t a, bfloat16_t b) {

    if ((a.bits & BFLOAT16_EXP_MASK) == 0) return b;
    if ((b.bits & BFLOAT16_EXP_MASK) == 0) return a;
    
    if ((a.bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK && 
        (a.bits & BFLOAT16_MANT_MASK) != 0) return a;
    if ((b.bits & BFLOAT16_EXP_MASK) == BFLOAT16_EXP_MASK && 
        (b.bits & BFLOAT16_MANT_MASK) != 0) return b;
    
    uint16_t sign_a, sign_b, sign_r;
    int16_t exp_a, exp_b, exp_r;
    uint16_t mant_a, mant_b, mant_r;
    
    bfloat16_unpack(a, &sign_a, &exp_a, &mant_a);
    bfloat16_unpack(b, &sign_b, &exp_b, &mant_b);
    
    int16_t exp_diff = exp_a - exp_b;
    if (exp_diff > 0) {
        if (exp_diff > 8) {
            return a;
        }
        mant_b >>= exp_diff;
        exp_r = exp_a;
    } else if (exp_diff < 0) {
        if (exp_diff < -8) {
            return b;
        }
        mant_a >>= -exp_diff;
        exp_r = exp_b;
    } else {
        exp_r = exp_a;
    }
    
    if (sign_a == sign_b) {
        sign_r = sign_a;
        mant_r = mant_a + mant_b;
        
        if (mant_r & 0x0100) {
            mant_r >>= 1;
            exp_r++;
        }
    } else {
        if (mant_a > mant_b) {
            sign_r = sign_a;
            mant_r = mant_a - mant_b;
        } else if (mant_b > mant_a) {
            sign_r = sign_b;
            mant_r = mant_b - mant_a;
        } else {
            bfloat16_t zero = {.bits = 0};
            return zero;
        }
        
        if (mant_r != 0) {
            while ((mant_r & 0x0080) == 0) {
                mant_r <<= 1;
                exp_r--;
            }
        } else {
            bfloat16_t zero = {.bits = 0};
            return zero;
        }
    }
    
    return bfloat16_pack(sign_r, exp_r, mant_r);
}

bfloat16_t bfloat16_subtract(bfloat16_t a, bfloat16_t b) {
  bfloat16_t neg_b = {.bits = b.bits ^ BFLOAT16_SIGN_MASK};
  return bfloat16_add(a, neg_b);
}

bfloat16_t bfloat16_multiply(bfloat16_t a, bfloat16_t b) {

  if ((a.bits & ~BFLOAT16_SIGN_MASK) == 0 || (b.bits & ~BFLOAT16_SIGN_MASK) == 0) {
        return (bfloat16_t){.bits = ((a.bits ^ b.bits) & BFLOAT16_SIGN_MASK)};
    }
    
    uint16_t sign_a, sign_b, sign_r;
    int16_t exp_a, exp_b, exp_r;
    uint16_t mant_a, mant_b, mant_r;
    
    bfloat16_unpack(a, &sign_a, &exp_a, &mant_a);
    bfloat16_unpack(b, &sign_b, &exp_b, &mant_b);
    
    sign_r = sign_a ^ sign_b;
        exp_r = exp_a + exp_b - BFLOAT16_EXP_BIAS;
    
    uint32_t mant_product = (uint32_t)mant_a * (uint32_t)mant_b;
    
    if (mant_product & 0x00010000) {
        mant_r = (mant_product >> 9) & 0x00FF;
        exp_r++;
    } else {
        mant_r = (mant_product >> 8) & 0x00FF;
    }
    
    return bfloat16_pack(sign_r, exp_r, mant_r);
}
