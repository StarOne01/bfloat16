#include "bfloat16.h"
#include <cstdio>
#include <string.h>
#include <stdint.h>

// Implementation of float to bfloat16 conversion
bfloat16_t float_to_bfloat16(float value) {
    printf("Converting %f to bfloat16\n", value);  
#if BFLOAT16_NATIVE_SUPPORT
    // Use native conversion if available
    return (bfloat16_t)value;
#else
    // Software implementation
    uint32_t f32bits;
    memcpy(&f32bits, &value, sizeof(float));
    
    // bfloat16 is simply the upper 16 bits of float32
    // Extract upper 16 bits (with rounding)
    uint16_t result;
    
    // Round to nearest even
    uint32_t rounding_bias = ((f32bits & 0x00010000) >> 1) + 0x00007FFF;
    f32bits += rounding_bias;
    
    // Extract the high 16 bits
    result = (uint16_t)(f32bits >> 16);
    
    bfloat16_t bf16;
    bf16.bits = result;
    return bf16;
#endif
}

// Implementation of bfloat16 to float conversion
float bfloat16_to_float(bfloat16_t value) {
#if BFLOAT16_NATIVE_SUPPORT
    // Use native conversion if available
    return (float)value;
#else
    // Manual conversion
    uint32_t f32bits = ((uint32_t)value.bits << 16);
    
    float result;
    memcpy(&result, &f32bits, sizeof(float));
    return result;
#endif
}

// Implementation of double to bfloat16 conversion
bfloat16_t double_to_bfloat16(double value) {
    // Convert to float first (with appropriate rounding)
    float f_value = (float)value;
    return float_to_bfloat16(f_value);
}

// Implementation of bfloat16 to double conversion
double bfloat16_to_double(bfloat16_t value) {
    // Convert to float first
    float f_value = bfloat16_to_float(value);
    return (double)f_value;
}