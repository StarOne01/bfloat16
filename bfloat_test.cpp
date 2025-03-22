#include <stdint.h>
#include <stdio.h>
#include "include/bfloat16.h"

int main() {
    bfloat16_t bf = float_to_bfloat16(1.022E-36);
    float f = bfloat16_to_float(bf);
    float f2 = bfloat16_to_float(bf);
    printf("Original: %f, Converted: %f\n", bfloat16_to_float(bf), f2);
    return 0;
}