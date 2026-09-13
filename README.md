A lightweight C++ implementation of the Brain Floating Point (bfloat16) format.

## Overview

bfloat16 is a 16-bit floating point format developed by Google Brain for use in machine learning applications. It preserves the dynamic range of 32-bit floating point (using the same 8-bit exponent) while reducing precision by storing only 7 bits of mantissa (compared to 23 bits in float32).

This implementation represents bfloat16 as:

- 1 sign bit
- 8 exponent bits (bias 127, same as float32)
- 7 mantissa bits

## Features

- C99 / C++17 compatible C API (`extern "C"`, usable from both languages)
- Portable 16-bit storage (`bfloat16_t.bits`) on every platform
- Conversions between bfloat16, float32 and float64 with round-to-nearest-even
- Basic arithmetic (`add`, `subtract`, `multiply`, `divide`, `negate`, `abs`,
  fused `fma`, mixed-precision `dot`) with IEEE 754 NaN / Inf / signed-zero handling
- Full comparison set (`eq`, `ne`, `lt`, `le`, `gt`, `ge`) plus classification
  (`isnan`, `isinf`, `isfinite`, `iszero`, `issubnormal`, `signbit`)
- Exponent utilities (`get_exponent`, `set_exponent`, `ldexp`, `exp2`)
- Lightweight C++ operator overloads (`+ - * /`, comparisons)
- CMake build with CTest suite (118 assertions)

## Layout

```text
include/bfloat16.h      public API (C + C++ overloads)
src/bfloat16_convo.cpp  float/double conversions
src/bfloat_arithemetics.cpp  arithmetic (widen-compute-narrow)
src/bfloat_comp.cpp     comparisons and classification
src/bfloat16_bias.cpp   exponent utilities
bfloat_test.cpp         test suite (also serves as a usage example)
CMakeLists.txt          build (static lib + tests + install rules)
```

## Building and testing

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Or without CMake:

```sh
g++ -Iinclude -Wall -Wextra -std=c++17 bfloat_test.cpp src/*.cpp -o bfloat_test
./bfloat_test
```

To use the library in your own project, add `include/` to your include path
and compile the four `src/*.cpp` files (or link the installed `libbfloat16.a`).

## Usage

```c
#include "bfloat16.h"

bfloat16_t a = float_to_bfloat16(1.5f);
bfloat16_t b = float_to_bfloat16(2.0f);
float sum = bfloat16_to_float(bfloat16_add(a, b)); /* 3.5f */
```

```c++
#include "bfloat16.h"

bfloat16_t a = float_to_bfloat16(1.5f);
bfloat16_t b = float_to_bfloat16(2.0f);
bfloat16_t c = a + b;              // C++ overloads
bool less = a < b;
```

## Implementation notes

- **Rounding:** every narrowing conversion uses round-to-nearest-even.
  `double -> bfloat16` rounds directly from the 64-bit pattern (no intermediate
  `float` step), so there is no double-rounding error.
- **Arithmetic:** operands widen exactly to `float`, the operation runs in
  `double`, and the result is rounded once to bfloat16. This gives
  correctly-rounded results up to the usual negligible double-rounding caveat,
  with Inf / NaN / signed-zero semantics inherited from IEEE 754 arithmetic.
  Overflow saturates to infinity.
- **NaN handling:** conversions preserve NaN payload top bits and force the
  quiet bit, so a NaN can never collapse into zero or infinity by truncation.
- **`set_exponent` vs `ldexp`:** `set_exponent` is raw bit manipulation
  (keeps sign + mantissa, truncates on the way into the denormal range);
  `ldexp`/`exp2` are true correctly-rounded arithmetic.
- **Native types:** storage is always the portable struct, even where the
  compiler offers `__bf16`. `BFLOAT16_HAS_NATIVE` reports its presence for
  informational purposes only.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Google Brain bfloat16 specification](https://cloud.google.com/tpu/docs/bfloat16)
- [TensorFlow bfloat16 implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/bfloat16/bfloat16.h)
- [NVIDIA's bfloat16 implementation](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
