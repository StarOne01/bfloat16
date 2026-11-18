[![CI](https://github.com/StarOne01/bfloat16/actions/workflows/ci.yml/badge.svg)](https://github.com/StarOne01/bfloat16/actions/workflows/ci.yml)

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
- CMake build with CTest suite (134 C++ assertions, C99 API test, Python
  exact-fma and ml_dtypes differential tests, plus opt-in exhaustive tests)

## Layout

```text
include/bfloat16.h      public API (C + C++ overloads)
src/bfloat16_convo.cpp  float/double conversions
src/bfloat_arithmetics.cpp   arithmetic (widen-compute-narrow)
src/bfloat_comp.cpp     comparisons and classification
src/bfloat16_bias.cpp   exponent utilities
bfloat_test.cpp         C++ test suite (also serves as a usage example)
test_c.c                C99 API test
test_fma_exact.py       exact-rational fma differential test
test_ml_dtypes.py       sampled ml_dtypes differential test
CMakeLists.txt          build (static lib + tests + install rules)
```

## Building and testing

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

CTest runs the C++ suite, `bfloat_test_c` (a C99 translation unit compiled
with `-std=c99 -Wall -Wextra`), an exact-rational fma differential test, and,
when Python's `ml_dtypes` is installed, a sampled arithmetic differential
test against that independent implementation.

The full exhaustive suite checks all 2^32 float inputs and all 2^32 bf16
operand pairs for add/subtract/multiply/divide against GCC's independent
`__bf16` arithmetic. It is opt-in because it takes a few minutes:

```sh
cmake -S . -B build -DBFLOAT16_EXHAUSTIVE_TESTS=ON
cmake --build build
ctest --test-dir build -L exhaustive --output-on-failure
```

Or without CMake:

```sh
g++ -Iinclude -Wall -Wextra -std=c++17 bfloat_test.cpp src/*.cpp -o bfloat_test -lm
./bfloat_test              # quick suite
./bfloat_test --exhaustive # exhaustive suite
```

To use the library in your own project, add `include/` to your include path
and compile the four `src/*.cpp` files (or link the installed `libbfloat16.a`).

## Verification

- Exhaustive `float_to_bfloat16` against GCC `__bf16` over all 2^32 float bit
  patterns.
- Exhaustive add, subtract, multiply, and divide against GCC `__bf16` over
  all 2^32 bf16 operand pairs for each operation.
- Fused multiply-add against an exact rational reference, including every
  addend for targeted tie cases such as `fma(15, 23, 2^-60)`.
- A deterministic sample cross-checked against Python `ml_dtypes`.
- The public header compiled and linked as strict C99.

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
- **Arithmetic:** add, subtract, multiply, and divide widen exactly to `double`
  and then round once to bfloat16. They are correctly rounded: the 53-bit
  intermediate is far wider than the 2p+2 = 18 bits required to make double
  rounding harmless for a p = 8 target. `fma` also rounds `a*b+c` correctly,
  using round-to-odd on the double result to preserve the residual when the
  exact value lies on a bf16 midpoint. Inf / NaN / signed-zero semantics are
  inherited from IEEE 754 arithmetic; overflow produces infinity, as in IEEE
  round-to-nearest.
- **`dot`:** each bf16 pair is multiplied exactly in `double`, then the
  products are accumulated left-to-right (`i = 0..n-1`) in a single `double`
  accumulator. Every addition rounds to `double`; terms are not reordered or
  reduced pairwise.
- **`-ffast-math`:** do not use `-ffast-math` or `-ffinite-math-only` when
  compiling this library. They let the compiler assume NaN and infinity
  cannot occur, which breaks the documented IEEE 754 behavior.
- **Platform caveats:** FTZ/DAZ modes flush bf16 subnormals to zero in the
  widening arithmetic and break the documented subnormal behavior. On 32-bit
  x86 without SSE2, x87 extended-precision evaluation can reintroduce double
  rounding; x86-64 and ARM64 are unaffected.
- **NaN handling:** conversions preserve NaN payload top bits and force the
  quiet bit, so a NaN can never collapse into zero or infinity by truncation.
- **`set_exponent` vs `ldexp`:** `set_exponent` is raw bit manipulation
  (keeps sign + mantissa, leaves NaN/Inf/zero unchanged, truncates on the way
  into the denormal range); `ldexp`/`exp2` are true correctly-rounded
  arithmetic.
- **Native types:** storage is always the portable struct, even where the
  compiler offers `__bf16`. `BFLOAT16_HAS_NATIVE` reports its presence for
  informational purposes only.

## Acknowledgements

Thanks to [tansy](https://github.com/tansy) for the early fixes and feedback
that started this work.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Google Brain bfloat16 specification](https://cloud.google.com/tpu/docs/bfloat16)
- [TensorFlow bfloat16 implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/bfloat16/bfloat16.h)
- [NVIDIA's bfloat16 implementation](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
