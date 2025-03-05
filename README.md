A lightweight C++ implementation of the Brain Floating Point (bfloat16) format.

## Overview

bfloat16 is a 16-bit floating point format developed by Google Brain for use in machine learning applications. It preserves the dynamic range of 32-bit floating point (using the same 8-bit exponent) while reducing precision by storing only 7 bits of mantissa (compared to 23 bits in float32).

## Features

- Header-only C++ implementation
- Conversions between bfloat16 and float32
- Basic arithmetic operations
- Standard mathematical functions
- IEEE 754 compatibility


## Implementation Details

This implementation represents bfloat16 as:
- 1 sign bit
- 8 exponent bits (same as float32)
- 7 mantissa bits

The format provides a balance between range and precision that is particularly suitable for neural network training and inference.

## Building and Installation

This is a header-only library. Simply include the header files in your project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Google Brain bfloat16 specification](https://cloud.google.com/tpu/docs/bfloat16)
- [TensorFlow bfloat16 implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/bfloat16/bfloat16.h)
- [NVIDIA's bfloat16 implementation](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.