#!/usr/bin/env python3
"""Second-reference differential test for the basic bfloat16 arithmetic.

The C++ exhaustive suite uses GCC's __bf16 arithmetic. This script checks a
deterministic sampled subset against Python's ml_dtypes, which is an
independent implementation of the bfloat16 storage format.

Usage: test_ml_dtypes.py path/to/libbfloat16-test-shared.so [pair-count]
"""

import ctypes
import sys


class BF16(ctypes.Structure):
    _fields_ = [("bits", ctypes.c_uint16)]


def load_library(path):
    lib = ctypes.CDLL(path)
    for name in (
        "bfloat16_add",
        "bfloat16_subtract",
        "bfloat16_multiply",
        "bfloat16_divide",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [BF16, BF16]
        fn.restype = BF16
    return lib


def nan_mask(bits):
    return ((bits & 0x7F80) == 0x7F80) & ((bits & 0x007F) != 0)


def main():
    if len(sys.argv) < 2:
        print("usage: test_ml_dtypes.py path/to/shared-lib [pair-count]",
              file=sys.stderr)
        return 2

    try:
        import numpy as np
        import ml_dtypes
    except ImportError as exc:
        print(f"ml_dtypes differential: skipped ({exc})")
        return 0

    count = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    rng = np.random.default_rng(0xBF16)

    # Random patterns plus an edge-value cross product: zeros, denormals,
    # normals, infinities, and quiet/signalling NaNs of both signs.
    edge_values = np.array(
        [
            0x0000, 0x8000,  # +/- zero
            0x0001, 0x8001,  # +/- smallest denormal
            0x007F, 0x807F,  # +/- largest denormal
            0x0080, 0x8080,  # +/- smallest normal
            0x3F80, 0xBF80,  # +/- 1.0
            0x7F7F, 0xFF7F,  # +/- largest finite
            0x7F80, 0xFF80,  # +/- infinity
            0x7FC0, 0xFFC0,  # +/- quiet NaN
            0x7F81, 0xFF81,  # +/- signalling NaN
        ],
        dtype=np.uint16,
    )
    a_bits = np.concatenate(
        [
            rng.integers(0, 65536, size=count, dtype=np.uint16),
            np.repeat(edge_values, edge_values.size),
        ]
    )
    b_bits = np.concatenate(
        [
            rng.integers(0, 65536, size=count, dtype=np.uint16),
            np.tile(edge_values, edge_values.size),
        ]
    )

    a_ref = a_bits.view(ml_dtypes.bfloat16)
    b_ref = b_bits.view(ml_dtypes.bfloat16)
    operations = [
        ("add", "bfloat16_add", np.add),
        ("subtract", "bfloat16_subtract", np.subtract),
        ("multiply", "bfloat16_multiply", np.multiply),
        ("divide", "bfloat16_divide", np.divide),
    ]

    lib = load_library(sys.argv[1])
    failures = 0

    for op_name, fn_name, np_op in operations:
        with np.errstate(all="ignore"):
            want = np.asarray(np_op(a_ref, b_ref)).view(np.uint16)
        got = np.empty_like(want)

        fn = getattr(lib, fn_name)
        for i in range(a_bits.size):
            got[i] = fn(BF16(int(a_bits[i])), BF16(int(b_bits[i]))).bits

        got_nan = nan_mask(got)
        want_nan = nan_mask(want)
        ok = np.where(
            got_nan | want_nan,
            got_nan & want_nan,
            got == want,
        )

        bad = np.nonzero(~ok)[0]
        if bad.size:
            failures += int(bad.size)
            print(f"FAIL ml_dtypes {op_name}: {bad.size} mismatches")
            for i in bad[:10]:
                print(
                    f"  a=0x{int(a_bits[i]):04X} b=0x{int(b_bits[i]):04X} "
                    f"got=0x{int(got[i]):04X} want=0x{int(want[i]):04X}"
                )

    total = a_bits.size * len(operations)
    if failures:
        print(f"ml_dtypes differential: {total} checks, {failures} failures")
        return 1
    print(f"ml_dtypes differential: {total} checks, 0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
