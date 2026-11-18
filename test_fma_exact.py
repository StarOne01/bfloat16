#!/usr/bin/env python3
"""Exact-rational differential test for bfloat16_fma.

This is independent of the C implementation: bf16 operands are decoded to
Python integers/Fractions, a*b+c is formed exactly, and the result is rounded
to bf16 with round-to-nearest-even. In addition to random triples it covers
every possible addend for a few targeted (a, b) pairs, including the
double-rounding case fma(15, 23, 2^-60).

Usage: test_fma_exact.py path/to/shared-lib [random-triple-count]
"""

import ctypes
import random
import sys
from fractions import Fraction


class BF16(ctypes.Structure):
    _fields_ = [("bits", ctypes.c_uint16)]


def decode(bits):
    sign = -1 if (bits >> 15) else 1
    exp = (bits >> 7) & 0xFF
    mant = bits & 0x7F
    if exp == 0xFF:
        return ("nan", 0) if mant else ("inf", sign)
    if exp == 0:
        if mant == 0:
            return ("zero", sign)
        return ("finite", sign, mant, -133)
    return ("finite", sign, 128 + mant, exp - 134)


def value(decoded):
    if decoded[0] == "zero":
        return Fraction(0)
    _, sign, sig, exp = decoded
    return sign * sig * Fraction(2) ** exp


def is_nan_bits(bits):
    return ((bits & 0x7F80) == 0x7F80) and ((bits & 0x007F) != 0)


def floor_log2(x):
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    return e


def round_to_bf16(x):
    if x == 0:
        return 0x0000
    sign = 0x8000 if x < 0 else 0
    ax = abs(x)
    e = floor_log2(ax)

    if e >= -126:
        scaled = ax / Fraction(2) ** (e - 7)
        n = scaled.numerator // scaled.denominator
        rem = scaled - n
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
            n += 1
        if n == 256:
            n = 128
            e += 1
        if e > 127:
            return sign | 0x7F80
        return sign | ((e + 127) << 7) | (n - 128)

    scaled = ax / Fraction(2) ** -133
    n = scaled.numerator // scaled.denominator
    rem = scaled - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
        n += 1
    if n >= 128:
        return sign | (1 << 7)
    return sign | n


def ref_fma(a_bits, b_bits, c_bits):
    a, b, c = decode(a_bits), decode(b_bits), decode(c_bits)
    if "nan" in (a[0], b[0], c[0]):
        return 0x7FC0
    if (a[0] == "inf" and b[0] == "zero") or (b[0] == "inf" and a[0] == "zero"):
        return 0x7FC0
    if a[0] == "inf" or b[0] == "inf":
        product_sign = a[1] * b[1]
        if c[0] == "inf" and c[1] != product_sign:
            return 0x7FC0
        return (0x8000 if product_sign < 0 else 0) | 0x7F80
    if c[0] == "inf":
        return (0x8000 if c[1] < 0 else 0) | 0x7F80

    product = value(a) * value(b)
    exact = product + value(c)
    if exact == 0:
        # In round-to-nearest an exact zero is +0, except that -0 combined
        # with -0 stays -0.
        if product == 0 and value(c) == 0:
            product_sign = -1 if (a[1] < 0) ^ (b[1] < 0) else 1
            if product_sign < 0 and c[1] < 0:
                return 0x8000
        return 0x0000
    return round_to_bf16(exact)


def main():
    if len(sys.argv) < 2:
        print("usage: test_fma_exact.py path/to/shared-lib [triple-count]",
              file=sys.stderr)
        return 2

    lib = ctypes.CDLL(sys.argv[1])
    fn = lib.bfloat16_fma
    fn.argtypes = [BF16, BF16, BF16]
    fn.restype = BF16

    random_count = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    rng = random.Random(0xBF16)

    triples = [
        (rng.getrandbits(16), rng.getrandbits(16), rng.getrandbits(16))
        for _ in range(random_count)
    ]

    # Exhaust every addend for these products, including the reported tie.
    targeted = [
        (0x4170, 0x41B8),  # 15 * 23 = 345, a bf16 midpoint
        (0xC170, 0x41B8),  # -15 * 23
        (0x7F7F, 0x3F80),  # max finite * 1
        (0x0001, 0x3F80),  # smallest subnormal * 1
    ]
    for a_bits, b_bits in targeted:
        triples.extend((a_bits, b_bits, c_bits) for c_bits in range(65536))

    failures = 0
    for a_bits, b_bits, c_bits in triples:
        got = fn(BF16(a_bits), BF16(b_bits), BF16(c_bits)).bits
        want = ref_fma(a_bits, b_bits, c_bits)
        ok = is_nan_bits(got) if is_nan_bits(want) else got == want
        if not ok:
            failures += 1
            if failures <= 10:
                print(
                    f"FAIL fma: a=0x{a_bits:04X} b=0x{b_bits:04X} "
                    f"c=0x{c_bits:04X} got=0x{got:04X} want=0x{want:04X}"
                )

    if failures:
        print(f"exact fma: {len(triples)} triples, {failures} failures")
        return 1
    print(f"exact fma: {len(triples)} triples, 0 failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
