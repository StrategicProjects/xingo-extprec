"""Concrete counterexamples to the mutated statements of check.sh.

A mutant rejected by the tools shows that the proof is sensitive to the
change; a counterexample shows, in addition, that the mutated statement is
false. Each case below runs the mutated algorithm in binary64 (Python floats
are binary64 with round-to-nearest-even; math.fma is a correctly rounded
FMA, Python >= 3.13) and compares with exact rationals. It also checks that
the original algorithm satisfies the proved statement on the same input.

Run: python3 -I proofs/mutants/counterexamples.py   (exit status 0 = all
mutated statements refuted, all originals satisfied).
"""
import math
import sys
from fractions import Fraction as Q

U = Q(2) ** -53
ETA = Q(2) ** -1074
failures = 0


def report(name, ok, detail):
    global failures
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        failures += 1


# --- Algorithm M3 (dd squared), src/dd.rs -----------------------------------
def m3(xh, xl, two=2.0):
    ch = xh * xh
    cl1 = math.fma(xh, xh, -ch)
    c = math.fma(two * xh, xl, cl1)
    zh = ch + c
    z = zh - ch
    return zh, c - z


def m3_relerr(xh, xl, two=2.0):
    zh, zl = m3(xh, xl, two)
    x = Q(xh) + Q(xl)
    return abs(Q(zh) + Q(zl) - x * x) / (x * x) / (U * U)


xh = float.fromhex("0x1.00ba07a649e0ep+0")
xl = float.fromhex("-0x1.fd95502f7b724p-54")
assert xh + xl == xh  # a normalized double-double
r = m3_relerr(xh, xl)
report("M3 original", r <= 3, f"relative error {float(r):.4f} u^2 <= 3 u^2")
report("m3-bound-2.5u2", r > Q(5, 2), f"same input: {float(r):.4f} u^2 > 2.5 u^2")
r = m3_relerr(xh, xl, two=1.0)
report("m3-no-factor-2", r > 3, f"same input: {float(r):.3e} u^2 > 3 u^2")


# --- Algorithm M1 (dd times f64), src/dd.rs ----------------------------------
def m1(xh, xl, y, keep_cl1=True):
    ch = xh * y
    cl1 = math.fma(xh, y, -ch)
    c = math.fma(xl, y, cl1 if keep_cl1 else 0.0)
    zh = ch + c
    z = zh - ch
    return zh, c - z


def m1_relerr(xh, xl, y, keep_cl1=True):
    zh, zl = m1(xh, xl, y, keep_cl1)
    p = (Q(xh) + Q(xl)) * Q(y)
    return abs(Q(zh) + Q(zl) - p) / abs(p) / (U * U)


xh = float.fromhex("0x1.25e7ad9085ab2p+0")
xl = float.fromhex("-0x1.ed2e01a3c6a1ep-54")
y = float.fromhex("0x1.13e0fe19e6ba3p+0")
assert xh + xl == xh
r = m1_relerr(xh, xl, y)
report("M1 original", r <= 2 * (1 + Q(2) ** -37), f"relative error {float(r):.4f} u^2 <= 2u^2 (1 + 2^-37)")
report("m1-bound-1.5u2", r > Q(3, 2), f"same input: {float(r):.4f} u^2 > 1.5 u^2")
r = m1_relerr(xh, xl, y, keep_cl1=False)
report("m1-no-cl1", r > 2 * (1 + Q(2) ** -37), f"same input: {float(r):.3e} u^2")


# --- Step C5 of dot2_err, src/kernels.rs --------------------------------------
def c5_beta(n, e, shift=3, underflow=6.0 * 2.0**-1022, inflate=1.0 + 4.0 * 2.0**-52):
    a = float(n + shift) * 2.0**-53
    delta = a / (1.0 - 2.0 * a)
    return (delta * e + underflow) * inflate


def c5_bound(n, e):
    return Q(n + 3) * U / (1 - (2 * n + 5) * U) * Q(e) + n * ETA / 2


cases = [
    ("c5-inflate-1", 527222, float.fromhex("0x1.0b060d0bfff9ep-885"), dict(inflate=1.0)),
    ("c5-underflow-0", 1, 0.0, dict(underflow=0.0)),
    ("c5-terms-n+2", 1, 1.0, dict(shift=2)),
]
for name, n, e, mutation in cases:
    t = c5_bound(n, e)
    report(f"C5 original (n={n}, e={e.hex()})", Q(c5_beta(n, e)) >= t, "beta >= bound")
    report(name, Q(c5_beta(n, e, **mutation)) < t, "mutated beta < bound")

sys.exit(1 if failures else 0)
