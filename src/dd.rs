//! # Chapter 11.2 — Double-double arithmetic
//!
//! ## Purpose
//! [`DoubleDouble`]: the field operations, the square root and the
//! comparisons of numbers `x = x_h + x_ℓ` held as two `f64`. Each operation
//! is a short, fixed sequence of `f64` operations built from the error-free
//! transformations of Chapter 11.1, with a relative error bound that was
//! *proven* in the literature; the bounds are data
//! ([`DoubleDouble::ERROR_BOUNDS`]) and the tests check every operation
//! against them in exact rational arithmetic.
//!
//! ## Notation
//! As in Chapters 11 and 11.1: `RN`, `u = 2⁻⁵³`, `η = 2⁻¹⁰⁷⁴`. A pair `(h,
//! ℓ)` is *normalized* when `h = RN(h + ℓ)`. `2Sum`, `Fast2Sum` and `2Prod`
//! are Algorithms S, F and P of Chapter 11.1. "JMP" is Joldes, Muller &
//! Popescu (2017), "MR" Muller & Rideau (2022), "LLMPR" Lefèvre, Louvet,
//! Muller, Picot & Rideau (2023).
//!
//! ## Algorithms
//! Inputs are normalized pairs `x = (x_h, x_ℓ)`, `y = (y_h, y_ℓ)` or an
//! `f64` `y`; the output is the pair `(z_h, z_ℓ)`. Subtraction is addition
//! of the (exact) negation.
//!
//! **Algorithm A1 (`dd + f64`; DWPlusFP, JMP Algorithm 4).**
//!
//! 1. `(s_h, s_ℓ) ← 2Sum(x_h, y)`.
//! 2. `v ← RN(x_ℓ + s_ℓ)`.
//! 3. `(z_h, z_ℓ) ← Fast2Sum(s_h, v)`.
//!
//! **Algorithm A2 (`dd + dd`; AccurateDWPlusDW, JMP Algorithm 6).**
//!
//! 1. `(s_h, s_ℓ) ← 2Sum(x_h, y_h)`; `(t_h, t_ℓ) ← 2Sum(x_ℓ, y_ℓ)`.
//! 2. `c ← RN(s_ℓ + t_h)`; `(v_h, v_ℓ) ← Fast2Sum(s_h, c)`.
//! 3. `w ← RN(t_ℓ + v_ℓ)`; `(z_h, z_ℓ) ← Fast2Sum(v_h, w)`.
//!
//! **Algorithm M1 (`dd · f64`; DWTimesFP3, JMP Algorithm 9).**
//!
//! 1. `(c_h, c_ℓ1) ← 2Prod(x_h, y)`.
//! 2. `c_ℓ3 ← RN(c_ℓ1 + x_ℓ y)` (one FMA).
//! 3. `(z_h, z_ℓ) ← Fast2Sum(c_h, c_ℓ3)`.
//!
//! **Algorithm M2 (`dd · dd`; DWTimesDW3, JMP Algorithm 12).**
//!
//! 1. `(c_h, c_ℓ1) ← 2Prod(x_h, y_h)`.
//! 2. `t_ℓ0 ← RN(x_ℓ y_ℓ)`; `t_ℓ1 ← RN(x_h y_ℓ + t_ℓ0)`; `c_ℓ2 ← RN(t_ℓ1 +
//!    x_ℓ y_h)` (two FMAs).
//! 3. `c_ℓ3 ← RN(c_ℓ1 + c_ℓ2)`; `(z_h, z_ℓ) ← Fast2Sum(c_h, c_ℓ3)`.
//!
//! **Algorithm M3 (`dd²`; DWPow2).** The specialization of the two-FMA
//! product (JMP Algorithm 11) to equal operands.
//!
//! 1. `(c_h, c_ℓ1) ← 2Prod(x_h, x_h)`.
//! 2. `c ← RN(2 x_h x_ℓ + c_ℓ1)` (one FMA; `2 x_h` is exact).
//! 3. `(z_h, z_ℓ) ← Fast2Sum(c_h, c)`.
//!
//! **Algorithm M4 (DWTimesFP1, JMP Algorithm 7).** Used inside D2 only.
//!
//! 1. `(c_h, c_ℓ1) ← 2Prod(x_h, y)`; `c_ℓ2 ← RN(x_ℓ y)`.
//! 2. `(t_h, t_ℓ1) ← Fast2Sum(c_h, c_ℓ2)`; `t_ℓ2 ← RN(t_ℓ1 + c_ℓ1)`.
//! 3. `(z_h, z_ℓ) ← Fast2Sum(t_h, t_ℓ2)`.
//!
//! **Algorithm D1 (`dd / f64`; DWDivFP3, JMP Algorithm 15).**
//!
//! 1. `t_h ← RN(x_h / y)`; `(π_h, π_ℓ) ← 2Prod(t_h, y)`.
//! 2. `δ_h ← RN(x_h − π_h)`; `δ′ ← RN(δ_h − π_ℓ)`; `δ ← RN(δ′ + x_ℓ)`.
//! 3. `t_ℓ ← RN(δ / y)`; `(z_h, z_ℓ) ← Fast2Sum(t_h, t_ℓ)`.
//!
//! **Algorithm D2 (`dd / dd`; DWDivDW2, JMP Algorithm 17).**
//!
//! 1. `t_h ← RN(x_h / y_h)`; `(r_h, r_ℓ) ← M4(y, t_h)`.
//! 2. `π_h ← RN(x_h − r_h)`; `δ_ℓ ← RN(x_ℓ − r_ℓ)`; `δ ← RN(π_h + δ_ℓ)`.
//! 3. `t_ℓ ← RN(δ / y_h)`; `(z_h, z_ℓ) ← Fast2Sum(t_h, t_ℓ)`.
//!
//! D2 keeps exact quotients exact: when `x / y` is an `f64` `q` (in
//! particular `x/x = 1`, `(q y)/y = q`, and `(a b)/b = a` for `x` the
//! exact product of two `f64`), `t_h` is `q` or a neighbour of it, the
//! residual `x − y t_h` is formed without error and `t_ℓ` is exactly `q −
//! t_h`. A division through a reciprocal (DWDivDW3, JMP Algorithm 18:
//! bound `9,8u²`, 31 flops) does not have this property, which is why it
//! is not used here (Chapter 11, "Why not DWDivDW3"). The property is
//! tested, not proven, for the cases named.
//!
//! **Algorithm R (`√dd`; SQRTDWtoDW, LLMPR Algorithm 8).** For `x > 0`.
//!
//! 1. `s_h ← RN(√x_h)`.
//! 2. `ρ₁ ← RN(x_h − s_h²)` (one FMA; exact); `ρ₂ ← RN(x_ℓ + ρ₁)`.
//! 3. `s_ℓ ← RN(ρ₂ / (2 s_h))`; `(z_h, z_ℓ) ← Fast2Sum(s_h, s_ℓ)`.
//!
//! **Step E (exceptional results), the last step of every algorithm.** Let
//! `g` be the plain `f64` result of the operation on the leading components
//! (`x_h + y_h`, `x_h y_h`, `x_h / y_h`, `√x_h`). If `z_h` is finite,
//! return `(z_h, z_ℓ)` — with `z_ℓ` replaced by `0` if it is not finite
//! (possible only in `from_sum`, invariant T1). If `z_h` is NaN, or `g` is
//! not finite, return `(g, 0)`: the IEEE result (`∞ − ∞` is NaN, `1/0 =
//! ∞`, `x/∞ = 0`, …), or the plain `f64` result when an intermediate
//! quantity overflowed although the result did not (operands within a
//! factor of four of `Ω`). Otherwise (`z_h` infinite with `g` finite: the
//! low-order terms carried the sum over `Ω`) return `(z_h, 0)`.
//!
//! ## Bounds, and where they hold
//! Let **the safe range** be: finite operands that are zero or of magnitude
//! in `[2⁻⁴⁸⁰, 2⁴⁸⁰]`. There no operation overflows and every `2Prod` is in
//! its exact range (invariant T2: the products formed are near `x_h y_h`,
//! `x_h` or `y_h t_h ≈ x_h`, all of magnitude at least `2⁻⁹⁶⁰`). In the
//! safe range the relative error `|z − z*| / |z*|` of the result `z`
//! against the exact `z*` is at most:
//!
//! | operation | algorithm | bound | source | flops |
//! |---|---|---|---|---|
//! | `dd + f64` | A1 | `2u²/(1 − 2u) < 2u² + 5u³` | JMP §3 (DWPlusFP) | 10 |
//! | `dd + dd` | A2 | `3u²/(1 − 4u) < 3u² + 13u³` | JMP Thm 3.1 | 20 |
//! | `dd · f64` | M1 | `2u²` | JMP (DWTimesFP3) | 6 |
//! | `dd · dd` | M2 | `4u²` | MR (JMP proved `5u²`) | 9 |
//! | `dd²` | M3 | `3u²` | Hubrecht & Melquiond 2026, §IV-B (binary64; see below) | 6 |
//! | `dd / f64` | D1 | `3u²` | JMP (DWDivFP3) | 10 |
//! | `dd / dd` | D2 | `15u² + 56u³` | JMP (DWDivDW2) | 18 |
//! | `√dd` | R | `25/8 u²` | LLMPR Theorem 3.6 | 8 |
//!
//! **The square root's constant.** LLMPR, Theorem 3.6 (read in the author
//! version, HAL hal-03482567v2; the proof is in their supplementary
//! material and formalized in Coq): "if `x = (x_h, x_ℓ)` is a double-word
//! number, `p ≥ 5`, `x ≥ 2^(2k)` where `k` is an integer larger than or
//! equal to `(e_min + p)/2`, and no overflow or underflow occurs, then the
//! relative error of Algorithm 8 is bounded by `(25/8) u²`, and that bound
//! is asymptotically optimal". For binary64 the condition is `x ≥ 2⁻⁹⁶⁸`,
//! which the safe range implies. A first-order analysis, written before
//! the text could be read, reproduces the constant. Take `1 ≤ x_h < 2`
//! (`ulp(x_h) = 2u`). Then `|s_h − √x_h| ≤ u`, `ρ₁` is exact with `|ρ₁| ≤
//! 2u√x_h`, and `r = x − s_h² = x_ℓ + ρ₁` has `|r| ≤ (2√x_h + 1)u < 4u`.
//! Three errors remain, relative to `√x`: rounding `ρ₂` (`≤ 2u²` absolute,
//! `u²/x_h` after the division by `2 s_h`), rounding `s_ℓ` (`|s_ℓ| < 2u`,
//! hence `≤ u²` absolute, `u²/√x_h`), and the truncation of the Newton
//! step, `r² / (8 x_h²) ≤ (2√x_h + 1)² u² / (8 x_h²)`. Their sum is
//! largest as `x_h → 1`: `1 + 1 + 9/8 = 25/8`. For `2 ≤ x_h < 4` the same
//! count gives at most `2,44 u²`. The tests check `25/8` against exact
//! arithmetic (largest error observed on 20 000 random operands: `2,69
//! u²`).
//!
//! **The square's constant.** Hubrecht & Melquiond (2026, §IV-B) bound the
//! relative error of exactly the code of M3 by `3u²`: a proof generated by
//! the tool Gappa for binary64 with round-to-nearest-even and an
//! *unbounded exponent range* (so it says nothing about underflow or
//! overflow, which the safe range excludes), for an operand `x_h = RN(x)`,
//! `x_ℓ = x − x_h`. Their statement is about `c_h + c` before the final
//! Fast2Sum, which is error-free here (`|c| ≤ |c_h|`). It is not a theorem
//! for a general precision. The tests check `3u²` against exact
//! arithmetic (largest error observed: `2,86 u²`).
//!
//! The additions never leave the model of the proofs when subnormal
//! numbers occur (invariant T1): their bounds hold on the whole finite
//! range short of overflow, where an operand of magnitude `Ω` counts as
//! overflow: an intermediate step of its 2Sum can overflow and step E then
//! returns the rounded `f64` sum (measured: relative error `2⁵² u²`).
//! *The corner* is exactly this: the first operand of the 2Sum (`x_h` of
//! `x + y`, `a` of `from_sum(a, b)`) is `±Ω` and `x_h + y` is a tie in the
//! top binade `[2¹⁰²³, Ω]`, rounded away from zero; the sum is then finite
//! and the error term NaN. Operands in the safe range (`≤ 2⁴⁸⁰`) never
//! reach it. `kernels::dot2_err` detects it and answers `β = +∞` (Chapter
//! 11.5, the proof of C5, "Overflow"), so its bound stays true there.
//!
//! *Not repaired, by decision* (issue xingo-extprec#6, closed as "won't
//! fix (cost)"). Recomputing the error term by an ordered Fast2Sum makes
//! the corner exact and keeps the bounds (a proof and exact-rational tests
//! were written in task I02, not merged), but no variant is free. The test as a branch of step E costs
//! nothing in a dependent chain, but it keeps the compiler from turning
//! step E into selects, so the loops it vectorizes on its own (the scalar
//! `axpy` and `gemm_sub`, the naive dense kernel) lose 35–67 %. As a select
//! on the error term it adds latency to every addition (+15–23 %). Both
//! were measured on x86-64-v3 (task I02).
//!
//! Outside the safe range the other operations
//! lose the bound — a `2Prod` that underflows contributes an absolute error
//! of order `η` — but keep the invariants below.
//!
//! The sloppy addition of the QD library (11 flops) is not offered: its
//! relative error is unbounded for operands of opposite signs (JMP §3).
//!
//! ## Invariants
//! (D1) *Normalized.* Every finite value satisfies `h = RN(h + ℓ)` (hence
//!      `|ℓ| ≤ ulp(h)/2`); `h = ±0` implies `ℓ = 0`.
//! (D2) *Non-finite.* If `h` is infinite then `ℓ = 0`; the value is
//!      "infinite" or "NaN" as `h` says, and the operations follow IEEE on
//!      `h` (step E).
//! (D3) *Order.* The derived lexicographic order on `(h, ℓ)` is the order
//!      of the values `h + ℓ`, total on the non-NaN values.
//! (D4) *Conversions.* `from_f64` is exact; `to_f64` is `h`, the `f64`
//!      nearest to the value.
//! (D5) *Determinism.* Every operation is the fixed sequence of `f64`
//!      operations above, followed by step E: no data-dependent choice of
//!      algorithm, no reassociation, the same bits on every platform.
//!
//! [`DoubleDouble::check_invariants`] tests (D1) and (D2);
//! [`DoubleDouble::from_parts`] asserts them in debug builds.
//!
//! ## Comparisons
//! This module compares `f64` values exactly (`==`, `<`): exactness is its
//! subject, as tolerances are the subject of `xingo_core::Tolerances`. No
//! tolerance appears anywhere in the crate.
//!
//! ## References
//! Joldes, Muller & Popescu, "Tight and rigorous error bounds for basic
//! building blocks of double-word arithmetic", ACM TOMS 44(2) (2017).
//! Muller & Rideau, "Formalization of double-word arithmetic, and comments
//! on 'Tight and rigorous error bounds for basic building blocks of
//! double-word arithmetic'", ACM TOMS 48(1) (2022). Lefèvre, Louvet,
//! Muller, Picot & Rideau, "Accurate calculation of Euclidean norms using
//! double-word arithmetic", ACM TOMS 49(1) (2023). Stloukal & Jézéquel,
//! "Lord of the Precision, the Two Floats: double-word arithmetic for
//! complex numbers", arXiv:2611.08371 (2026). Hubrecht & Melquiond,
//! "Verifying code that uses error-free transformations", 33rd IEEE
//! Symposium on Computer Arithmetic (ARITH 2026). Hida, Li & Bailey,
//! "Algorithms for quad-double precision floating point arithmetic",
//! ARITH-15 (2001) — read, not copied.

#![allow(clippy::float_cmp)] // exact comparisons are the subject (see "Comparisons")

use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::eft::{fast_two_sum, two_prod, two_sum};

/// A double-double number: the exact sum `hi + lo` of two `f64`, normalized
/// (invariant D1). About 106 bits of significand (32 decimal digits), the
/// exponent range of `f64`.
///
/// `#[repr(C)]` with `hi` first: a slice `&[DoubleDouble]` is an array of
/// `(hi, lo)` pairs, the layout the SIMD kernels read.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
#[repr(C)]
pub struct DoubleDouble {
    hi: f64,
    lo: f64,
}

/// The proven relative error bound of one operation of [`DoubleDouble`],
/// as a multiple of `u² = 2⁻¹⁰⁶`: an `f64` that is **at least** the full
/// proven expression of the chapter's table (rounded up where the
/// expression has higher-order terms), so that `units_of_u2 · u²` is itself
/// a rigorous bound.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ErrorBound {
    /// The operation, as written in the chapter's table (`"dd + dd"`, …).
    pub operation: &'static str,
    /// `c` in "relative error `≤ c u²`", with `c` at least the proven
    /// expression (see [`DoubleDouble::ERROR_BOUNDS`]).
    pub units_of_u2: f64,
}

/// Step E: the exceptional results (the chapter). `g` is the `f64` result
/// of the operation on the leading components.
#[inline]
fn finish(zh: f64, zl: f64, g: f64) -> DoubleDouble {
    if zh.is_finite() {
        // `zl` is finite after a Fast2Sum; after a bare 2Sum it is not when
        // an operand is ±Ω (invariant T1 of Chapter 11.1).
        DoubleDouble {
            hi: zh,
            lo: if zl.is_finite() { zl } else { 0.0 },
        }
    } else if zh.is_nan() || !g.is_finite() {
        DoubleDouble { hi: g, lo: 0.0 }
    } else {
        DoubleDouble { hi: zh, lo: 0.0 }
    }
}

/// Algorithm M4 (DWTimesFP1), without step E: its caller, D2, ends in one.
#[inline]
fn times_fp1(x: DoubleDouble, y: f64) -> (f64, f64) {
    let (ch, cl1) = two_prod(x.hi, y);
    let cl2 = x.lo * y;
    let (th, tl1) = fast_two_sum(ch, cl2);
    let tl2 = tl1 + cl1;
    fast_two_sum(th, tl2)
}

/// `x · 2ᵏ` for an `f64`, by at most three multiplications by powers of
/// two; exact unless the result is subnormal or overflows.
#[inline]
fn scale_f64(mut x: f64, k: i32) -> f64 {
    // 2^1000 and 2^-1000.
    const UP: f64 = f64::from_bits((1023 + 1000) << 52);
    const DOWN: f64 = f64::from_bits((1023 - 1000) << 52);
    // Beyond ±2200 every nonzero finite value overflows or vanishes.
    let mut k = k.clamp(-2200, 2200);
    while k > 1000 {
        x *= UP;
        k -= 1000;
    }
    while k < -1000 {
        x *= DOWN;
        k += 1000;
    }
    let biased = u64::try_from(1023 + k).expect("|k| <= 1000");
    x * f64::from_bits(biased << 52)
}

impl DoubleDouble {
    /// Zero.
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };
    /// One.
    pub const ONE: Self = Self { hi: 1.0, lo: 0.0 };
    /// `2⁻¹⁰⁴`, the "machine epsilon" (`Real::EPSILON`).
    pub const EPSILON: Self = Self {
        hi: 4.930_380_657_631_324e-32,
        lo: 0.0,
    };
    /// Positive infinity (invariant D2).
    pub const INFINITY: Self = Self {
        hi: f64::INFINITY,
        lo: 0.0,
    };
    /// A NaN (invariant D2).
    pub const NAN: Self = Self {
        hi: f64::NAN,
        lo: 0.0,
    };
    // The 19 stable constants of `core::f64::consts` (Chapter 11.11, L2.4):
    // each is the nearest double-double `(RN(c), RN(c − RN(c)))`, relative
    // error at most `u²(1 + u)`, checked bit for bit by the tests against two
    // independent oracles.
    /// `π`, the nearest double-double: `3.141592653589793e0 + 1.2246467991473532e-16`.
    pub const PI: Self = Self {
        hi: f64::from_bits(0x4009_21fb_5444_2d18),
        lo: f64::from_bits(0x3ca1_a626_3314_5c07),
    };
    /// `2π`, the nearest double-double: `6.283185307179586e0 + 2.4492935982947064e-16`.
    pub const TAU: Self = Self {
        hi: f64::from_bits(0x4019_21fb_5444_2d18),
        lo: f64::from_bits(0x3cb1_a626_3314_5c07),
    };
    /// `π/2`, the nearest double-double: `1.5707963267948966e0 + 6.123233995736766e-17`.
    pub const FRAC_PI_2: Self = Self {
        hi: f64::from_bits(0x3ff9_21fb_5444_2d18),
        lo: f64::from_bits(0x3c91_a626_3314_5c07),
    };
    /// `π/3`, the nearest double-double: `1.0471975511965979e0 + -1.072081766451091e-16`.
    pub const FRAC_PI_3: Self = Self {
        hi: f64::from_bits(0x3ff0_c152_382d_7366),
        lo: f64::from_bits(0xbc9e_e691_3347_c2a6),
    };
    /// `π/4`, the nearest double-double: `7.853981633974483e-1 + 3.061616997868383e-17`.
    pub const FRAC_PI_4: Self = Self {
        hi: f64::from_bits(0x3fe9_21fb_5444_2d18),
        lo: f64::from_bits(0x3c81_a626_3314_5c07),
    };
    /// `π/6`, the nearest double-double: `5.235987755982989e-1 + -5.360408832255455e-17`.
    pub const FRAC_PI_6: Self = Self {
        hi: f64::from_bits(0x3fe0_c152_382d_7366),
        lo: f64::from_bits(0xbc8e_e691_3347_c2a6),
    };
    /// `π/8`, the nearest double-double: `3.9269908169872414e-1 + 1.5308084989341915e-17`.
    pub const FRAC_PI_8: Self = Self {
        hi: f64::from_bits(0x3fd9_21fb_5444_2d18),
        lo: f64::from_bits(0x3c71_a626_3314_5c07),
    };
    /// `1/π`, the nearest double-double: `3.183098861837907e-1 + -1.9678676675182486e-17`.
    pub const FRAC_1_PI: Self = Self {
        hi: f64::from_bits(0x3fd4_5f30_6dc9_c883),
        lo: f64::from_bits(0xbc76_b01e_c541_7056),
    };
    /// `2/π`, the nearest double-double: `6.366197723675814e-1 + -3.935735335036497e-17`.
    pub const FRAC_2_PI: Self = Self {
        hi: f64::from_bits(0x3fe4_5f30_6dc9_c883),
        lo: f64::from_bits(0xbc86_b01e_c541_7056),
    };
    /// `2/√π`, the nearest double-double: `1.1283791670955126e0 + 1.533545961316588e-17`.
    pub const FRAC_2_SQRT_PI: Self = Self {
        hi: f64::from_bits(0x3ff2_0dd7_5042_9b6d),
        lo: f64::from_bits(0x3c71_ae3a_914f_ed80),
    };
    /// `√2`, the nearest double-double: `1.4142135623730951e0 + -9.667293313452913e-17`.
    pub const SQRT_2: Self = Self {
        hi: f64::from_bits(0x3ff6_a09e_667f_3bcd),
        lo: f64::from_bits(0xbc9b_dd34_13b2_6456),
    };
    /// `1/√2`, the nearest double-double: `7.071067811865476e-1 + -4.833646656726457e-17`.
    pub const FRAC_1_SQRT_2: Self = Self {
        hi: f64::from_bits(0x3fe6_a09e_667f_3bcd),
        lo: f64::from_bits(0xbc8b_dd34_13b2_6456),
    };
    /// `e`, the nearest double-double: `2.718281828459045e0 + 1.4456468917292502e-16`.
    pub const E: Self = Self {
        hi: f64::from_bits(0x4005_bf0a_8b14_5769),
        lo: f64::from_bits(0x3ca4_d57e_e2b1_013a),
    };
    /// `log₂ e`, the nearest double-double: `1.4426950408889634e0 + 2.0355273740931033e-17`.
    pub const LOG2_E: Self = Self {
        hi: f64::from_bits(0x3ff7_1547_652b_82fe),
        lo: f64::from_bits(0x3c77_77d0_ffda_0d24),
    };
    /// `log₂ 10`, the nearest double-double: `3.321928094887362e0 + 1.661617516973592e-16`.
    pub const LOG2_10: Self = Self {
        hi: f64::from_bits(0x400a_934f_0979_a371),
        lo: f64::from_bits(0x3ca7_f249_5fb7_fa6d),
    };
    /// `log₁₀ e`, the nearest double-double: `4.342944819032518e-1 + 1.098319650216765e-17`.
    pub const LOG10_E: Self = Self {
        hi: f64::from_bits(0x3fdb_cb7b_1526_e50e),
        lo: f64::from_bits(0x3c69_5355_baaa_fad3),
    };
    /// `log₁₀ 2`, the nearest double-double: `3.010299956639812e-1 + -2.8037281277851704e-18`.
    pub const LOG10_2: Self = Self {
        hi: f64::from_bits(0x3fd3_4413_509f_79ff),
        lo: f64::from_bits(0xbc49_dc1d_a994_fd21),
    };
    /// `ln 2`, the nearest double-double: `6.931471805599453e-1 + 2.3190468138462996e-17`.
    pub const LN_2: Self = Self {
        hi: f64::from_bits(0x3fe6_2e42_fefa_39ef),
        lo: f64::from_bits(0x3c7a_bc9e_3b39_803f),
    };
    /// `ln 10`, the nearest double-double: `2.302585092994046e0 + -2.1707562233822494e-16`.
    pub const LN_10: Self = Self {
        hi: f64::from_bits(0x4002_6bb1_bbb5_5516),
        lo: f64::from_bits(0xbcaf_48ad_494e_a3e9),
    };
    /// The bounds of the chapter's table, for the tests and for callers
    /// that propagate errors. The subtractions have the bounds of the
    /// additions; `1 / dd` is `dd / dd`. Each `units_of_u2` is an `f64` at
    /// least the proven expression, with `u = 2⁻⁵³`:
    ///
    /// | operation | proven bound | `units_of_u2` |
    /// |---|---|---|
    /// | `dd + f64` | `2u²/(1 − 2u)` | `2 + 2⁻⁵⁰` (the expression rounded up) |
    /// | `dd + dd` | `3u²/(1 − 4u)` | `3 + 2⁻⁴⁹` (rounded up) |
    /// | `dd · f64` | `2u²` | `2` (exact) |
    /// | `dd · dd` | `4u²` | `4` (exact) |
    /// | `dd²` | `3u²` | `3` (exact) |
    /// | `dd / f64` | `3u²` | `3` (exact) |
    /// | `dd / dd` | `15u² + 56u³` | `15 + 2⁻⁴⁷` (rounded up) |
    /// | `√dd` | `(25/8) u²` | `3.125` (exact) |
    ///
    /// The test `error_bounds_dominate_the_proven_expressions` checks each
    /// row in exact rational arithmetic.
    pub const ERROR_BOUNDS: &'static [ErrorBound] = &[
        ErrorBound {
            operation: "dd + f64",
            // 2/(1 − 2u) = 2 + 2⁻⁵¹ + …, rounded up: 2 + 2⁻⁵⁰.
            units_of_u2: f64::from_bits(0x4000_0000_0000_0002),
        },
        ErrorBound {
            operation: "dd + dd",
            // 3/(1 − 4u) = 3 + 3·2⁻⁵¹ + …, rounded up: 3 + 2⁻⁴⁹.
            units_of_u2: f64::from_bits(0x4008_0000_0000_0004),
        },
        ErrorBound {
            operation: "dd · f64",
            units_of_u2: 2.0,
        },
        ErrorBound {
            operation: "dd · dd",
            units_of_u2: 4.0,
        },
        ErrorBound {
            operation: "dd²",
            units_of_u2: 3.0,
        },
        ErrorBound {
            operation: "dd / f64",
            units_of_u2: 3.0,
        },
        ErrorBound {
            operation: "dd / dd",
            // 15 + 56u = 15 + 3.5·2⁻⁴⁹, rounded up: 15 + 2⁻⁴⁷.
            units_of_u2: f64::from_bits(0x402e_0000_0000_0004),
        },
        ErrorBound {
            operation: "√dd",
            units_of_u2: 3.125,
        },
    ];

    /// The value of an `f64`, exactly.
    #[must_use]
    pub const fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    /// `a + b` exactly, for any two `f64` ([`two_sum`]).
    #[inline]
    #[must_use]
    pub fn from_sum(a: f64, b: f64) -> Self {
        let (s, e) = two_sum(a, b);
        finish(s, e, s)
    }

    /// `a · b` exactly, for any two `f64` in the range of [`two_prod`]
    /// (outside it: within `η/2`, and still normalized).
    #[inline]
    #[must_use]
    pub fn from_product(a: f64, b: f64) -> Self {
        let (p, e) = two_prod(a, b);
        // A no-op in the exact range of `two_prod`; below it the rounded
        // error term can be exactly half an ulp of `p` on the wrong side
        // of the tie (invariant T3), and this exact sum renormalizes.
        let (zh, zl) = fast_two_sum(p, e);
        finish(zh, zl, p)
    }

    /// The pair `(hi, lo)` as given, **requires** it normalized (D1); a
    /// `debug_assert!`. For deserialization and tests.
    ///
    /// The requirement is unchecked in release builds, and every bound of
    /// this chapter assumes it: use [`DoubleDouble::from_sum`] for an
    /// arbitrary pair.
    #[inline]
    #[must_use]
    pub fn from_parts(hi: f64, lo: f64) -> Self {
        let x = Self { hi, lo };
        debug_assert!(
            x.check_invariants().is_ok(),
            "from_parts({hi:e}, {lo:e}): {:?}",
            x.check_invariants()
        );
        x
    }

    /// The pair `(hi, lo)` as given, unchecked: for the SIMD kernels, which
    /// store pairs computed by the algorithms of this chapter (Chapter
    /// 11.6), and for the literal tables of Chapter 11.11.
    #[inline]
    pub(crate) const fn from_raw(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    /// The leading component: the `f64` nearest to the value (D4).
    #[must_use]
    pub const fn hi(self) -> f64 {
        self.hi
    }

    /// The trailing component, `|lo| ≤ ulp(hi)/2`.
    #[must_use]
    pub const fn lo(self) -> f64 {
        self.lo
    }

    /// The nearest `f64` (the leading component).
    #[must_use]
    pub const fn to_f64(self) -> f64 {
        self.hi
    }

    /// `|x|`, exact.
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        if self.hi.is_sign_negative() {
            -self
        } else {
            self
        }
    }

    /// `√x`; NaN for negative `x`, `±0` for `±0`.
    #[inline]
    #[must_use]
    pub fn sqrt(self) -> Self {
        if self.hi == 0.0 {
            return self;
        }
        // Algorithm R.
        let sh = self.hi.sqrt();
        let rho1 = (-sh).mul_add(sh, self.hi);
        let rho2 = self.lo + rho1;
        let sl = rho2 / (2.0 * sh);
        let (zh, zl) = fast_two_sum(sh, sl);
        finish(zh, zl, sh)
    }

    /// `1 / x`.
    #[inline]
    #[must_use]
    pub fn recip(self) -> Self {
        Self::ONE / self
    }

    /// `x²`, cheaper than `x · x` (one product fewer), same bound.
    #[inline]
    #[must_use]
    pub fn square(self) -> Self {
        // Algorithm M3.
        let (ch, cl1) = two_prod(self.hi, self.hi);
        let c = (2.0 * self.hi).mul_add(self.lo, cl1);
        let (zh, zl) = fast_two_sum(ch, c);
        finish(zh, zl, ch)
    }

    /// `x · 2ᵏ`, exact barring overflow and underflow.
    #[inline]
    #[must_use]
    pub fn ldexp(self, k: i32) -> Self {
        let h = scale_f64(self.hi, k);
        let l = scale_f64(self.lo, k);
        // Exact scalings keep the pair normalized; when a component became
        // subnormal the sum is exact (T1) and renormalizes it.
        let (zh, zl) = fast_two_sum(h, l);
        finish(zh, zl, h)
    }

    /// Neither infinite nor NaN.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.hi.is_finite()
    }

    /// NaN (D2).
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.hi.is_nan()
    }

    /// The larger value, with the NaN rule of `f64::max`.
    #[inline]
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        if self.is_nan() || self < other {
            other
        } else {
            self
        }
    }

    /// The smaller value, with the NaN rule of `f64::min`.
    #[inline]
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        if self.is_nan() || other < self {
            other
        } else {
            self
        }
    }

    /// Invariants (D1) and (D2).
    ///
    /// # Errors
    /// The text of the violated invariant.
    pub fn check_invariants(self) -> Result<(), &'static str> {
        if self.hi.is_nan() {
            return Ok(());
        }
        if self.hi.is_infinite() {
            return if self.lo == 0.0 {
                Ok(())
            } else {
                Err("D2: an infinite value has a nonzero trailing component")
            };
        }
        if !self.lo.is_finite() {
            return Err("D1: a finite value has a non-finite trailing component");
        }
        if self.hi == 0.0 && self.lo != 0.0 {
            return Err("D1: a zero leading component with a nonzero trailing one");
        }
        if self.hi + self.lo != self.hi {
            return Err("D1: the pair is not normalized, hi != fl(hi + lo)");
        }
        Ok(())
    }
}

impl Add for DoubleDouble {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        // Algorithm A2.
        let (sh, sl) = two_sum(self.hi, rhs.hi);
        let (th, tl) = two_sum(self.lo, rhs.lo);
        let c = sl + th;
        let (vh, vl) = fast_two_sum(sh, c);
        let w = tl + vl;
        let (zh, zl) = fast_two_sum(vh, w);
        finish(zh, zl, sh)
    }
}

/// Mixed operation with an `f64` right operand: the cheaper algorithm of
/// the chapter's table, not a promotion.
impl Add<f64> for DoubleDouble {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self {
        // Algorithm A1.
        let (sh, sl) = two_sum(self.hi, rhs);
        let v = self.lo + sl;
        let (zh, zl) = fast_two_sum(sh, v);
        finish(zh, zl, sh)
    }
}

impl Sub for DoubleDouble {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

/// Mixed operation with an `f64` right operand: the cheaper algorithm of
/// the chapter's table, not a promotion.
impl Sub<f64> for DoubleDouble {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self {
        self + (-rhs)
    }
}

impl Mul for DoubleDouble {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Algorithm M2.
        let (ch, cl1) = two_prod(self.hi, rhs.hi);
        let tl0 = self.lo * rhs.lo;
        let tl1 = self.hi.mul_add(rhs.lo, tl0);
        let cl2 = self.lo.mul_add(rhs.hi, tl1);
        let cl3 = cl1 + cl2;
        let (zh, zl) = fast_two_sum(ch, cl3);
        finish(zh, zl, ch)
    }
}

/// Mixed operation with an `f64` right operand: the cheaper algorithm of
/// the chapter's table, not a promotion.
impl Mul<f64> for DoubleDouble {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        // Algorithm M1.
        let (ch, cl1) = two_prod(self.hi, rhs);
        let cl3 = self.lo.mul_add(rhs, cl1);
        let (zh, zl) = fast_two_sum(ch, cl3);
        finish(zh, zl, ch)
    }
}

impl Div for DoubleDouble {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // Algorithm D2.
        let th = self.hi / rhs.hi;
        let (rh, rl) = times_fp1(rhs, th);
        let pih = self.hi - rh;
        let dl = self.lo - rl;
        let d = pih + dl;
        let tl = d / rhs.hi;
        let (zh, zl) = fast_two_sum(th, tl);
        finish(zh, zl, th)
    }
}

/// Mixed operation with an `f64` right operand: the cheaper algorithm of
/// the chapter's table, not a promotion.
impl Div<f64> for DoubleDouble {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        // Algorithm D1.
        let th = self.hi / rhs;
        let (pih, pil) = two_prod(th, rhs);
        let dh = self.hi - pih;
        let dt = dh - pil;
        let d = dt + self.lo;
        let tl = d / rhs;
        let (zh, zl) = fast_two_sum(th, tl);
        finish(zh, zl, th)
    }
}

macro_rules! assign {
    ($tra:ident, $fa:ident, $op:tt) => {
        impl $tra for DoubleDouble {
            #[inline]
            fn $fa(&mut self, rhs: Self) {
                *self = *self $op rhs;
            }
        }
    };
}
assign!(AddAssign, add_assign, +);
assign!(SubAssign, sub_assign, -);
assign!(MulAssign, mul_assign, *);
assign!(DivAssign, div_assign, /);

impl Neg for DoubleDouble {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

/// Left to right from [`DoubleDouble::ZERO`] (requirement R3 of `Real`).
impl Sum for DoubleDouble {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl From<f64> for DoubleDouble {
    fn from(x: f64) -> Self {
        Self::from_f64(x)
    }
}

/// `Real` for double-double (requirements R1 as reworded in E2, R2–R4 by
/// invariants D3–D5): the containers and factorizations of `xingo-sparse`
/// instantiate in `DoubleDouble` unchanged.
#[cfg(feature = "sparse")]
impl xingo_sparse::Real for DoubleDouble {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;
    const EPSILON: Self = Self::EPSILON;
    #[inline]
    fn from_f64(x: f64) -> Self {
        Self::from_f64(x)
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self.hi
    }
    #[inline]
    fn abs(self) -> Self {
        Self::abs(self)
    }
    #[inline]
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }
    #[inline]
    fn recip(self) -> Self {
        Self::recip(self)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
    #[inline]
    fn is_finite(self) -> bool {
        Self::is_finite(self)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        Self::max(self, other)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        Self::min(self, other)
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // L2.4 (Chapter 11.11): the tests may use the libm of `f64` (clippy.toml)
mod tests {
    use super::*;

    const U: f64 = 1.110_223_024_625_156_5e-16; // 2^-53
    type Dd = DoubleDouble;

    fn dd(hi: f64, lo: f64) -> Dd {
        Dd::from_parts(hi, lo)
    }

    #[test]
    fn constants() {
        assert_eq!(Dd::EPSILON.hi(), 2f64.powi(-104));
        assert_eq!(Dd::ZERO + Dd::ONE, Dd::ONE);
        assert!(Dd::NAN.is_nan() && !Dd::INFINITY.is_finite());
        assert_eq!(Dd::default(), Dd::ZERO);
        assert_eq!(Dd::ERROR_BOUNDS.len(), 8);
    }

    #[test]
    fn textbook_cases() {
        // 1 + 2^-80 is not an f64 but is a double-double.
        let tiny = 2f64.powi(-80);
        let x = Dd::ONE + tiny;
        assert_eq!((x.hi(), x.lo()), (1.0, tiny));
        assert_eq!(x - Dd::ONE, Dd::from_f64(tiny));
        assert_eq!(x - 1.0, Dd::from_f64(tiny));
        // (1 + 2^-30)(1 − 2^-30) = 1 − 2^-60 needs 61 bits.
        let e = 2f64.powi(-30);
        assert_eq!(
            Dd::from_f64(1.0 + e) * Dd::from_f64(1.0 - e),
            dd(1.0, -2f64.powi(-60))
        );
        assert_eq!(Dd::from_f64(1.0 + e) * (1.0 - e), dd(1.0, -2f64.powi(-60)));
        // Exact small integers.
        let six = Dd::from_f64(2.0) * Dd::from_f64(3.0);
        assert_eq!(six, Dd::from_f64(6.0));
        assert_eq!(six / 3.0, Dd::from_f64(2.0));
        assert_eq!(six / Dd::from_f64(3.0), Dd::from_f64(2.0));
        assert_eq!(Dd::from_f64(9.0).sqrt(), Dd::from_f64(3.0));
        assert_eq!(Dd::from_f64(3.0).square(), Dd::from_f64(9.0));
        assert_eq!(Dd::from_f64(4.0).recip(), Dd::from_f64(0.25));
        // 1/3: the trailing component is the error of the f64 quotient.
        let third = Dd::ONE / 3.0;
        assert_eq!(third.hi(), 1.0 / 3.0);
        assert_eq!(third.lo(), (1.0 / 3.0) * U / 2.0);
        assert_eq!(Dd::ONE / Dd::from_f64(3.0), third);
        // Exact quotients stay exact (the reason for Algorithm D2).
        assert_eq!(Dd::from_f64(53.0) / Dd::from_f64(53.0), Dd::ONE);
        assert_eq!(third / third, Dd::ONE);
        // √2 squared is 2 to the last bit of 106.
        let r = Dd::from_f64(2.0).sqrt();
        let err = (r.square() - 2.0).abs();
        assert!(err <= Dd::EPSILON * 8.0, "{err:?}");
    }

    #[test]
    fn exact_constructors() {
        assert_eq!(Dd::from_sum(1.0, U), dd(1.0, U));
        let x = 1.0 + f64::EPSILON;
        assert_eq!(
            Dd::from_product(x, x),
            dd(1.0 + 2.0 * f64::EPSILON, f64::EPSILON * f64::EPSILON)
        );
        assert_eq!(Dd::from(0.5), Dd::from_f64(0.5));
        assert_eq!(Dd::from_f64(0.1).to_f64().to_bits(), 0.1_f64.to_bits());
        assert!(Dd::from_sum(f64::MAX, f64::MAX).check_invariants().is_ok());
        // The spurious overflow of 2Sum at Ω (invariant T1): the leading
        // component survives.
        let b = -3.0 * 2f64.powi(970);
        assert!(!two_sum(f64::MAX, b).1.is_finite());
        assert_eq!(Dd::from_sum(f64::MAX, b), Dd::from_f64(f64::MAX + b));
        assert_eq!(Dd::from_f64(f64::MAX) + b, Dd::from_f64(f64::MAX + b));
        assert_eq!(Dd::from_sum(f64::MAX, f64::MAX), Dd::INFINITY);
        assert_eq!(Dd::from_product(f64::MAX, -2.0), -Dd::INFINITY);
    }

    #[test]
    fn assignments_and_sum() {
        let mut x = Dd::from_f64(1.5);
        x += Dd::ONE;
        x -= Dd::from_f64(0.5);
        x *= Dd::from_f64(4.0);
        x /= Dd::from_f64(2.0);
        assert_eq!(x, Dd::from_f64(4.0));
        let tiny = 2f64.powi(-70);
        let s: Dd = [1.0, tiny, -1.0].into_iter().map(Dd::from_f64).sum();
        assert_eq!(s, Dd::from_f64(tiny));
        assert_eq!(core::iter::empty::<Dd>().sum::<Dd>(), Dd::ZERO);
    }

    #[test]
    fn order_abs_max_min() {
        let tiny = 2f64.powi(-80);
        let a = Dd::ONE - tiny;
        let b = Dd::ONE;
        let c = Dd::ONE + tiny;
        assert!(a < b && b < c && a < c);
        assert!(c > a);
        assert_eq!((-c).abs(), c);
        assert_eq!(Dd::from_f64(-0.0).abs().hi().to_bits(), 0.0_f64.to_bits());
        assert_eq!(a.max(c), c);
        assert_eq!(a.min(c), a);
        assert_eq!(Dd::NAN.max(a), a);
        assert_eq!(a.max(Dd::NAN), a);
        assert_eq!(Dd::NAN.min(a), a);
        assert_eq!(a.min(Dd::NAN), a);
        assert!(Dd::NAN.partial_cmp(&a).is_none());
        assert!(-Dd::INFINITY < a && a < Dd::INFINITY);
    }

    #[test]
    fn non_finite_values_follow_ieee() {
        let inf = Dd::INFINITY;
        let one = Dd::ONE;
        assert_eq!(inf + one, inf);
        assert_eq!(inf + 1.0, inf);
        assert_eq!(one - inf, -inf);
        assert!((inf - inf).is_nan());
        assert_eq!(inf * one, inf);
        assert_eq!(inf * -2.0, -inf);
        assert!((inf * Dd::ZERO).is_nan());
        assert_eq!(one / Dd::ZERO, inf);
        assert_eq!(-one / 0.0, -inf);
        assert!((Dd::ZERO / Dd::ZERO).is_nan());
        assert_eq!(one / inf, Dd::ZERO);
        assert_eq!(one / f64::INFINITY, Dd::ZERO);
        assert!((inf / inf).is_nan());
        assert_eq!(inf.sqrt(), inf);
        assert!((-one).sqrt().is_nan());
        assert_eq!(inf.square(), inf);
        assert_eq!(Dd::ZERO.recip(), inf);
        assert!((Dd::NAN + one).is_nan() && (one * Dd::NAN).is_nan());
        assert!((Dd::NAN / one).is_nan() && Dd::NAN.sqrt().is_nan());
        // Overflow gives a well-formed infinity.
        let big = Dd::from_f64(f64::MAX);
        for z in [
            big + big,
            big * big,
            big / Dd::from_f64(1e-300),
            big.square(),
        ] {
            assert_eq!(z, inf);
            assert!(z.check_invariants().is_ok());
        }
        assert_eq!(inf.ldexp(-5), inf);
        assert_eq!(big.ldexp(1), inf);
    }

    #[test]
    fn signed_zero_square_roots() {
        assert_eq!(Dd::ZERO.sqrt().hi().to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            Dd::from_f64(-0.0).sqrt().hi().to_bits(),
            (-0.0_f64).to_bits()
        );
    }

    #[test]
    fn ldexp_is_exact() {
        let x = Dd::ONE / 3.0;
        let y = x.ldexp(700);
        assert_eq!(
            (y.hi(), y.lo()),
            (x.hi() * 2f64.powi(700), x.lo() * 2f64.powi(700))
        );
        assert_eq!(y.ldexp(-700), x);
        assert_eq!(x.ldexp(1500).ldexp(-1500), Dd::INFINITY);
        assert_eq!(x.ldexp(-900).ldexp(900), x);
        assert_eq!(x.ldexp(i32::MIN), Dd::ZERO);
        assert_eq!(x.ldexp(i32::MAX), Dd::INFINITY);
        assert_eq!(Dd::ZERO.ldexp(i32::MAX), Dd::ZERO);
        // Into the subnormal range the pair stays normalized.
        assert!(x.ldexp(-1060).check_invariants().is_ok());
    }

    #[test]
    fn check_invariants_rejects_bad_pairs() {
        let bad = |hi, lo| Dd { hi, lo }.check_invariants().is_err();
        assert!(bad(1.0, 1.0)); // not normalized
        assert!(bad(1.0, f64::EPSILON)); // lo = ulp(hi): hi + lo != hi
        assert!(bad(0.0, 1e-300));
        assert!(bad(f64::INFINITY, 1.0));
        assert!(bad(1.0, f64::NAN));
        assert!(Dd { hi: 1.0, lo: U }.check_invariants().is_ok()); // the tie
        assert!(
            Dd {
                hi: f64::NAN,
                lo: 3.0
            }
            .check_invariants()
            .is_ok()
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "from_parts")]
    fn from_parts_asserts_normalization() {
        let _ = Dd::from_parts(1.0, 1.0);
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn real_is_implemented() {
        use xingo_sparse::Real;
        fn third<T: Real>() -> T {
            T::ONE / T::from_f64(3.0)
        }
        let t = third::<Dd>();
        assert_eq!(t, Dd::ONE / 3.0);
        assert_eq!(Real::to_f64(t).to_bits(), (1.0_f64 / 3.0).to_bits());
        assert_eq!(<Dd as Real>::EPSILON, Dd::EPSILON);
        assert_eq!(
            Real::mul_add(t, Dd::from_f64(3.0), Dd::ONE),
            t * Dd::from_f64(3.0) + Dd::ONE
        );
        assert_eq!(Real::abs(-t), t);
        assert_eq!(Real::recip(Dd::from_f64(8.0)), Dd::from_f64(0.125));
        assert_eq!(Real::sqrt(Dd::from_f64(16.0)), Dd::from_f64(4.0));
        assert!(Real::is_finite(t) && !Real::is_finite(Dd::INFINITY));
        assert_eq!(Real::max(t, Dd::ONE), Dd::ONE);
        assert_eq!(Real::min(t, Dd::ONE), t);
    }
}
