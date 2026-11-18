//! # Chapter 11.9 — Intervals by computed directed rounding
//!
//! ## Purpose
//! [`Interval`]: closed intervals with `f64` endpoints whose operations
//! *contain* the exact result (invariant I1) and are as tight as the
//! endpoints allow (I3) — in ordinary round-to-nearest arithmetic, without
//! touching the rounding mode of the processor (question E3 of Chapter 11,
//! D-0109). They are the last step of a certificate: Chapter 11.10 turns a
//! double-double residual and its error bound into an interval, and the
//! caller tests it with the `certainly_*` predicates.
//!
//! ## Notation
//! `F` is the set of finite `f64` values, `RN` rounding to nearest, `u =
//! 2⁻⁵³`, `η = 2⁻¹⁰⁷⁴`, `Ω` the largest element of `F`. For a real `t`,
//! `▽t = max{f ∈ F ∪ {−∞} : f ≤ t}` and `△t = min{f ∈ F ∪ {+∞} : f ≥ t}`
//! are the two *directed roundings*; so `▽t = Ω` for every `t ≥ Ω`
//! (overflow never produces `+∞` as a lower endpoint) and `▽t = −∞` for `t
//! < −Ω`. `pred` and `succ` are the neighbours in `F ∪ {±∞}`
//! (`f64::next_down`, `f64::next_up`). An interval `X = [x̲, x̄]` has `x̲ ∈
//! F ∪ {−∞}`, `x̄ ∈ F ∪ {+∞}`, `x̲ ≤ x̄`, and stands for the set of **real**
//! numbers `{x ∈ R : x̲ ≤ x ≤ x̄}`: an infinite endpoint says "unbounded",
//! it is not an element. `∅` is the empty set.
//!
//! ## Algorithms
//! Every operation reduces to one or two calls of a *directed primitive*:
//! `a ∘ b` rounded `▽` or `△`, for endpoints `a`, `b`.
//!
//! **Algorithm N (directed rounding of an operation).** Input: endpoints
//! `a`, `b`, an operation `∘ ∈ {+, ·, /}` or `√`, a direction.
//!
//! N1. \[Infinite and zero operands.\] `a + b` with an infinite operand is
//!     that operand (the other is never the opposite infinity: N-pre
//!     below). `a · b` is `0` when either is zero — **also when the other
//!     is infinite**: an infinite endpoint bounds a set of reals, and
//!     every real times zero is zero — and otherwise, with an infinite
//!     operand, the infinity of the product's sign. `a / b` is `0` for `a
//!     = 0` or `b = ±∞`, and the infinity of the quotient's sign for `a =
//!     ±∞`. `√0 = 0`, `√(+∞) = +∞`. Return.
//! N2. \[Nearest.\] `v ← RN(a ∘ b)`, one `f64` operation.
//! N3. \[Overflow.\] If `v = +∞`: return `Ω` for `▽`, `+∞` for `△`. If `v =
//!     −∞`: return `−∞` for `▽`, `−Ω` for `△`.
//! N4. \[Residual sign.\] Compute `σ`, a number with the sign of the exact
//!     `a ∘ b − v` (zero iff `v` is exact):
//!     * `+`: the error term of Fast2Sum on the operands ordered by
//!       magnitude, exact whenever `v` is finite (invariant T1 of Chapter
//!       11.1);
//!     * `·`, with `|v| ≥ 2⁻⁹⁶⁸`: `σ = RN(a b − v)`, one FMA, exact
//!       (invariant T2: `|a b| ≥ 2⁻⁹⁶⁹`);
//!     * `/`, with `|a| ≥ 2⁻⁹⁶⁸`: `ρ = RN(a − v b)`, one FMA, and `σ = ρ`
//!       for `b > 0`, `−ρ` for `b < 0`;
//!     * `√`, with `a ≥ 2⁻⁹⁰⁰`: `σ = RN(a − v²)`, one FMA;
//!     * below those thresholds: step N6.
//! N5. \[Step.\] Return `pred(v)` if the direction is `▽` and `σ < 0`,
//!     `succ(v)` if it is `△` and `σ > 0`, and `v` otherwise.
//! N6. \[Underflow range.\] The FMA residual of N4 may itself be rounded
//!     there (to zero, in the worst case: the sign is lost), so the
//!     operation is redone `2²⁰⁰` times larger, where it is exact, and
//!     brought back by Algorithm U:
//!     * `·`: if `v = 0` (so `0 < |a b| ≤ η/2`) the result is known from
//!       the signs: `(0, η)` as `(▽, △)` for a positive product, `(−η, 0)`
//!       for a negative one. Otherwise `(w, δ) ← 2Prod(2²⁰⁰ s, ℓ)` for
//!       `s`, `ℓ` the operand of smaller and of larger magnitude, and
//!       return `U(w, δ)`.
//!     * `/`: `a′ ← 2²⁰⁰ a`, `w ← RN(a′ / b)`, `δ ← ± RN(a′ − w b)` with
//!       the sign rule of N4; return `U(w, δ)`.
//!     * `√`: return `2⁻¹⁰⁰` times the result of N2–N5 for `2²⁰⁰ a` (both
//!       scalings exact: a square root is never subnormal).
//!
//! **Algorithm U (directed rounding of `(w + δ) 2⁻²⁰⁰`).** Input `w ∈ F`,
//! `δ` with `|δ| < ulp(w)` of which only the sign is used, a direction.
//!
//! U1. `v ← RN(2⁻²⁰⁰ w)` (rounded only when `v` is subnormal).
//! U2. `τ ← w − 2²⁰⁰ v`, exact. `σ ← τ` if `τ ≠ 0`, else `σ ← δ`.
//! U3. Step N5 on `(v, σ)`.
//!
//! **Algorithm I (the interval operations).** With `X = [a, b]`, `Y = [c,
//! d]` non-empty (an empty operand gives `∅`):
//!
//! I1. \[`+`, `−`.\] `X + Y = [▽(a + c), △(b + d)]`, `X − Y = [▽(a − d),
//!     △(b − c)]`.
//! I2. \[`·`.\] By the signs of the operands, two products (four when both
//!     intervals contain zero in their interior): the table below, the
//!     first entry of each cell rounded `▽`, the second `△`.
//! I3. \[`/`, `0 ∉ Y`.\] Likewise: for `c > 0`, `[a/d, b/c]`, `[a/c, b/c]`
//!     or `[a/c, b/d]` as `X` is `≥ 0`, mixed or `≤ 0`; for `d < 0`,
//!     `[b/d, a/c]`, `[b/d, a/d]` or `[b/c, a/d]`.
//! I4. \[`/`, `0 ∈ Y`.\] The hull of `{x / y : x ∈ X, y ∈ Y, y ≠ 0}`: `[0,
//!     0]` if `X = [0, 0]` and `Y ≠ [0, 0]`; a half-line when zero is an
//!     endpoint of `Y` and `X` has one sign (`Y = [0, d]`: `[▽(a/d), +∞]`
//!     for `a ≥ 0`, `[−∞, △(b/d)]` for `b ≤ 0`; `Y = [c, 0]`: `[−∞,
//!     △(a/c)]` for `a ≥ 0`, `[▽(b/c), +∞]` for `b ≤ 0`); `[−∞, +∞]`
//!     otherwise — in particular for `Y = [0, 0]`, where no quotient is
//!     defined and the set is empty: "nothing is known" is returned rather
//!     than `∅`, so that a division by an exact zero can never *prove*
//!     anything downstream (invariant C3 of Chapter 11).
//! I5. \[`√`, `x²`, `|x|`.\] `√X = [▽√max(a, 0), △√b]`, or `∅` if `b < 0`.
//!     `X² = [▽a², △b²]` for `a ≥ 0`, `[▽b², △a²]` for `b ≤ 0`, `[0,
//!     △max(a², b²)]` otherwise. `|X|` and `−X` are exact.
//! I6. \[A point operand.\] `X ∘ y` for an `f64` `y` is `X ∘ [y, y]` with
//!     the case of I2–I3 chosen by the sign of `y` (two directed
//!     operations). A NaN or infinite `y` is not a real number: the result
//!     is `[−∞, +∞]` (C3), never a panic.
//!
//! The table of step I2 (rows: the sign of `X = [a, b]`; columns: that of
//! `Y = [c, d]`):
//!
//! | | `c ≥ 0` | `c < 0 < d` | `d ≤ 0` |
//! |---|---|---|---|
//! | `a ≥ 0` | `[ac, bd]` | `[bc, bd]` | `[bc, ad]` |
//! | `a < 0 < b` | `[ad, bd]` | `[min(ad, bc), max(ac, bd)]` | `[bc, ac]` |
//! | `b ≤ 0` | `[ad, bc]` | `[ad, ac]` | `[bd, ac]` |
//!
//! ## Proofs
//! *(N-pre) The primitives never see `∞ − ∞`, `0/0`, `∞/∞` or a NaN.* A
//! lower endpoint is never `+∞` and an upper one never `−∞` (I2), and
//! Algorithm I adds lower to lower and upper to upper (for `−`, after
//! negating the subtrahend); the table of I2 and step N1 remove `0 · ∞`;
//! in I3 the divisor is never zero, and whenever the divisor can be
//! infinite the dividend is an endpoint that the sign case makes finite
//! (`a` in `a/d` with `a ≥ 0`; `b` in `b/c` with `b ≤ 0`; …).
//!
//! *(N) Steps N2–N5 return the directed rounding.* `v = RN(t)` for the
//! exact `t = a ∘ b`, so `pred(v) ≤ t ≤ succ(v)`: `▽t` is `v` when `v ≤ t`
//! and `pred(v)` otherwise, and that is the sign of `t − v`. It remains
//! that `σ` has that sign.
//!
//! * `+`: Fast2Sum with `|hi| ≥ |lo|` returns the exact error for any
//!   finite sum, subnormal or not (T1).
//! * `·`: `|v| ≥ 2⁻⁹⁶⁸` gives `|a b| ≥ 2⁻⁹⁶⁸(1 − u) > 2⁻⁹⁶⁹`, the exact
//!   range of 2ProdFMA (T2).
//! * `/`: let `q = v` and `t = a − q b`. With `ulp` of a subnormal equal
//!   to `η`, `t` is an integer multiple of `g = ulp(q) ulp(b)` and `|t| ≤
//!   ½ ulp(q) |b| < 2⁵² g`, so `t ∈ F` as soon as `g ≥ η`. For normal `q`
//!   and `b` that is `e_q + e_b ≥ −970`, implied by `|a| ≥ 2⁻⁹⁶⁸` (`e_q +
//!   e_b ≥ e_a − 1`); a subnormal `q` forces `|b| > 2⁵⁴` and a subnormal
//!   `b` forces `|q| > 2⁵⁴`, and `g ≥ η` again. So the FMA returns `t`
//!   exactly, and `a/b − q = t/b`.
//! * `√`: with `s = v`, `t = a − s²` is a multiple of `ulp(s)²` and `|t| ≤
//!   ulp(s) s + ¼ ulp(s)² `, at most `2⁵³` such units; so `t ∈ F` when
//!   `ulp(s)² ≥ η`, i.e. `a ≥ 2⁻⁹⁷⁰`. (The thresholds `2⁻⁹⁶⁸` and `2⁻⁹⁰⁰`
//!   are sufficient, not sharp.)
//!
//! *(U) Algorithm U.* `2²⁰⁰ v` is exact (`|v| ≤ 2⁻⁷⁶⁷`). If `v` is normal,
//! U1 was exact and `τ = 0`. Otherwise `2²⁰⁰ v` is `w` rounded to the grid
//! `2⁻⁸⁷⁴ Z`, both are multiples of `ulp(w)` and `|τ| ≤ |w|`, so `τ ∈ F`
//! and the subtraction is exact. The exact value is `2⁻²⁰⁰(w + δ)`, and
//! `(w + δ) − 2²⁰⁰ v = τ + δ`: a nonzero `τ` is a multiple of `ulp(w) >
//! |δ|` and decides the sign; if `τ = 0`, `δ` does. In step N6 the
//! hypotheses hold: for the product, `v ≠ 0` gives `|a b| > η/2`, so
//! `2²⁰⁰ |a b| > 2⁻⁸⁷⁵` is in the exact range of 2Prod, and `|s| < 2⁻⁴⁸³`
//! makes `2²⁰⁰ s` exact; for the quotient, `|a′| ≥ 2⁻⁸⁷⁴` puts `a′ − w b`
//! under the argument of (N) — when `w` underflows the residual may be
//! rounded, but it is a nonzero multiple of `η ulp(b)` with `|b| > 2¹⁴⁸`,
//! far above `η`: its sign survives, which is all U2 reads.
//!
//! *(I) Inclusion and tightness.* `+`, `−`, `·` and `/` (for `0 ∉ Y`) are
//! monotone in each argument on the boxes that the sign cases delimit, so
//! the exact range over `X × Y` is the interval between the two endpoint
//! expressions of I1–I3; `▽` of the lower one and `△` of the upper one are
//! by definition the tightest endpoints in `F ∪ {±∞}` (I3, with no
//! exception for underflow or overflow: steps N3 and N6). When an operand
//! endpoint is infinite, the corresponding expression is the limit, and N1
//! returns it. I4 and I5 follow the same way; I6 is I2–I3 for `c = d`.
//!
//! ## Exceptional cases, in one place
//! * **Overflow.** `▽` saturates at `Ω`, `△` goes to `+∞` (N3); a result
//!   may be a half-line, never invalid and never empty.
//! * **Subnormals and underflow.** Sums are exact there. Products and
//!   quotients whose FMA residual could be rounded take step N6 and are
//!   still the exact directed roundings (a result of magnitude below `η`
//!   becomes `[0, η]` or `[−η, 0]`, keeping its sign).
//! * **Zero in the divisor.** I4: a half-line or `[−∞, +∞]`; no panic.
//! * **NaN.** No interval holds a NaN (I2), and no operation on valid
//!   intervals produces one (N-pre). A NaN enters only through a
//!   constructor or a point operand: [`Interval::new`] returns an error,
//!   [`Interval::point`] and [`Interval::mid_rad`] panic, an operator
//!   and `From<f64>` return `[−∞, +∞]`, [`Interval::enclose`] returns
//!   `[−∞, +∞]`,
//!   [`Interval::contains`] and the `certainly_*` predicates return
//!   `false`.
//! * **The empty set.** Stored as `[+∞, −∞]`; every arithmetic operation
//!   with an empty operand returns `∅`; `hull` ignores it; `mid`, `width`,
//!   `mag` and `mig` are NaN. **The `certainly_*` predicates are `false`
//!   on `∅`**, although "every element of `∅` is `≤ b`" is vacuously
//!   true: they are the last step of a certificate, and an empty enclosure
//!   means that a hypothesis failed upstream, not that a claim was proved
//!   (C3). `is_subset_of`, a set relation, is `true` for `∅ ⊆ Y`.
//!
//! ## Printing
//! **Algorithm D (decimal output, `{}` and `{:e}`).** Input: a non-empty
//! `X = [a, b]`, the precision `p` of the format, if any.
//!
//! D1. \[No precision.\] Print each endpoint as `f64` prints it: the
//!     shortest decimal that reads back as the same `f64`. The text
//!     *identifies* the interval (it is not an enclosure by decimals: the
//!     shortest decimal of `a` may lie on either side of `a`).
//! D2. \[Exact expansion.\] With a precision: write `|a|` exactly — an
//!     `f64` has at most 1074 fractional and 767 significant decimal
//!     digits, and the formatting of `f64` is exact at that length.
//! D3. \[Directed truncation.\] Keep `p` fractional digits (`{}`) or `1 +
//!     p` significant digits (`{:e}`); if a discarded digit is nonzero and
//!     the direction points away from zero (`▽` for a negative `a`, `△`
//!     for a positive `b`), add one unit in the last kept place, carrying
//!     (in `{:e}` a carry out of the leading digit is `1.00…` with the
//!     exponent one higher).
//!
//! (D) *The printed pair, read as exact decimals, contains `X`*, and each
//! printed endpoint is within one unit of its last digit of the true one:
//! truncation moves toward zero, the added unit moves past the value by
//! less than one unit. An infinite endpoint prints as `-inf` or `inf`, `∅`
//! as `∅`. With a precision, the only other flag honoured is `+`; the
//! width and the fill are not.
//!
//! ## Invariants
//! (I1) *Inclusion.* `x ∈ X`, `y ∈ Y` and `x ∘ y` defined imply `x ∘ y ∈ X
//!      ∘ Y`.
//! (I2) *Well-formed.* A non-empty interval has `lo ≤ hi`, no NaN, `lo <
//!      +∞`, `hi > −∞`; `∅` is the pair `[+∞, −∞]` and nothing else.
//!      [`Interval::check_invariants`]; a `debug_assert!` after every
//!      operation.
//! (I3) *Tightness.* `+`, `−`, `·`, `/` (divisor without zero), `√` and
//!      `x²` return the hull of the exact range rounded outward to `F`:
//!      each endpoint is the directed rounding of the exact one. In
//!      particular an operation on points returns an interval of at most
//!      **one ulp** (two adjacent `f64`), and a point when the result is
//!      an `f64`. The tests check both against exact rationals.
//!
//! (I4) *Operator forms.* `X ∘= Y`, `X ∘= y`, `y ∘ X` return, bit for bit,
//!      `X ∘ Y`, `X ∘ y` and `[y, y] ∘ X` (with `[−∞, +∞]` for a `y` that
//!      is not a real number, as in step I6).
//!
//! ## Cost
//! Per directed endpoint: one operation, one EFT or FMA and a branch — 4
//! flops for a sum, 2 for a product, quotient or root, against 1 with a
//! hardware rounding mode. No allocation.
//!
//! ## References
//! Rump, Zimmermann, Boldo & Melquiond, "Computing predecessor and
//! successor in rounding to nearest", BIT 49 (2009) — directed results
//! from round-to-nearest ones; Boldo & Daumas, "Representable correcting
//! terms for possibly underflowing floating point operations", ARITH-16
//! (2003) — when the residuals of `/` and `√` are representable; Muller et
//! al., *Handbook of Floating-Point Arithmetic*, 2nd ed. (2018), §4.3–4.4
//! and §5.2; IEEE Std 1788-2015 (the set-based model: an infinite
//! endpoint is not an element; division by an interval containing zero;
//! `mid` of an unbounded interval); Hickey, Ju & van Emden, "Interval
//! arithmetic: from principles to implementation", J. ACM 48 (2001) — the
//! sign table of I2 and the convention `0 · ∞ = 0` for endpoints. **None
//! of these was reread for this chapter**: the statements used are proved
//! above (Proofs N, U, I) from invariants T1–T2 of Chapter 11.1, and the
//! tests check every operation against exact rational arithmetic.

#![allow(clippy::float_cmp)] // exact comparisons of endpoints are the subject

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::dd::DoubleDouble;
use crate::eft::{fast_two_sum, two_prod};

/// `η`, the smallest positive subnormal.
const ETA: f64 = f64::from_bits(1);
/// `2⁻⁹⁶⁸`: at or above it (for `|RN(a b)|`, resp. for the dividend `|a|`)
/// the FMA residual of step N4 is exact.
const EXACT_RESIDUAL: f64 = f64::from_bits((1023 - 968) << 52);
/// `2⁻⁹⁰⁰`: the same for the radicand (a round number above `2⁻⁹⁷⁰` whose
/// scaling by `2²⁰⁰` is exact).
const EXACT_SQRT_RESIDUAL: f64 = f64::from_bits((1023 - 900) << 52);
/// `2²⁰⁰`, `2⁻²⁰⁰` and `2⁻¹⁰⁰`: the scalings of steps N6 and U.
const UPSCALE: f64 = f64::from_bits((1023 + 200) << 52);
const DOWNSCALE: f64 = f64::from_bits((1023 - 200) << 52);
const DOWNSCALE_SQRT: f64 = f64::from_bits((1023 - 100) << 52);

/// A rounding direction: `▽` or `△`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dir {
    Down,
    Up,
}

/// Step N5: `v` moved one step in the direction `dir` when the exact value
/// lies strictly on that side (`sigma` has the sign of `exact − v`).
#[inline]
fn step(v: f64, sigma: f64, dir: Dir) -> f64 {
    debug_assert!(v.is_finite() && !sigma.is_nan(), "step({v:e}, {sigma:e})");
    match dir {
        Dir::Down if sigma < 0.0 => v.next_down(),
        Dir::Up if sigma > 0.0 => v.next_up(),
        _ => v,
    }
}

/// Step N3: the directed rounding of a finite exact value whose nearest
/// rounding `v` overflowed.
#[inline]
fn overflow(v: f64, dir: Dir) -> f64 {
    debug_assert!(v.is_infinite());
    match (dir, v > 0.0) {
        (Dir::Down, true) => f64::MAX,
        (Dir::Up, false) => f64::MIN,
        _ => v,
    }
}

/// Algorithm U: the directed rounding of `(w + δ) 2⁻²⁰⁰`.
#[inline]
fn descale(w: f64, delta: f64, dir: Dir) -> f64 {
    let v = w * DOWNSCALE; // U1
    let tau = w - v * UPSCALE; // U2, exact
    step(v, if tau == 0.0 { delta } else { tau }, dir) // U3
}

/// Algorithm N for `a + b`.
#[inline]
fn add_dir(a: f64, b: f64, dir: Dir) -> f64 {
    let v = a + b; // N2
    debug_assert!(!v.is_nan(), "add_dir({a:e}, {b:e})");
    if a.is_infinite() || b.is_infinite() {
        return v; // N1
    }
    if v.is_infinite() {
        return overflow(v, dir); // N3
    }
    // N4.
    let (hi, lo) = if a.abs() >= b.abs() { (a, b) } else { (b, a) };
    let (_, sigma) = fast_two_sum(hi, lo);
    step(v, sigma, dir) // N5
}

/// Algorithm N for `a · b`.
#[inline]
fn mul_dir(a: f64, b: f64, dir: Dir) -> f64 {
    debug_assert!(!a.is_nan() && !b.is_nan(), "mul_dir({a:e}, {b:e})");
    if a == 0.0 || b == 0.0 {
        return 0.0; // N1, 0 · ∞ = 0 included
    }
    let v = a * b; // N2
    if a.is_infinite() || b.is_infinite() {
        return v; // N1
    }
    if v.is_infinite() {
        return overflow(v, dir); // N3
    }
    if v.abs() >= EXACT_RESIDUAL {
        return step(v, a.mul_add(b, -v), dir); // N4, N5
    }
    // N6.
    if v == 0.0 {
        return match (dir, (a > 0.0) == (b > 0.0)) {
            (Dir::Up, true) => ETA,
            (Dir::Down, false) => -ETA,
            _ => 0.0,
        };
    }
    let (s, l) = if a.abs() <= b.abs() { (a, b) } else { (b, a) };
    let (w, delta) = two_prod(s * UPSCALE, l);
    descale(w, delta, dir)
}

/// Algorithm N for `a / b`, `b ≠ 0`.
#[inline]
fn div_dir(a: f64, b: f64, dir: Dir) -> f64 {
    debug_assert!(
        !a.is_nan() && !b.is_nan() && b != 0.0,
        "div_dir({a:e}, {b:e})"
    );
    debug_assert!(a.is_finite() || b.is_finite(), "div_dir: ∞ / ∞");
    if a == 0.0 || b.is_infinite() {
        return 0.0; // N1
    }
    let v = a / b; // N2
    if a.is_infinite() {
        return v; // N1
    }
    if v.is_infinite() {
        return overflow(v, dir); // N3
    }
    // The sign of `a/b − q` from `ρ = a − q b`.
    let sigma = |rho: f64| if b > 0.0 { rho } else { -rho };
    if a.abs() >= EXACT_RESIDUAL {
        return step(v, sigma((-v).mul_add(b, a)), dir); // N4, N5
    }
    // N6.
    let a2 = a * UPSCALE;
    let w = a2 / b;
    descale(w, sigma((-w).mul_add(b, a2)), dir)
}

/// Algorithm N for `√a`, `a ≥ 0` or `a = +∞`.
#[inline]
fn sqrt_dir(a: f64, dir: Dir) -> f64 {
    debug_assert!(a >= 0.0, "sqrt_dir({a:e})");
    if a == 0.0 {
        return 0.0; // N1
    }
    if a.is_infinite() {
        return a; // N1
    }
    // N6: both scalings are exact.
    let (x, back) = if a < EXACT_SQRT_RESIDUAL {
        (a * UPSCALE, DOWNSCALE_SQRT)
    } else {
        (a, 1.0)
    };
    let v = x.sqrt(); // N2
    step(v, (-v).mul_add(v, x), dir) * back // N4, N5
}

/// A closed interval `[lo, hi]` with `f64` endpoints, or the empty set
/// (invariants I1–I3). Operations round outward by computed directed
/// rounding (Chapter 11.9).
///
/// An interval is a set of *real* numbers: an infinite endpoint means
/// "unbounded on that side". Not `xingo_sparse::Real`: intervals have no
/// total order and their operations no single rounding (D-0080, Q2). Not
/// `PartialOrd` either: the comparisons are the named predicates below,
/// whose meaning cannot be misread.
///
/// ```
/// use xingo_extprec::Interval;
/// let tenth = Interval::point(1.0) / 10.0; // encloses the real 1/10
/// assert!(tenth.inf() < tenth.sup() && tenth.contains(0.1));
/// let one = tenth * 10.0;
/// assert!(one.contains(1.0) && one.certainly_within(1.0 + 1e-15));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    lo: f64,
    hi: f64,
}

/// `lo > hi`, or an endpoint is NaN, or `lo = +∞`, or `hi = −∞`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidInterval;

impl Interval {
    /// `[−∞, +∞]`: "nothing is known" (invariant C3).
    pub const ENTIRE: Self = Self {
        lo: f64::NEG_INFINITY,
        hi: f64::INFINITY,
    };
    /// The empty set (stored as `[+∞, −∞]`; every operation propagates it).
    pub const EMPTY: Self = Self {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    /// `[0, 0]`.
    pub const ZERO: Self = Self { lo: 0.0, hi: 0.0 };

    /// The interval of the pair, which the caller proved well-formed.
    #[inline]
    fn raw(lo: f64, hi: f64) -> Self {
        let x = Self { lo, hi };
        debug_assert!(
            x.check_invariants().is_ok() && !x.is_empty(),
            "[{lo:e}, {hi:e}]: {:?}",
            x.check_invariants()
        );
        x
    }

    /// `[lo, hi]`.
    ///
    /// # Errors
    /// [`InvalidInterval`] when (I2) fails.
    pub fn new(lo: f64, hi: f64) -> Result<Self, InvalidInterval> {
        // False for a NaN endpoint too.
        if lo <= hi && lo < f64::INFINITY && hi > f64::NEG_INFINITY {
            Ok(Self { lo, hi })
        } else {
            Err(InvalidInterval)
        }
    }

    /// `[x, x]`. Panics on NaN — and on `±∞`, which is not a real number
    /// (it would violate (I2)).
    #[must_use]
    pub fn point(x: f64) -> Self {
        assert!(x.is_finite(), "Interval::point({x}): not a real number");
        Self { lo: x, hi: x }
    }

    /// `[m − r, m + r]` rounded outward; `r ≥ 0` (`r = +∞` gives
    /// [`Interval::ENTIRE`], and so does an infinite `m`). Panics if `m`
    /// or `r` is NaN or `r < 0`.
    #[must_use]
    pub fn mid_rad(m: f64, r: f64) -> Self {
        assert!(
            !m.is_nan() && r >= 0.0,
            "Interval::mid_rad({m}, {r}): NaN or a negative radius"
        );
        if m.is_infinite() || r.is_infinite() {
            return Self::ENTIRE;
        }
        Self::raw(add_dir(m, -r, Dir::Down), add_dir(m, r, Dir::Up))
    }

    /// The tightest interval containing the double-double `x`: `[hi, hi]`
    /// if `lo = 0`, else the two neighbours of `x` in `F`. An infinite or
    /// NaN `x` — the trace of an overflow or of an invalid operation, not
    /// a real number — gives [`Interval::ENTIRE`] (invariant C3).
    #[must_use]
    pub fn enclose(x: DoubleDouble) -> Self {
        let (h, l) = (x.hi(), x.lo());
        if !h.is_finite() || !l.is_finite() {
            return Self::ENTIRE;
        }
        // (D1): `|l| ≤ ulp(h)/2`, so `h + l` lies between `h` and the
        // neighbour of `h` on the side of `l`.
        if l > 0.0 {
            Self::raw(h, h.next_up())
        } else if l < 0.0 {
            Self::raw(h.next_down(), h)
        } else {
            Self::raw(h, h)
        }
    }

    /// The lower endpoint (`+∞` for the empty set).
    #[must_use]
    pub const fn inf(self) -> f64 {
        self.lo
    }

    /// The upper endpoint (`−∞` for the empty set).
    #[must_use]
    pub const fn sup(self) -> f64 {
        self.hi
    }

    /// The empty set.
    #[inline]
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.lo > self.hi
    }

    /// An `f64` in the interval nearest its midpoint (finite when the
    /// interval is bounded; `0` for [`Interval::ENTIRE`]). For a
    /// half-line, the finite `f64` farthest out (`±Ω`, as in IEEE 1788);
    /// NaN for the empty set.
    #[must_use]
    pub fn mid(self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        match (self.lo.is_finite(), self.hi.is_finite()) {
            (false, false) => 0.0,
            (false, true) => f64::MIN,
            (true, false) => f64::MAX,
            // `(lo + hi)/2`, or `lo/2 + hi/2` when the sum overflows:
            // monotone rounding keeps either inside `[lo, hi]`.
            (true, true) => f64::midpoint(self.lo, self.hi),
        }
    }

    /// `sup − inf` rounded up (`+∞` when unbounded, NaN when empty).
    #[must_use]
    pub fn width(self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        add_dir(self.hi, -self.lo, Dir::Up)
    }

    /// `max{|x| : x ∈ X}`, the magnitude (exact; `+∞` when unbounded, NaN
    /// when empty).
    #[must_use]
    pub fn mag(self) -> f64 {
        if self.is_empty() {
            return f64::NAN;
        }
        self.lo.abs().max(self.hi.abs())
    }

    /// `min{|x| : x ∈ X}`, the mignitude (exact; NaN when empty).
    #[must_use]
    pub fn mig(self) -> f64 {
        if self.is_empty() {
            f64::NAN
        } else if self.lo > 0.0 {
            self.lo
        } else if self.hi < 0.0 {
            -self.hi
        } else {
            0.0
        }
    }

    /// `x ∈ X` (`false` for a NaN or infinite `x`: the elements are real
    /// numbers).
    #[must_use]
    pub fn contains(self, x: f64) -> bool {
        x.is_finite() && self.lo <= x && x <= self.hi
    }

    /// `X ⊆ Y` (`∅` is a subset of every interval).
    #[must_use]
    pub fn is_subset_of(self, other: Self) -> bool {
        self.is_empty() || (other.lo <= self.lo && self.hi <= other.hi)
    }

    /// Every element of `X` is `≤ b`: **certainly** `x ≤ b`. `false` for
    /// the empty set and for a NaN `b` (Chapter 11.9: nothing was proved).
    #[must_use]
    pub fn certainly_le(self, b: f64) -> bool {
        !self.is_empty() && self.hi <= b
    }

    /// Every element of `X` is `≥ a`: **certainly** `x ≥ a`. `false` for
    /// the empty set and for a NaN `a`.
    #[must_use]
    pub fn certainly_ge(self, a: f64) -> bool {
        !self.is_empty() && self.lo >= a
    }

    /// `X ⊆ [−t, t]`: **certainly** `|x| ≤ t` (a verified tolerance test).
    /// `false` for the empty set and for a NaN `t`.
    #[must_use]
    pub fn certainly_within(self, t: f64) -> bool {
        !self.is_empty() && -t <= self.lo && self.hi <= t
    }

    /// The hull `[min inf, max sup]` (the union's enclosure).
    #[must_use]
    pub fn hull(self, other: Self) -> Self {
        // The stored form of `∅`, `[+∞, −∞]`, is the neutral element.
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// The intersection (possibly [`Interval::EMPTY`]).
    #[must_use]
    pub fn intersection(self, other: Self) -> Self {
        let (lo, hi) = (self.lo.max(other.lo), self.hi.min(other.hi));
        if lo <= hi {
            Self { lo, hi }
        } else {
            Self::EMPTY
        }
    }

    /// `{|x| : x ∈ X}`.
    #[must_use]
    pub fn abs(self) -> Self {
        if self.is_empty() || self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            -self
        } else {
            Self::raw(0.0, (-self.lo).max(self.hi))
        }
    }

    /// `{x² : x ∈ X}` (tighter than `X · X` when `0 ∈ X`).
    #[must_use]
    pub fn square(self) -> Self {
        if self.is_empty() {
            return self;
        }
        let (a, b) = (self.lo, self.hi);
        if a >= 0.0 {
            Self::raw(mul_dir(a, a, Dir::Down), mul_dir(b, b, Dir::Up))
        } else if b <= 0.0 {
            Self::raw(mul_dir(b, b, Dir::Down), mul_dir(a, a, Dir::Up))
        } else {
            let m = (-a).max(b);
            Self::raw(0.0, mul_dir(m, m, Dir::Up))
        }
    }

    /// `{√x : x ∈ X, x ≥ 0}` ([`Interval::EMPTY`] when `sup < 0`).
    #[must_use]
    pub fn sqrt(self) -> Self {
        if self.is_empty() || self.hi < 0.0 {
            return Self::EMPTY;
        }
        Self::raw(
            sqrt_dir(self.lo.max(0.0), Dir::Down),
            sqrt_dir(self.hi, Dir::Up),
        )
    }

    /// Invariant (I2).
    ///
    /// # Errors
    /// The text of the violated invariant.
    pub fn check_invariants(self) -> Result<(), &'static str> {
        if self.lo.is_nan() || self.hi.is_nan() {
            return Err("I2: an endpoint is NaN");
        }
        if self.lo == f64::INFINITY && self.hi == f64::NEG_INFINITY {
            return Ok(()); // ∅
        }
        if self.lo > self.hi {
            return Err("I2: lo > hi in an interval that is not the stored empty set");
        }
        if self.lo == f64::INFINITY {
            return Err("I2: the lower endpoint is +∞");
        }
        if self.hi == f64::NEG_INFINITY {
            return Err("I2: the upper endpoint is −∞");
        }
        Ok(())
    }
}

/// Step I1.
impl Add for Interval {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        Self::raw(
            add_dir(self.lo, rhs.lo, Dir::Down),
            add_dir(self.hi, rhs.hi, Dir::Up),
        )
    }
}

/// Step I1.
impl Sub for Interval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        Self::raw(
            add_dir(self.lo, -rhs.hi, Dir::Down),
            add_dir(self.hi, -rhs.lo, Dir::Up),
        )
    }
}

/// Step I2.
impl Mul for Interval {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        let (a, b, c, d) = (self.lo, self.hi, rhs.lo, rhs.hi);
        let dn = |x, y| mul_dir(x, y, Dir::Down);
        let up = |x, y| mul_dir(x, y, Dir::Up);
        let (lo, hi) = if a >= 0.0 {
            if c >= 0.0 {
                (dn(a, c), up(b, d))
            } else if d <= 0.0 {
                (dn(b, c), up(a, d))
            } else {
                (dn(b, c), up(b, d))
            }
        } else if b <= 0.0 {
            if c >= 0.0 {
                (dn(a, d), up(b, c))
            } else if d <= 0.0 {
                (dn(b, d), up(a, c))
            } else {
                (dn(a, d), up(a, c))
            }
        } else if c >= 0.0 {
            (dn(a, d), up(b, d))
        } else if d <= 0.0 {
            (dn(b, c), up(a, c))
        } else {
            (dn(a, d).min(dn(b, c)), up(a, c).max(up(b, d)))
        };
        Self::raw(lo, hi)
    }
}

/// Steps I3 and I4: when the divisor contains zero the result is
/// [`Interval::ENTIRE`], or a half-line when zero is an endpoint of the
/// divisor and the dividend has one sign — the hull of the quotients by
/// the nonzero elements, never a panic. (`[0, 0] / Y` is `[0, 0]` unless
/// `Y = [0, 0]`.)
impl Div for Interval {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if self.is_empty() || rhs.is_empty() {
            return Self::EMPTY;
        }
        let (a, b, c, d) = (self.lo, self.hi, rhs.lo, rhs.hi);
        let dn = |x, y| div_dir(x, y, Dir::Down);
        let up = |x, y| div_dir(x, y, Dir::Up);
        if c == 0.0 && d == 0.0 {
            return Self::ENTIRE; // I4
        }
        if a == 0.0 && b == 0.0 {
            return Self::ZERO;
        }
        if c > 0.0 {
            // I3.
            if a >= 0.0 {
                Self::raw(dn(a, d), up(b, c))
            } else if b <= 0.0 {
                Self::raw(dn(a, c), up(b, d))
            } else {
                Self::raw(dn(a, c), up(b, c))
            }
        } else if d < 0.0 {
            if a >= 0.0 {
                Self::raw(dn(b, d), up(a, c))
            } else if b <= 0.0 {
                Self::raw(dn(b, c), up(a, d))
            } else {
                Self::raw(dn(b, d), up(a, d))
            }
        } else if c == 0.0 {
            // I4, Y = [0, d] with d > 0.
            if a >= 0.0 {
                Self::raw(dn(a, d), f64::INFINITY)
            } else if b <= 0.0 {
                Self::raw(f64::NEG_INFINITY, up(b, d))
            } else {
                Self::ENTIRE
            }
        } else if d == 0.0 {
            // I4, Y = [c, 0] with c < 0.
            if a >= 0.0 {
                Self::raw(f64::NEG_INFINITY, up(a, c))
            } else if b <= 0.0 {
                Self::raw(dn(b, c), f64::INFINITY)
            } else {
                Self::ENTIRE
            }
        } else {
            Self::ENTIRE
        }
    }
}

/// With a point right operand (`[x, x]`, exact): half the work (step I6).
/// A NaN or infinite `x` gives [`Interval::ENTIRE`].
impl Add<f64> for Interval {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        if self.is_empty() {
            return self;
        }
        if !rhs.is_finite() {
            return Self::ENTIRE;
        }
        Self::raw(
            add_dir(self.lo, rhs, Dir::Down),
            add_dir(self.hi, rhs, Dir::Up),
        )
    }
}

/// With a point right operand (`[x, x]`, exact): half the work (step I6).
/// A NaN or infinite `x` gives [`Interval::ENTIRE`].
impl Sub<f64> for Interval {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        self + (-rhs)
    }
}

/// With a point right operand (`[x, x]`, exact): half the work (step I6).
/// A NaN or infinite `x` gives [`Interval::ENTIRE`].
impl Mul<f64> for Interval {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        if self.is_empty() {
            return self;
        }
        if !rhs.is_finite() {
            return Self::ENTIRE;
        }
        let (a, b) = if rhs >= 0.0 {
            (self.lo, self.hi)
        } else {
            (self.hi, self.lo)
        };
        Self::raw(mul_dir(a, rhs, Dir::Down), mul_dir(b, rhs, Dir::Up))
    }
}

/// With a point right operand (`[x, x]`, exact): half the work (step I6).
/// A NaN, infinite or zero `x` gives [`Interval::ENTIRE`].
impl Div<f64> for Interval {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        if self.is_empty() {
            return self;
        }
        if !rhs.is_finite() || rhs == 0.0 {
            return Self::ENTIRE;
        }
        let (a, b) = if rhs > 0.0 {
            (self.lo, self.hi)
        } else {
            (self.hi, self.lo)
        };
        Self::raw(div_dir(a, rhs, Dir::Down), div_dir(b, rhs, Dir::Up))
    }
}

/// The assigning forms: `X ∘= Y` is `X = X ∘ Y`, `X ∘= y` is `X = X ∘ y`
/// (invariant I4 of Chapter 11.9).
macro_rules! assign_ops {
    ($($tr:ident, $f:ident, $op:tt;)*) => {$(
        impl $tr for Interval {
            fn $f(&mut self, rhs: Self) {
                *self = *self $op rhs;
            }
        }
        impl $tr<f64> for Interval {
            fn $f(&mut self, rhs: f64) {
                *self = *self $op rhs;
            }
        }
    )*};
}
assign_ops! {
    AddAssign, add_assign, +;
    SubAssign, sub_assign, -;
    MulAssign, mul_assign, *;
    DivAssign, div_assign, /;
}

/// The point `[y, y]` of a left scalar operand, or what the operation
/// returns without computing: `∅` for an empty `x`, [`Interval::ENTIRE`]
/// for a `y` that is not a real number (step I6).
#[inline]
fn left_point(y: f64, x: Interval) -> Result<Interval, Interval> {
    if x.is_empty() {
        Err(Interval::EMPTY)
    } else if y.is_finite() {
        Ok(Interval { lo: y, hi: y })
    } else {
        Err(Interval::ENTIRE)
    }
}

/// With a point left operand: `y + X` is `[y, y] + X` (invariant I4). A
/// NaN or infinite `y` gives [`Interval::ENTIRE`].
impl Add<Interval> for f64 {
    type Output = Interval;
    fn add(self, rhs: Interval) -> Interval {
        left_point(self, rhs).map_or_else(|r| r, |p| p + rhs)
    }
}

/// With a point left operand: `y − X` is `[y, y] − X` (invariant I4). A
/// NaN or infinite `y` gives [`Interval::ENTIRE`].
impl Sub<Interval> for f64 {
    type Output = Interval;
    fn sub(self, rhs: Interval) -> Interval {
        left_point(self, rhs).map_or_else(|r| r, |p| p - rhs)
    }
}

/// With a point left operand: `y · X` is `[y, y] · X` (invariant I4). A
/// NaN or infinite `y` gives [`Interval::ENTIRE`].
impl Mul<Interval> for f64 {
    type Output = Interval;
    fn mul(self, rhs: Interval) -> Interval {
        left_point(self, rhs).map_or_else(|r| r, |p| p * rhs)
    }
}

/// With a point left operand: `y / X` is `[y, y] / X` (invariant I4; steps
/// I3 and I4 when `X` contains zero). A NaN or infinite `y` gives
/// [`Interval::ENTIRE`].
impl Div<Interval> for f64 {
    type Output = Interval;
    fn div(self, rhs: Interval) -> Interval {
        left_point(self, rhs).map_or_else(|r| r, |p| p / rhs)
    }
}

/// `{−x : x ∈ X}`, exact.
impl Neg for Interval {
    type Output = Self;
    fn neg(self) -> Self {
        // `∅ = [+∞, −∞]` is its own negative.
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

/// The interval of one `f64`, **total**: `[x, x]` for a finite `x`, and
/// [`Interval::ENTIRE`] for a NaN or an infinite `x`.
///
/// This differs from [`Interval::point`], which panics on a NaN or an
/// infinity, on purpose: `From` must not fail (the standard library's
/// contract for the trait), and "nothing is known" is what the operators
/// already return for a scalar operand that is not a real number — so
/// `Interval::from(y) ∘ X` equals `y ∘ X` for every `y` and every
/// non-empty `X`. Use `point` when a non-real argument is a bug to be
/// caught, `from` when it is data to be carried.
impl From<f64> for Interval {
    fn from(x: f64) -> Self {
        if x.is_finite() {
            Self { lo: x, hi: x }
        } else {
            Self::ENTIRE
        }
    }
}

impl fmt::Display for InvalidInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid interval: the endpoints are not lo <= hi with lo < +inf and hi > -inf")
    }
}

impl core::error::Error for InvalidInterval {}

/// Fractional digits that write any `f64` exactly (it is a multiple of
/// `2⁻¹⁰⁷⁴`).
const EXACT_FRACTION_DIGITS: usize = 1074;
/// Digits after the leading one that write any `f64` exactly in
/// scientific notation (767 significant digits at most; a round number
/// above it).
const EXACT_MANTISSA_DIGITS: usize = 800;

/// Adds one unit in the last place of a string of ASCII decimal digits;
/// `true` if the carry leaves the leading digit (the digits are then all
/// zero).
fn increment(digits: &mut [u8]) -> bool {
    for d in digits.iter_mut().rev() {
        if *d == b'9' {
            *d = b'0';
        } else {
            *d += 1;
            return false;
        }
    }
    true
}

/// Whether step D3 adds a unit: the discarded digits are not all zero and
/// the direction points away from zero.
fn away(x: f64, discarded: &[u8], dir: Dir) -> bool {
    discarded.iter().any(|&d| d != b'0') && ((x > 0.0) == (dir == Dir::Up))
}

/// Steps D2 and D3 for `{:.p}`: `x` with `p` fractional digits, rounded in
/// the direction `dir`.
fn write_fixed(f: &mut fmt::Formatter<'_>, x: f64, p: usize, dir: Dir) -> fmt::Result {
    if !x.is_finite() || p >= EXACT_FRACTION_DIGITS {
        // Exact as `f64` prints it (`inf`, or zero padding).
        return fmt::Display::fmt(&x, f);
    }
    let exact = format!("{:.*}", EXACT_FRACTION_DIGITS, x.abs()); // D2
    let (int, frac) = exact.split_once('.').expect("a fractional part");
    let mut kept: Vec<u8> = int.bytes().chain(frac.bytes().take(p)).collect();
    let carry = away(x, &frac.as_bytes()[p..], dir) && increment(&mut kept); // D3
    let (int, frac) = kept.split_at(int.len());
    if x.is_sign_negative() {
        f.write_str("-")?;
    } else if f.sign_plus() {
        f.write_str("+")?;
    }
    if carry {
        f.write_str("1")?;
    }
    // ASCII digits.
    f.write_str(core::str::from_utf8(int).expect("ASCII digits"))?;
    if p > 0 {
        f.write_str(".")?;
        f.write_str(core::str::from_utf8(frac).expect("ASCII digits"))?;
    }
    Ok(())
}

/// Steps D2 and D3 for `{:.pe}`: `x` with `1 + p` significant digits,
/// rounded in the direction `dir`.
fn write_exp(f: &mut fmt::Formatter<'_>, x: f64, p: usize, dir: Dir) -> fmt::Result {
    if !x.is_finite() || p >= EXACT_MANTISSA_DIGITS {
        return fmt::LowerExp::fmt(&x, f);
    }
    let exact = format!("{:.*e}", EXACT_MANTISSA_DIGITS, x.abs()); // D2
    let (mantissa, exponent) = exact.split_once('e').expect("an exponent");
    let mut exponent: i32 = exponent.parse().expect("a decimal exponent");
    let digits: Vec<u8> = mantissa.bytes().filter(|&c| c != b'.').collect();
    let (kept, discarded) = digits.split_at(1 + p);
    let mut kept = kept.to_vec();
    if away(x, discarded, dir) && increment(&mut kept) {
        // D3: 9.99… became 10.0…, that is 1.00… with the next exponent.
        kept[0] = b'1';
        exponent += 1;
    }
    if x.is_sign_negative() {
        f.write_str("-")?;
    } else if f.sign_plus() {
        f.write_str("+")?;
    }
    f.write_str(core::str::from_utf8(&kept[..1]).expect("ASCII digits"))?;
    if p > 0 {
        f.write_str(".")?;
        f.write_str(core::str::from_utf8(&kept[1..]).expect("ASCII digits"))?;
    }
    write!(f, "e{exponent}")
}

/// `[a, b]`, or `∅` (Algorithm D of Chapter 11.9). Without a precision
/// each endpoint is the shortest text that reads back as the same `f64`
/// (with the flags of the formatter): the output identifies the interval.
/// **With a precision (`{:.3}`) the endpoints are rounded outward**: the
/// printed pair, read as exact decimals, contains the interval. The width
/// and the fill are then ignored.
impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("∅");
        }
        let p = f.precision();
        f.write_str("[")?;
        match p {
            None => fmt::Display::fmt(&self.lo, f)?,
            Some(p) => write_fixed(f, self.lo, p, Dir::Down)?,
        }
        f.write_str(", ")?;
        match p {
            None => fmt::Display::fmt(&self.hi, f)?,
            Some(p) => write_fixed(f, self.hi, p, Dir::Up)?,
        }
        f.write_str("]")
    }
}

/// `[a, b]` in scientific notation (`{:e}`), or `∅`; as `Display`:
/// shortest round-trip endpoints without a precision, **outward-rounded**
/// endpoints with one (`{:.3e}`: four significant digits).
impl fmt::LowerExp for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("∅");
        }
        let p = f.precision();
        f.write_str("[")?;
        match p {
            None => fmt::LowerExp::fmt(&self.lo, f)?,
            Some(p) => write_exp(f, self.lo, p, Dir::Down)?,
        }
        f.write_str(", ")?;
        match p {
            None => fmt::LowerExp::fmt(&self.hi, f)?,
            Some(p) => write_exp(f, self.hi, p, Dir::Up)?,
        }
        f.write_str("]")
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // L2.4 (Chapter 11.11): the tests may use the libm of `f64` (clippy.toml)
mod tests {
    use super::*;

    const MAX: f64 = f64::MAX;
    const INF: f64 = f64::INFINITY;

    fn iv(lo: f64, hi: f64) -> Interval {
        Interval::new(lo, hi).expect("a valid interval")
    }

    #[test]
    fn constructors_enforce_i2() {
        assert_eq!(
            Interval::new(1.0, 2.0).map(|x| (x.inf(), x.sup())),
            Ok((1.0, 2.0))
        );
        for (lo, hi) in [
            (2.0, 1.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (INF, INF),
            (-INF, -INF),
            (INF, -INF),
        ] {
            assert_eq!(Interval::new(lo, hi), Err(InvalidInterval), "[{lo}, {hi}]");
        }
        assert_eq!(Interval::new(-INF, INF), Ok(Interval::ENTIRE));
        for x in [
            Interval::ENTIRE,
            Interval::EMPTY,
            Interval::ZERO,
            iv(-INF, 3.0),
        ] {
            assert_eq!(x.check_invariants(), Ok(()));
        }
        for (lo, hi) in [
            (2.0, 1.0),
            (f64::NAN, 1.0),
            (INF, INF),
            (-INF, -INF),
            (INF, 0.0),
        ] {
            assert!(
                Interval { lo, hi }.check_invariants().is_err(),
                "[{lo}, {hi}]"
            );
        }
        assert!(Interval::EMPTY.is_empty() && !Interval::ZERO.is_empty());
        assert_eq!(Interval::point(-0.0), Interval::ZERO);
    }

    #[test]
    #[should_panic(expected = "not a real number")]
    fn point_rejects_nan() {
        let _ = Interval::point(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "not a real number")]
    fn point_rejects_infinity() {
        let _ = Interval::point(INF);
    }

    #[test]
    #[should_panic(expected = "negative radius")]
    fn mid_rad_rejects_a_negative_radius() {
        let _ = Interval::mid_rad(0.0, -1.0);
    }

    #[test]
    fn mid_rad_rounds_outward() {
        assert_eq!(Interval::mid_rad(1.0, 0.0), Interval::point(1.0));
        assert_eq!(Interval::mid_rad(1.0, 0.5), iv(0.5, 1.5));
        // 1 ± 2^-60 is not representable: one step out on each side.
        let x = Interval::mid_rad(1.0, 2f64.powi(-60));
        assert_eq!((x.inf(), x.sup()), (1.0_f64.next_down(), 1.0_f64.next_up()));
        assert_eq!(Interval::mid_rad(1.0, INF), Interval::ENTIRE);
        assert_eq!(Interval::mid_rad(-INF, 1.0), Interval::ENTIRE);
        assert_eq!(Interval::mid_rad(MAX, MAX), iv(0.0, INF));
    }

    #[test]
    fn enclose_is_the_two_neighbours() {
        let u = 2f64.powi(-53);
        assert_eq!(
            Interval::enclose(DoubleDouble::from_f64(3.0)),
            Interval::point(3.0)
        );
        let up = Interval::enclose(DoubleDouble::from_sum(1.0, u * u));
        assert_eq!((up.inf(), up.sup()), (1.0, 1.0_f64.next_up()));
        let down = Interval::enclose(DoubleDouble::from_sum(1.0, -u * u));
        assert_eq!((down.inf(), down.sup()), (1.0_f64.next_down(), 1.0));
        let top = Interval::enclose(DoubleDouble::from_sum(MAX, 1.0));
        assert_eq!((top.inf(), top.sup()), (MAX, INF));
        assert_eq!(Interval::enclose(DoubleDouble::NAN), Interval::ENTIRE);
        assert_eq!(Interval::enclose(DoubleDouble::INFINITY), Interval::ENTIRE);
        assert_eq!(Interval::enclose(-DoubleDouble::INFINITY), Interval::ENTIRE);
    }

    #[test]
    fn sums_round_outward_by_one_step() {
        let u = 2f64.powi(-53);
        // 1 + 2^-53 is a tie: RN gives 1, the exact value is above.
        let s = Interval::point(1.0) + u;
        assert_eq!((s.inf(), s.sup()), (1.0, 1.0_f64.next_up()));
        // 1 − 2^-53 is an f64; 1 − 2^-54 is a tie, rounded to 1.
        assert_eq!(
            Interval::point(1.0) - u,
            Interval::point(1.0_f64.next_down())
        );
        let s = Interval::point(1.0) - 0.5 * u;
        assert_eq!((s.inf(), s.sup()), (1.0_f64.next_down(), 1.0));
        // Exact sums stay points, subnormal ones included.
        assert_eq!(
            Interval::point(3.0) + Interval::point(4.0),
            Interval::point(7.0)
        );
        assert_eq!(Interval::point(ETA) + ETA, Interval::point(2.0 * ETA));
        assert_eq!(Interval::point(ETA) - ETA, Interval::ZERO);
        assert_eq!(iv(1.0, 2.0) - iv(1.0, 2.0), iv(-1.0, 1.0));
    }

    #[test]
    fn overflow_saturates_the_inner_endpoint() {
        // N3: a lower endpoint never becomes +∞, an upper one never −∞.
        let s = Interval::point(MAX) + MAX;
        assert_eq!((s.inf(), s.sup()), (MAX, INF));
        let s = Interval::point(-MAX) - MAX;
        assert_eq!((s.inf(), s.sup()), (-INF, -MAX));
        let p = Interval::point(MAX) * 2.0;
        assert_eq!((p.inf(), p.sup()), (MAX, INF));
        let p = Interval::point(MAX) * Interval::point(-2.0);
        assert_eq!((p.inf(), p.sup()), (-INF, -MAX));
        let q = Interval::point(MAX) / 0.5;
        assert_eq!((q.inf(), q.sup()), (MAX, INF));
        let q = Interval::point(1.0) / Interval::point(-ETA);
        assert_eq!((q.inf(), q.sup()), (-INF, -MAX));
        // MAX + ulp/4 does not overflow in RN; the upper endpoint does.
        let s = Interval::point(MAX) + 2f64.powi(969);
        assert_eq!((s.inf(), s.sup()), (MAX, INF));
        assert_eq!(Interval::point(MAX).square().sup(), INF);
        assert_eq!(Interval::point(MAX).square().inf(), MAX);
    }

    #[test]
    fn products_in_the_underflow_range_are_still_tight() {
        let x = 1.0 + f64::EPSILON;
        // Total underflow keeps the sign (step N6).
        let p = Interval::point(1e-200) * 1e-200;
        assert_eq!((p.inf(), p.sup()), (0.0, ETA));
        let p = Interval::point(1e-200) * -1e-200;
        assert_eq!((p.inf(), p.sup()), (-ETA, 0.0));
        // η/2 exactly: RN is 0 (tie to even).
        let p = Interval::point(ETA) * 0.5;
        assert_eq!((p.inf(), p.sup()), (0.0, ETA));
        // An exact subnormal product is a point.
        // 2^-500 · 2^-560 = 2^-1060 = 2^14 η.
        assert_eq!(
            Interval::point(2f64.powi(-500)) * 2f64.powi(-560),
            Interval::point(16384.0 * ETA)
        );
        assert_eq!(Interval::point(3.0 * ETA) * 1.0, Interval::point(3.0 * ETA));
        // (1 + 2^-52)² 2^-972: the FMA residual 2^-1076 rounds to zero and
        // would lose the sign; the product is just above RN.
        let a = x * 2f64.powi(-486);
        let p = Interval::point(a) * a;
        let v = a * a;
        assert_eq!(a.mul_add(a, -v), 0.0);
        assert_eq!((p.inf(), p.sup()), (v, v.next_up()));
        // 1.5 η rounds to 2η (tie to even): the exact value is below.
        let p = Interval::point(3.0 * ETA) * 0.5;
        assert_eq!((p.inf(), p.sup()), (ETA, 2.0 * ETA));
        // 2.5 η rounds to 2η: the exact value is above.
        let p = Interval::point(5.0 * ETA) * -0.5;
        assert_eq!((p.inf(), p.sup()), (-3.0 * ETA, -2.0 * ETA));
    }

    #[test]
    fn quotients_round_outward_everywhere() {
        let t = Interval::point(1.0) / 3.0;
        assert_eq!(t.sup(), t.inf().next_up());
        assert!(t.inf() * 3.0 <= 1.0 && t.contains(1.0 / 3.0));
        assert_eq!(
            Interval::point(6.0) / Interval::point(-3.0),
            Interval::point(-2.0)
        );
        // Quotient underflow: η/3 is below η/2.
        let q = Interval::point(ETA) / 3.0;
        assert_eq!((q.inf(), q.sup()), (0.0, ETA));
        let q = Interval::point(-ETA) / 3.0;
        assert_eq!((q.inf(), q.sup()), (-ETA, 0.0));
        // 3η/2 = 1.5η: a tie rounded to 2η.
        let q = Interval::point(3.0 * ETA) / 2.0;
        assert_eq!((q.inf(), q.sup()), (ETA, 2.0 * ETA));
        // An exact subnormal quotient is a point.
        assert_eq!(Interval::point(6.0 * ETA) / 3.0, Interval::point(2.0 * ETA));
        // Tiny dividend, tiny divisor: an ordinary quotient through N6.
        let q = Interval::point(2f64.powi(-1000)) / (3.0 * 2f64.powi(-1000));
        assert_eq!(q, Interval::point(1.0) / 3.0);
        // q = (1 + 2^-52) 2^-1000 is RN(a / b) for b = 1 + 2^-52 and a = (1 +
        // 2^-51) 2^-1000, and a − q b = −2^-1104: the FMA residual rounds
        // to zero and would lose the sign. Step N6 keeps it.
        let (a, b) = (
            (1.0 + 2.0 * f64::EPSILON) * 2f64.powi(-1000),
            1.0 + f64::EPSILON,
        );
        let v = a / b;
        assert_eq!(v, b * 2f64.powi(-1000));
        assert_eq!((-v).mul_add(b, a), 0.0);
        let q = Interval::point(a) / b;
        assert_eq!((q.inf(), q.sup()), (v.next_down(), v));
        let q = Interval::point(a) / Interval::point(-b);
        assert_eq!((q.inf(), q.sup()), (-v, (-v).next_up()));
        // A quotient that underflows completely.
        let q = Interval::point(2f64.powi(-1000)) / -2f64.powi(1000);
        assert_eq!((q.inf(), q.sup()), (-ETA, 0.0));
    }

    #[test]
    fn division_by_an_interval_with_zero() {
        let one_two = iv(1.0, 2.0);
        assert_eq!(one_two / iv(-1.0, 1.0), Interval::ENTIRE);
        assert_eq!(one_two / Interval::ZERO, Interval::ENTIRE);
        assert_eq!(Interval::ZERO / Interval::ZERO, Interval::ENTIRE);
        assert_eq!(one_two / 0.0, Interval::ENTIRE);
        assert_eq!(Interval::ZERO / iv(-1.0, 1.0), Interval::ZERO);
        assert_eq!(one_two / iv(0.0, 4.0), iv(0.25, INF));
        assert_eq!(-one_two / iv(0.0, 4.0), iv(-INF, -0.25));
        assert_eq!(one_two / iv(-4.0, 0.0), iv(-INF, -0.25));
        assert_eq!(-one_two / iv(-4.0, 0.0), iv(0.25, INF));
        assert_eq!(iv(-1.0, 2.0) / iv(0.0, 4.0), Interval::ENTIRE);
        assert_eq!(iv(0.0, 2.0) / iv(0.0, 4.0), iv(0.0, INF));
    }

    #[test]
    fn infinite_endpoints_bound_sets_of_reals() {
        let pos = iv(0.0, INF);
        // 0 · ∞ = 0 for endpoints (step N1).
        assert_eq!(pos * Interval::ZERO, Interval::ZERO);
        assert_eq!(Interval::ENTIRE * 0.0, Interval::ZERO);
        assert_eq!(pos * iv(2.0, 3.0), pos);
        assert_eq!(pos * iv(-3.0, 2.0), Interval::ENTIRE);
        assert_eq!(pos + iv(1.0, 2.0), iv(1.0, INF));
        assert_eq!(pos - pos, Interval::ENTIRE);
        assert_eq!(iv(1.0, INF) / iv(2.0, INF), pos);
        assert_eq!(iv(-INF, -1.0) / iv(2.0, 4.0), iv(-INF, -0.25));
        assert_eq!(iv(4.0, INF).sqrt(), iv(2.0, INF));
        assert_eq!(iv(-INF, 3.0).square(), pos);
        assert_eq!(iv(-INF, 3.0).abs(), pos);
        assert_eq!(-iv(-INF, 3.0), iv(-3.0, INF));
        // A non-real point operand: nothing is known.
        for bad in [f64::NAN, INF, -INF] {
            assert_eq!(iv(1.0, 2.0) + bad, Interval::ENTIRE);
            assert_eq!(iv(1.0, 2.0) - bad, Interval::ENTIRE);
            assert_eq!(iv(1.0, 2.0) * bad, Interval::ENTIRE);
            assert_eq!(iv(1.0, 2.0) / bad, Interval::ENTIRE);
        }
    }

    #[test]
    fn the_empty_set_propagates_and_proves_nothing() {
        let e = Interval::EMPTY;
        let x = iv(1.0, 2.0);
        for r in [e + x, x + e, e - x, x - e, e * x, x * e, e / x, x / e] {
            assert!(r.is_empty());
        }
        for r in [e + 1.0, e - 1.0, e * 1.0, e / 1.0, e / 0.0, e + f64::NAN] {
            assert!(r.is_empty());
        }
        assert!((-e).is_empty() && e.abs().is_empty() && e.square().is_empty());
        assert!(e.sqrt().is_empty() && iv(-2.0, -1.0).sqrt().is_empty());
        assert_eq!(e.hull(x), x);
        assert_eq!(x.hull(e), x);
        assert!(e.hull(e).is_empty() && e.intersection(x).is_empty());
        assert!(x.intersection(iv(3.0, 4.0)).is_empty());
        assert!(e.mid().is_nan() && e.width().is_nan() && e.mag().is_nan() && e.mig().is_nan());
        assert!(!e.contains(0.0));
        assert!(e.is_subset_of(x) && e.is_subset_of(e) && !x.is_subset_of(e));
        // Chapter 11.9: the predicates of a certificate are false on ∅.
        assert!(!e.certainly_le(INF) && !e.certainly_ge(-INF) && !e.certainly_within(INF));
        assert_eq!(e.check_invariants(), Ok(()));
        assert_eq!(e.to_string(), "∅");
    }

    #[test]
    fn square_roots_are_tight() {
        assert_eq!(iv(4.0, 9.0).sqrt(), iv(2.0, 3.0));
        assert_eq!(iv(-4.0, 9.0).sqrt(), iv(0.0, 3.0));
        assert_eq!(Interval::ZERO.sqrt(), Interval::ZERO);
        let r = Interval::point(2.0).sqrt();
        assert_eq!(r.sup(), r.inf().next_up());
        assert!(r.inf() * r.inf() < 2.0 && r.sup() * r.sup() > 2.0);
        // Subnormal radicands: √η = 2^-537 exactly; √(2η) is irrational.
        assert_eq!(
            Interval::point(ETA).sqrt(),
            Interval::point(2f64.powi(-537))
        );
        let r = Interval::point(2.0 * ETA).sqrt();
        let two = Interval::point(2.0).sqrt();
        assert_eq!(
            (r.inf(), r.sup()),
            (two.inf() * 2f64.powi(-537), two.sup() * 2f64.powi(-537))
        );
        let r = Interval::point(MAX).sqrt();
        assert_eq!(r.sup(), r.inf().next_up());
    }

    #[test]
    fn squares_know_about_zero() {
        assert_eq!(iv(-2.0, 3.0).square(), iv(0.0, 9.0));
        assert_eq!(iv(-3.0, 2.0).square(), iv(0.0, 9.0));
        assert_eq!(iv(-3.0, -2.0).square(), iv(4.0, 9.0));
        assert_eq!(iv(2.0, 3.0).square(), iv(4.0, 9.0));
        // X · X does not: it is wider.
        assert_eq!(iv(-2.0, 3.0) * iv(-2.0, 3.0), iv(-6.0, 9.0));
    }

    #[test]
    fn the_sign_table_of_the_product() {
        let (p, m, n) = (iv(2.0, 3.0), iv(-2.0, 3.0), iv(-3.0, -2.0));
        assert_eq!(p * p, iv(4.0, 9.0));
        assert_eq!(p * m, iv(-6.0, 9.0));
        assert_eq!(p * n, iv(-9.0, -4.0));
        assert_eq!(m * p, iv(-6.0, 9.0));
        assert_eq!(m * m, iv(-6.0, 9.0));
        assert_eq!(m * n, iv(-9.0, 6.0));
        assert_eq!(n * p, iv(-9.0, -4.0));
        assert_eq!(n * m, iv(-9.0, 6.0));
        assert_eq!(n * n, iv(4.0, 9.0));
        assert_eq!(iv(-5.0, 1.0) * iv(-1.0, 7.0), iv(-35.0, 7.0));
        assert_eq!(p * -2.0, iv(-6.0, -4.0));
        assert_eq!(m * 2.0, iv(-4.0, 6.0));
        // The quotient's table.
        assert_eq!(iv(4.0, 8.0) / iv(2.0, 4.0), iv(1.0, 4.0));
        assert_eq!(iv(-8.0, 4.0) / iv(2.0, 4.0), iv(-4.0, 2.0));
        assert_eq!(iv(-8.0, -4.0) / iv(2.0, 4.0), iv(-4.0, -1.0));
        assert_eq!(iv(4.0, 8.0) / iv(-4.0, -2.0), iv(-4.0, -1.0));
        assert_eq!(iv(-8.0, 4.0) / iv(-4.0, -2.0), iv(-2.0, 4.0));
        assert_eq!(iv(-8.0, -4.0) / iv(-4.0, -2.0), iv(1.0, 4.0));
        assert_eq!(iv(-8.0, 4.0) / -2.0, iv(-2.0, 4.0));
    }

    #[test]
    fn queries() {
        let x = iv(-1.0, 3.0);
        assert_eq!((x.mid(), x.width(), x.mag(), x.mig()), (1.0, 4.0, 3.0, 0.0));
        assert_eq!(iv(2.0, 3.0).mig(), 2.0);
        assert_eq!(iv(-3.0, -2.0).mig(), 2.0);
        assert_eq!(iv(-3.0, -2.0).mag(), 3.0);
        assert_eq!(Interval::ENTIRE.mid(), 0.0);
        assert_eq!(iv(1.0, INF).mid(), MAX);
        assert_eq!(iv(-INF, 1.0).mid(), -MAX);
        assert_eq!(iv(-MAX, MAX).mid(), 0.0);
        assert_eq!(iv(MAX, MAX).mid(), MAX);
        assert_eq!(iv(1.0, INF).width(), INF);
        assert_eq!(iv(-MAX, MAX).width(), INF);
        assert_eq!(Interval::ENTIRE.mag(), INF);
        // The width is rounded up.
        assert_eq!(iv(-2f64.powi(-60), 1.0).width(), 1.0_f64.next_up());
        assert!(x.contains(-1.0) && x.contains(3.0) && !x.contains(3.5));
        assert!(!x.contains(f64::NAN) && !Interval::ENTIRE.contains(INF));
        assert!(iv(0.0, 1.0).is_subset_of(x) && !x.is_subset_of(iv(0.0, 1.0)));
        assert!(x.is_subset_of(Interval::ENTIRE));
        assert!(x.certainly_le(3.0) && !x.certainly_le(2.9));
        assert!(x.certainly_ge(-1.0) && !x.certainly_ge(-0.9));
        assert!(x.certainly_within(3.0) && !x.certainly_within(2.9));
        assert!(!x.certainly_le(f64::NAN) && !x.certainly_ge(f64::NAN));
        assert!(!x.certainly_within(f64::NAN) && !x.certainly_within(-5.0));
        assert_eq!(x.hull(iv(5.0, 6.0)), iv(-1.0, 6.0));
        assert_eq!(x.intersection(iv(2.0, 6.0)), iv(2.0, 3.0));
        assert_eq!(x.intersection(iv(3.0, 6.0)), Interval::point(3.0));
        assert_eq!(x.abs(), iv(0.0, 3.0));
        assert_eq!(iv(-3.0, -1.0).abs(), iv(1.0, 3.0));
        assert_eq!(iv(1.0, 3.0).abs(), iv(1.0, 3.0));
    }

    #[test]
    fn display() {
        assert_eq!(iv(-1.5, 2.0).to_string(), "[-1.5, 2]");
        assert_eq!(Interval::ENTIRE.to_string(), "[-inf, inf]");
        assert_eq!(Interval::point(0.1).to_string(), "[0.1, 0.1]");
        assert_eq!(format!("{:e}", iv(-1.5, 2000.0)), "[-1.5e0, 2e3]");
        assert_eq!(format!("{:e}", Interval::EMPTY), "∅");
        assert_eq!(format!("{:.2}", Interval::EMPTY), "∅");
    }

    /// Algorithm D: with a precision the decimals round outward.
    #[test]
    fn display_with_a_precision_rounds_outward() {
        // f64 rounds 0.126 to 0.13 and 0.124 to 0.12; here the lower
        // endpoint goes down and the upper one up, whatever the digit.
        assert_eq!(format!("{:.2}", iv(0.124, 0.126)), "[0.12, 0.13]");
        assert_eq!(format!("{:.2}", iv(0.121, 0.129)), "[0.12, 0.13]");
        assert_eq!(format!("{:.2}", iv(-0.129, -0.121)), "[-0.13, -0.12]");
        assert_eq!(format!("{:.0}", iv(-0.5, 0.5)), "[-1, 1]");
        assert_eq!(format!("{:.0}", iv(1.0, 2.0)), "[1, 2]");
        // The decimal 0.1 is below the f64 0.1 = 0.1000000000000000055…
        assert_eq!(format!("{:.1}", Interval::point(0.1)), "[0.1, 0.2]");
        assert_eq!(format!("{:.3}", Interval::point(0.125)), "[0.125, 0.125]");
        assert_eq!(format!("{:.1}", iv(9.95, 9.96)), "[9.9, 10.0]");
        assert_eq!(format!("{:.1}", iv(-99.99, 99.99)), "[-100.0, 100.0]");
        // Zeros keep their sign, as f64 prints them.
        assert_eq!(format!("{:.2}", iv(-0.0, 0.0)), "[-0.00, 0.00]");
        assert_eq!(format!("{:.2}", iv(0.0, 0.0)), "[0.00, 0.00]");
        assert_eq!(format!("{:+.1}", iv(-1.0, 1.0)), "[-1.0, +1.0]");
        // Subnormals: the smallest positive one is far below 10^-3.
        assert_eq!(format!("{:.3}", iv(-ETA, ETA)), "[-0.001, 0.001]");
        assert_eq!(format!("{:.3}", iv(0.0, ETA)), "[0.000, 0.001]");
        assert_eq!(format!("{:.2}", Interval::ENTIRE), "[-inf, inf]");
        assert_eq!(format!("{:.1}", iv(1.0, INF)), "[1.0, inf]");
        // The whole expansion when the precision asks for it.
        let s = format!("{:.1100}", Interval::point(ETA));
        assert_eq!(s, format!("[{ETA:.1100}, {ETA:.1100}]"));

        // Scientific notation.
        assert_eq!(format!("{:.2e}", iv(1234.0, 5678.0)), "[1.23e3, 5.68e3]");
        assert_eq!(
            format!("{:.2e}", iv(-5678.0, -1234.0)),
            "[-5.68e3, -1.23e3]"
        );
        assert_eq!(format!("{:.0e}", iv(1.5, 2.5)), "[1e0, 3e0]");
        assert_eq!(format!("{:.1e}", iv(0.0996, 9.96)), "[9.9e-2, 1.0e1]");
        assert_eq!(format!("{:.1e}", iv(-9.96, -0.0)), "[-1.0e1, -0.0e0]");
        assert_eq!(format!("{:.3e}", iv(0.0, 0.0)), "[0.000e0, 0.000e0]");
        assert_eq!(format!("{:.2e}", iv(-ETA, ETA)), "[-4.95e-324, 4.95e-324]");
        assert_eq!(format!("{:.2e}", iv(-MAX, MAX)), "[-1.80e308, 1.80e308]");
        assert_eq!(format!("{:.1e}", Interval::ENTIRE), "[-inf, inf]");
        assert_eq!(
            format!("{:.3e}", Interval::point(1.0)),
            "[1.000e0, 1.000e0]"
        );
    }

    #[test]
    fn from_f64_is_total() {
        for x in [
            0.0,
            -0.0,
            0.1,
            -7.5,
            ETA,
            -ETA,
            MAX,
            -MAX,
            f64::MIN_POSITIVE,
        ] {
            let (p, f) = (Interval::point(x), Interval::from(x));
            assert_eq!(p.inf().to_bits(), f.inf().to_bits(), "{x:e}");
            assert_eq!(p.sup().to_bits(), f.sup().to_bits(), "{x:e}");
            let g: Interval = x.into();
            assert_eq!(g, f);
        }
        for x in [f64::NAN, INF, -INF] {
            assert_eq!(Interval::from(x), Interval::ENTIRE, "{x}");
            // The same answer as the operators with that scalar.
            assert_eq!(Interval::from(x) + iv(1.0, 2.0), x + iv(1.0, 2.0));
        }
    }

    #[test]
    fn errors_are_errors() {
        let e: Box<dyn core::error::Error> = Box::new(InvalidInterval);
        assert!(e.to_string().starts_with("invalid interval"));
    }

    /// Invariant (I4) on the cases of the sign tables and the exceptions.
    #[test]
    fn operator_forms_agree_bit_for_bit() {
        let same = |p: Interval, q: Interval| {
            p.inf().to_bits() == q.inf().to_bits() && p.sup().to_bits() == q.sup().to_bits()
        };
        let xs = [
            iv(2.0, 3.0),
            iv(-2.0, 3.0),
            iv(-3.0, -2.0),
            iv(0.0, 0.1),
            iv(-0.1, 0.0),
            Interval::ZERO,
            iv(1.0, INF),
            Interval::ENTIRE,
            Interval::EMPTY,
        ];
        let ys = [0.1, -0.3, 0.0, 7.0, MAX, ETA, f64::NAN, INF, -INF];
        for &x in &xs {
            for &z in &xs {
                let mut t = x;
                t += z;
                assert!(same(t, x + z));
                t = x;
                t -= z;
                assert!(same(t, x - z));
                t = x;
                t *= z;
                assert!(same(t, x * z));
                t = x;
                t /= z;
                assert!(same(t, x / z));
            }
            for &y in &ys {
                let mut t = x;
                t += y;
                assert!(same(t, x + y));
                t = x;
                t -= y;
                assert!(same(t, x - y));
                t = x;
                t *= y;
                assert!(same(t, x * y));
                t = x;
                t /= y;
                assert!(same(t, x / y));
                // A left scalar is the point [y, y]; not a real number:
                // ENTIRE (∅ for an empty X).
                let point = |op: fn(Interval, Interval) -> Interval| {
                    if x.is_empty() {
                        Interval::EMPTY
                    } else if y.is_finite() {
                        op(Interval::point(y), x)
                    } else {
                        Interval::ENTIRE
                    }
                };
                assert!(same(y + x, point(|p, q| p + q)), "{y} + {x:?}");
                assert!(same(y - x, point(|p, q| p - q)), "{y} - {x:?}");
                assert!(same(y * x, point(|p, q| p * q)), "{y} * {x:?}");
                assert!(same(y / x, point(|p, q| p / q)), "{y} / {x:?}");
            }
        }
        assert_eq!(1.0 - iv(0.25, 0.5), iv(0.5, 0.75));
        assert_eq!(1.0 / iv(2.0, 4.0), iv(0.25, 0.5));
        assert_eq!(1.0 / iv(0.0, 4.0), iv(0.25, INF));
        assert_eq!(1.0 / iv(-1.0, 1.0), Interval::ENTIRE);
        assert_eq!(0.0 / iv(-1.0, 1.0), Interval::ZERO);
    }
}
