//! # Chapter 11.5 — Slice kernels (scalar reference)
//!
//! ## Purpose
//! The inner loops of extended-precision linear algebra: compensated sums
//! and dot products of `f64` data that return a double-double (Ogita, Rump
//! & Oishi 2005), one of them with a *rigorous* error bound, and the
//! double-double `dot`, `axpy` and small-block `gemm`. This chapter is the
//! **scalar reference**: it fixes, operation by operation, what each kernel
//! computes. The SIMD versions (L2.2b, `pulp`) must return the same bits
//! (invariant K1), so the order of accumulation is part of each
//! definition, not an implementation detail.
//!
//! ## Notation
//! `u = 2⁻⁵³`, `η = 2⁻¹⁰⁷⁴`, `γ_k = k u / (1 − k u)`. `2Sum` and `2Prod`
//! are the error-free transformations of Chapter 11.1, `⊕ ⊖ ⊗` the
//! double-double operations of Chapter 11.2 (Algorithms A2, M2, M1). `W =
//! 4` is the number of accumulators — *lanes* — on every target: the width
//! of AVX2; narrower targets emulate it, wider ones use it, so that there
//! is one answer everywhere. Element `i` belongs to lane `i mod W`.
//!
//! ## Algorithms
//! **Algorithm C (compensated accumulation: `sum2`, `dot2`, `dot2_err`).**
//! Input `x ∈ Fⁿ` (and `y ∈ Fⁿ`). Output a double-double `r` (and `β`).
//!
//! C1. \[Lanes.\] `p_k ← 0`, `s_k ← 0`, `e_k ← 0` for `k = 0, …, W − 1`.
//! C2. \[Accumulate.\] For `i = 0, …, n − 1`, with `k = i mod W`: form the
//!     term — `sum2`: `(p_k, q) ← 2Sum(p_k, x_i)`, `t ← q`; `dot2` and
//!     `dot2_err`: `(h, r) ← 2Prod(x_i, y_i)`, `(p_k, q) ← 2Sum(p_k, h)`,
//!     `t ← RN(q + r)` — then `s_k ← RN(s_k + t)` and, for `dot2_err`,
//!     `e_k ← RN(e_k + |t|)`.
//! C3. \[Combine.\] `p ← p_0`, `s ← s_0`, `e ← e_0`. For `k = 1, …, W − 1`:
//!     `(p, q) ← 2Sum(p, p_k)`; `s ← RN(s + RN(s_k + q))`; `e ← RN(e +
//!     RN(e_k + |q|))`.
//! C4. \[Result.\] `r ← 2Sum(p, s)`, with step E of Chapter 11.2.
//! C5. \[Bound, `dot2_err` only.\] If `n = 0`, `β ← 0`. If `2(n + 3)u ≥ 1`,
//!     or `p`, `s`, `e` or `r` is not finite, `β ← +∞`. Otherwise, in `f64`
//!     arithmetic, `δ ← (n + 3)u / (1 − 2(n + 3)u)` and `β ← ((δ e + 3η/u)
//!     (1 + 8u))`.
//!
//! **Algorithm B (double-double BLAS: `dot`, `dot_mixed`, `axpy`,
//! `gemm_sub`).**
//!
//! B1. \[`dot`, `dot_mixed`.\] `a_k ← 0`; for `i = 0, …, n − 1`: `a_k ← a_k
//!     ⊕ (x_i ⊗ y_i)` with `k = i mod W`; return `((a_0 ⊕ a_1) ⊕ a_2) ⊕
//!     a_3`. In `dot_mixed`, `y_i` is an `f64` and `⊗` is Algorithm M1.
//! B2. \[`axpy`.\] `y_i ← y_i ⊕ (α ⊗ x_i)` for each `i` (no accumulation).
//! B3. \[`gemm_sub`.\] For `j = 0, …, n − 1`, for `l = 0, …, k − 1`, for `i
//!     = 0, …, m − 1`: `c_ij ← c_ij ⊖ (a_il ⊗ b_lj)`. Each `c_ij` receives
//!     its `k` updates in increasing `l`; the loop over `i` is the one a
//!     SIMD version vectorizes.
//!
//! ## Bounds
//! Sum2 and Dot2 are analysed by Ogita, Rump & Oishi for the sequential
//! order; their argument uses the order only through "a floating-point sum
//! of `m` terms, in any order, errs by at most `γ_{m−1} Σ|terms|`" (their
//! (2.4)) and so carries over to Algorithm C, with `n + 3` in place of `n`
//! (the three `2Sum` of step C3 add three error terms). Barring overflow:
//!
//! * `sum2`: `|r − Σx_i| ≤ γ²_{n+3} Σ|x_i|` (their Proposition 4.5 has
//!   `γ²_{n−1}` for the sequential order, plus `u|Σx_i|` for rounding the
//!   result to one `f64`, which C4 does not do);
//! * `dot2`: `|r − xᵀy| ≤ γ²_{n+5} |x|ᵀ|y| + n η/2` (their Proposition 5.5:
//!   `γ²_n`, and `5nη` for the Dekker product in place of the FMA).
//!
//! **Proof of step C5.** Barring overflow, and with every `2Prod` exact,
//! `xᵀy = p + Σ_i (q_i + r_i) + Σ_c q_c` exactly, where the `q_c` are the
//! three errors of step C3. `s` is a floating-point sum, in some order, of
//! the `N = n + 3` terms `t_i = RN(q_i + r_i)` and `q_c`, so with `E =
//! Σ|t_i| + Σ|q_c|`:
//! `|xᵀy − p − s| ≤ u Σ|t_i| + γ_{N−1} E ≤ γ_{n+3} E`. And `e` is a
//! floating-point sum of the same `N` nonnegative magnitudes, so `E ≤ (1 +
//! u)^{n+2} e`. Hence `|xᵀy − p − s| ≤ (n + 3)u / (1 − (2n + 5)u) · e`. A
//! `2Prod` out of its exact range adds at most `η/2` (invariant T2 of
//! Chapter 11.1): `n η/2` in all. Step C5 evaluates an upper bound of that
//! in round-to-nearest: `δ` is computed from exact operands by one division
//! (`δ ≥ (1 − u)` times the exact coefficient above, whose denominator is
//! the larger `1 − (2n + 5)u`), the product `δ e` errs by at most a factor
//! `(1 − u)` and `η/2`, the two remaining operations by `(1 − u)` each (the
//! term `3η/u = 3 · 2⁻¹⁰²¹` keeps them out of the subnormal range), and `(1
//! + 8u)(1 − u)⁴ ≥ 1`, `3η/u − η/2 ≥ n η/2 + η/2`. Overflow anywhere
//! leaves an infinity or a NaN in `p`, `s` or `e` (the operations are
//! additions, products and absolute values: none maps a non-finite value
//! back to a finite one), which C5 tests. This is Algorithm 5.8 and
//! Corollary 5.9 of Ogita, Rump & Oishi adapted to the lanes, to the FMA
//! product and to a double-double result (their bound has one more term,
//! `u|res|`, for the final rounding to `f64`).
//!
//! The kernels of Algorithm B inherit the bounds of `⊕` and `⊗`: each
//! term errs by at most `4u²` relative, each accumulation by `3u² + 13u³`.
//!
//! ## Invariants
//! (K1) *One answer.* Every kernel is the fixed sequence of `f64`
//!      operations above for a given `n`; a SIMD implementation returns the
//!      same bits. [`backend`] names the implementation in use.
//! (K2) *Rigour.* `dot2_err` returns `(r, β)` with `|xᵀy − r| ≤ β` for
//!      every input; `β = +∞` when a hypothesis fails (invariant C3 of
//!      Chapter 11).
//!
//! ## References
//! Ogita, Rump & Oishi, "Accurate sum and dot product", SIAM J. Sci.
//! Comput. 26 (2005): Algorithm 4.4 (Sum2) and Proposition 4.5, Algorithm
//! 5.3 (Dot2) and Proposition 5.5, Algorithm 5.8 (Dot2Err) and Corollary
//! 5.9. Higham, *Accuracy and Stability of Numerical Algorithms* (2002),
//! §4.2 (the order-independent bound).

use crate::dd::DoubleDouble;
use crate::eft::{two_prod, two_sum};

/// `W`, the number of lanes (the chapter).
const LANES: usize = 4;

/// `2⁻⁵³`.
const U: f64 = 0.5 * f64::EPSILON;

/// The state of Algorithm C: the lanes `p_k`, `s_k`, `e_k`.
#[derive(Clone, Copy, Default)]
struct Compensated {
    p: [f64; LANES],
    s: [f64; LANES],
    e: [f64; LANES],
}

impl Compensated {
    /// Step C2 for the term `h + r` (`r = 0` for a sum) of element `i`.
    #[inline]
    fn push(&mut self, i: usize, h: f64, r: f64) {
        let k = i % LANES;
        let (p, q) = two_sum(self.p[k], h);
        let t = q + r;
        self.p[k] = p;
        self.s[k] += t;
        self.e[k] += t.abs();
    }

    /// Step C3: `(p, s, e)`.
    #[inline]
    fn combine(&self) -> (f64, f64, f64) {
        let (mut p, mut s, mut e) = (self.p[0], self.s[0], self.e[0]);
        for k in 1..LANES {
            let (pn, q) = two_sum(p, self.p[k]);
            p = pn;
            s += self.s[k] + q;
            e += self.e[k] + q.abs();
        }
        (p, s, e)
    }
}

/// `Σ xᵢ` of `f64` values as a double-double, as accurate as if summed
/// in twice the working precision (Sum2 of Ogita, Rump & Oishi, over `W =
/// 4` lanes: Algorithm C of Chapter 11.5): `|result − Σxᵢ| ≤ γ²_{n+3}
/// Σ|xᵢ|` barring overflow.
#[must_use]
pub fn sum2(x: &[f64]) -> DoubleDouble {
    let mut acc = Compensated::default();
    for (i, &xi) in x.iter().enumerate() {
        acc.push(i, xi, 0.0);
    }
    let (p, s, _) = acc.combine();
    DoubleDouble::from_sum(p, s)
}

/// `xᵀy` of `f64` vectors as a double-double (Dot2, over `W = 4` lanes:
/// Algorithm C of Chapter 11.5): `|result − xᵀy| ≤ γ²_{n+5} |x|ᵀ|y| + n
/// η/2` barring overflow. Panics if the lengths differ.
#[must_use]
pub fn dot2(x: &[f64], y: &[f64]) -> DoubleDouble {
    assert_eq!(x.len(), y.len(), "dot2: the lengths differ");
    let mut acc = Compensated::default();
    for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
        let (h, r) = two_prod(xi, yi);
        acc.push(i, h, r);
    }
    let (p, s, _) = acc.combine();
    DoubleDouble::from_sum(p, s)
}

/// As [`dot2`], with a rigorous bound: returns `(r, β)` with `|xᵀy −
/// r| ≤ β` (Dot2Err; step C5 of Chapter 11.5 and its proof). `β = +∞`
/// when a hypothesis fails (overflow, NaN, `2(n + 3)u ≥ 1`): invariant C3.
/// Panics if the lengths differ.
#[must_use]
pub fn dot2_err(x: &[f64], y: &[f64]) -> (DoubleDouble, f64) {
    /// `3η/u = 3 · 2⁻¹⁰²¹`.
    const UNDERFLOW: f64 = 6.0 * f64::MIN_POSITIVE;
    /// `1 + 8u`.
    const INFLATE: f64 = 1.0 + 4.0 * f64::EPSILON;
    /// `2(n + 3)u < 1` iff `n + 3 < 2⁵²`.
    const MAX_TERMS: u64 = 1 << 52;

    assert_eq!(x.len(), y.len(), "dot2_err: the lengths differ");
    let mut acc = Compensated::default();
    for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
        let (h, r) = two_prod(xi, yi);
        acc.push(i, h, r);
    }
    let (p, s, e) = acc.combine();
    let r = DoubleDouble::from_sum(p, s);
    // C5.
    if x.is_empty() {
        return (r, 0.0);
    }
    let terms = u64::try_from(x.len()).map_or(u64::MAX, |n| n.saturating_add(3));
    let finite = p.is_finite() && s.is_finite() && e.is_finite() && r.is_finite();
    if terms >= MAX_TERMS || !finite {
        return (r, f64::INFINITY);
    }
    #[allow(clippy::cast_precision_loss)] // exact: terms < 2^52
    let a = terms as f64 * U;
    let delta = a / (1.0 - 2.0 * a);
    (r, (delta * e + UNDERFLOW) * INFLATE)
}

/// `xᵀy` of double-double vectors, over `W = 4` lanes combined left to
/// right (step B1 of Chapter 11.5). Panics if the lengths differ.
#[must_use]
pub fn dot(x: &[DoubleDouble], y: &[DoubleDouble]) -> DoubleDouble {
    assert_eq!(x.len(), y.len(), "dot: the lengths differ");
    let mut acc = [DoubleDouble::ZERO; LANES];
    for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
        acc[i % LANES] += xi * yi;
    }
    ((acc[0] + acc[1]) + acc[2]) + acc[3]
}

/// `xᵀy` with `x` double-double and `y` `f64` (a row of an `f64`
/// matrix against an extended vector: the residual of a polished
/// solution); the order of [`dot`]. Panics if the lengths differ.
#[must_use]
pub fn dot_mixed(x: &[DoubleDouble], y: &[f64]) -> DoubleDouble {
    assert_eq!(x.len(), y.len(), "dot_mixed: the lengths differ");
    let mut acc = [DoubleDouble::ZERO; LANES];
    for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
        acc[i % LANES] += xi * yi;
    }
    ((acc[0] + acc[1]) + acc[2]) + acc[3]
}

/// `y ← y + α x`, element by element (step B2 of Chapter 11.5). Panics if
/// the lengths differ.
pub fn axpy(alpha: DoubleDouble, x: &[DoubleDouble], y: &mut [DoubleDouble]) {
    assert_eq!(x.len(), y.len(), "axpy: the lengths differ");
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

/// `C ← C − A B` for small column-major blocks (`A` is `m × k`, `B` is
/// `k × n`, `C` is `m × n`); the update of a supernodal block. Each entry
/// of `C` receives its `k` updates in increasing column of `A` (step B3 of
/// Chapter 11.5). Panics if a slice has the wrong length.
pub fn gemm_sub(
    m: usize,
    k: usize,
    n: usize,
    a: &[DoubleDouble],
    b: &[DoubleDouble],
    c: &mut [DoubleDouble],
) {
    assert_eq!(a.len(), m * k, "gemm_sub: A is not m × k");
    assert_eq!(b.len(), k * n, "gemm_sub: B is not k × n");
    assert_eq!(c.len(), m * n, "gemm_sub: C is not m × n");
    if m == 0 {
        return;
    }
    for (j, cj) in c.chunks_exact_mut(m).enumerate() {
        for l in 0..k {
            let blj = b[l + j * k];
            for (cij, &ail) in cj.iter_mut().zip(&a[l * m..(l + 1) * m]) {
                *cij -= ail * blj;
            }
        }
    }
}

/// The name of the implementation the dispatch selected (`"scalar"`,
/// `"avx2"`, `"avx512"`, `"neon"`), for benchmarks and bug reports. Only
/// the scalar reference exists so far (L2.2a).
#[must_use]
pub fn backend() -> &'static str {
    "scalar"
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the tests compare exact values
mod tests {
    use super::*;

    type Dd = DoubleDouble;

    #[test]
    fn empty_inputs() {
        assert_eq!(sum2(&[]), Dd::ZERO);
        assert_eq!(dot2(&[], &[]), Dd::ZERO);
        assert_eq!(dot2_err(&[], &[]), (Dd::ZERO, 0.0));
        assert_eq!(dot(&[], &[]), Dd::ZERO);
        assert_eq!(dot_mixed(&[], &[]), Dd::ZERO);
        axpy(Dd::ONE, &[], &mut []);
        gemm_sub(0, 0, 0, &[], &[], &mut []);
        gemm_sub(0, 2, 3, &[], &[Dd::ONE; 6], &mut []);
        assert_eq!(backend(), "scalar");
    }

    #[test]
    fn sum2_survives_cancellation() {
        // The plain f64 sum loses the small terms.
        let x = [1e100, 1.0, -1e100, 2.0];
        assert_eq!(x.iter().sum::<f64>(), 2.0);
        assert_eq!(sum2(&x), Dd::from_f64(3.0));
        let tiny = 2f64.powi(-60);
        assert_eq!(sum2(&[1.0, tiny, -1.0]), Dd::from_f64(tiny));
        // The result is the unrounded pair (p, s) of step C4.
        assert_eq!(sum2(&[1.0, tiny]), Dd::from_parts(1.0, tiny));
        // More terms than lanes.
        let y: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(sum2(&y), Dd::from_f64(55.0));
    }

    #[test]
    fn dot2_survives_cancellation() {
        let big = 2f64.powi(60);
        let x = [big, 1.0 + f64::EPSILON, -big];
        let y = [big, 1.0 + f64::EPSILON, big];
        assert_eq!(x.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>(), 0.0);
        // "As if in twice the working precision": the leading component is
        // right, the error is within the bound of Dot2Err.
        let exact = Dd::from_product(x[1], y[1]);
        let r = dot2(&x, &y);
        assert_eq!(r.hi(), exact.hi());
        let (r_err, beta) = dot2_err(&x, &y);
        assert_eq!(r_err, r);
        assert!(beta.is_finite() && beta < 1e-10, "{beta:e}");
        assert!((r - exact).abs().hi() <= beta);
    }

    #[test]
    fn dot2_err_reports_failed_hypotheses() {
        let (r, beta) = dot2_err(&[1e200, 1.0], &[1e200, 1.0]);
        assert!(r.check_invariants().is_ok());
        assert_eq!(beta, f64::INFINITY);
        let (r, beta) = dot2_err(&[f64::NAN], &[1.0]);
        assert!(r.is_nan());
        assert_eq!(beta, f64::INFINITY);
        // Exact data still have a positive bound: the underflow term.
        let (r, beta) = dot2_err(&[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(r, Dd::from_f64(11.0));
        assert!(beta > 0.0 && beta < 1e-300);
    }

    #[test]
    fn dd_kernels_follow_their_definitions() {
        let x: Vec<Dd> = (1..=7).map(|i| Dd::ONE / f64::from(i)).collect();
        let y: Vec<Dd> = (1..=7).map(|i| Dd::from_f64(f64::from(i)).sqrt()).collect();
        let yf: Vec<f64> = y.iter().map(|v| v.hi()).collect();
        // B1: lanes 0..4, then left to right.
        let lanes = |term: &dyn Fn(usize) -> Dd| {
            let mut a = [Dd::ZERO; 4];
            for i in 0..7 {
                a[i % 4] += term(i);
            }
            ((a[0] + a[1]) + a[2]) + a[3]
        };
        assert_eq!(dot(&x, &y), lanes(&|i| x[i] * y[i]));
        assert_eq!(dot_mixed(&x, &yf), lanes(&|i| x[i] * yf[i]));
        // B2.
        let alpha = Dd::ONE / 7.0;
        let mut z = y.clone();
        axpy(alpha, &x, &mut z);
        for i in 0..7 {
            assert_eq!(z[i], y[i] + alpha * x[i]);
        }
        // B3 on a 2 × 3 times 3 × 2 product.
        let a = &x[..6];
        let b = &y[..6];
        let mut c = vec![Dd::ONE; 4];
        gemm_sub(2, 3, 2, a, b, &mut c);
        for j in 0..2 {
            for i in 0..2 {
                let mut want = Dd::ONE;
                for l in 0..3 {
                    want -= a[i + 2 * l] * b[l + 3 * j];
                }
                assert_eq!(c[i + 2 * j], want);
            }
        }
    }

    #[test]
    #[should_panic(expected = "the lengths differ")]
    fn length_mismatch_panics() {
        let _ = dot2(&[1.0], &[]);
    }
}
