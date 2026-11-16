//! # Chapter 11.5 — Slice kernels: definitions, scalar reference, dispatch
//!
//! ## Purpose
//! The inner loops of extended-precision linear algebra: compensated sums
//! and dot products of `f64` data that return a double-double (Ogita, Rump
//! & Oishi 2005), one of them with a *rigorous* error bound, and the
//! double-double `dot`, `axpy` and small-block `gemm`. This chapter fixes,
//! operation by operation, what each kernel computes, and holds the
//! **scalar reference** implementation of that definition (the private
//! module `reference`). The SIMD versions (Chapter 11.6, feature `simd`)
//! return the same bits (invariant K1), so the order of accumulation is
//! part of each definition, not an implementation detail. The public
//! functions of this module choose between the two at run time
//! (Algorithm X below).
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
//!     its `k` updates in increasing `l`; the loop over `i` — the *column
//!     update* `c ← c ⊖ a ⊗ t` of one column of `C` by one column of `A`
//!     and one scalar `t = b_lj` — is the one a SIMD version vectorizes.
//!     The entries of `C` do not interact, so the loops over `j`, `l` and
//!     `i` may be blocked for the cache in any way that keeps `l`
//!     increasing for each `(i, j)`: the kernels run in blocks of
//!     `BLOCK_ROWS` rows and `BLOCK_K` columns of `A`, the scheme of
//!     `xingo_sparse::NaiveDense` (Chapter 5.16).
//!
//! **Algorithm X (dispatch).** For a call of a kernel `κ`:
//!
//! X1. \[Feature.\] Without the feature `simd`, run the reference `κ`.
//! X2. \[Detect.\] On `aarch64`, if NEON is available (it always is), run
//!     the SIMD `κ` of Chapter 11.6 with two-lane vectors; on `x86` and
//!     `x86_64`, if AVX2 and FMA are available, run it with four-lane
//!     vectors. The detection is `pulp`'s (cached after the first call).
//! X3. \[Fallback.\] Otherwise run the reference `κ`.
//!
//! [`backend`] reports the choice of X1–X3. AVX-512 is deliberately not a
//! backend: `W = 4` is the width of AVX2, and a wider accumulator would be
//! a different kernel (another order of accumulation).
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
//!      operations above for a given `n`; the SIMD implementation returns
//!      the same bits for **every** input — finite or not, normalized or
//!      not — with one reservation: where a result is a NaN, both are NaN,
//!      and the sign and payload of that NaN are not specified (Rust does
//!      not specify them for `f64` arithmetic either). Chapter 11.6 has the
//!      argument; the tests of this module compare the two implementations
//!      bit for bit on random and on exceptional data. [`backend`] names
//!      the implementation in use.
//! (K2) *Rigour.* `dot2_err` returns `(r, β)` with `|xᵀy − r| ≤ β` for
//!      every input; `β = +∞` when a hypothesis fails (invariant C3 of
//!      Chapter 11).
//!
//! ## References
//! Ogita, Rump & Oishi, "Accurate sum and dot product", SIAM J. Sci.
//! Comput. 26 (2005): Algorithm 4.4 (Sum2) and Proposition 4.5, Algorithm
//! 5.3 (Dot2) and Proposition 5.5, Algorithm 5.8 (Dot2Err) and Corollary
//! 5.9. Higham, *Accuracy and Stability of Numerical Algorithms* (2002),
//! §4.2 (the order-independent bound). Chapter 5.16 of `xingo-sparse`
//! (the blocked loop order of `gemm_sub`).

use crate::dd::DoubleDouble;
use crate::eft::two_sum;

/// `W`, the number of lanes (the chapter).
pub(crate) const LANES: usize = 4;

/// Rows of `C` and `A` handled per block of step B3 (a cache-blocking
/// size, not a numerical threshold; not tuned): `BLOCK_ROWS × BLOCK_K`
/// double-doubles of `A` — 128 KiB — stay in cache while the columns of `C`
/// pass over them.
pub(crate) const BLOCK_ROWS: usize = 128;
/// Columns of `A` (the summation index `l`) per block; see [`BLOCK_ROWS`].
pub(crate) const BLOCK_K: usize = 64;

/// `2⁻⁵³`.
const U: f64 = 0.5 * f64::EPSILON;

/// The state of Algorithm C: the lanes `p_k`, `s_k`, `e_k`.
#[derive(Clone, Copy, Default)]
pub(crate) struct Compensated {
    pub(crate) p: [f64; LANES],
    pub(crate) s: [f64; LANES],
    pub(crate) e: [f64; LANES],
}

impl Compensated {
    /// Step C2 for the term `h + r` (`r = 0` for a sum) of element `i`.
    #[inline]
    pub(crate) fn push(&mut self, i: usize, h: f64, r: f64) {
        let k = i % LANES;
        let (p, q) = two_sum(self.p[k], h);
        let t = q + r;
        self.p[k] = p;
        self.s[k] += t;
        self.e[k] += t.abs();
    }

    /// Step C3: `(p, s, e)`.
    #[inline]
    pub(crate) fn combine(&self) -> (f64, f64, f64) {
        let (mut p, mut s, mut e) = (self.p[0], self.s[0], self.e[0]);
        for k in 1..LANES {
            let (pn, q) = two_sum(p, self.p[k]);
            p = pn;
            s += self.s[k] + q;
            e += self.e[k] + q.abs();
        }
        (p, s, e)
    }

    /// Steps C3 and C4: the result `r`.
    #[inline]
    pub(crate) fn result(&self) -> DoubleDouble {
        let (p, s, _) = self.combine();
        DoubleDouble::from_sum(p, s)
    }

    /// Steps C3–C5 for `n` elements: the result `r` and the bound `β`.
    #[inline]
    pub(crate) fn result_with_bound(&self, n: usize) -> (DoubleDouble, f64) {
        /// `3η/u = 3 · 2⁻¹⁰²¹`.
        const UNDERFLOW: f64 = 6.0 * f64::MIN_POSITIVE;
        /// `1 + 8u`.
        const INFLATE: f64 = 1.0 + 4.0 * f64::EPSILON;
        /// `2(n + 3)u < 1` iff `n + 3 < 2⁵²`.
        const MAX_TERMS: u64 = 1 << 52;

        let (p, s, e) = self.combine();
        let r = DoubleDouble::from_sum(p, s);
        // C5.
        if n == 0 {
            return (r, 0.0);
        }
        let terms = u64::try_from(n).map_or(u64::MAX, |n| n.saturating_add(3));
        let finite = p.is_finite() && s.is_finite() && e.is_finite() && r.is_finite();
        if terms >= MAX_TERMS || !finite {
            return (r, f64::INFINITY);
        }
        #[allow(clippy::cast_precision_loss)] // exact: terms < 2^52
        let a = terms as f64 * U;
        let delta = a / (1.0 - 2.0 * a);
        (r, (delta * e + UNDERFLOW) * INFLATE)
    }
}

/// The column update of step B3, `c_i ← c_i ⊖ (a_i ⊗ t)` for every `i`:
/// the one loop that the kernels of Algorithm B built on `gemm` vectorize
/// (and the dense kernels of Chapter 11.7 with them). Implemented by the
/// scalar [`reference::Reference`] and by the vectors of Chapter 11.6.
pub(crate) trait ColumnUpdate: Copy {
    /// `c ← c ⊖ a ⊗ t`; `c` and `a` have the same length.
    fn column_sub(self, c: &mut [DoubleDouble], a: &[DoubleDouble], t: DoubleDouble);
}

/// Step B3 with the blocks of the chapter: `c_ij ← c_ij ⊖ Σ_l a_il ⊗
/// t(j, l)` for the `m × k` block `a` and the `m × n` block `c`
/// (column-major), `l` increasing for each entry.
#[inline(always)] // into `pulp`'s `vectorize` (Chapter 11.6)
#[allow(clippy::inline_always)]
pub(crate) fn gemm_blocked<U: ColumnUpdate>(
    update: U,
    m: usize,
    k: usize,
    a: &[DoubleDouble],
    c: &mut [DoubleDouble],
    t: impl Fn(usize, usize) -> DoubleDouble,
) {
    debug_assert_eq!(a.len(), m * k, "gemm_blocked: A is not m × k");
    debug_assert!(m == 0 || c.len().is_multiple_of(m), "gemm_blocked: C");
    if m == 0 {
        return;
    }
    let mut i0 = 0;
    while i0 < m {
        let i1 = (i0 + BLOCK_ROWS).min(m);
        let mut l0 = 0;
        while l0 < k {
            let l1 = (l0 + BLOCK_K).min(k);
            for (j, cj) in c.chunks_exact_mut(m).enumerate() {
                let cj = &mut cj[i0..i1];
                for l in l0..l1 {
                    update.column_sub(cj, &a[l * m + i0..l * m + i1], t(j, l));
                }
            }
            l0 = l1;
        }
        i0 = i1;
    }
}

/// The scalar reference kernels: Algorithms C and B of the chapter, one
/// `f64` operation at a time. They are the definition of the kernels, the
/// oracle of the SIMD versions (invariant K1) and the fallback of the
/// dispatch (Algorithm X).
pub(crate) mod reference {
    use super::{ColumnUpdate, Compensated, LANES, gemm_blocked};
    use crate::dd::DoubleDouble;
    use crate::eft::two_prod;

    /// The scalar column update (step B3).
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Reference;

    impl ColumnUpdate for Reference {
        #[inline]
        fn column_sub(self, c: &mut [DoubleDouble], a: &[DoubleDouble], t: DoubleDouble) {
            debug_assert_eq!(c.len(), a.len(), "column_sub: the lengths differ");
            for (ci, &ai) in c.iter_mut().zip(a) {
                *ci -= ai * t;
            }
        }
    }

    /// Steps C1–C2 of `sum2`.
    pub(crate) fn sum2(x: &[f64]) -> Compensated {
        let mut acc = Compensated::default();
        for (i, &xi) in x.iter().enumerate() {
            acc.push(i, xi, 0.0);
        }
        acc
    }

    /// Steps C1–C2 of `dot2` and `dot2_err`.
    pub(crate) fn dot2(x: &[f64], y: &[f64]) -> Compensated {
        let mut acc = Compensated::default();
        for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
            let (h, r) = two_prod(xi, yi);
            acc.push(i, h, r);
        }
        acc
    }

    /// Step B1, double-double `y`.
    pub(crate) fn dot(x: &[DoubleDouble], y: &[DoubleDouble]) -> DoubleDouble {
        let mut acc = [DoubleDouble::ZERO; LANES];
        for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
            acc[i % LANES] += xi * yi;
        }
        ((acc[0] + acc[1]) + acc[2]) + acc[3]
    }

    /// Step B1, `f64` `y`.
    pub(crate) fn dot_mixed(x: &[DoubleDouble], y: &[f64]) -> DoubleDouble {
        let mut acc = [DoubleDouble::ZERO; LANES];
        for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
            acc[i % LANES] += xi * yi;
        }
        ((acc[0] + acc[1]) + acc[2]) + acc[3]
    }

    /// Step B2.
    pub(crate) fn axpy(alpha: DoubleDouble, x: &[DoubleDouble], y: &mut [DoubleDouble]) {
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi += alpha * xi;
        }
    }

    /// Step B3 (`b` is `k × n`, column-major).
    pub(crate) fn gemm_sub(
        m: usize,
        k: usize,
        a: &[DoubleDouble],
        b: &[DoubleDouble],
        c: &mut [DoubleDouble],
    ) {
        gemm_blocked(Reference, m, k, a, c, |j, l| b[l + j * k]);
    }
}

/// Algorithm X, steps X1–X2: if a SIMD backend is compiled in and available,
/// **returns from the enclosing function** with the value of `$simd`, an
/// expression in which `$lanes` is the backend's four-lane arithmetic
/// (`crate::simd::Lanes`), evaluated inside `pulp`'s `vectorize` (which
/// enables the target features for the code inlined into it). Otherwise
/// falls through to step X3, written after the macro call.
macro_rules! dispatch {
    (|$lanes:ident| $simd:expr) => {
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        if let Some(token) = pulp::aarch64::Neon::try_new() {
            let $lanes = $crate::simd::X2::new(token);
            return pulp::Simd::vectorize(
                token,
                #[inline(always)]
                || $simd,
            );
        }
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        if let Some(token) = pulp::x86::V3::try_new() {
            let $lanes = $crate::simd::X4::new(token);
            return pulp::Simd::vectorize(
                token,
                #[inline(always)]
                || $simd,
            );
        }
    };
}
// Chapter 11.7 dispatches its loops with the same macro.
#[cfg(feature = "sparse")]
pub(crate) use dispatch;

/// `Σ xᵢ` of `f64` values as a double-double, as accurate as if summed
/// in twice the working precision (Sum2 of Ogita, Rump & Oishi, over `W =
/// 4` lanes: Algorithm C of Chapter 11.5): `|result − Σxᵢ| ≤ γ²_{n+3}
/// Σ|xᵢ|` barring overflow.
#[must_use]
pub fn sum2(x: &[f64]) -> DoubleDouble {
    dispatch!(|lanes| crate::simd::sum2(lanes, x).result());
    reference::sum2(x).result()
}

/// `xᵀy` of `f64` vectors as a double-double (Dot2, over `W = 4` lanes:
/// Algorithm C of Chapter 11.5): `|result − xᵀy| ≤ γ²_{n+5} |x|ᵀ|y| + n
/// η/2` barring overflow. Panics if the lengths differ.
#[must_use]
pub fn dot2(x: &[f64], y: &[f64]) -> DoubleDouble {
    assert_eq!(x.len(), y.len(), "dot2: the lengths differ");
    dispatch!(|lanes| crate::simd::dot2::<_, false>(lanes, x, y).result());
    reference::dot2(x, y).result()
}

/// As [`dot2`], with a rigorous bound: returns `(r, β)` with `|xᵀy −
/// r| ≤ β` (Dot2Err; step C5 of Chapter 11.5 and its proof). `β = +∞`
/// when a hypothesis fails (overflow, NaN, `2(n + 3)u ≥ 1`): invariant C3.
/// Panics if the lengths differ.
#[must_use]
pub fn dot2_err(x: &[f64], y: &[f64]) -> (DoubleDouble, f64) {
    assert_eq!(x.len(), y.len(), "dot2_err: the lengths differ");
    dispatch!(|lanes| crate::simd::dot2::<_, true>(lanes, x, y).result_with_bound(x.len()));
    reference::dot2(x, y).result_with_bound(x.len())
}

/// `xᵀy` of double-double vectors, over `W = 4` lanes combined left to
/// right (step B1 of Chapter 11.5). Panics if the lengths differ.
#[must_use]
pub fn dot(x: &[DoubleDouble], y: &[DoubleDouble]) -> DoubleDouble {
    assert_eq!(x.len(), y.len(), "dot: the lengths differ");
    dispatch!(|lanes| crate::simd::dot(lanes, x, y));
    reference::dot(x, y)
}

/// `xᵀy` with `x` double-double and `y` `f64` (a row of an `f64`
/// matrix against an extended vector: the residual of a polished
/// solution); the order of [`dot`]. Panics if the lengths differ.
#[must_use]
pub fn dot_mixed(x: &[DoubleDouble], y: &[f64]) -> DoubleDouble {
    assert_eq!(x.len(), y.len(), "dot_mixed: the lengths differ");
    dispatch!(|lanes| crate::simd::dot_mixed(lanes, x, y));
    reference::dot_mixed(x, y)
}

/// `y ← y + α x`, element by element (step B2 of Chapter 11.5). Panics if
/// the lengths differ.
pub fn axpy(alpha: DoubleDouble, x: &[DoubleDouble], y: &mut [DoubleDouble]) {
    assert_eq!(x.len(), y.len(), "axpy: the lengths differ");
    dispatch!(|lanes| crate::simd::axpy(lanes, alpha, x, y));
    reference::axpy(alpha, x, y);
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
    dispatch!(|lanes| gemm_blocked(lanes, m, k, a, c, |j, l| b[l + j * k]));
    reference::gemm_sub(m, k, a, b, c);
}

/// The name of the implementation the dispatch selected (`"scalar"`,
/// `"avx2"`, `"neon"`), for benchmarks and bug reports: Algorithm X of
/// Chapter 11.5. `"scalar"` without the feature `simd`, or when the CPU
/// has neither NEON nor AVX2 with FMA. (`"avx512"`, which the contract
/// listed, is never returned: the chapter says why.)
#[must_use]
pub fn backend() -> &'static str {
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    if pulp::aarch64::Neon::try_new().is_some() {
        return "neon";
    }
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    if pulp::x86::V3::try_new().is_some() {
        return "avx2";
    }
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
    }

    /// Algorithm X: what `backend` must say on this build and this CPU.
    #[test]
    fn backend_names_the_dispatch() {
        #[cfg(not(feature = "simd"))]
        let expected = "scalar";
        // NEON is part of the baseline of every aarch64 target of Rust.
        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        let expected = "neon";
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        let expected = if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            "avx2"
        } else {
            "scalar"
        };
        #[cfg(all(
            feature = "simd",
            not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))
        ))]
        let expected = "scalar";
        assert_eq!(backend(), expected);
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
        assert_eq!(reference::dot(&x, &y), lanes(&|i| x[i] * y[i]));
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

    // ---- Invariant (K1): the dispatched kernels against the reference ----

    /// A deterministic generator (xorshift64*), so that every length is
    /// tried with the same data on every run.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        /// A random significand and sign with an exponent in `±span`.
        fn scaled(&mut self, span: i32) -> f64 {
            #[allow(clippy::cast_precision_loss)] // 53 random bits
            let m = 1.0 + (self.next() >> 11) as f64 / (1_u64 << 53) as f64;
            let e = i32::try_from(self.next() % (2 * u64::from(span.unsigned_abs()) + 1))
                .expect("small")
                - span;
            let v = m * 2f64.powi(e);
            if self.next() & 1 == 0 { v } else { -v }
        }

        fn special(&mut self) -> f64 {
            const SPECIAL: [f64; 10] = [
                0.0,
                -0.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                f64::MAX,
                f64::MIN,
                f64::MIN_POSITIVE,
                5e-324,
                -2.5e-310,
            ];
            SPECIAL[usize::try_from(self.next() % 10).expect("small")]
        }

        /// One `f64` of the given kind of data: 0 moderate (the declared
        /// domain), 1 wide exponents (products overflow and underflow), 2
        /// any bit pattern, 3 moderate with exceptional values mixed in.
        fn f64(&mut self, kind: u8) -> f64 {
            match kind {
                0 => self.scaled(30),
                1 => self.scaled(600),
                2 => {
                    if self.next().is_multiple_of(8) {
                        self.special()
                    } else {
                        f64::from_bits(self.next())
                    }
                }
                _ => {
                    if self.next().is_multiple_of(16) {
                        self.special()
                    } else {
                        self.scaled(30)
                    }
                }
            }
        }

        /// A valid double-double (invariants D1, D2) of the given kind.
        fn dd(&mut self, kind: u8) -> Dd {
            let h = self.f64(kind);
            let l = self.f64(0) * h * 2f64.powi(-84);
            Dd::from_sum(h, l)
        }

        fn f64s(&mut self, kind: u8, n: usize) -> Vec<f64> {
            (0..n).map(|_| self.f64(kind)).collect()
        }

        fn dds(&mut self, kind: u8, n: usize) -> Vec<Dd> {
            (0..n).map(|_| self.dd(kind)).collect()
        }
    }

    /// The same bits, or a NaN on both sides (invariant K1).
    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    fn same_dd(a: Dd, b: Dd) -> bool {
        same(a.hi(), b.hi()) && same(a.lo(), b.lo())
    }

    fn same_dds(a: &[Dd], b: &[Dd]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(&p, &q)| same_dd(p, q))
    }

    /// (K1) for the six vector kernels on one data set; the name of the
    /// first kernel that differs.
    fn k1_vectors(kind: u8, n: usize, seed: u64) -> Result<(), String> {
        let mut rng = Rng(seed | 1);
        let (x, y) = (rng.f64s(kind, n), rng.f64s(kind, n));
        let (p, q) = (rng.dds(kind, n), rng.dds(kind, n));
        let alpha = rng.dd(kind);
        let fail = |what: &str| Err(format!("{what}: kind {kind}, n = {n}, seed {seed}"));

        if !same_dd(sum2(&x), reference::sum2(&x).result()) {
            return fail("sum2");
        }
        if !same_dd(dot2(&x, &y), reference::dot2(&x, &y).result()) {
            return fail("dot2");
        }
        let (r, beta) = dot2_err(&x, &y);
        let (r0, beta0) = reference::dot2(&x, &y).result_with_bound(n);
        if !same_dd(r, r0) || !same(beta, beta0) {
            return fail("dot2_err");
        }
        if !same_dd(dot(&p, &q), reference::dot(&p, &q)) {
            return fail("dot");
        }
        if !same_dd(dot_mixed(&p, &x), reference::dot_mixed(&p, &x)) {
            return fail("dot_mixed");
        }
        let (mut z, mut z0) = (q.clone(), q.clone());
        axpy(alpha, &p, &mut z);
        reference::axpy(alpha, &p, &mut z0);
        if !same_dds(&z, &z0) {
            return fail("axpy");
        }
        for v in &z {
            v.check_invariants()
                .map_err(|e| format!("axpy: {e} (kind {kind}, n = {n}, seed {seed})"))?;
        }
        Ok(())
    }

    /// (K1) for `gemm_sub`, against the blocked reference and against the
    /// plain triple loop of step B3.
    fn k1_gemm(kind: u8, m: usize, k: usize, n: usize, seed: u64) -> Result<(), String> {
        let mut rng = Rng(seed | 1);
        let a = rng.dds(kind, m * k);
        let b = rng.dds(kind, k * n);
        let c = rng.dds(kind, m * n);
        let (mut c1, mut c2, mut c3) = (c.clone(), c.clone(), c);
        gemm_sub(m, k, n, &a, &b, &mut c1);
        reference::gemm_sub(m, k, &a, &b, &mut c2);
        for j in 0..n {
            for l in 0..k {
                for i in 0..m {
                    c3[i + j * m] -= a[i + l * m] * b[l + j * k];
                }
            }
        }
        if same_dds(&c1, &c2) && same_dds(&c2, &c3) {
            Ok(())
        } else {
            Err(format!(
                "gemm_sub: kind {kind}, {m} × {k} × {n}, seed {seed}"
            ))
        }
    }

    /// Every length `0..=67` (all residues modulo `W`, with and without
    /// chunks) and some large ones, on every kind of data.
    #[test]
    fn k1_every_length_and_kind() {
        for kind in 0..4_u8 {
            for n in (0..=67).chain([255, 256, 1000, 4099, 65_539]) {
                let seeds = if n <= 67 { 6 } else { 2 };
                for s in 0..seeds {
                    let seed = 0x9e37_79b9_7f4a_7c15_u64
                        .wrapping_mul(u64::from(kind) + 1)
                        .wrapping_add((n as u64) << 8)
                        .wrapping_add(s);
                    k1_vectors(kind, n, seed).unwrap();
                }
            }
        }
    }

    /// `gemm_sub` on shapes that are empty, smaller than a chunk, and
    /// larger than a cache block in both directions.
    #[test]
    fn k1_gemm_shapes() {
        let shapes = [
            (0, 3, 2),
            (3, 0, 2),
            (3, 2, 0),
            (1, 1, 1),
            (3, 2, 5),
            (4, 4, 4),
            (7, 5, 3),
            (67, 9, 4),
            (BLOCK_ROWS + 5, 3, 2),
            (9, BLOCK_K + 7, 2),
            (2 * BLOCK_ROWS + 3, BLOCK_K + 1, 3),
        ];
        for kind in 0..4_u8 {
            for (s, &(m, k, n)) in shapes.iter().enumerate() {
                k1_gemm(kind, m, k, n, 77 + 1000 * u64::from(kind) + s as u64).unwrap();
            }
        }
    }

    /// The slow paths of Chapter 11.6 are taken and are right: a single
    /// non-finite element, or one overflowing product, in each lane.
    #[test]
    fn k1_exceptional_values_in_each_lane() {
        let bad = [
            Dd::INFINITY,
            -Dd::INFINITY,
            Dd::NAN,
            Dd::from_f64(f64::MAX),
            Dd::from_f64(1e200),
        ];
        for (b, &v) in bad.iter().enumerate() {
            for at in 0..12 {
                let mut rng = Rng(1 + at as u64 + 100 * b as u64);
                let mut p = rng.dds(0, 13);
                let q = rng.dds(0, 13);
                p[at] = v;
                let big = Dd::from_f64(1e200);
                for alpha in [Dd::ONE, big, Dd::ZERO] {
                    let (mut z, mut z0) = (q.clone(), q.clone());
                    axpy(alpha, &p, &mut z);
                    reference::axpy(alpha, &p, &mut z0);
                    assert!(same_dds(&z, &z0), "axpy, value {b} at {at}");
                    let (mut c, mut c0) = (q.clone(), q.clone());
                    gemm_sub(13, 1, 1, &p, &[alpha], &mut c);
                    reference::gemm_sub(13, 1, &p, &[alpha], &mut c0);
                    assert!(same_dds(&c, &c0), "gemm_sub, value {b} at {at}");
                }
                assert!(same_dd(dot(&p, &q), reference::dot(&p, &q)));
                assert!(same_dd(dot(&p, &p), reference::dot(&p, &p)));
                let y: Vec<f64> = q.iter().map(|d| d.hi() * 1e150).collect();
                assert!(same_dd(dot_mixed(&p, &y), reference::dot_mixed(&p, &y)));
                let x: Vec<f64> = p.iter().map(|d| d.hi()).collect();
                assert!(same_dd(sum2(&x), reference::sum2(&x).result()));
                assert!(same_dd(dot2(&x, &y), reference::dot2(&x, &y).result()));
                let (r, beta) = dot2_err(&x, &y);
                let (r0, beta0) = reference::dot2(&x, &y).result_with_bound(13);
                assert!(same_dd(r, r0) && same(beta, beta0));
            }
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(2000))]
        /// (K1), scalar × SIMD, bit for bit, on random lengths, kinds of
        /// data and seeds.
        #[test]
        fn k1_simd_equals_scalar(kind in 0_u8..4, n in 0_usize..=67, seed in proptest::prelude::any::<u64>()) {
            proptest::prop_assert_eq!(k1_vectors(kind, n, seed), Ok(()));
        }

        #[test]
        fn k1_gemm_simd_equals_scalar(
            kind in 0_u8..4,
            m in 0_usize..40,
            k in 0_usize..7,
            n in 0_usize..5,
            seed in proptest::prelude::any::<u64>(),
        ) {
            proptest::prop_assert_eq!(k1_gemm(kind, m, k, n, seed), Ok(()));
        }
    }
}
