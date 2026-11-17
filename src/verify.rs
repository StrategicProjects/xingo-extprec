//! # Chapter 11.10 — Verified residuals and polishing
//!
//! ## Purpose
//! Two things a solver wants to know about an approximate solution `x` of
//! `A x = b`, for the `f64` data as stored (C1 of Chapter 11):
//!
//! 1. *how large the residual really is* — Algorithm V returns, row by
//!    row, an interval that **contains** `a_iᵀx − b_i`, a theorem and not
//!    an estimate (invariant V1);
//! 2. *a better `x`* — Algorithm P refines `x` in double-double around a
//!    solve the caller already has (its `f64` LU), after which Algorithm V
//!    certifies the residual that was reached.
//!
//! The module verifies **residuals**. It does not bound `‖x − A⁻¹b‖`: that
//! needs a rigorous bound of `‖A⁻¹‖` (Rump's `verifylss`), which Chapter
//! 11 leaves out of v0.1.
//!
//! ## Notation
//! `A ∈ F^{m×n}` in compressed rows (`xingo_sparse::CsrMatrix<f64>`), `a_i`
//! its row `i` with `k_i` stored entries, `x ∈ F^n` or `x ∈ DD^n` (each
//! component the exact sum `h_j + ℓ_j`), `b ∈ F^m`. `u = 2⁻⁵³`. `Dot2Err`
//! is `kernels::dot2_err` (Chapter 11.5, Algorithm C): for `f64` vectors
//! `p`, `q` it returns a double-double `r` and an `f64` `β` with `|pᵀq − r|
//! ≤ β` for **every** input, `β = +∞` when a hypothesis of the bound fails
//! (invariant K2). `▽`, `△` are the directed roundings of Chapter 11.9.
//! `‖·‖∞` is the largest absolute value of a vector and the largest
//! absolute row sum of a matrix.
//!
//! ## Algorithm V (verified residual)
//! Input: `A`, `x`, `b`. Output: intervals `R_i ∋ a_iᵀx − b_i`.
//!
//! V1. \[Row.\] For each row `i`, gather two `f64` vectors `p`, `q` whose
//!     dot product is the residual, *exactly, as real numbers*:
//!     * `x ∈ F^n`: the `k_i` pairs `(a_ij, x_j)` and the pair `(−1, b_i)`;
//!     * `x ∈ DD^n`: the pairs `(a_ij, h_j)`, then the pairs `(a_ij,
//!       ℓ_j)`, then `(−1, b_i)` — `2 k_i + 1` in all;
//!     * activities: the `k_i` pairs `(a_ij, x_j)` alone.
//! V2. \[Accumulate.\] `(r, β) ← Dot2Err(p, q)`.
//! V3. \[Enclose.\] If `β` or `r` is not finite, `R_i ← [−∞, +∞]`.
//!     Otherwise, with `r = h + ℓ`, `R_i ← [▽(h + ▽(ℓ − β)), △(h + △(ℓ +
//!     β))]`: the radius is applied to the *trailing* component, where it
//!     is of the right size, before the leading one is added.
//!
//! **Proof of (V1).** `pᵀq` is the residual exactly: for a double-double
//! `x`, `Σ_j a_ij (h_j + ℓ_j) − b_i = Σ_j a_ij h_j + Σ_j a_ij ℓ_j + (−1)
//! b_i`, an identity of real numbers, and the value of `x_j` *is* `h_j +
//! ℓ_j`. So the double-double case needs no bound of its own — it is
//! `Dot2Err` on a vector twice as long (this replaces the "second-order
//! terms" that the contract's sketch of step V2 announced). By (K2),
//! `|pᵀq − r| ≤ β`, i.e. `h + (ℓ − β) ≤ pᵀq ≤ h + (ℓ + β)`, and each of
//! the two directed sums of V3 on either side moves outward. When `β =
//! +∞` nothing is claimed (C3). ∎ (The contract's sketch had `enclose(r) +
//! [−β, β]`, which is also correct and one to two ulps wider: it rounds
//! `r` to its neighbours before the radius is added.)
//!
//! **Width.** For a finite result, `width(R_i) ≤ 2 ulp(r) + 2β (1 + 2u)`
//! — typically one ulp, two when `r` is an `f64` — and `β` is of order
//! `k² u² Σ_j |a_ij x_j|` (Chapter 11.5, step C5): the enclosure of a
//! residual of size `ρ` computed from terms of size `S` has
//! a relative width of about `u + k² u² S/ρ`. An `f64` solution leaves `ρ
//! ≈ κ u S` at best, so its residual is enclosed to about 16 digits; a
//! polished one has `ρ ≈ k u² S`, of the order of the bound itself, and
//! the certificate then reads "`|a_iᵀx − b_i| ≤ c k² u² S`" — which is the
//! most that double-double accumulation can prove.
//!
//! **Cost.** About 25 flops per stored entry for `x ∈ F^n` and 50 for `x ∈
//! DD^n`, and two gather buffers of the length of the longest row.
//! [`residual_csc`] first converts to compressed rows (`O(nnz)` time and
//! memory).
//!
//! ## Algorithm P (polishing: iterative refinement with extended residuals)
//! Input: a square `A`, `b`, an approximate solution `x ∈ DD^n` (an `f64`
//! solution has zero trailing components), a *solve* `s ↦ d ≈ A⁻¹ s` in
//! `f64` (a closure: the caller's LU), the parameters `τ`, `ρ` and the
//! iteration limit of [`PolishOptions`]. Output: `x`, polished in place,
//! and a [`PolishReport`].
//!
//! P1. \[Residual.\] `r ← b − A x` in double-double ([`residual_dd`]); `ν ←
//!     ‖r‖∞`. On the first pass, `ν₀ ← ν`.
//! P2. \[Stop?\] In this order:
//!     (a) if `ν` is not finite: restore the previous `x` (if any) and
//!         stop, *not finite*;
//!     (b) if `ν > ν_prev` (after a correction): the correction made
//!         things worse — restore the previous `x` and stop, *stagnated*;
//!     (c) if `|r_i| ≤ τ S_i` in **every** row `i`, with `S_i = Σ_j |a_ij
//!         x_j| + |b_i|` the size of the terms of that row (the
//!         componentwise test; "How step P2 (c) rounds" below): stop,
//!         *converged*;
//!     (d) if `ν > ρ ν_prev` (after a correction): stop, *stagnated* (the
//!         new `x` is kept: it is not worse);
//!     (e) if the iteration limit is reached: stop, *iteration limit*.
//! P3. \[Correct.\] `s ← RN(r)` (the leading components); `d ← solve(s)`.
//!     If some `d_i` is not finite, stop, *not finite* (`x` is the current
//!     one). Otherwise save `x`, set `x ← x + d` in double-double, `ν_prev
//!     ← ν`, and return to P1.
//!
//! **How step P2 (c) rounds.** The test is per row because a global scale
//! `‖A‖∞‖x‖∞ + ‖b‖∞` — the contract's — is set by the largest row of a
//! badly scaled model and declares the small rows converged while they
//! still carry a residual far above double-double level (measured on the
//! Netlib bases in task L2.3). The right-hand side must not be
//! *over*estimated, or convergence would be declared unduly. It is
//! computed in `f64` as `t_i = RN(RN(τ Ŝ_i) · d_i)`, with `Ŝ_i` the plain
//! left-to-right sum of `|b_i|` and the `k` products `RN(|a_ij| |h_j|)`
//! (`h_j` the leading component of `x_j`) and the deflation `d_i = RN(1 −
//! (2k + 10) u)`. Barring underflow, `Ŝ_i ≤ S_i (1 + u)^{k+2}` for the
//! exact `S_i = Σ_j |a_ij x_j| + |b_i|` (one rounding per product, `|h_j|
//! ≤ (1 + u)|x_j|`, `k` additions of nonnegative terms), two more
//! roundings give `t_i ≤ τ S_i (1 + u)^{k+4} d_i`, and the test compares
//! the leading component `|RN(r_i)| ≥ |r_i|/(1 + u)`. Since `(1 +
//! u)^{k+6} ≤ 1/(1 − (k + 6)u) ≤ 1/(1 − (2k + 10)u)`, **`|RN(r_i)| ≤ t_i`
//! implies `|r_i| ≤ τ S_i`** for the double-double `r_i` of step P1 — the
//! rounding of the test is always against convergence. (A product that
//! underflows adds at most `η/2` to `Ŝ_i`: rows whose terms are all below
//! `2⁻¹⁰²²` are outside this statement. A row of `2⁵¹` entries or more
//! never passes, and neither does a row whose `t_i` overflows: an infinite
//! bound proves nothing about `τ S_i`.) The `r_i` themselves are computed, not enclosed: P
//! decides when to *stop*, Algorithm V what is *proved*.
//!
//! A row with `Ŝ_i = 0` — an empty row with `b_i = 0`, or terms that all
//! underflow — needs no rule of its own: the test reads `|RN(r_i)| ≤ 0`,
//! which holds exactly when the residual of that row is zero (and for
//! finite data it is). No quotient is formed.
//!
//! **Convergence by rows, progress by the norm: a measurement.** Steps
//! (b) and (d) compare the *norm* `ν = ‖r‖∞` of successive passes, not a
//! row-scaled quantity. The other choice — progress measured as `ω =
//! max_i |r_i| / S_i`, the scale of test (c) — **was implemented, measured
//! on the optimal bases of the 93 Netlib LPs and rejected** (task L2.3,
//! 2026-11-16; same `τ`, `ρ` and iteration limit throughout):
//!
//! | convergence / progress | converged | stagnated | largest certified `‖r‖∞` after P |
//! |---|---:|---:|---:|
//! | global / global (the contract) | 93 | 0 | `2.879e-20` |
//! | by rows / global (**this version**) | 84 | 9 | `6.778e-22` |
//! | by rows / `ω` | 83 | 10 | `2.879e-20` |
//!
//! With progress in `ω`, seven bases ended with a certified residual
//! larger than with progress in the norm (greenbea `6.778e-22 → 2.879e-20`,
//! maros `6.975e-24 → 1.088e-22`, dfl001 `7.740e-28 → 1.445e-27`, and
//! cycle, stocfor2, scsd1, wood1p by 2 % or less), one with a smaller one
//! (scsd8, by 1 %), and none ended above what the contract's criterion
//! gives.
//!
//! *A limitation of the row scale, measured.* The cause was read off an
//! instrumented run (eleven bases): when the *exact* solution has `S_i =
//! 0` in some row — a nonbasic variable at a zero bound, in its row `x_v =
//! 0`; a constraint all of whose terms vanish — every iterate that is not
//! exactly right there has `|r_i| = S_i` up to rounding, however small
//! `r_i` is. So `ω` was exactly `1` before the first correction on all
//! eleven, and still between `0.96` and `1` after it on eight of the ten
//! that stagnated (on rows with `|r_i| = S_i` between `1e-34` and
//! `1e-29`): it did not halve, and P stopped after one pass where the
//! norm criterion goes on for one or two more. The same rows are,
//! presumably (this version was not instrumented), why test (c) leaves
//! nine bases *stagnated* here: they pass only once their residual is
//! exactly zero. That outcome is harmless — the point
//! reached is kept and Algorithm V certifies it — and all nine have a
//! certified residual no larger than under the contract's criterion,
//! except scsd8 (`1.899e-29 → 1.919e-29`).
//!
//! *An open lever, not measured.* The literature on componentwise
//! backward error replaces the denominator of such rows by a norm-based
//! one — here it would be `max(S_i, θ ‖a_i‖₁ ‖x‖∞)` for a threshold `θ`
//! (Arioli, Demmel & Duff, 1989: **cited from memory**, not reread). It
//! was **not implemented and not measured**; it needs a threshold, to be
//! chosen by a campaign like the one above.
//!
//! **What P guarantees.** (P-mono) `ν` at exit is at most `ν₀`: steps (a)
//! and (b) undo the only corrections that could increase it. Nothing
//! else: P is numerical, not rigorous — its residuals are double-double
//! values without a bound, its norms plain `f64` — and the certificate of
//! its result is Algorithm V. Higham (2002, ch. 12): with residuals in
//! twice the working precision and a solve of relative accuracy `κ(A) u <
//! 1`, each pass contracts the error by about `κ(A) u`, until the residual
//! reaches the accuracy of its own evaluation, `≈ k u² (|A||x| + |b|)`;
//! because `x` is *stored* in double-double here, the forward error keeps
//! falling below `u` (to about `κ(A) k u²`), which a refinement that
//! stores `x` in `f64` cannot do.
//!
//! ## Invariants
//! (V1) *Residuals.* [`residual`]`(A, x, b)[i] ∋ a_iᵀx − b_i`, and likewise
//!      [`residual_of_dd`], [`residual_csc`] and [`activity`] (`∋ a_iᵀx`),
//!      for every row and every input; an unbounded interval when a
//!      hypothesis fails (C3). Checked against exact rationals.
//! (V2) *Well-formed.* Every interval returned satisfies (I2) of Chapter
//!      11.9 and is not empty (a `debug_assert!`).
//! (P1) *Monotone.* [`PolishReport::residual_after`] `≤`
//!      [`PolishReport::residual_before`] whenever the latter is finite (a
//!      `debug_assert!`).
//! (P2) *Untouched on failure.* With [`PolishStop::NotFinite`] and zero
//!      iterations, `x` is the input.
//!
//! ## References
//! Ogita, Rump & Oishi, "Accurate sum and dot product", SIAM J. Sci.
//! Comput. 26 (2005), Algorithm 5.8 and Corollary 5.9 (through Chapter
//! 11.5, which adapts and proves them); Higham, *Accuracy and Stability of
//! Numerical Algorithms*, 2nd ed. (2002), ch. 12 (iterative refinement;
//! §12.1 for the mixed-precision form) — **cited from memory**, the
//! statement above is the qualitative one and nothing in the code depends
//! on its constants; Rump, "Verification methods: rigorous results using
//! floating-point arithmetic", Acta Numerica 19 (2010) — the distinction
//! between a verified residual and a verified solution (*not reread*; no
//! bound of this chapter is taken from it).

use xingo_sparse::{CscMatrix, CsrMatrix, Row};

use crate::dd::DoubleDouble;
use crate::interval::Interval;
use crate::kernels;

/// What went wrong before any arithmetic. (`Display` and
/// `core::error::Error` are implemented.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// `x` or `b` has the wrong length for `A` (for [`polish`], also: `A`
    /// is not square — `expected` is its number of rows, `got` its number
    /// of columns).
    Dimension {
        /// What was expected.
        expected: usize,
        /// What was given.
        got: usize,
    },
}

impl core::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Dimension { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl core::error::Error for VerifyError {}

/// `Ok` iff `got == expected`.
fn dimension(expected: usize, got: usize) -> Result<(), VerifyError> {
    if got == expected {
        Ok(())
    } else {
        Err(VerifyError::Dimension { expected, got })
    }
}

/// Step V3: the interval of `r ± β`.
fn enclosure(r: DoubleDouble, beta: f64) -> Interval {
    // False for a NaN `β` too. A non-finite `r` comes with `β = +∞` (K2);
    // it is tested all the same.
    let known = beta.is_finite() && beta >= 0.0 && r.hi().is_finite() && r.lo().is_finite();
    let x = if known {
        Interval::mid_rad(r.lo(), beta) + r.hi()
    } else {
        Interval::ENTIRE
    };
    debug_assert!(
        x.check_invariants().is_ok() && !x.is_empty(),
        "V2: {x:?} from r = {r:?}, β = {beta:e}"
    );
    x
}

/// Algorithm V over the rows of `a`: `fill(i, cols, vals, p, q)` appends
/// to the empty `(p, q)` the pairs of row `i` (step V1).
fn rows(
    a: &CsrMatrix<f64>,
    mut fill: impl FnMut(usize, &[u32], &[f64], &mut Vec<f64>, &mut Vec<f64>),
) -> Vec<Interval> {
    let (mut p, mut q) = (Vec::new(), Vec::new());
    (0..a.n_rows())
        .map(|i| {
            p.clear();
            q.clear();
            let (cols, vals) = a.row(Row::from_usize(i));
            fill(i, cols, vals, &mut p, &mut q); // V1
            let (r, beta) = kernels::dot2_err(&p, &q); // V2
            enclosure(r, beta) // V3
        })
        .collect()
}

/// Algorithm V for an `f64` solution: `R_i ∋ a_iᵀx − b_i` (invariant V1 of
/// Chapter 11.10), for the stored `f64` values. A row whose bound cannot
/// be established (overflow, a NaN or an infinity in the data) gets
/// [`Interval::ENTIRE`].
///
/// # Errors
/// [`VerifyError::Dimension`].
///
/// # Panics
/// If `a` violates the invariants of its type (a column index out of
/// range).
pub fn residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> Result<Vec<Interval>, VerifyError> {
    dimension(a.n_cols(), x.len())?;
    dimension(a.n_rows(), b.len())?;
    Ok(rows(a, |i, cols, vals, p, q| {
        p.extend_from_slice(vals);
        q.extend(cols.iter().map(|&j| x[j as usize]));
        p.push(-1.0);
        q.push(b[i]);
    }))
}

/// Algorithm V for a double-double (polished) solution: `R_i ∋ a_iᵀx −
/// b_i` for the exact values `hi + lo` of the components of `x`.
///
/// # Errors
/// [`VerifyError::Dimension`].
///
/// # Panics
/// If `a` violates the invariants of its type.
pub fn residual_of_dd(
    a: &CsrMatrix<f64>,
    x: &[DoubleDouble],
    b: &[f64],
) -> Result<Vec<Interval>, VerifyError> {
    dimension(a.n_cols(), x.len())?;
    dimension(a.n_rows(), b.len())?;
    // V1: the leading terms first, then the trailing ones.
    Ok(rows(a, |i, cols, vals, p, q| {
        p.extend_from_slice(vals);
        q.extend(cols.iter().map(|&j| x[j as usize].hi()));
        p.extend_from_slice(vals);
        q.extend(cols.iter().map(|&j| x[j as usize].lo()));
        p.push(-1.0);
        q.push(b[i]);
    }))
}

/// As [`residual`] for a column-compressed matrix (one scatter pass;
/// `O(nnz)` extra memory for the row gather).
///
/// # Errors
/// [`VerifyError::Dimension`].
///
/// # Panics
/// If `a` violates the invariants of its type.
pub fn residual_csc(
    a: &CscMatrix<f64>,
    x: &[f64],
    b: &[f64],
) -> Result<Vec<Interval>, VerifyError> {
    dimension(a.n_cols(), x.len())?;
    dimension(a.n_rows(), b.len())?;
    residual(&a.to_csr(), x, b)
}

/// `b − A x` in double-double, not rigorous (step P1): each component
/// within `≈ k u²` relative to `Σ|a_ij x_j| + |b_i|`, `k` the number of
/// entries of the row (`kernels::dot_mixed`, then one subtraction). Note
/// the sign: the *interval* residuals of this module enclose `A x − b`.
///
/// # Errors
/// [`VerifyError::Dimension`].
///
/// # Panics
/// If `a` violates the invariants of its type.
pub fn residual_dd(
    a: &CsrMatrix<f64>,
    x: &[DoubleDouble],
    b: &[f64],
) -> Result<Vec<DoubleDouble>, VerifyError> {
    dimension(a.n_cols(), x.len())?;
    dimension(a.n_rows(), b.len())?;
    let mut gathered = Vec::new();
    Ok(b.iter()
        .enumerate()
        .map(|(i, &bi)| {
            let (cols, vals) = a.row(Row::from_usize(i));
            gathered.clear();
            gathered.extend(cols.iter().map(|&j| x[j as usize]));
            DoubleDouble::from_f64(bi) - kernels::dot_mixed(&gathered, vals)
        })
        .collect())
}

/// A rigorous upper bound of `‖R‖∞ = max_i mag(R_i)` (`+∞` if some `R_i`
/// is unbounded; `0` for no rows). An **empty** `R_i` also gives `+∞`: no
/// function of this module returns one, and an empty enclosure proves
/// nothing (Chapter 11.9).
#[must_use]
pub fn norm_inf_bound(r: &[Interval]) -> f64 {
    r.iter().fold(0.0, |norm: f64, x| {
        if x.is_empty() {
            f64::INFINITY
        } else {
            norm.max(x.mag())
        }
    })
}

/// Row activities with bounds, for feasibility certificates: `S_i ∋
/// a_iᵀx`, to be tested with `certainly_le(u_i + t)` and
/// `certainly_ge(l_i − t)`.
///
/// # Errors
/// [`VerifyError::Dimension`].
///
/// # Panics
/// If `a` violates the invariants of its type.
pub fn activity(a: &CsrMatrix<f64>, x: &[f64]) -> Result<Vec<Interval>, VerifyError> {
    dimension(a.n_cols(), x.len())?;
    Ok(rows(a, |_, cols, vals, p, q| {
        p.extend_from_slice(vals);
        q.extend(cols.iter().map(|&j| x[j as usize]));
    }))
}

/// The parameters of Algorithm P. No literal lives in the kernel: the
/// defaults are here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolishOptions {
    /// `τ` of step P2: convergence is `|r_i| ≤ τ (Σ_j |a_ij x_j| + |b_i|)`
    /// in every row `i` — relative to the terms of *that* row, not to a
    /// global norm (default `2⁻⁹⁰`).
    pub tol: f64,
    /// `ρ` of step P2: stop unless the norm `‖r‖∞` fell below `ρ` times
    /// its previous value (default `0.5`). Progress is measured in the
    /// norm although convergence is tested row by row: Chapter 11.10 has
    /// the measurement behind that choice.
    pub min_progress: f64,
    /// Iteration limit (default 10).
    pub max_iter: u32,
}

impl Default for PolishOptions {
    fn default() -> Self {
        Self {
            // 2⁻⁹⁰ = 2¹⁶ u²: sixteen binary digits above the accuracy `≈ 3k
            // u²` of the double-double residual of a row of `k` entries,
            // so reachable for rows of up to thousands of entries.
            tol: f64::from_bits((1023 - 90) << 52),
            min_progress: 0.5,
            max_iter: 10,
        }
    }
}

/// Why Algorithm P stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolishStop {
    /// The test of step P2 (c) holds in every row.
    Converged,
    /// The norm of the residual stopped shrinking, or a correction
    /// increased it and was undone, before every row passed the test of
    /// convergence: `κ(A) u ≳ 1`, the solve is not a solve of `A` — or,
    /// **commonly and harmlessly**, the residual is already at the
    /// accuracy of double-double while some row whose exact terms are all
    /// zero cannot pass a relative test (9 of the 93 optimal bases of
    /// Netlib; Chapter 11.10). The point reached is kept.
    Stagnated,
    /// [`PolishOptions::max_iter`] corrections were made.
    IterationLimit,
    /// A residual or a correction was not finite.
    NotFinite,
}

/// The outcome of Algorithm P.
#[derive(Clone, Debug, PartialEq)]
pub struct PolishReport {
    /// Why it stopped.
    pub stop: PolishStop,
    /// Corrections applied (and kept: one undone by step P2 does not
    /// count).
    pub iterations: u32,
    /// `‖b − A x⁰‖∞` (double-double residual, as an `f64`).
    pub residual_before: f64,
    /// `‖b − A x‖∞` at the end.
    pub residual_after: f64,
}

/// `‖v‖∞` of the leading components; NaN if a component is NaN.
fn norm_inf(v: &[DoubleDouble]) -> f64 {
    v.iter().fold(0.0, |norm: f64, c| {
        let h = c.hi().abs();
        // `f64::max` would drop a NaN.
        if h.is_nan() || norm.is_nan() {
            f64::NAN
        } else {
            norm.max(h)
        }
    })
}

/// `2⁻⁵³`.
const U: f64 = 0.5 * f64::EPSILON;

/// Step P2 (c): `|RN(r_i)| ≤ t_i` in every row, with `t_i` rounded against
/// convergence (the chapter, "How step P2 (c) rounds").
fn rows_converged(
    a: &CsrMatrix<f64>,
    x: &[DoubleDouble],
    b: &[f64],
    r: &[DoubleDouble],
    tol: f64,
) -> bool {
    r.iter().zip(b).enumerate().all(|(i, (ri, bi))| {
        let (cols, vals) = a.row(Row::from_usize(i));
        let mut s = bi.abs();
        for (&j, v) in cols.iter().zip(vals) {
            s += v.abs() * x[j as usize].hi().abs();
        }
        #[allow(clippy::cast_precision_loss)] // a deflation: any rounding is covered
        let deflate = 1.0 - (2 * cols.len() + 10) as f64 * U;
        let t = tol * s * deflate;
        // False for a NaN on either side, and for a bound that overflowed:
        // an infinite `t` says nothing about `τ S_i`.
        t.is_finite() && ri.hi().abs() <= t
    })
}

/// Algorithm P. `x` enters as the approximate solution (trailing
/// components zero for an `f64` solution) and leaves polished; `solve`
/// overwrites its argument `s` with `≈ A⁻¹ s` in `f64`.
///
/// The residual norm never increases (invariant P1 of Chapter 11.10): a
/// correction that makes it worse is undone. The result is *not* a
/// certificate; [`residual_of_dd`] certifies it.
///
/// # Errors
/// [`VerifyError::Dimension`], also when `a` is not square.
///
/// # Panics
/// If `a` violates the invariants of its type.
pub fn polish(
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [DoubleDouble],
    solve: &mut dyn FnMut(&mut [f64]),
    opts: &PolishOptions,
) -> Result<PolishReport, VerifyError> {
    dimension(a.n_cols(), x.len())?;
    dimension(a.n_rows(), b.len())?;
    dimension(a.n_rows(), a.n_cols())?;
    let mut previous: Vec<DoubleDouble> = x.to_vec();
    let mut s = vec![0.0; b.len()];
    let mut iterations = 0_u32;
    let mut nu_prev = f64::INFINITY;
    let mut nu_first = f64::NAN;
    let (stop, nu) = loop {
        // P1.
        let r = residual_dd(a, x, b)?;
        let nu = norm_inf(&r);
        if iterations == 0 {
            nu_first = nu;
        }
        // P2.
        if !nu.is_finite() || nu > nu_prev {
            let stop = if nu.is_finite() {
                PolishStop::Stagnated // (b)
            } else {
                PolishStop::NotFinite // (a)
            };
            if iterations > 0 {
                x.copy_from_slice(&previous);
                iterations -= 1;
                break (stop, nu_prev);
            }
            break (stop, nu);
        }
        if rows_converged(a, x, b, &r, opts.tol) {
            break (PolishStop::Converged, nu); // (c)
        }
        if iterations > 0 && nu > opts.min_progress * nu_prev {
            break (PolishStop::Stagnated, nu); // (d)
        }
        if iterations >= opts.max_iter {
            break (PolishStop::IterationLimit, nu); // (e)
        }
        // P3.
        for (si, ri) in s.iter_mut().zip(&r) {
            *si = ri.hi();
        }
        solve(&mut s);
        if !s.iter().all(|d| d.is_finite()) {
            break (PolishStop::NotFinite, nu);
        }
        previous.copy_from_slice(x);
        for (xi, &di) in x.iter_mut().zip(&s) {
            // The `dd + f64` addition (Chapter 11.2), not a promotion.
            #[allow(clippy::assign_op_pattern)]
            {
                *xi = *xi + di;
            }
        }
        nu_prev = nu;
        iterations += 1;
    };
    debug_assert!(
        !nu_first.is_finite() || nu <= nu_first,
        "P1: the residual grew from {nu_first:e} to {nu:e}"
    );
    Ok(PolishReport {
        stop,
        iterations,
        residual_before: nu_first,
        residual_after: nu,
    })
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the tests compare exact values
mod tests {
    use super::*;

    type Dd = DoubleDouble;

    /// `[[2, 1], [1, 3]]`.
    fn small() -> CsrMatrix<f64> {
        CsrMatrix::from_raw_parts(2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![2.0, 1.0, 1.0, 3.0])
            .expect("a valid matrix")
    }

    #[test]
    fn exact_data_give_point_like_residuals() {
        let a = small();
        // x = (1, 2): A x = (4, 7).
        let r = residual(&a, &[1.0, 2.0], &[4.0, 6.0]).unwrap();
        // Only the underflow term of Dot2Err widens them.
        assert!(r[0].contains(0.0) && r[0].width() < 1e-300);
        assert!(r[1].contains(1.0) && r[1].width() < 1e-15);
        assert!(r[1].certainly_within(1.0 + 1e-15) && !r[1].certainly_within(0.5));
        let s = activity(&a, &[1.0, 2.0]).unwrap();
        assert!(s[0].contains(4.0) && s[1].contains(7.0));
        assert!(s[1].certainly_le(7.0 + 1e-12) && s[1].certainly_ge(7.0 - 1e-12));
        let n = norm_inf_bound(&r);
        assert!((1.0..1.0 + 1e-15).contains(&n));
        assert_eq!(
            residual_csc(&a.to_csc(), &[1.0, 2.0], &[4.0, 6.0]).unwrap(),
            r
        );
        let x = [Dd::from_f64(1.0), Dd::from_f64(2.0)];
        assert_eq!(residual_of_dd(&a, &x, &[4.0, 6.0]).unwrap(), r);
        assert_eq!(
            residual_dd(&a, &x, &[4.0, 6.0]).unwrap(),
            vec![Dd::ZERO, Dd::from_f64(-1.0)]
        );
    }

    #[test]
    fn a_residual_below_the_f64_grid_is_seen() {
        // (1 + 2^-52)·(1 + 2^-52) − (1 + 2^-51) = 2^-104: zero in f64.
        let x = 1.0 + f64::EPSILON;
        let a = CsrMatrix::from_raw_parts(1, vec![0, 1], vec![0], vec![x]).unwrap();
        let r = residual(&a, &[x], &[1.0 + 2.0 * f64::EPSILON]).unwrap();
        let exact = f64::EPSILON * f64::EPSILON;
        assert!(r[0].contains(exact) && r[0].certainly_ge(exact * 0.5));
        // A double-double x: the trailing component is part of the value.
        let xd = [Dd::from_sum(1.0, exact)];
        let one = CsrMatrix::from_raw_parts(1, vec![0, 1], vec![0], vec![1.0]).unwrap();
        let r = residual_of_dd(&one, &xd, &[1.0]).unwrap();
        assert!(r[0].contains(exact) && r[0].certainly_ge(exact * 0.5));
    }

    #[test]
    fn failed_hypotheses_claim_nothing() {
        let a = small();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let r = residual(&a, &[bad, 1.0], &[0.0, 0.0]).unwrap();
            assert_eq!(r, vec![Interval::ENTIRE; 2], "{bad}");
            assert_eq!(norm_inf_bound(&r), f64::INFINITY);
            let s = activity(&a, &[1.0, bad]).unwrap();
            assert_eq!(s, vec![Interval::ENTIRE; 2], "{bad}");
        }
        // Overflow in one row only: 2 · 10³⁰⁸ in the first, not the second.
        let r = residual(&a, &[1e308, 1.0], &[0.0, 0.0]).unwrap();
        assert_eq!(r[0], Interval::ENTIRE);
        assert!(r[1].contains(1e308) && r[1].width() < 1e293);
        let r = residual(&a, &[1.0, 1.0], &[f64::NAN, 0.0]).unwrap();
        assert_eq!(r[0], Interval::ENTIRE);
        assert!(r[1].contains(4.0));
        let r = residual_of_dd(&a, &[Dd::NAN, Dd::ONE], &[0.0, 0.0]).unwrap();
        assert_eq!(r, vec![Interval::ENTIRE; 2]);
        assert_eq!(norm_inf_bound(&[]), 0.0);
        assert_eq!(
            norm_inf_bound(&[Interval::ZERO, Interval::EMPTY]),
            f64::INFINITY
        );
        assert_eq!(enclosure(Dd::ONE, f64::NAN), Interval::ENTIRE);
    }

    #[test]
    fn dimensions_are_checked_first() {
        let a = small();
        let err = |expected, got| VerifyError::Dimension { expected, got };
        assert_eq!(residual(&a, &[1.0], &[0.0, 0.0]), Err(err(2, 1)));
        assert_eq!(residual(&a, &[1.0, 1.0], &[0.0]), Err(err(2, 1)));
        assert_eq!(
            residual_csc(&a.to_csc(), &[1.0; 3], &[0.0; 2]),
            Err(err(2, 3))
        );
        assert_eq!(residual_of_dd(&a, &[Dd::ONE], &[0.0; 2]), Err(err(2, 1)));
        assert_eq!(residual_dd(&a, &[Dd::ONE; 2], &[0.0; 3]), Err(err(2, 3)));
        assert_eq!(activity(&a, &[]), Err(err(2, 0)));
        let mut x = [Dd::ONE; 2];
        let o = PolishOptions::default();
        assert_eq!(polish(&a, &[0.0], &mut x, &mut |_| {}, &o), Err(err(2, 1)));
        // 1 × 2: not square.
        let wide = CsrMatrix::from_raw_parts(2, vec![0, 2], vec![0, 1], vec![1.0, 1.0]).unwrap();
        assert_eq!(
            polish(&wide, &[0.0], &mut x, &mut |_| {}, &o),
            Err(err(1, 2))
        );
    }

    #[test]
    fn errors_are_errors() {
        let e: Box<dyn core::error::Error> = Box::new(VerifyError::Dimension {
            expected: 3,
            got: 2,
        });
        assert_eq!(e.to_string(), "dimension mismatch: expected 3, got 2");
    }

    /// Step P2 (c) is per row: a row of small terms is not declared
    /// converged because another row is large.
    #[test]
    fn convergence_is_tested_row_by_row() {
        // diag(1e20, 1), b = (1e20, 1): x = (1, 1). Start with the second
        // component off by 1e-10: its residual, 1e-10, is 5e-31 of the
        // global scale ‖A‖‖x‖ + ‖b‖ = 2e20 (below τ ≈ 8e-28) and 5e-11 of
        // its own row.
        let a = CsrMatrix::from_raw_parts(2, vec![0, 1, 2], vec![0, 1], vec![1e20, 1.0]).unwrap();
        let b = [1e20, 1.0];
        let mut x = [Dd::ONE, Dd::from_f64(1.0 + 1e-10)];
        let mut solve = |s: &mut [f64]| s[0] /= 1e20;
        let rep = polish(&a, &b, &mut x, &mut solve, &PolishOptions::default()).unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::Converged, 1));
        assert_eq!(x, [Dd::ONE, Dd::ONE]);
        // The threshold is exactly relative to the row: 2 x = 2 with x one
        // ulp off has |r| = 2ε against Σ|a x| + |b| ≈ 4.
        let two = CsrMatrix::from_raw_parts(1, vec![0, 1], vec![0], vec![2.0]).unwrap();
        let start = [Dd::from_f64(1.0 + f64::EPSILON)];
        let loose = PolishOptions {
            tol: 0.6 * f64::EPSILON,
            ..PolishOptions::default()
        };
        let tight = PolishOptions {
            tol: 0.4 * f64::EPSILON,
            ..PolishOptions::default()
        };
        let mut x = start;
        let rep = polish(&two, &[2.0], &mut x, &mut |s| s[0] /= 2.0, &loose).unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::Converged, 0));
        let mut x = start;
        let rep = polish(&two, &[2.0], &mut x, &mut |s| s[0] /= 2.0, &tight).unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::Converged, 1));
        assert_eq!(x, [Dd::ONE]);
    }

    /// Progress is measured in the norm (step P2 (d)) while convergence is
    /// tested row by row.
    #[test]
    fn progress_is_measured_in_the_norm() {
        let a = CsrMatrix::from_raw_parts(2, vec![0, 1, 2], vec![0, 1], vec![1e20, 1.0]).unwrap();
        let b = [1e20, 1.0];
        // Row 1 keeps a residual of 1e20 ε that the "solve" never touches;
        // row 2 is solved exactly by the first correction. The norm ‖r‖∞,
        // set by row 1, does not move: stagnated, the correction kept.
        let start = [Dd::from_f64(1.0 + f64::EPSILON), Dd::from_f64(1.5)];
        let mut x = start;
        let mut solve = |s: &mut [f64]| s[0] = 0.0;
        let rep = polish(&a, &b, &mut x, &mut solve, &PolishOptions::default()).unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::Stagnated, 1));
        assert_eq!(x, [start[0], Dd::ONE]);
        assert_eq!(rep.residual_after, rep.residual_before);
    }

    /// Rows whose scale `S_i` is zero need no rule of their own in step
    /// P2 (c): an empty row with `b_i = 0` and a row whose only term
    /// underflows completely pass with a zero residual, and only then.
    #[test]
    fn rows_of_zero_scale() {
        // Row 0 is empty, row 1 is 1e-200 x_1 = 0 with x_1 = 1e-200, row 2
        // is 2 x_2 = 2.
        let a =
            CsrMatrix::from_raw_parts(3, vec![0, 0, 1, 2], vec![1, 2], vec![1e-200, 2.0]).unwrap();
        let b = [0.0, 0.0, 2.0];
        let x = [Dd::ZERO, Dd::from_f64(1e-200), Dd::ONE];
        let r = residual_dd(&a, &x, &b).unwrap();
        assert_eq!(r, vec![Dd::ZERO; 3]);
        assert!(rows_converged(&a, &x, &b, &r, 2f64.powi(-90)));
        let mut y = x;
        let rep = polish(
            &a,
            &b,
            &mut y,
            &mut |_| panic!("no solve"),
            &PolishOptions::default(),
        )
        .unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::Converged, 0));
        // A nonzero residual against a zero scale never passes, whatever
        // τ; nor does a NaN.
        let fake = [Dd::from_f64(1e-300), Dd::ZERO, Dd::ZERO];
        assert!(!rows_converged(&a, &x, &b, &fake, 1e300));
        let fake = [Dd::ZERO, Dd::ZERO, Dd::NAN];
        assert!(!rows_converged(&a, &x, &b, &fake, 1e300));
        // A scale that overflows proves nothing: the row does not pass,
        // although each product and the residual are finite (1.5e308 −
        // 1.5e308, whose scale 3e308 is not an `f64`).
        let big =
            CsrMatrix::from_raw_parts(2, vec![0, 2], vec![0, 1], vec![1e150, -1e150]).unwrap();
        let xb = [Dd::from_f64(1.5e158), Dd::from_f64(1.5e158)];
        let rb = residual_dd(&big, &xb, &[0.0]).unwrap();
        assert_eq!(rb, vec![Dd::ZERO]);
        assert!(!rows_converged(&big, &xb, &[0.0], &rb, 2f64.powi(-90)));
        let rb = [Dd::from_f64(1e290)];
        assert!(!rows_converged(&big, &xb, &[0.0], &rb, 2f64.powi(-90)));
    }

    #[test]
    fn defaults_are_those_of_the_contract() {
        let o = PolishOptions::default();
        assert_eq!(
            (o.tol, o.min_progress, o.max_iter),
            (2f64.powi(-90), 0.5, 10)
        );
    }

    /// The exact inverse of [`small`], times `damping`.
    fn solve_small(damping: f64) -> impl FnMut(&mut [f64]) {
        move |s: &mut [f64]| {
            let (p, q) = (s[0], s[1]);
            s[0] = damping * (3.0 * p - q) / 5.0;
            s[1] = damping * (2.0 * q - p) / 5.0;
        }
    }

    #[test]
    fn polish_converges_and_reports() {
        let a = small();
        let b = [1.0, 1.0]; // x = (2/5, 1/5): not representable
        let mut x = [Dd::from_f64(0.4), Dd::from_f64(0.2)];
        let rep = polish(
            &a,
            &b,
            &mut x,
            &mut solve_small(1.0),
            &PolishOptions::default(),
        )
        .unwrap();
        assert_eq!(rep.stop, PolishStop::Converged);
        assert!(rep.iterations >= 1 && rep.iterations <= 3, "{rep:?}");
        assert!(
            rep.residual_before > 1e-17 && rep.residual_after < 1e-30,
            "{rep:?}"
        );
        assert!(x[0].lo() != 0.0 && x[1].lo() != 0.0);
        let cert = norm_inf_bound(&residual_of_dd(&a, &x, &b).unwrap());
        assert!(cert < 1e-30, "{cert:e}");
        // An exact solution converges without a correction.
        let mut x = [Dd::ONE, Dd::from_f64(2.0)];
        let rep = polish(
            &a,
            &[4.0, 7.0],
            &mut x,
            &mut |_| panic!("no solve"),
            &PolishOptions::default(),
        )
        .unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::Converged, 0));
        assert_eq!((rep.residual_before, rep.residual_after), (0.0, 0.0));
        // The empty system.
        let empty = CsrMatrix::from_raw_parts(0, vec![0], vec![], vec![]).unwrap();
        let rep = polish(&empty, &[], &mut [], &mut |_| {}, &PolishOptions::default()).unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::Converged, 0));
    }

    #[test]
    fn polish_stops_for_each_reason() {
        let a = small();
        let b = [1.0, 1.0];
        let start = [Dd::from_f64(0.5), Dd::from_f64(0.5)];
        let o = PolishOptions::default();

        // A "solve" that is not one (the identity): the residual grows,
        // the correction is undone (P2 (b)).
        let mut x = start;
        let rep = polish(&a, &b, &mut x, &mut |_| {}, &o).unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::Stagnated, 0));
        assert_eq!(x, start);
        assert_eq!(rep.residual_after, rep.residual_before);
        assert_eq!(rep.residual_before, 1.0);

        // A solve that is 40 % right: the residual shrinks by 0.6 > ρ per
        // pass; the first correction is kept (P2 (d)).
        let mut x = start;
        let rep = polish(&a, &b, &mut x, &mut solve_small(0.4), &o).unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::Stagnated, 1));
        assert!(rep.residual_after < rep.residual_before && x != start);

        // The same with ρ = 0.9: the limit of 10 corrections (P2 (e)).
        let slow = PolishOptions {
            min_progress: 0.9,
            ..o
        };
        let mut x = start;
        let rep = polish(&a, &b, &mut x, &mut solve_small(0.4), &slow).unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::IterationLimit, 10));
        assert!(rep.residual_after < 0.01 * rep.residual_before, "{rep:?}");
        // No correction at all with a limit of zero.
        let none = PolishOptions { max_iter: 0, ..o };
        let mut x = start;
        let rep = polish(&a, &b, &mut x, &mut |_| panic!("no solve"), &none).unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::IterationLimit, 0));
        assert_eq!(x, start);

        // A correction that is not finite: x untouched (P2).
        let mut x = start;
        let rep = polish(&a, &b, &mut x, &mut |s| s[0] = f64::NAN, &o).unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::NotFinite, 0));
        assert_eq!(x, start);
        // A residual that is not finite from the start.
        let mut x = [Dd::NAN, Dd::ONE];
        let rep = polish(&a, &b, &mut x, &mut |_| panic!("no solve"), &o).unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::NotFinite, 0));
        assert!(rep.residual_before.is_nan() && rep.residual_after.is_nan());
        // A correction that overflows the next residual is undone (P2 (a)).
        let mut x = start;
        let rep = polish(&a, &b, &mut x, &mut |s| s[0] = 1e308, &o).unwrap();
        assert_eq!((rep.stop, rep.iterations), (PolishStop::NotFinite, 0));
        assert_eq!(x, start);
        assert_eq!(rep.residual_after, rep.residual_before);
    }
}
