//! # Chapter 11.10, continued — Algorithm E′: a rigorous bound of the error through triangular factors
//!
//! ## Purpose
//! Algorithm E (the parent chapter) bounds `‖x̃ − A⁻¹b‖∞` with `n`
//! transposed solves (`2n` with refinement) and an `O(n)` scan per row:
//! quadratic in `n` (measured: 21 s at `n = 10⁴`, 271 s at `3 · 10⁴`).
//! Algorithm E′ proves the same statement from triangular factors `L`,
//! `U` and permutations `P`, `Q` with `P A Q ≈ L U` — those of
//! `xingo_sparse::Lu::triangular_factors` (Chapter 5.3 of `xingo-sparse`,
//! "Reading the factors"), or any others — at the cost of **one product
//! `L U`** (the flops of the factorization), **four triangular solves with
//! comparison matrices**, one ordinary solve and one product with `A`.
//! Nothing is assumed about the quality of the factors: the defect `P A Q −
//! L U`, which holds the rounding of the elimination *and* the entries the
//! elimination dropped as negligible (`drop_tol`), is **enclosed**, not
//! supposed small. Bad factors can only make the certificate fail (`β ≥
//! 1`) or the bound loose (invariant E1′).
//!
//! ## Notation
//! As in the parent chapter: `F` the `f64` numbers, `u = 2⁻⁵³`, `▽`/`△`
//! the directed roundings of Chapter 11.9, `mag(X) = max{|x| : x ∈ X}`,
//! `‖·‖∞` the max norm and the max absolute row sum, `𝟙` the vector of
//! ones, inequalities between vectors and matrices entrywise, `|M|` the
//! entrywise absolute value. The factors are the exported ones, read as
//! **exact real numbers**: `p_s = row_perm.fwd[s]`, `q_t = col_perm.fwd[t]`,
//! `P` and `Q` the permutation matrices with `(P v)_s = v_{p_s}` and `(Q
//! y)_{q_t} = y_t`, so that `(P A Q)_{st} = a_{p_s q_t}`; `L` is `l` plus
//! the unit diagonal, `U` is `u` (diagonal included). "Step numbering" is
//! the index `s`, `t` of `L` and `U`; "positions" the index of `x`. `F =
//! Pᵀ L U Qᵀ` (the matrix the factors represent), `E = P A Q − L U` (the
//! defect, exact). For a triangular `T` with nonzero diagonal, `⟨T⟩` is its
//! comparison matrix: `|t_kk|` on the diagonal, `−|t_kt|` off it.
//!
//! ## Three lemmas
//! **Lemma C (comparison matrices).** If `T` is triangular with nonzero
//! diagonal, `T` and `⟨T⟩` are invertible and `|T⁻¹| ≤ ⟨T⟩⁻¹`.
//!
//! *Proof.* Write `T = D (I − N)` with `D` the diagonal of `T` and `N = I
//! − D⁻¹T`, strictly triangular, so `Nⁿ = 0` and `T⁻¹ = (Σ_{k<n} Nᵏ) D⁻¹`.
//! Likewise `⟨T⟩ = |D| (I − |N|)` (the off-diagonal entries of `D⁻¹T` are
//! `t_kt / t_kk`), with `|N|` strictly triangular, so `⟨T⟩⁻¹ = (Σ_{k<n}
//! |N|ᵏ) |D|⁻¹`. The triangle inequality and `|XY| ≤ |X||Y|` give `|T⁻¹| ≤
//! Σ |N|ᵏ |D⁻¹| = ⟨T⟩⁻¹`. ∎ (In particular `⟨T⟩⁻¹ ≥ 0`.)
//!
//! **Lemma U (upward comparison solves).** Let `v ≥ 0` be a vector of `F ∪
//! {+∞}` in step numbering. Procedure `CompSolve(v)`: (1) for `k = 0, 1,
//! …, n − 1`: if `v_k ≠ 0`, for every stored `l_sk` (`s > k`): `v_s ←
//! △(v_s + △(|l_sk| · v_k))`; (2) for `k = n − 1, …, 0`: `v_k ← △(v_k /
//! |u_kk|)`; if `v_k ≠ 0`, for every stored `u_sk` with `s < k`: `v_s ←
//! △(v_s + △(|u_sk| · v_k))`. Its output `v̂` satisfies `v̂ ≥ ⟨U⟩⁻¹ ⟨L⟩⁻¹
//! v` (an infinite component is an upper bound too).
//!
//! *Proof.* The exact `y = ⟨L⟩⁻¹ v` satisfies `y_s = v_s + Σ_{k<s} |l_sk|
//! y_k`. Phase (1) is the column-oriented forward substitution: when
//! column `k` is visited, `v_k` has received every contribution it will
//! ever receive (only columns `k′ < k` write into row `k`), so call it
//! `ŷ_k`. By induction on `k`, `ŷ_k ≥ y_k ≥ 0`: `ŷ_s` is `v_s` plus the
//! terms `△(|l_sk| ŷ_k) ≥ |l_sk| y_k ≥ 0`, added with `△`; each `△` is
//! at least its exact argument and `+` is monotone, so the computed sum is
//! at least the exact one, in any order. A skipped column has `ŷ_k = 0`,
//! hence `y_k = 0` and zero contributions. Phase (2) is the same argument
//! for `z = ⟨U⟩⁻¹ y`, `z_k = (y_k + Σ_{t>k} |u_kt| z_t) / |u_kk|`, columns
//! visited in decreasing order, with `△(a / |u_kk|) ≥ a / |u_kk|` and the
//! input `ŷ ≥ y` (monotone in the input since `⟨U⟩⁻¹ ≥ 0`, Lemma C). The
//! directed operations are those of Algorithm N (Chapter 11.9): exact
//! directed roundings, under- and overflow included (`△` returns `+∞` on
//! overflow, and `+∞` propagates as `+∞`; `0 · ∞` never arises, since
//! stored values are nonzero). ∎
//!
//! **Lemma D (the defect).** Step E′1 below returns `ē` with `ē_s ≥ Σ_t
//! |E_{st}|` for every `s`.
//!
//! *Proof.* `(L U)_{st} = Σ_k L_{sk} U_{kt}` and a term is nonzero only if
//! `u_kt` is stored in column `t` of `u` and either `s = k` (the implicit
//! `L_{kk} = 1`) or `l_sk` is stored in column `k` of `l`. `(P A Q)_{st} =
//! a_{p_s q_t}` is nonzero only if it is stored in column `q_t` of `A`, at
//! row `p_s`, i.e. `s = P⁻¹(p_s)` (`row_perm.inv`). Step E′1 gathers, for
//! each `(s, t)`, exactly those terms as pairs — `(l_sk, u_kt)`, `(1,
//! u_kt)`, `(−1, a_{p_s q_t})` — whose dot product is `−E_{st}`, an
//! identity of real numbers; an `(s, t)` with no pair has `E_{st} = 0`. By
//! (K2) of Chapter 11.5 and the proof of (V1), the interval of step V3
//! contains `−E_{st}` (or is `[−∞, +∞]` when a hypothesis of Dot2Err
//! fails), so its magnitude is at least `|E_{st}|`; the magnitudes are
//! added with `△`, which bounds the exact sum of nonnegative terms. ∎
//!
//! ## Algorithm E′
//! Input: a square `A ∈ F^{n×n}` (compressed rows; one compressed-column
//! copy is made), the enclosure `[r] ∋ A x̃ − b` (from `residual` or
//! `residual_of_dd`), the factors, and the switch `first_order_correction`
//! of [`FactorBoundOptions`] (default on). Output: a [`ForwardErrorBound`]
//! with `contraction = β` and `solves = 0`.
//!
//! E′0. \[Checks.\] `A` satisfies the invariants of its type
//!      (`CsrMatrix::check_invariants`: valid pointers, column indices in
//!      range and strictly increasing in each row, stored values finite
//!      and nonzero), else a **panic**, in every build — the fields of
//!      `CsrMatrix` are public too, and with a column repeated in a row
//!      "the real matrix the stored `f64` represent" is not defined (task
//!      I06, review R06, observation 1). `A` square, `[r]` and the factors
//!      of its order, else
//!      `VerifyError::Dimension`; the factors satisfy (T1) of Chapter 5.3
//!      of `xingo-sparse` (`TriangularFactors::check_invariants`: valid
//!      permutations, `l` strictly lower, `u` upper with its diagonal
//!      present, stored values finite and nonzero), else
//!      `VerifyError::InvalidFactors` — the theorem needs them, and the
//!      fields are public. An empty `[r_i]`: not certified (*not finite*).
//! E′1. \[Defect.\] For each column `t` of `U`: for every stored `u_kt`,
//!      the pair `(1, u_kt)` for row `k` and the pairs `(l_sk, u_kt)` for
//!      the stored `l_sk`; for every stored `a_{i q_t}`, the pair `(−1,
//!      a_{i q_t})` for row `P⁻¹(i)`. Group the pairs by row `s` (a counting
//!      sort, rows in the order of their first appearance, pairs in the
//!      order above); for each row, `(d, β) ← Dot2Err`, `ē_s ← △(ē_s +
//!      mag(V3(d, β)))`.
//! E′2. \[Contraction.\] `ŵ ← CompSolve(ē)`; `w_{q_t} ← ŵ_t`; `β ← max_p
//!      w_p` (exact). If not `β < 1`: not certified (*not contracting*).
//! E′3. \[First-order correction, optional.\] If on: `z ← Q U⁻¹ L⁻¹ P
//!      mid([r])` in `f64` with the factors (not rigorous: it only chooses
//!      `z`); if a component of `mid([r])` or of `z` is not finite, `z ←
//!      0`. With `z ≠ 0`: `[r₂]_i ← [r_i] − V3(Dot2Err((a_ij)_j, (z_j)_j))`
//!      in `Interval`; if some `[r₂]_i` is unbounded while every `[r_i]` is
//!      bounded, `z ← 0`. With `z = 0`: `[r₂] ← [r]`.
//! E′4. \[Second order.\] `ĝ ← CompSolve(P mag([r₂]))`; `g_{q_t} ← ĝ_t`;
//!      `γ ← max_p g_p`. If `γ = +∞`: not certified (*not finite*).
//! E′5. \[Bounds.\] `ε_d ← sup([γ, γ] / (1 − [β, β]))` in `Interval`; for
//!      each `p`: `ε′_p ← △(△(|z_p| + g_p) + △(w_p · ε_d))`; `ε ←
//!      min(△(‖z‖∞ + ε_d), max_p ε′_p)`; if `ε = +∞`: *not finite*. `ε_p ←
//!      min(ε, ε′_p)`.
//!
//! **Theorem E′.** Let the factors satisfy (T1). If Algorithm E′ ends
//! certified, then `A` is nonsingular and, for **every** `b ∈ ℝⁿ` and `x̃ ∈
//! ℝⁿ` with `A x̃ − b ∈ [r]` componentwise, `x* = A⁻¹b` satisfies `‖x̃ −
//! x*‖∞ ≤ ε` and `|x̃_p − x*_p| ≤ ε_p` for `p = 1, …, n`.
//!
//! *Proof.* (i) *`F` is invertible:* `det L = 1` and `det U = Π u_kk ≠ 0`
//! (T1). `F⁻¹ = Q U⁻¹ L⁻¹ P`, and since a permutation matrix equals its
//! absolute value, `|F⁻¹| ≤ Q |U⁻¹| |L⁻¹| P ≤ Q Ĝ P` with `Ĝ = ⟨U⟩⁻¹
//! ⟨L⟩⁻¹ ≥ 0` (Lemma C). (ii) *Contraction:* `A − F = Pᵀ E Qᵀ`; let `C =
//! F⁻¹ (A − F) = Q U⁻¹ L⁻¹ E Qᵀ`. Then `|C| ≤ Q Ĝ |E| Qᵀ`, and as `Qᵀ𝟙 =
//! 𝟙`, row `q_t` of `|C| 𝟙` is at most `(Ĝ |E| 𝟙)_t ≤ (Ĝ ē)_t ≤ ŵ_t =
//! w_{q_t}` (Lemma D, `Ĝ ≥ 0`, Lemma U). So `‖C‖∞ ≤ β < 1`. (iii)
//! *Nonsingularity:* `A = F (I + C)` with `‖C‖∞ < 1`: `I + C` is
//! invertible (Neumann series), and so is `A`; `x*` exists. (iv) *The
//! residual of the correction:* `z` is a vector of finite `f64`, i.e. of
//! reals (E′3 replaces any other by `0`). With `e = x̃ − x*`, `A e = A x̃ −
//! b =: ρ ∈ [r]`. Let `d = e − z`; `A d = ρ − A z`. By (K2) and (V1) the
//! interval `V3(…)` of E′3 contains `(A z)_i`, so by (I1) `(A d)_i ∈
//! [r₂]_i`, and `|A d| ≤ mag([r₂])`; with `z = 0` this is `ρ ∈ [r]`. (v)
//! *The bound of `d`:* from `F (I + C) d = A d`, `d = F⁻¹ A d − C d`, so
//! `|d| ≤ Q Ĝ P mag([r₂]) + |C| |d| ≤ Q Ĝ P mag([r₂]) + Q Ĝ |E| 𝟙
//! ‖d‖∞`. Component `q_t` of the first term is at most `ĝ_t = g_{q_t}`
//! (Lemma U), of the second at most `w_{q_t} ‖d‖∞` (as in ii). Hence
//! `|d_p| ≤ g_p + w_p ‖d‖∞` for every position `p`; the maximum over `p`
//! gives `‖d‖∞ ≤ γ + β ‖d‖∞`, so `‖d‖∞ ≤ γ / (1 − β) ≤ ε_d` — the quotient
//! in `Interval` (I1), with `▽(1 − β) > 0` because `β < 1` is an `f64`.
//! If the quotient overflows, `ε_d = +∞`; then `β > 0`, some `w_p > 0`,
//! `ε′_p = +∞` and `△(‖z‖∞ + ε_d) = +∞`, so `ε = +∞` and E′5 gives no
//! certificate: below, `ε_d` is finite (and `△(w_p · ε_d)` never meets the
//! convention `0 · ∞ = 0`).
//! (vi) *The bound of `e`:* `|e_p| ≤ |z_p| + |d_p| ≤ |z_p| + g_p + w_p
//! ε_d ≤ ε′_p` (each `△` at least its exact argument, all terms
//! nonnegative), so `‖e‖∞ ≤ max_p ε′_p`; also `‖e‖∞ ≤ ‖z‖∞ + ‖d‖∞ ≤
//! △(‖z‖∞ + ε_d)`. So `‖e‖∞ ≤ ε`, and `|e_p| ≤ min(ε, ε′_p) = ε_p`. ∎
//!
//! For an exactly singular `A` no factors give `β < 1` (it would make `A`
//! invertible), so Algorithm E′ cannot certify it (invariant E2′).
//!
//! **Where each rounding is.**
//!
//! | quantity | computed as | enclosed by |
//! |---|---|---|
//! | `L`, `U`, `P`, `Q` | given (`f64`, (T1) checked) | nothing needed: they *are* the numbers |
//! | `E_{st}` | Dot2Err on the exact pairs of E′1 | (K2), V3: Lemma D |
//! | `E_{st}` outside the gathered pattern | not computed | exactly zero (Lemma D) |
//! | `ē`, `w`, `g` | `△` of Algorithm N on nonnegative terms | Lemma U, Lemma D |
//! | `z` | `f64` solve with the factors, no bound | nothing needed: `d = e − z` for this `z` |
//! | `A z` | Dot2Err, V3 | (K2), (V1) |
//! | `[r₂] = [r] − [A z]` | `Interval` | (I1) |
//! | `γ/(1 − β)`, `ε′_p`, `ε` | `Interval`, `△` | (I1), Algorithm N |
//!
//! **What the design sketch (task I04) left open, and how it is closed
//! here.** (a) "Any factors serve" is true only of factors satisfying
//! (T1): the fields of `TriangularFactors` are public, and an entry of `l`
//! on or above the diagonal is silently misread by a triangular solve —
//! step E′0 checks (T1). (b) Enclosing `L U` by sums in `Interval`, as the
//! sketch had it, is valid but its width, about `k u (|L||U|)_{st}` for
//! `k` terms, is of the size of the defect itself: Dot2Err per entry
//! encloses `E_{st}` to a few ulps of it plus `≈ k² u² (|L||U|)_{st}`. (c)
//! The enclosure of the corrected residual is `[r] − [A z]`, not
//! `residual(A, z, b)` (which encloses `A z − b`); a non-finite `z` must
//! be replaced (step E′3). (d) The norm bound is also `max_p ε′_p`, never
//! larger than `‖z‖∞ + ε_d` in real arithmetic, before the roundings
//! (`|z_p| + g_p + w_p ε_d ≤ ‖z‖∞ + γ + β ε_d = ‖z‖∞ + ε_d`); the computed
//! `ε′_p` may pass `△(‖z‖∞ + ε_d)` by a few ulps, so E′5 takes the
//! minimum of the two. (e) The cost of E′1 is not linear in `nnz(L) +
//! nnz(U)` but in the number of products of the factorization, `Σ_t Σ_{k :
//! u_kt ≠ 0} (1 + nnz(l_{:k})) + nnz(A)`; no enclosure of the defect can
//! avoid forming every entry of `L U`.
//!
//! **Cost.** E′1: `N_E = Σ_t Σ_{k : u_kt stored} (1 + nnz(l_{:k})) +
//! nnz(A)` pairs of Dot2Err (≈ 25 flops each) — the multiplications of the
//! elimination that produced the factors, plus `A` — and a counting sort
//! per column. E′2 and E′4: `nnz(L) + nnz(U)` upward products and sums
//! each (Algorithm N: ≈ 4 flops per sum, 2 per product). E′3: one `f64`
//! solve (`2 (nnz(L) + nnz(U))` flops) and `nnz(A)` pairs of Dot2Err.
//! Memory: `O(n + nnz(A))` (a compressed-column copy of `A`, a few vectors
//! of length `n`) and the pairs of the largest column of `L U`. No term is
//! `n²` unless the factors are dense.
//!
//! **How tight, and when it fails.** With the correction, the first-order
//! part `z ≈ A⁻¹ ρ` is computed, not bounded, and `Ĝ` multiplies only the
//! residual of `z` and the width of `[r]`: when certified, the bound of an
//! `f64` solution is the true error to six digits or better. Without it,
//! the bound is that of `Ĝ |ρ|`, which loses every cancellation of signs
//! in `A⁻¹ ρ` (two to five orders of magnitude looser, measured). **The
//! limitation is `β`.** Lemma C is sharp only when the signs of the
//! factors are those of an M-matrix; for general sparse factors `‖Ĝ 𝟙‖∞`
//! grows exponentially with the depth of the elimination while `‖|F⁻¹|
//! 𝟙‖∞` does not. Measured (task I06, the factors of `xingo_sparse::Lu`):
//!
//! | systems | certified by E′ | by Algorithm E | `‖Ĝ𝟙‖∞` vs `‖\|F⁻¹\|𝟙‖∞` |
//! |---|---:|---:|---|
//! | random sparse, `n ≤ 12`, unspoiled factors (1 302 systems × 2 solutions, four option sets) | 2 586 of 2 604 | 9 more of the 1 302 `f64` ones | — |
//! | random sparse, `n = 50` (80, four option sets) | 38 of 80 | 80 | `10⁸`–`10¹²` vs `10²`–`10⁶` |
//! | random sparse, `n = 100` / `200` / `400` (80, 80, 79) | 19 / 1 / 0 | all | at `n = 400`: `10¹⁹`–`10⁴⁵` vs `10³`–`10⁹` |
//! | Netlib KKT matrices of task I05 that `Lu` factorizes (83) | 14 (12 of the 64 with `n ≤ 3000`) | 53 of those 64 | `β` median `3 · 10¹⁵` |
//!
//! A larger pivot threshold (`rel_pivot_tol = 1`, so `|l| ≤ 1`) lowers
//! `‖Ĝ𝟙‖∞` by a few orders of magnitude but not its growth; scaling cannot
//! change it (`⟨D T D′⟩ = D ⟨T⟩ D′` for positive diagonals). Where it
//! certifies, Algorithm E′ is far cheaper than Algorithm E (fit2p, `n =
//! 16 525`: 13 ms; maros-r7, `n = 12 544`, `N_E = 6 · 10⁸`: 1.9 s); where
//! it does not, Algorithm E decides, at its `n²` cost. Bounds for large
//! sparse systems that are not H-matrices remain an open problem of
//! verified computing (Rump 2010, cited from memory).
//!
//! ## Invariants
//! (E1′) *Never false.* [`ForwardErrorOutcome::Certified`] implies Theorem
//!       E′ for the stored data, whatever the factors (satisfying (T1)).
//!       Checked against exact rationals with real, perturbed and
//!       adversarial factors.
//! (E2′) *Nonsingular.* Certified implies `A` nonsingular; an exactly
//!       singular `A` is never certified (checked).
//! (E3) *Shape.* As for Algorithm E, with `α` read as `β` (a
//!       `debug_assert!`).
//! (E5′) *Deterministic.* The same bits for the same data: no closure,
//!       every loop in a fixed order, Dot2Err the same on every path (K1).
//!
//! ## References
//! Neumaier, *Interval Methods for Systems of Equations* (1990), ch. 3, for
//! comparison matrices and `|T⁻¹| ≤ ⟨T⟩⁻¹` — **cited from memory**; Lemma C
//! is proved above. Rump, "Verification methods", Acta Numerica 19 (2010),
//! for bounds through an approximate factorization — **cited from memory,
//! not reread**; the statement, the constants and the proof of Theorem E′
//! are this chapter's, from (K2), (V1), (I1) and Algorithm N only.

use xingo_sparse::{Col, CscMatrix, CsrMatrix, Row, TriangularFactors};

use super::{ForwardErrorBound, ForwardErrorOutcome, VerifyError, dimension, enclosure, upper};
use crate::interval::{Interval, add_up, div_up, mul_up};
use crate::kernels;

/// The parameters of Algorithm E′ (Chapter 11.10, "Algorithm E′").
/// `#[non_exhaustive]`: build it from [`FactorBoundOptions::default`] and
/// set the fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FactorBoundOptions {
    /// Step E′3: subtract the first-order correction `z = F⁻¹ mid([r])`
    /// before bounding — one `f64` solve with the factors and one enclosed
    /// product with `A` (default `true`). Without it the bound is that of
    /// `⟨U⟩⁻¹⟨L⟩⁻¹ |r|`, which loses the cancellations of `A⁻¹ r`.
    pub first_order_correction: bool,
}

impl Default for FactorBoundOptions {
    fn default() -> Self {
        Self {
            first_order_correction: true,
        }
    }
}

/// Algorithm E′: from `r ∋ A x − b` (from [`super::residual`] or
/// [`super::residual_of_dd`]) and triangular factors `P A Q ≈ L U` (for
/// instance `xingo_sparse::Lu::triangular_factors` of `A`), proves that `A`
/// is nonsingular and bounds `x − A⁻¹b` (Theorem E′ of Chapter 11.10): if
/// the result is certified, then for every `x` and `b` with `A x − b ∈ r`
/// componentwise, `‖x − A⁻¹b‖∞ ≤ norm_inf` and `|x_p − (A⁻¹b)_p| ≤
/// componentwise[p]`, for the stored `f64` values of `A`.
///
/// The bound is a theorem **whatever** the factors are, as long as they
/// are triangular with valid permutations (checked): the defect `P A Q −
/// L U` — rounding and dropped entries — is enclosed entry by entry, and
/// poor factors can only prevent the certificate or loosen the bound.
/// [`ForwardErrorBound::contraction`] reports `β ≥ ‖I − F⁻¹A‖∞` (`F = Pᵀ L
/// U Qᵀ`) and [`ForwardErrorBound::solves`] is `0`. The cost is one
/// product `L U` (the multiplications of the factorization), four
/// triangular solves with comparison matrices and one with the factors,
/// and two passes over `A` — not the `n` solves of
/// [`super::forward_error_bound`]. When `β ≥ 1` the result is
/// [`ForwardErrorOutcome::NotContracting`], and
/// [`super::forward_error_bound`] may still certify, at its cost.
///
/// # When it certifies
/// Only when `β < 1`, and `β` is bounded through the comparison matrices
/// `⟨U⟩⁻¹⟨L⟩⁻¹`, which grow exponentially with the depth of the
/// elimination for general sparse factors (Chapter 11.10, "How tight, and
/// when it fails"). **Measured** with the factors of `xingo_sparse::Lu`:
/// random sparse systems were certified at 48 % for `n = 50`, 24 % for `n =
/// 100`, 1 % for `n = 200` and 0 % for `n = 400` (Algorithm E certified
/// all of them); 14 of the 83 Netlib KKT matrices that `Lu` factorizes (on
/// `n` up to 16 525). So `β < 1` fails for most general sparse systems of
/// order ≳ 200. Where it certifies, the bound of an `f64` solution is the
/// true error to six digits, at a small fraction of the cost of
/// [`super::forward_error_bound`]. **On
/// [`ForwardErrorOutcome::NotContracting`], fall back to
/// [`super::forward_error_bound`]:**
///
/// ```no_run
/// # use xingo_extprec::Interval;
/// # use xingo_extprec::verify::{
/// #     FactorBoundOptions, ForwardErrorBound, ForwardErrorOptions, ForwardErrorOutcome,
/// #     VerifyError, forward_error_bound, forward_error_bound_lu,
/// # };
/// # use xingo_sparse::{CsrMatrix, Lu, LuFactor, SparseVec};
/// fn certify(a: &CsrMatrix<f64>, r: &[Interval], lu: &mut Lu<f64>)
///     -> Result<ForwardErrorBound, VerifyError> {
///     if let Some(f) = lu.triangular_factors() {
///         let e = forward_error_bound_lu(a, r, &f, &FactorBoundOptions::default())?;
///         if e.outcome != ForwardErrorOutcome::NotContracting {
///             return Ok(e);
///         }
///     }
///     // n transposed solves: the BTRAN of the same LU.
///     let mut solve_t = |c: &mut [f64]| {
///         let mut v = SparseVec::from_dense(c);
///         lu.btran(&mut v);
///         c.copy_from_slice(v.values());
///     };
///     forward_error_bound(a, r, &mut solve_t, &ForwardErrorOptions::default())
/// }
/// ```
///
/// ```
/// use xingo_extprec::verify::{FactorBoundOptions, forward_error_bound_lu, residual};
/// use xingo_sparse::{CsrMatrix, Lu, LuFactor};
///
/// // [[2, 1], [1, 3]] x = (1, 1): x* = (2/5, 1/5), not representable.
/// let a = CsrMatrix::from_raw_parts(2, vec![0, 2, 4], vec![0, 1, 0, 1],
///     vec![2.0, 1.0, 1.0, 3.0]).unwrap();
/// let mut lu = Lu::<f64>::default();
/// lu.factorize(&a.to_csc()).unwrap();
/// let factors = lu.triangular_factors().unwrap();
/// let (b, x) = ([1.0, 1.0], [0.4, 0.2]);
/// let r = residual(&a, &x, &b).unwrap();
/// let e = forward_error_bound_lu(&a, &r, &factors, &FactorBoundOptions::default())
///     .unwrap();
/// assert!(e.is_certified() && e.contraction < 1e-15 && e.solves == 0);
/// // |0.4 − 2/5| and |0.2 − 1/5| are below 2⁻⁵⁴ and 2⁻⁵⁵.
/// assert!(e.norm_inf < 1e-16 && e.componentwise[1] < 1e-16);
/// ```
///
/// # Errors
/// [`VerifyError::Dimension`] if `a` is not square, or `r` or the factors
/// do not have its order; [`VerifyError::InvalidFactors`] if the factors
/// violate their invariants (T1).
///
/// # Panics
/// If `a` violates the invariants of its type (`CsrMatrix::check_invariants`:
/// for instance a column repeated in a row, or a NaN stored), in release
/// builds too: they are checked first, in `O(nnz(A))`.
pub fn forward_error_bound_lu(
    a: &CsrMatrix<f64>,
    r: &[Interval],
    factors: &TriangularFactors<f64>,
    opts: &FactorBoundOptions,
) -> Result<ForwardErrorBound, VerifyError> {
    use ForwardErrorOutcome::{Certified, NotContracting, NotFinite};
    // E′0. `A` first: `n_rows` reads its pointers.
    if let Err(e) = a.check_invariants() {
        panic!("forward_error_bound_lu: `a` violates the invariants of its type: {e}");
    }
    dimension(a.n_rows(), a.n_cols())?;
    let n = a.n_rows();
    dimension(n, r.len())?;
    dimension(n, factors.order())?;
    factors
        .check_invariants()
        .map_err(|e| VerifyError::InvalidFactors(e.to_string()))?;
    if r.iter().any(|x| x.is_empty()) {
        return Ok(ForwardErrorBound::refused(NotFinite, n, 0.0, 0));
    }
    let bound = bound(a, r, factors, opts.first_order_correction)
        .unwrap_or_else(|(outcome, beta)| ForwardErrorBound::refused(outcome, n, beta, 0));
    debug_assert!(
        bound.check_invariants().is_ok(),
        "{:?}: {bound:?}",
        bound.check_invariants()
    );
    debug_assert!(n > 0 || bound.outcome == Certified);
    debug_assert!(bound.outcome != NotContracting || bound.contraction >= 1.0);
    Ok(bound)
}

/// Steps E′1–E′5 for checked data; `Err((outcome, β))` when there is no
/// certificate (`β = 0` if it was not reached).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(β < 1)` refuses a NaN too
fn bound(
    a: &CsrMatrix<f64>,
    r: &[Interval],
    f: &TriangularFactors<f64>,
    correct: bool,
) -> Result<ForwardErrorBound, (ForwardErrorOutcome, f64)> {
    use ForwardErrorOutcome::{Certified, NotContracting, NotFinite};
    let n = a.n_rows();
    let q = &f.col_perm.fwd;
    // E′1, E′2.
    let mut v = defect_row_sums(&a.to_csc(), f);
    comparison_solve(f, &mut v);
    let mut w = vec![0.0; n];
    for (t, &vt) in v.iter().enumerate() {
        w[q[t] as usize] = upper(vt);
    }
    let beta = w.iter().fold(0.0_f64, |m, &x| m.max(x));
    if !(beta < 1.0) {
        return Err((NotContracting, beta));
    }
    // E′3.
    let mut z = vec![0.0; n];
    let mut r2 = r.to_vec();
    if correct && let Some(zc) = first_order(f, r) {
        let corrected = corrected_residual(a, r, &zc);
        let bounded = |x: &[Interval]| x.iter().all(|y| y.mag().is_finite());
        if bounded(&corrected) || !bounded(r) {
            z = zc;
            r2 = corrected;
        }
    }
    // E′4.
    let p = &f.row_perm.fwd;
    for (s, vs) in v.iter_mut().enumerate() {
        *vs = upper(r2[p[s] as usize].mag());
    }
    comparison_solve(f, &mut v);
    let mut g = vec![0.0; n];
    for (t, &vt) in v.iter().enumerate() {
        g[q[t] as usize] = upper(vt);
    }
    let gamma = g.iter().fold(0.0_f64, |m, &x| m.max(x));
    if !gamma.is_finite() {
        return Err((NotFinite, beta));
    }
    // E′5.
    let eps_d = upper((Interval::point(gamma) / (1.0 - Interval::point(beta))).sup());
    let z_norm = z.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
    let parts: Vec<f64> = (0..n)
        .map(|p| upper(add_up(add_up(z[p].abs(), g[p]), mul_up(w[p], eps_d))))
        .collect();
    let eps = upper(add_up(z_norm, eps_d)).min(parts.iter().fold(0.0_f64, |m, &x| m.max(x)));
    if !eps.is_finite() {
        return Err((NotFinite, beta));
    }
    let componentwise = parts.iter().map(|&x| x.min(eps)).collect();
    Ok(ForwardErrorBound {
        outcome: Certified,
        norm_inf: eps,
        componentwise,
        contraction: beta,
        solves: 0,
    })
}

/// Step E′1 (Lemma D): `ē_s ≥ Σ_t |(P A Q − L U)_{st}|`, in step
/// numbering. `a` is `A` by columns.
fn defect_row_sums(a: &CscMatrix<f64>, f: &TriangularFactors<f64>) -> Vec<f64> {
    let n = f.order();
    let p_inv = &f.row_perm.inv;
    let mut ebar = vec![0.0_f64; n];
    // Per column `t`: `count[s]` pairs of row `s`, `start[s]` their offset,
    // `touched` the rows with a pair, in the order of their first pair.
    let mut count = vec![0_usize; n];
    let mut start = vec![0_usize; n];
    let mut touched: Vec<usize> = Vec::new();
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for t in 0..n {
        let (u_rows, u_vals) = f.u.col(Col::from_usize(t));
        let (a_rows, a_vals) = a.col(Col::from_usize(f.col_perm.fwd[t] as usize));
        // Counting pass.
        let mut bump = |s: usize| {
            if count[s] == 0 {
                touched.push(s);
            }
            count[s] += 1;
        };
        for &k in u_rows {
            bump(k as usize);
            for &s in f.l.col(Col::from_usize(k as usize)).0 {
                bump(s as usize);
            }
        }
        for &i in a_rows {
            bump(p_inv[i as usize] as usize);
        }
        let mut total = 0;
        for &s in &touched {
            start[s] = total;
            total += count[s];
        }
        xs.clear();
        xs.resize(total, 0.0);
        ys.clear();
        ys.resize(total, 0.0);
        // Filling pass: `start[s]` advances to the end of row `s`.
        let mut put = |s: usize, x: f64, y: f64| {
            xs[start[s]] = x;
            ys[start[s]] = y;
            start[s] += 1;
        };
        for (&k, &ukt) in u_rows.iter().zip(u_vals) {
            put(k as usize, 1.0, ukt);
            let (l_rows, l_vals) = f.l.col(Col::from_usize(k as usize));
            for (&s, &lsk) in l_rows.iter().zip(l_vals) {
                put(s as usize, lsk, ukt);
            }
        }
        for (&i, &aij) in a_rows.iter().zip(a_vals) {
            put(p_inv[i as usize] as usize, -1.0, aij);
        }
        // One Dot2Err per entry of column `t` of `L U − P A Q`.
        for &s in &touched {
            let seg = start[s] - count[s]..start[s];
            let (d, beta) = kernels::dot2_err(&xs[seg.clone()], &ys[seg]);
            ebar[s] = add_up(ebar[s], upper(enclosure(d, beta).mag()));
            count[s] = 0;
        }
        touched.clear();
    }
    ebar
}

/// `CompSolve` of Lemma U: `v ← △(⟨U⟩⁻¹ ⟨L⟩⁻¹ v)` for `v ≥ 0` in step
/// numbering, each component an upper bound of the exact one.
fn comparison_solve(f: &TriangularFactors<f64>, v: &mut [f64]) {
    debug_assert!(
        v.iter().all(|&x| x >= 0.0),
        "Lemma U: a negative or NaN input"
    );
    let n = f.order();
    for k in 0..n {
        let yk = v[k];
        if yk == 0.0 {
            continue;
        }
        let (rows, vals) = f.l.col(Col::from_usize(k));
        for (&s, &lsk) in rows.iter().zip(vals) {
            let s = s as usize;
            debug_assert!(s > k, "(T1): L strictly lower");
            v[s] = add_up(v[s], mul_up(lsk.abs(), yk));
        }
    }
    for k in (0..n).rev() {
        let (rows, vals) = f.u.col(Col::from_usize(k));
        let last = rows.len() - 1;
        debug_assert_eq!(rows[last] as usize, k, "(T1): the diagonal of U last");
        let yk = div_up(v[k], vals[last].abs());
        v[k] = yk;
        if yk == 0.0 {
            continue;
        }
        for (&s, &usk) in rows[..last].iter().zip(&vals[..last]) {
            let s = s as usize;
            v[s] = add_up(v[s], mul_up(usk.abs(), yk));
        }
    }
}

/// Step E′3, the choice of `z = Q U⁻¹ L⁻¹ P mid([r])` in `f64` (positions);
/// `None` when an input or output component is not finite. Not rigorous,
/// and it need not be: the proof holds for every real `z`.
fn first_order(f: &TriangularFactors<f64>, r: &[Interval]) -> Option<Vec<f64>> {
    let n = f.order();
    let mut y: Vec<f64> = f
        .row_perm
        .fwd
        .iter()
        .map(|&i| r[i as usize].mid())
        .collect();
    if !y.iter().all(|v| v.is_finite()) {
        return None;
    }
    for k in 0..n {
        let yk = y[k];
        if yk == 0.0 {
            continue;
        }
        let (rows, vals) = f.l.col(Col::from_usize(k));
        for (&s, &lsk) in rows.iter().zip(vals) {
            y[s as usize] -= lsk * yk;
        }
    }
    let mut z = vec![0.0; n];
    for k in (0..n).rev() {
        let (rows, vals) = f.u.col(Col::from_usize(k));
        let last = rows.len() - 1;
        let zk = y[k] / vals[last];
        z[f.col_perm.fwd[k] as usize] = zk;
        if zk == 0.0 {
            continue;
        }
        for (&s, &usk) in rows[..last].iter().zip(&vals[..last]) {
            y[s as usize] -= usk * zk;
        }
    }
    z.iter().all(|v| v.is_finite()).then_some(z)
}

/// Step E′3, the enclosure: `[r₂]_i = [r_i] − V3(Dot2Err(a_i, z))`, which
/// contains `ρ_i − (A z)_i` for every `ρ_i ∈ [r_i]`.
fn corrected_residual(a: &CsrMatrix<f64>, r: &[Interval], z: &[f64]) -> Vec<Interval> {
    let mut zs = Vec::new();
    (0..a.n_rows())
        .map(|i| {
            let (cols, vals) = a.row(Row::from_usize(i));
            zs.clear();
            zs.extend(cols.iter().map(|&j| z[j as usize]));
            let (d, beta) = kernels::dot2_err(vals, &zs);
            r[i] - enclosure(d, beta)
        })
        .collect()
}
