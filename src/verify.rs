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
//!    certifies the residual that was reached;
//! 3. *how far `x` is from the exact solution* — Algorithm E proves, from
//!    the enclosure of Algorithm V and `n` transposed solves (`2n` with
//!    refinement), that a square `A` is nonsingular and bounds `‖x −
//!    A⁻¹b‖∞` and every `|x_p − (A⁻¹b)_p|`, again a theorem (Theorem E).
//! 4. *the same, for large systems* — Algorithm E′ (the submodule `lu`,
//!    [`forward_error_bound_lu`]) proves the statement of Theorem E from
//!    triangular factors `P A Q ≈ L U` and their comparison matrices, at
//!    the cost of one product `L U` and a few triangular solves instead of
//!    `n` solves (Theorem E′; its chapter follows this one).
//!
//! Algorithm E bounds the error of **one** approximate solution of **one**
//! square system with point data. Verified *solution sets* of interval
//! systems (Rump's `verifylss` on interval data), interval LU and Krawczyk
//! iteration stay out of v0.1 (Chapter 11, "Out of scope").
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
//! For `x ∈ DD^n` ([`residual_of_dd`]) four more steps narrow `R_i` — the
//! *distillation* of the residual (issue #5 of the mirror; "Why the
//! polished residual needs more" below). `N = 2 k_i + 1` is the number of
//! pairs of V1, `(p_j, q_j)` the pairs.
//!
//! V4. \[Split.\] `(π_j, ε_j) ← 2Prod(p_j, q_j)` for `j = 1, …, N`; `w ←
//!     (π_1, ε_1, π_2, ε_2, …, π_N, ε_N)`, `2N` numbers. If one of them is
//!     not finite, keep the `R_i` of V3 and stop.
//! V5. \[Distil, twice.\] Two passes of *VecSum*: `σ ← w_1`; for `j = 2, …,
//!     L` (`L` the current length of `w`): `(σ, w_{j−1}) ← 2Sum(σ, w_j)`;
//!     then `w ← (w_1, …, w_{L−1})`. The two final `σ` are `σ_1`, `σ_2`; `w`
//!     ends with `M = 2N − 2` numbers. If a `σ` or a `w_j` is not finite,
//!     keep the `R_i` of V3 and stop.
//! V6. \[Tail.\] `t ← RN(… RN(RN(w_1 + w_2) + w_3) … + w_M)` and `a ←` the
//!     same sum of the `|w_j|` (`t = a = 0` if `M = 0`). With `m = max(M,
//!     1) − 1`: if `2 m u ≥ 1`, or `t` or `a` is not finite, keep the `R_i`
//!     of V3 and stop. Otherwise `β′ ← sup([δ]·[a] + [φ])`, `[δ] ← [m u] /
//!     (1 − 2[m u])`, in `Interval` (`m u` is an `f64` exactly for `m <
//!     2⁵²`), with `φ = (3η/u)(1 + 8u)`, the floor of `β` in step C5 of
//!     Chapter 11.5 (an `f64` exactly).
//! V7. \[Combine.\] `(s, τ) ← 2Sum(σ_1, σ_2)`; if `s` or `τ` is not finite,
//!     keep the `R_i` of V3 and stop. `R′_i ← ((t ± β′) + τ) + s` in
//!     `Interval` (`t ± β′` is `Interval::mid_rad`), and `R_i ← R_i ∩ R′_i`.
//!
//! **Proof of (V1).** `pᵀq` is the residual exactly: for a double-double
//! `x`, `Σ_j a_ij (h_j + ℓ_j) − b_i = Σ_j a_ij h_j + Σ_j a_ij ℓ_j + (−1)
//! b_i`, an identity of real numbers, and the value of `x_j` *is* `h_j +
//! ℓ_j`. So the double-double case needs no bound of its own — it is
//! `Dot2Err` on a vector twice as long (this replaces the "second-order
//! terms" that the contract's sketch of step V2 announced). By (K2),
//! `|pᵀq − r| ≤ β`, i.e. `h + (ℓ − β) ≤ pᵀq ≤ h + (ℓ + β)`, and each of
//! the two directed sums of V3 on either side moves outward. When `β =
//! +∞` nothing is claimed (C3). ∎ (Until review X0.3i, (K2) failed when an
//! intermediate step of the last 2Sum of Dot2Err overflowed at `±Ω`, and
//! this proof with it; the enclosures found there still contained the
//! residual, by the margin of V3. Now `β = +∞` in that corner and the row is
//! `ENTIRE`: Chapter 11.5, the proof of C5, "Overflow".) (The contract's sketch had `enclose(r) +
//! [−β, β]`, which is also correct and one to two ulps wider: it rounds
//! `r` to its neighbours before the radius is added.)
//!
//! **Proof of (V1) for steps V4–V7.** Write `s* = pᵀq`, the exact residual.
//! (a) *Split.* By (T2) of Chapter 11.1, with `π_j` finite, `p_j q_j = π_j +
//! ε_j − μ_j` with `μ_j = 0` when the product is in the exact range and
//! `|μ_j| ≤ η/2` otherwise; so `s* = Σ w − μ`, `|μ| ≤ N η/2`. (b) *Distil.*
//! By (T1), a 2Sum whose outputs are finite is exact (its only failure, at
//! `|a| = Ω`, leaves the error term not finite, and V5 tests every one), so
//! each pass keeps the exact sum: `Σ w⁽⁰⁾ = σ_1 + Σ w⁽¹⁾ = σ_1 + σ_2 + Σ
//! w⁽²⁾`. (c) *Tail.* `t` is a floating-point sum of the `M` numbers of
//! `w⁽²⁾`. An addition is exact when its result is subnormal, so every
//! addition errs by at most `u` times its result, underflow or not; none
//! overflowed, since an infinite partial sum cannot come back finite and
//! `t`, `a` are finite. Hence `|Σ w⁽²⁾ − t| ≤ γ_m Σ|w_j|` (the
//! order-independent bound of Higham §4.2, `m = M − 1` additions), and
//! since each of the `m` additions of `a` is `RN(c) = c/(1 + θ)`, `|θ| ≤
//! u`, `Σ|w_j| ≤ (1 + u)^m a`. With `m u < 1/2`, `(1 + u)^m ≤ e^{mu} ≤ 1/(1
//! − mu)` and `(1 − mu)² ≥ 1 − 2mu`, so `γ_m (1 + u)^m ≤ mu/(1 − 2mu) = δ`,
//! and `|Σ w⁽²⁾ − t| ≤ δ a`. `[δ]` contains `δ` (I1; `m u` is exact), and
//! `φ > 3 · 2⁵³ η ≥ N η/2` (`N ≤ m/2 + 2 < 2⁵¹ + 2`), so `δ a + N η/2 ≤
//! β′`. (d) *Combine.* `σ_1 + σ_2 = s + τ` exactly (T1, both
//! finite). So `s* = s + τ + t + (Σ w⁽²⁾ − t) − μ` lies in `((t ± β′) +
//! τ) + s` (I1), i.e. in `R′_i`; and in the `R_i` of V3 by the proof
//! above; so in their intersection. ∎ (When a step keeps the `R_i` of V3,
//! nothing new is claimed.) The intersection makes V4–V7 **never widen**
//! the enclosure of V3 (invariant V3), and leaves it bit for bit when
//! `R′_i ⊇ R_i`. That is the case when every operation of V2 and V4–V7 is
//! exact and the residual is an `f64`: both are then that `f64` widened by
//! `φ` — the floor is shared so that an exact row keeps its V3 bits.
//!
//! **Width.** For a finite result of V3, `width(R_i) ≤ 2 ulp(r) + 2β (1 +
//! 2u)` — typically one ulp, two when `r` is an `f64` — and `β` is of
//! order `k² u² Σ_j |a_ij x_j|` (Chapter 11.5, step C5): the enclosure of a
//! residual of size `ρ` computed from terms of size `S` has
//! a relative width of about `u + k² u² S/ρ`. An `f64` solution leaves `ρ
//! ≈ κ u S` at best, so its residual is enclosed to about 16 digits by V3
//! alone. A polished one has `ρ ≈ k u² S`, of the order of `β` itself: V3
//! then only proves "`|a_iᵀx − b_i| ≤ c k² u² S`".
//!
//! *Why the polished residual needs more, and what V4–V7 give.* VecSum
//! leaves error terms bounded by `γ_{L−1}` times the sum of the magnitudes
//! it was given (Ogita, Rump & Oishi, Lemma 4.2 — cited from memory; the
//! proof above does not use it), so after two passes `Σ|w⁽²⁾| ≲ 4N²u² S`
//! and `β′ ≲ 8 N³ u³ S`, against `β ≈ N² u² S` for Dot2Err: one more factor
//! `N u` — the accuracy of a sum in *three* times the working precision
//! (SumK with `K = 3`, *ibid.* Algorithm 4.8), with an a posteriori bound
//! in place of theirs. The other terms of the width are the outward
//! roundings of V7, a few ulps of `s*` when `|s*| ≳ β′`. With `ρ ≈ k u² S`
//! the relative width falls from `≈ k` to `≈ u + k² u`. **Measured** on
//! the polished optimal bases of the 93 Netlib LPs (task X0.5, rows of up
//! to 6 185 entries): the largest half-width over `‖r‖∞` of a basis went
//! from `126` (median) and `9.4 · 10⁴` (largest) with V3 alone to at most
//! `7.2 · 10⁻¹¹` with V4–V7. A third pass of V5 changed none of the bounds
//! of Algorithm E below at three digits, so the chapter stops at two.
//!
//! **Cost.** About 25 flops per stored entry for `x ∈ F^n`; for `x ∈ DD^n`
//! about 50 for V1–V3 and 65 more for V4–V7 (one 2Prod, four 2Sum and the
//! tail per pair, two pairs per entry), and two gather buffers of the
//! length of the longest row and one of four times that. Measured (task
//! X0.5, M2 Pro under load, eight polished Netlib systems of 2 392 to
//! 18 301 rows): [`residual_of_dd`] takes 3.0 to 4.5 times as long as V1–V3
//! alone (whose Dot2Err is vectorized; V4–V7 are scalar), 0.4 to 3.4 ms — at most 5 % of the time of
//! Algorithm E on the same bases, except where the basis is far smaller
//! than the system (fit2d: `m = 25`, Algorithm E under 1 ms).
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
//! ## Algorithm E (a rigorous bound of the error, by an implicit approximate inverse)
//! Input: a square `A ∈ F^{n×n}` (compressed rows; one compressed-column
//! copy is made), the enclosure `[r]` of the residual `A x̃ − b` of an
//! approximate solution `x̃` — `n` intervals, from [`residual`] (`x̃ ∈ F^n`)
//! or [`residual_of_dd`] (`x̃ ∈ DD^n`) — a *transposed solve* `c ↦ ≈ A⁻ᵀc`
//! in `f64` (a closure: the BTRAN of the caller's LU) and the switch
//! `refine_rows` of [`ForwardErrorOptions`] (default on). Output
//! ([`ForwardErrorBound`]): `ε`, the `ε_p`, `α`, the number of solves and
//! the outcome.
//!
//! *Notation.* `e_p` is the `p`-th unit vector and `δ_pq` the Kronecker
//! delta; `a_iq` are the stored entries and `a_{:q}` the column `q`. A
//! vector `y` of row `p` has components `y_i = h_i + ℓ_i` (two `f64`, the
//! exact sum; `ℓ = 0` without refinement). `R` is the matrix whose row `p`
//! is the `y` that step E3 **actually used**, taken as exact real numbers:
//! `R` is not "the inverse", it is whatever came out, and the theorem
//! holds for every `R`. `C = I − RA` (exact, real), `x* = A⁻¹b`, `e = x̃ −
//! x*`. `mag(X) = max{|x| : x ∈ X}`.
//!
//! E0. \[Dimensions; trivial cases.\] If `A` is not square or `[r]` does not
//!     have `n` entries: [`VerifyError::Dimension`], before any arithmetic.
//!     If `n = 0`: certified, `ε = 0`, `α = 0`, no solve. If some `[r_i]`
//!     is empty: not certified (*not finite*), no solve.
//! E1. \[Row of `R`.\] For `p = 1, …, n`: `h ← e_p`, `ℓ ← 0`;
//!     `solve_transpose(h)`. If some `h_i` is not finite: not certified
//!     (*not finite*), stop.
//! E2. \[Refinement, optional.\] If `refine_rows`: with `Q_p` of step E3
//!     for this `y`, `d_q ← RN(δ_pq − Dot2(a_{:q}, h))` for `q ∈ Q_p` and
//!     `d_q ← 0` elsewhere (`Dot2` of Chapter 11.5, *not* rigorous: it only
//!     chooses `R`); `solve_transpose(d)`; if every `d_i` and every
//!     `fl(h_i + d_i)` is finite, `(h_i, ℓ_i) ← h_i ⊕ d_i` (the addition
//!     `dd + f64` of Chapter 11.2, A1: the exact sum `h_i + d_i` unless
//!     `|h_i| = Ω`, where step E may return `(fl(h_i + d_i), 0)` — either
//!     way a pair of finite `f64`); otherwise `y` stays the one of E1.
//!     Nothing of this step enters the proof: `R` is the pairs that came
//!     out, exact sums or not.
//! E3. \[Row of `C`.\] `S_p = {i : h_i ≠ 0 or ℓ_i ≠ 0}`, `Q_p = {q : a_iq is
//!     stored for some i ∈ S_p} ∪ {p}`. For each `q ∈ Q_p`, gather the
//!     pairs `(a_iq, h_i)` for the stored `a_iq`, then — if step E2 set
//!     `ℓ` — the pairs `(a_iq, ℓ_i)`, and `(−1, δ_pq)`; `(s, β) ←
//!     Dot2Err`; `M_q ← mag(V3(s, β))`. `ρ_p ← sup(Σ_{q∈Q_p} M_q)`, the sum
//!     in `Interval`.
//! E4. \[Row of `R[r]`.\] `G ← Σ_{i∈S_p} ([r_i]·h_i + [r_i]·ℓ_i)` in
//!     `Interval` (a term with a zero factor is omitted); `g_p ← mag(G)`.
//! E5. \[Early stop.\] `α ← max(α, ρ_p)` (exact). If not `α < 1`: not
//!     certified (*not contracting*), stop; the `α` reported is that of the
//!     rows examined.
//! E6. \[Norm.\] After the `n` rows: `γ ← max_p g_p`; if `γ = +∞`: *not
//!     finite*. `ε ← sup([γ, γ] / (1 − [α, α]))` in `Interval`; if `ε = +∞`:
//!     *not finite*.
//! E7. \[Components.\] `ε_p ← min(ε, sup([g_p, g_p] + [ρ_p, ρ_p]·ε))`, in
//!     `Interval`.
//!
//! **Theorem E.** If Algorithm E ends certified, `A` is nonsingular and,
//! for **every** `b ∈ ℝⁿ` and `x̃ ∈ ℝⁿ` with `A x̃ − b ∈ [r]` componentwise
//! — in use, the `b` and `x̃` whose residual [`residual`] enclosed — `x* =
//! A⁻¹b` satisfies `‖x̃ − x*‖∞ ≤ ε` and `|x̃_p − x*_p| ≤ ε_p` for `p = 1, …,
//! n`.
//!
//! *Proof.* (o) *`R` is a real matrix:* every `h_i` and `ℓ_i` used is a
//! finite `f64` — E1 refuses a non-finite `h`, E2 keeps a correction only
//! when every `d_i` and `fl(h_i + d_i)` is finite, and the addition A1
//! never leaves a non-finite component then (step E of Chapter 11.2). (i) *Each `M_q` dominates `|c_pq|`.* As real numbers, `Σ_i a_iq
//! (h_i + ℓ_i) − δ_pq = Σ_i a_iq h_i + Σ_i a_iq ℓ_i + (−1) δ_pq` (when E2
//! did not set `ℓ`, `ℓ = 0` and the second sum is omitted) is exactly the
//! dot product of the pairs of E3, and it equals `−c_pq` (`c_pq = δ_pq −
//! Σ_i y_i a_iq`). By (K2) of Chapter 11.5, `|−c_pq − s| ≤ β`, with `β =
//! +∞` when a hypothesis of Dot2Err fails (a length with `2(k + 3)u ≥ 1`,
//! an overflow — of an intermediate step at `±Ω` too, since X0.3i — a NaN) — then
//! V3 is `ENTIRE`, `M_q = ρ_p = +∞` and E5 refuses. By the proof of (V1)
//! the interval V3 contains `−c_pq`, hence `|c_pq| ≤ M_q`.
//! (ii) *The columns outside `Q_p` have `c_pq = 0` exactly:* for `q ∉ Q_p`,
//! `q ≠ p`, and every product `a_iq y_i` has a zero factor (either `a_iq`
//! is not stored or `i ∉ S_p`, i.e. `y_i = h_i + ℓ_i = 0`).
//! (iii) *`ρ_p ≥ Σ_q |c_pq|`:* a sum of nonnegative reals each dominated by
//! an `f64`, computed in `Interval` (by (I1) its `sup` dominates the exact
//! sum). Hence `‖C‖∞ = max_p Σ_q |c_pq| ≤ α < 1`.
//! (iv) *Nonsingularity.* `RA = I − C` with `‖C‖∞ < 1` is invertible
//! (Neumann series), so `A` and `R` are invertible; `x*` exists and is
//! unique.
//! (v) *`g_p ≥ |(R r)_p|` for every `r ∈ [r]`:* `(R r)_p = Σ_i (h_i + ℓ_i)
//! r_i = Σ_i h_i r_i + Σ_i ℓ_i r_i`; each term lies in `[r_i]·h_i` or
//! `[r_i]·ℓ_i` (I1, with a point operand), the omitted terms (`i ∉ S_p`, or
//! a zero factor) are zero, and the sum lies in the interval sum (I1).
//! (vi) *The bound.* `A e = A x̃ − b = r ∈ [r]`; multiplying by `R`, `(I −
//! C) e = R r`, i.e. `e = R r + C e`. So `‖e‖∞ ≤ ‖R r‖∞ + α ‖e‖∞`, and since
//! `α < 1`, `‖e‖∞ ≤ ‖R r‖∞/(1 − α) ≤ γ/(1 − α) ≤ ε` — the quotient in
//! `Interval` (I1, I3), with `▽(1 − α) > 0` because `α < 1` is an `f64`.
//! Componentwise, `|e_p| ≤ |(R r)_p| + Σ_q |c_pq| |e_q| ≤ g_p + ρ_p ‖e‖∞ ≤
//! g_p + ρ_p ε`, whose `sup` in `Interval` is an upper bound; and `|e_p| ≤
//! ‖e‖∞ ≤ ε`, so the minimum of the two bounds is one. ∎
//!
//! Nothing in the proof depends on how `y` was obtained: a wrong, noisy or
//! adversarial `solve_transpose` can only make `α ≥ 1` (no certificate) or
//! the bound loose, never false (invariant E1). For an exactly singular `A`
//! there is no `R` with `‖I − RA‖∞ < 1` (it would make `A` invertible), so
//! Algorithm E cannot certify it (invariant E2).
//!
//! **Where each rounding is.** This is where textbook versions of the bound
//! fail:
//!
//! | quantity | computed as | enclosed by |
//! |---|---|---|
//! | `R` | `f64` (E1) or double-double (E2), no bound | nothing needed: `R` *is* the numbers that came out |
//! | `c_pq` | Dot2Err on the exact pairs of E3 | (K2) and V3: `\|c_pq\| ≤ M_q`, underflow included in `β` |
//! | `c_pq`, `q ∉ Q_p` | not computed | exactly zero (proof, ii) |
//! | `ρ_p = Σ M_q` | `Interval` | (I1), outward, exact under underflow (I3) |
//! | `α = max ρ_p` | `f64::max` | exact |
//! | `r = A x̃ − b` | [`residual`], [`residual_of_dd`] | (V1) |
//! | `(R r)_p` | `Interval`, interval × point, `h` and `ℓ` apart | (I1) |
//! | `γ/(1 − α)`, `g_p + ρ_p ε` | `Interval` | (I1), (I3) |
//!
//! The errors this avoids: computing `I − RA` in floating point and taking
//! the norm of the result (the rounding of `RA` is of the order of `C`
//! itself); using `‖R‖ ‖r‖` with `r` recomputed in `f64` (noise of the
//! size of the residual itself); forgetting the underflow of the products
//! (here it is in `β`); and accumulating the products of a double-double
//! `R` in `f64`, whose error `≈ n u Σ|R||A|` undoes the refinement — hence
//! `h` and `ℓ` enter Dot2Err as separate pairs, the device of (V1) for a
//! double-double `x`.
//!
//! **How tight, and why.** For an `f64` solution the error *is* `R r` up to
//! `C e`, and `C` is small: in the measurements of the contract (task X0.3,
//! decision log D-0125) `ε/‖e‖∞` was between 1.00 and 1.04 on every
//! certified case with `α ≤ 0.04`, and 1.00 on all 93 optimal bases of
//! Netlib. For a polished solution the same holds since steps V4–V7 of
//! Algorithm V (task X0.5): before them its exact residual was of the
//! order of the width of its Dot2Err enclosure ("Width" above), `g_p`
//! reflected `|R| rad([r])`, and the overestimation `ε̂/‖e‖∞` went from
//! 1.4 to `7.7 · 10⁴` (median about 230 over the 88 bases with a nonzero
//! error). Measured term by term on those bases, nothing else contributed:
//! `α ≤ 1.2 · 10⁻⁷`, so `1/(1 − α)` and `ρ_p ε` are invisible at three
//! digits, and with `[r]` replaced by a tight enclosure of the *exact*
//! residual (computed in rationals) the ratio was already 1.000 on every
//! basis — `|R| |r|` does not overestimate `|R r|` there. With V4–V7 the
//! ratio is 1.000 at three digits on all 88 too (the true error bracketed
//! rigorously, by Algorithm E itself on `A e = A x̂ − b` with the exact
//! residual), and at most `1 + 5 · 10⁻⁶` on the dense test matrices of
//! `κ ≤ 10¹⁴` (before: up to 572). One solve per row
//! leaves `α` of the order of `κ(A) u`; the refinement E2 lowers it by
//! orders of magnitude (`7.5 · 10⁻³ → 3.1 · 10⁻⁶` at `κ = 10¹⁴`, order
//! 20): measured, it certifies dense matrices up to `κ ≈ 10¹⁶`, where one
//! solve per row stops near `10¹⁴`.
//!
//! **A basis of a linear program.** The right-hand side `b = −K_N β_N` of
//! a basis is not made of `f64` values. Pass `A = B` and, as `[r]`, the
//! enclosure of the first `m` rows of the square system of order `n + m`
//! of `xingo-simplex`'s `residual_certificates` test (`K z = 0`, `z_v =
//! β_v` for the nonbasic `v`), evaluated with the nonbasic variables set
//! **exactly** to their bounds: those rows are then `B z_B − b` exactly,
//! and Theorem E holds for the exact real `b`.
//!
//! **Cost.** `n` calls of `solve_transpose` (`2n` with `refine_rows`), plus
//! `Σ_p Σ_{q∈Q_p} (nnz(a_{:q}) + 1)` pairs of Dot2Err (about 25 flops each;
//! up to twice as many with refinement) — `n · nnz(A)` at worst — and an
//! `O(n)` scan per row for `S_p`. Memory `O(n + nnz(A))`: one
//! compressed-column copy, a few vectors of length `n`, buffers of the
//! length of the longest column. Measured (X0.3, a loaded machine): 10.8 s
//! summed over the 93 optimal bases of Netlib with refinement, the largest
//! (dfl001, `m = 6 071`) 3.7 s; on a sparse synthetic family `n = 10⁴` in
//! 21 s and `3 · 10⁴` in 271 s — quadratic in `n`, because each solve and
//! each scan is `O(n)` with dense vectors.
//!
//! **What stays out.** Bounds in the 2-norm (by `σ_min`; rejected by the
//! measurements of X0.3); an explicit `R`; a componentwise criterion with a
//! vector `v ≠ 𝟙` (Perron, H-matrices: another pass of `n` solves);
//! Krawczyk iteration; parallel rows; a bound for a subset of the
//! components (`α` is global); interval `A` or `b`. The transposed system
//! needs no function of its own: call again with `Aᵀ` and the FTRAN as the
//! solve. Bounds through the LU factors with comparison matrices are
//! Algorithm E′ (the next chapter), now that `xingo-sparse` exports its
//! factors (`Lu::triangular_factors`).
//!
//! ## Invariants
//! (V1) *Residuals.* [`residual`]`(A, x, b)[i] ∋ a_iᵀx − b_i`, and likewise
//!      [`residual_of_dd`], [`residual_csc`] and [`activity`] (`∋ a_iᵀx`),
//!      for every row and every input; an unbounded interval when a
//!      hypothesis fails (C3). Checked against exact rationals.
//! (V2) *Well-formed.* Every interval returned satisfies (I2) of Chapter
//!      11.9 and is not empty (a `debug_assert!`).
//! (V3) *Never wider.* Each row of [`residual_of_dd`] is contained in the
//!      enclosure of steps V1–V3 for the same pairs (by construction, the
//!      intersection of V7; a `debug_assert!`; an empty intersection, which
//!      only a defect could produce, falls back to the enclosure of V3).
//!      Checked against exact rationals, with the width `≤ 8u|s*| +
//!      2⁻¹²⁸S + 2⁻¹⁰⁰⁰`, and on a row built so that the tail bound `β′`
//!      is needed to within a factor 1.7 (`tests/data/tail_bound_row.txt`).
//! (P1) *Monotone.* [`PolishReport::residual_after`] `≤`
//!      [`PolishReport::residual_before`] whenever the latter is finite (a
//!      `debug_assert!`).
//! (P2) *Untouched on failure.* With [`PolishStop::NotFinite`] and zero
//!      iterations, `x` is the input.
//! (E1) *Never false.* [`ForwardErrorOutcome::Certified`] implies Theorem
//!      E for the stored data, whatever the closure (a wrong one can only
//!      prevent the certificate). Checked against exact rationals, with
//!      adversarial closures.
//! (E2) *Nonsingular.* Certified implies `A` nonsingular; an exactly
//!      singular `A` is never certified (checked).
//! (E3) *Shape.* Certified ⇔ `ε` finite ⇔ every `ε_p` finite; then `0 ≤
//!      ε_p ≤ ε` and `0 ≤ α < 1`. Not certified ⇒ `ε = +∞` and every `ε_p =
//!      +∞` (a `debug_assert!`).
//! (E4) *Deterministic.* The same bits for the same data and the same
//!      closure (K1, D5), if the closure is a function of its argument.
//!      The BTRAN of `xingo_sparse::Lu` is not, by design: it chooses
//!      between its hypersparse and its dense path by a moving average of
//!      the density of the previous results, so two equal calls can sum in
//!      different orders (on one Netlib basis, share1b, they returned other
//!      bits); the same sequence of calls gives the same bits. Each result
//!      is a theorem for its own `R` either way.
//!
//! ## References
//! Ogita, Rump & Oishi, "Accurate sum and dot product", SIAM J. Sci.
//! Comput. 26 (2005), Algorithm 5.8 and Corollary 5.9 (through Chapter
//! 11.5, which adapts and proves them), and VecSum and SumK (their
//! Algorithms 4.3 and 4.8, Lemma 4.2), the model of steps V4–V7 — **cited
//! from memory**; the a posteriori bound of V6 and its proof are this
//! chapter's; Higham, *Accuracy and Stability of
//! Numerical Algorithms*, 2nd ed. (2002), ch. 12 (iterative refinement;
//! §12.1 for the mixed-precision form) — **cited from memory**, the
//! statement above is the qualitative one and nothing in the code depends
//! on its constants; Rump, "Verification methods: rigorous results using
//! floating-point arithmetic", Acta Numerica 19 (2010) — the distinction
//! between a verified residual and a verified solution, and (§10) the bound
//! `‖I − RA‖ ≤ α < 1 ⇒ ‖x̃ − x*‖ ≤ ‖R(A x̃ − b)‖/(1 − α)` for an
//! approximate inverse `R` that Algorithm E applies one implicit row at a
//! time (*cited from memory, not reread*; the constants and the proof of
//! Theorem E are this chapter's, from (K2), (V1) and (I1)–(I3) only).
//! Before release the proof and the code of Algorithm E were reviewed by an
//! independent reviewer (task X0.3, 262 000 adversarial cases of order at
//! most 8 checked in exact arithmetic): no false bound; the review found
//! the corner of (K2) fixed in Chapter 11.5 and the textual corrections of
//! steps E2 and (o) above.

use xingo_sparse::{Col, CscMatrix, CsrMatrix, Row};

use crate::dd::DoubleDouble;
use crate::eft::{two_prod, two_sum};
use crate::interval::Interval;
use crate::kernels;

mod lu;
pub use lu::{FactorBoundOptions, forward_error_bound_lu};

/// What went wrong before any arithmetic. (`Display` and
/// `core::error::Error` are implemented.)
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerifyError {
    /// `x` or `b` has the wrong length for `A` (for [`polish`], also: `A`
    /// is not square — `expected` is its number of rows, `got` its number
    /// of columns; for [`forward_error_bound`], also: `A` not square or `r`
    /// of the wrong length).
    Dimension {
        /// What was expected.
        expected: usize,
        /// What was given.
        got: usize,
    },
    /// The factors given to [`forward_error_bound_lu`] violate their
    /// invariants (T1) (`xingo_sparse::TriangularFactors::check_invariants`):
    /// the message of that check. *Provisional name (task I06).*
    InvalidFactors(String),
}

impl core::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Dimension { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidFactors(why) => write!(f, "invalid triangular factors: {why}"),
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

/// One pass of VecSum (step V5): replaces `w` by the errors of the cascade
/// of 2Sum over it (one shorter) and returns the sum; `None` if a value
/// produced is not finite (then nothing is claimed: (T1)).
fn vec_sum(w: &mut Vec<f64>) -> Option<f64> {
    let Some(&first) = w.first() else {
        return Some(0.0);
    };
    let mut sigma = first;
    for j in 1..w.len() {
        let (s, e) = two_sum(sigma, w[j]);
        if !(s.is_finite() && e.is_finite()) {
            return None;
        }
        w[j - 1] = e;
        sigma = s;
    }
    w.pop();
    Some(sigma)
}

/// `φ = (3η/u)(1 + 8u)`, the underflow term of step V6: the floor of `β` in
/// step C5 of Chapter 11.5, exactly (the product is an `f64`).
const FLOOR: f64 = 6.0 * f64::MIN_POSITIVE * (1.0 + 4.0 * f64::EPSILON);

/// Steps V4–V7 for the pairs `(p, q)`: `R′ ∋ pᵀq`, or `None` when a step
/// keeps the enclosure of V3. `w` is a buffer.
fn distilled(p: &[f64], q: &[f64], w: &mut Vec<f64>) -> Option<Interval> {
    /// `2(m u) < 1` iff `m < 2⁵²`.
    // `u64`, not `usize`: `2⁵²` does not fit a 32-bit `usize` (i686, wasm32).
    const MAX_TERMS: u64 = 1 << 52;
    // V4.
    w.clear();
    for (&pj, &qj) in p.iter().zip(q) {
        let (pi, eps) = two_prod(pj, qj);
        if !(pi.is_finite() && eps.is_finite()) {
            return None;
        }
        w.push(pi);
        w.push(eps);
    }
    // V5.
    let sigma_1 = vec_sum(w)?;
    let sigma_2 = vec_sum(w)?;
    // V6: `t` and `a` left to right, as the chapter states.
    let (mut t, mut a) = (0.0_f64, 0.0_f64);
    for &v in w.iter() {
        t += v;
        a += v.abs();
    }
    let m = w.len().max(1) - 1;
    if u64::try_from(m).map_or(true, |m| m >= MAX_TERMS) || !(t.is_finite() && a.is_finite()) {
        return None;
    }
    #[allow(clippy::cast_precision_loss)] // exact: m < 2⁵²
    let mu = m as f64 * U;
    let delta = Interval::point(mu) / (1.0 - Interval::point(2.0 * mu));
    let beta = upper((delta * Interval::point(a) + Interval::point(FLOOR)).sup());
    if !beta.is_finite() {
        return None;
    }
    // V7.
    let (s, tau) = two_sum(sigma_1, sigma_2);
    if !(s.is_finite() && tau.is_finite()) {
        return None;
    }
    Some(Interval::mid_rad(t, beta) + tau + s)
}

/// Algorithm V over the rows of `a`: `fill(i, cols, vals, p, q)` appends
/// to the empty `(p, q)` the pairs of row `i` (step V1); with `distil`,
/// steps V4–V7 narrow each row.
fn rows(
    a: &CsrMatrix<f64>,
    distil: bool,
    mut fill: impl FnMut(usize, &[u32], &[f64], &mut Vec<f64>, &mut Vec<f64>),
) -> Vec<Interval> {
    let (mut p, mut q, mut w) = (Vec::new(), Vec::new(), Vec::new());
    (0..a.n_rows())
        .map(|i| {
            p.clear();
            q.clear();
            let (cols, vals) = a.row(Row::from_usize(i));
            fill(i, cols, vals, &mut p, &mut q); // V1
            let (r, beta) = kernels::dot2_err(&p, &q); // V2
            let x = enclosure(r, beta); // V3
            if !distil {
                return x;
            }
            match distilled(&p, &q, &mut w) {
                None => x,
                Some(narrow) => {
                    let y = x.intersection(narrow);
                    debug_assert!(
                        !y.is_empty() && y.is_subset_of(x),
                        "V1/V3: {x:?} ∩ {narrow:?} in row {i}"
                    );
                    // Both contain the residual, so the intersection is never
                    // empty; should a defect of V4–V7 make it so, a release
                    // build falls back to the enclosure of V3 (V2).
                    if y.is_empty() { x } else { y }
                }
            }
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
    Ok(rows(a, false, |i, cols, vals, p, q| {
        p.extend_from_slice(vals);
        q.extend(cols.iter().map(|&j| x[j as usize]));
        p.push(-1.0);
        q.push(b[i]);
    }))
}

/// Algorithm V for a double-double (polished) solution: `R_i ∋ a_iᵀx −
/// b_i` for the exact values `hi + lo` of the components of `x`. Steps
/// V4–V7 of Chapter 11.10 distil the residual as if it were summed in
/// three times the working precision, so that the width follows a residual
/// of double-double size: a few ulps of it plus `≈ 8N³u³ Σ_j |a_ij x_j|`
/// for `N = 2k + 1` pairs, never wider than the Dot2Err enclosure of steps
/// V1–V3 (invariant V3).
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
    // V1: the leading terms first, then the trailing ones; V4–V7.
    Ok(rows(a, true, |i, cols, vals, p, q| {
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
    Ok(rows(a, false, |_, cols, vals, p, q| {
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
#[non_exhaustive]
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
#[non_exhaustive]
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

/// The parameters of Algorithm E.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForwardErrorOptions {
    /// Refine each row of the implicit inverse once in double-double
    /// (step E2): up to twice the solves (default `true`). Measured on dense
    /// matrices, it certified up to `κ(A) ≈ 1e16` where one solve per row
    /// stopped near `1e14` — a measurement, not a guarantee: whether a
    /// matrix is certified depends on the solve and the data, not on `κ`
    /// alone.
    pub refine_rows: bool,
}

impl Default for ForwardErrorOptions {
    fn default() -> Self {
        Self { refine_rows: true }
    }
}

/// Why Algorithm E stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ForwardErrorOutcome {
    /// The bounds are a theorem (Chapter 11.10, Theorem E).
    Certified,
    /// `‖I − RA‖∞ < 1` could not be shown: `A` singular or too
    /// ill-conditioned for the solve, or the solve is not one of `Aᵀ`.
    NotContracting,
    /// A solve, the residual enclosure or the bound was not finite.
    NotFinite,
}

/// The outcome of Algorithm E.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ForwardErrorBound {
    /// Why it stopped.
    pub outcome: ForwardErrorOutcome,
    /// `ε ≥ ‖x − x*‖∞` (`+∞` unless certified).
    pub norm_inf: f64,
    /// `ε_p ≥ |x_p − x*_p|`, `0 ≤ ε_p ≤ ε` (all `+∞` unless certified).
    pub componentwise: Vec<f64>,
    /// `α ≥ ‖I − RA‖∞` over the rows examined: `< 1` when certified;
    /// `≥ 1` when [`ForwardErrorOutcome::NotContracting`]; with
    /// [`ForwardErrorOutcome::NotFinite`], the value reached when it
    /// stopped (`0` if no row was examined).
    pub contraction: f64,
    /// Calls of `solve_transpose`.
    pub solves: usize,
}

impl ForwardErrorBound {
    /// The outcome is [`ForwardErrorOutcome::Certified`].
    pub fn is_certified(&self) -> bool {
        self.outcome == ForwardErrorOutcome::Certified
    }

    /// No certificate: every bound `+∞` (invariant E3).
    fn refused(outcome: ForwardErrorOutcome, n: usize, contraction: f64, solves: usize) -> Self {
        Self {
            outcome,
            norm_inf: f64::INFINITY,
            componentwise: vec![f64::INFINITY; n],
            contraction,
            solves,
        }
    }

    /// Invariant E3.
    fn check_invariants(&self) -> Result<(), &'static str> {
        let finite = self.norm_inf.is_finite();
        if self.is_certified() {
            if !(finite && self.norm_inf >= 0.0) {
                return Err("E3: a certificate with an infinite or negative ε");
            }
            if !(0.0..1.0).contains(&self.contraction) {
                return Err("E3: a certificate with α outside [0, 1)");
            }
            if !self
                .componentwise
                .iter()
                .all(|&c| c >= 0.0 && c <= self.norm_inf)
            {
                return Err("E3: some ε_p outside [0, ε]");
            }
        } else if finite || self.componentwise.iter().any(|c| c.is_finite()) {
            return Err("E3: a finite bound without a certificate");
        }
        Ok(())
    }
}

/// An upper bound as computed, with a NaN read as `+∞`: it proves nothing
/// (invariant C3 of Chapter 11), and `f64::max` and `f64::min` would drop
/// it.
fn upper(x: f64) -> f64 {
    if x.is_nan() { f64::INFINITY } else { x }
}

/// `δ_pq`.
fn kronecker(p: usize, q: usize) -> f64 {
    if p == q { 1.0 } else { 0.0 }
}

/// Step E3, the sets: `support ← S_p` (the `i` with `h_i ≠ 0` or `ℓ_i ≠
/// 0`) and `cols ← Q_p` (the columns stored in the rows of `S_p`, and
/// `p`), each without repetition; `mark` is all `false` on entry and on
/// exit.
fn row_sets(
    a: &CsrMatrix<f64>,
    p: usize,
    (h, l): (&[f64], &[f64]),
    mark: &mut [bool],
    support: &mut Vec<usize>,
    cols: &mut Vec<usize>,
) {
    support.clear();
    cols.clear();
    for (i, (&hi, &li)) in h.iter().zip(l).enumerate() {
        if hi != 0.0 || li != 0.0 {
            support.push(i);
            for &q in a.row(Row::from_usize(i)).0 {
                let q = q as usize;
                if !mark[q] {
                    mark[q] = true;
                    cols.push(q);
                }
            }
        }
    }
    if !mark[p] {
        mark[p] = true;
        cols.push(p);
    }
    for &q in cols.iter() {
        mark[q] = false;
    }
}

/// Algorithm E: from `r ∋ A x − b` (from [`residual`] or
/// [`residual_of_dd`]) and a transposed solve (`c ← ≈ A⁻ᵀ c`, `f64`),
/// proves that `A` is nonsingular and bounds `x − A⁻¹b` (Theorem E of
/// Chapter 11.10): if the result is certified, then for the `b` whose
/// residual `r` encloses — precisely, for every `x` and `b` with `A x − b ∈
/// r` componentwise — `‖x − A⁻¹b‖∞ ≤ norm_inf` and `|x_p − (A⁻¹b)_p| ≤
/// componentwise[p]`, for the stored `f64` values of `A`.
///
/// The bound is a theorem **whatever** `solve_transpose` returns: a wrong
/// solve can only prevent the certificate or loosen the bound. It is called
/// at most `n` times (at most `2n` with
/// [`ForwardErrorOptions::refine_rows`]; fewer when it stops early), each
/// time on a vector of length `n`.
///
/// ```
/// use xingo_extprec::verify::{
///     ForwardErrorOptions, forward_error_bound, residual,
/// };
/// use xingo_sparse::CsrMatrix;
///
/// // [[2, 1], [1, 3]] x = (1, 1): x* = (2/5, 1/5), not representable.
/// let a = CsrMatrix::from_raw_parts(2, vec![0, 2, 4], vec![0, 1, 0, 1],
///     vec![2.0, 1.0, 1.0, 3.0]).unwrap();
/// let (b, x) = ([1.0, 1.0], [0.4, 0.2]);
/// // Aᵀ = A: the transposed solve is the exact inverse, rounded.
/// let mut solve_t = |c: &mut [f64]| {
///     let (p, q) = (c[0], c[1]);
///     c[0] = (3.0 * p - q) / 5.0;
///     c[1] = (2.0 * q - p) / 5.0;
/// };
/// let r = residual(&a, &x, &b).unwrap();
/// let e = forward_error_bound(&a, &r, &mut solve_t, &ForwardErrorOptions::default())
///     .unwrap();
/// assert!(e.is_certified() && e.contraction < 1e-15);
/// // |0.4 − 2/5| and |0.2 − 1/5| are below 2⁻⁵⁴ and 2⁻⁵⁵.
/// assert!(e.norm_inf < 1e-16 && e.componentwise[1] < 1e-16);
/// ```
///
/// # Errors
/// [`VerifyError::Dimension`] if `a` is not square or `r` does not have
/// one entry per row.
///
/// # Panics
/// If `a` violates the invariants of its type.
pub fn forward_error_bound(
    a: &CsrMatrix<f64>,
    r: &[Interval],
    solve_transpose: &mut dyn FnMut(&mut [f64]),
    opts: &ForwardErrorOptions,
) -> Result<ForwardErrorBound, VerifyError> {
    use ForwardErrorOutcome::{Certified, NotContracting, NotFinite};
    // E0.
    dimension(a.n_rows(), a.n_cols())?;
    let n = a.n_rows();
    dimension(n, r.len())?;
    if r.iter().any(|x| x.is_empty()) {
        return Ok(ForwardErrorBound::refused(NotFinite, n, 0.0, 0));
    }
    let bound =
        forward_error_rows(a, r, solve_transpose, opts.refine_rows).unwrap_or_else(|refusal| {
            let (outcome, alpha, solves) = refusal;
            ForwardErrorBound::refused(outcome, n, alpha, solves)
        });
    debug_assert!(
        bound.check_invariants().is_ok(),
        "{:?}: {bound:?}",
        bound.check_invariants()
    );
    debug_assert!(n > 0 || bound.outcome == Certified);
    debug_assert!(bound.outcome != NotContracting || bound.contraction >= 1.0);
    Ok(bound)
}

/// Steps E1–E7 for a square `a` and a nonempty `[r]` of the right length;
/// `Err((outcome, α, solves))` when there is no certificate.
// Every upper bound goes through `upper`, so a NaN — none arises today:
// the operations of `Interval` on non-empty operands return none — would
// count as `+∞` and refuse; `!(α < 1)` refuses one too.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn forward_error_rows(
    a: &CsrMatrix<f64>,
    r: &[Interval],
    solve_transpose: &mut dyn FnMut(&mut [f64]),
    refine_rows: bool,
) -> Result<ForwardErrorBound, (ForwardErrorOutcome, f64, usize)> {
    use ForwardErrorOutcome::{Certified, NotContracting, NotFinite};
    let n = a.n_rows();
    let csc = a.to_csc();
    // `y = h + ℓ`, the row of `R`; `d` the correction of step E2.
    let (mut h, mut l, mut d) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut mark = vec![false; n];
    let (mut support, mut cols) = (Vec::new(), Vec::new());
    let (mut pp, mut qq) = (Vec::new(), Vec::new());
    let (mut rho, mut g) = (vec![0.0; n], vec![0.0; n]);
    let mut alpha = 0.0_f64;
    let mut solves = 0_usize;
    for p in 0..n {
        // E1.
        h.fill(0.0);
        l.fill(0.0);
        h[p] = 1.0;
        solve_transpose(&mut h);
        solves += 1;
        if !h.iter().all(|v| v.is_finite()) {
            return Err((NotFinite, alpha, solves));
        }
        // E2: only chooses `R`; nothing here enters the proof. `refined`:
        // the trailing components `ℓ` were set (otherwise all zero).
        let mut refined = false;
        if refine_rows {
            row_sets(a, p, (&h, &l), &mut mark, &mut support, &mut cols);
            d.fill(0.0);
            for &q in &cols {
                let (rows, vals) = csc.col(Col::from_usize(q));
                qq.clear();
                qq.extend(rows.iter().map(|&i| h[i as usize]));
                d[q] = (DoubleDouble::from_f64(kronecker(p, q)) - kernels::dot2(vals, &qq)).hi();
            }
            solve_transpose(&mut d);
            solves += 1;
            // (Q2 of X0.3i: also discard a correction whose sum overflows.)
            if d.iter()
                .zip(&h)
                .all(|(&di, &hi)| di.is_finite() && (hi + di).is_finite())
            {
                refined = true;
                for ((hi, li), &di) in h.iter_mut().zip(l.iter_mut()).zip(&d) {
                    let s = DoubleDouble::from_f64(*hi) + di;
                    (*hi, *li) = (s.hi(), s.lo());
                }
            }
        }
        // Proof of Theorem E, (o): `R` is real.
        debug_assert!(
            h.iter().chain(&l).all(|v| v.is_finite()),
            "(o): a non-finite component in row {p} of R"
        );
        // E3.
        row_sets(a, p, (&h, &l), &mut mark, &mut support, &mut cols);
        let mut sum = Interval::ZERO;
        for &q in &cols {
            let (rows, vals) = csc.col(Col::from_usize(q));
            pp.clear();
            qq.clear();
            pp.extend_from_slice(vals);
            qq.extend(rows.iter().map(|&i| h[i as usize]));
            if refined {
                pp.extend_from_slice(vals);
                qq.extend(rows.iter().map(|&i| l[i as usize]));
            }
            pp.push(-1.0);
            qq.push(kronecker(p, q));
            let (s, beta) = kernels::dot2_err(&pp, &qq);
            sum += enclosure(s, beta).mag();
        }
        rho[p] = upper(sum.sup());
        // E4.
        let mut acc = Interval::ZERO;
        for &i in &support {
            for y in [h[i], l[i]] {
                if y != 0.0 {
                    acc += r[i] * y;
                }
            }
        }
        g[p] = upper(acc.mag());
        // E5.
        alpha = alpha.max(rho[p]);
        if !(alpha < 1.0) {
            return Err((NotContracting, alpha, solves));
        }
    }
    // E6.
    let gamma = g.iter().fold(0.0_f64, |m, &v| m.max(v));
    if !gamma.is_finite() {
        return Err((NotFinite, alpha, solves));
    }
    let eps = upper((Interval::point(gamma) / (1.0 - Interval::point(alpha))).sup());
    if !eps.is_finite() {
        return Err((NotFinite, alpha, solves));
    }
    // E7.
    let componentwise = g
        .iter()
        .zip(&rho)
        .map(|(&gp, &rp)| upper((Interval::point(gp) + Interval::point(rp) * eps).sup()).min(eps))
        .collect();
    Ok(ForwardErrorBound {
        outcome: Certified,
        norm_inf: eps,
        componentwise,
        contraction: alpha,
        solves,
    })
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // L2.4 (Chapter 11.11): the tests may use the libm of `f64` (clippy.toml)
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
