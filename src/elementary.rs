//! # Chapter 11.11 — Elementary functions of double-doubles
//!
//! ## Purpose
//! The elementary functions of [`DoubleDouble`] (task L2.4; contract
//! ratified as D-0121): the exponentials, logarithms and powers, the
//! circular functions and their inverses, the hyperbolic functions and
//! their inverses — the names and special cases of `f64` — and the 19
//! constants of `core::f64::consts`. Each function has a declared error
//! bound, a row of [`DoubleDouble::FUNCTION_BOUNDS`], and each bound is
//! **proven** below: the four kernels (exponential, logarithm, sine and
//! cosine, arctangent) directly, the other functions by composition. No
//! bound was fitted to a measurement; the tests measure against two
//! independent oracles and report the largest errors seen.
//!
//! ## Notation
//! As in Chapters 11.1 and 11.2: `RN` is rounding to nearest even, `u =
//! 2⁻⁵³`, `u² = 2⁻¹⁰⁶`, `η = 2⁻¹⁰⁷⁴`, `Ω` the largest finite `f64`.
//! `MAX_dd = (Ω, ℓ)`, with `ℓ` the largest `f64` below `ulp(Ω)/2`, is the
//! largest finite double-double. The operations A1 (`dd + f64`), A2 (`dd +
//! dd`), M1 (`dd · f64`), M2 (`dd · dd`), M3 (`dd²`), D1 (`dd / f64`), D2
//! (`dd / dd`) and R (`√dd`) are those of Chapter 11.2, with their proven
//! relative bounds `2u²`, `3u²`, `2u²`, `4u²`, `3u²`, `3u²`, `15u²`,
//! `25/8 u²` (first-order constants; the exact expressions are in that
//! chapter). A *table entry* is the nearest double-double of its value,
//! with relative error at most `u²(1 + u)`. "`x ∈ dd`" means a normalized
//! double-double (invariant D1).
//!
//! ## How the bounds are stated
//! A function `f` with declared constants `(c, c₂, G, A)` (one row of
//! [`DoubleDouble::FUNCTION_BOUNDS`]) returns, for every finite argument
//! whose result `f(x)` has magnitude in `[2⁻⁹⁶⁸, Ω]`, a value `z` with
//!
//! ```text
//! |z − f(x)| ≤ (c + c₂·G) u² |f(x)| + A,
//! ```
//!
//! and for results of smaller magnitude the same plus `η` (the trailing
//! component loses bits to the subnormal range; Chapter 11.2, "safe
//! range"). `G` is `|n|` in `powi`, `|y ln x|` in `powf`, `|tan x|` in
//! `tan` (where `c₂ = 2⁻⁹⁴` turns `A(1 + tan²x)` into this form) and absent
//! elsewhere; `A` is `2⁻²⁰⁰` for the circular functions and their pair
//! and `0` elsewhere. Nothing is correctly rounded.
//!
//! ## How the bounds are proved (the error calculus)
//! For a computed `ẑ` of an exact `z` write `ε(ẑ) ≤ c` for `|ẑ − z| ≤ c u²
//! |z|`, and work to first order in `u²`: every product of two such
//! errors is below `(64u²)² < 2⁻¹⁹⁹`, and a proof that sums at most a few
//! hundred of them at a constant of at most a few hundred `u²` is changed by
//! less than one part in `2⁸⁰`. Each proven constant `c*` is therefore
//! honest up to the factor `1 + 2⁻⁸⁰`, and every declared constant exceeds
//! its `c*` by more than that. The rules, all from Chapter 11.2:
//!
//! * **(C1) Sums.** If `ε(â) ≤ α`, `ε(b̂) ≤ β`, A2 returns `ŝ` with `ε(ŝ) ≤
//!   w_a α + w_b β + 3`, where `w_a = |a|/|a + b|`, `w_b = |b|/|a + b|` are
//!   the *weights* (both at most 1 when `a` and `b` have the same sign;
//!   their sum is the cancellation factor otherwise). With an exact `f64`
//!   `b`, A1 gives `w_a α + 2`.
//! * **(C2) Products and quotients.** M2: `α + β + 4`; M1 (exact `f64`
//!   factor): `α + 2`; M3: `2α + 3`; D2: `α + β + 15`; D1: `α + 3`;
//!   R: `α/2 + 25/8`.
//! * **(C3) Exact steps** add nothing: negation, `ldexp` (in the normal
//!   range), [`DoubleDouble::from_sum`], [`DoubleDouble::from_product`]
//!   in its range, Sterbenz subtractions `a − b` of `f64` with `b/2 ≤ a ≤
//!   2b`, and the operations whose result the text shows to be exact.
//! * **(C4) Arguments.** If `f` is evaluated at `â` instead of `a` by an
//!   algorithm with `ε ≤ φ` against `f(â)`, then against `f(a)` the bound is
//!   `φ + κ ε(â)`, where `κ` bounds the condition number `|a f′(a)/f(a)|`
//!   between `a` and `â`; or, in absolute terms, `|f(â) − f(a)| ≤ |â − a|
//!   max|f′|`.
//! * **(C5) A tiny addend (Lemma S).** If `b ∈ F` and `|b| ≤ 2⁻¹⁰ u |a_h|`,
//!   A1 returns `a + b` with absolute error at most `u²|a_h|(1 + 2⁻⁹)`.
//!   *Proof.* `|b| < ulp(a_h)/2`, so step 1 of A1 is `(s_h, s_ℓ) = (a_h,
//!   b)`, exactly; step 2 rounds `v = a_ℓ + b`, `|v| ≤ (u + 2⁻¹⁰u)|a_h|`,
//!   with error at most `u|v|`; step 3 is exact (Chapter 11.1, T1). ∎
//! * **(C6) Horner (Lemma H).** Let `Q = q₀`, `q_k = c_k + s q_{k+1}`
//!   (`k = M − 1, …, 0`), `q_M = c_M`, be evaluated with A2 and M2 from
//!   table coefficients (`ε ≤ σ_k`, `σ_k ∈ {0, 1}`) and a computed `ŝ`
//!   with `ε(ŝ) ≤ ε_s`, and suppose `|s q_{k+1}| ≤ ρ |q_k|` and `|c_k| ≤
//!   (1 + ρ)|q_k|` for every `k` (the terms of an alternating or decreasing
//!   series with ratio at most `ρ`). Then `ε(q_k) ≤ (1 + ρ)σ_k + ρ(ε_s + 4 +
//!   ε(q_{k+1})) + 3`, and by induction from `k = M`, `ε(q_k) ≤ E` for every
//!   `k ≥ 1` with `E = ((1 + ρ)σ̄ + ρ(ε_s + 4) + 3)/(1 − ρ)`, `σ̄ = max σ_k`;
//!   `ε(Q) ≤ (1 + ρ)σ₀ + ρ(ε_s + 4 + E) + 3`. ∎
//! * **(C7) Truncation.** A series cut after its `N`-th term is charged the
//!   bound of its remainder, relative to the value, as one more term.
//!
//! A product `two_prod(a, b)` with `|a b| < 2⁻⁹⁶⁹` is not exact (Chapter
//! 11.1, T2) and leaves an absolute error up to `η`; each proof says why
//! that does not matter where it can happen (an intermediate far below the
//! scale of the result, or a branch for tiny arguments that avoids it).
//!
//! ## Algorithm E (`exp`, `exp2`: Tang's table-driven exponential)
//! Input `x ∈ dd`, finite, `−745.2 ≤ x_h ≤ 709.79` (outside: `+0`, `+∞`,
//! NaN by the table of special cases). `ℓ = ln 2/64`; `C₁, C₂, C₃` its three
//! parts of 53 bits (`ℓ − C₁ − C₂ − C₃ < 2⁻¹⁶⁶`); `T_j` the table entry of
//! `2^(j/64)`.
//!
//! E1. [Index.] `n ← round(x_h · RN(64/ln 2))`, an integer with `|n| ≤
//!     2¹⁷`; `k ← ⌊n/64⌋`, `j ← n − 64k ∈ [0, 64)`.
//! E2. [Reduce.] `r ← ((x − n C₁) − n C₂) − n C₃` with each `n Cᵢ` an exact
//!     product (`from_product`) and each subtraction A2; if `n = 0`, `r ← x`.
//!     Then `x = n ℓ + r` up to the error `δ_r` below, and `|r| ≤ ρ_E =
//!     0.005 43` (`ℓ/2 = 0.005 415` plus the rounding of `n`).
//! E3. [Polynomial.] `p ← P(r)`, where `P(r) = r + r²Q(r)` and `Q(r) =
//!     Σ_{m=2}^{11} r^(m−2)/m!` by Horner (Lemma H); `P(r) ← r` when `|r_h| <
//!     2⁻⁴⁰⁰` (then `|eʳ − 1 − r| ≤ r² < 2⁻⁴⁰⁰|r|`).
//! E4. [Reconstruct.] `z ← (T_j + T_j·p) · 2ᵏ` (M2, A2, `ldexp`).
//!
//! `exp2` replaces E1–E2 by: `n ← round(64 x_h)`, `y ← (x_h − n/64) + x_ℓ`
//! (exact: `x_h − n/64` is a Sterbenz difference scaled by `2⁻⁶`, and
//! `from_sum` adds `x_ℓ` exactly), `r ← y · LN_2` (M2), `|y| ≤ 1/128`. For an
//! integer `x`, `y = r = p = 0` and the result is `1 · 2ᵏ`, exact for `k ∈
//! [−1074, 1023]`.
//!
//! **Theorem E.** For results in `[2⁻⁹⁶⁸, Ω]`, `ε(exp) ≤ 4.6` and
//! `ε(exp2) ≤ 4.6`; below, the bound plus `η`.
//!
//! *Proof.* (i) *Reduction.* The exact `r* = x − nℓ`. The first difference
//! has the exact value `r₁ = r* + n(C₂ + C₃) + nτ`, `|nC₂| ≤ 2¹⁷ · 2⁻⁶⁰`
//! (`C₂ ≤ ulp(C₁)/2`), so `|r₁| ≤ ρ_E + 2⁻⁴²`; each of the three A2 errs by
//! at most `3u²` times its result, so `δ_r = |r − r*| ≤ 3u²(3ρ_E + 2⁻⁴¹) +
//! 2¹⁷ · 2⁻¹⁶⁶ < 0.05u²`. As an absolute error of the argument of `eʳ` it is
//! a relative error of the result (C4): `0.05`. For `exp2`, `r = y LN_2`
//! with `ε ≤ 1 + 4` (table, M2), an absolute `5u² ρ_E < 0.03u²`.
//! (ii) *Polynomial.* In Lemma H, `s = r` is exact (we bound against
//! `P(r)` at the computed `r`), the coefficient ratio is `|r|/(m + 1) ≤
//! ρ_E/3`, so `ρ = 0.0019`, `σ₀ = 0` (`1/2` is exact) and `σ = 1`: `E =
//! (1.0019 + 0.0076 + 3)/0.9981 = 4.02`, `ε(Q) ≤ 0.0019 · 8.02 + 3 = 3.02`.
//! `ε(r²) = 3`, `ε(r²Q) ≤ 3 + 3.02 + 4 = 10.02`. In `r + r²Q` the weights are
//! `|r|/|P| ≤ 1/(1 − ρ_E/2) ≤ 1.0028` (exact term) and `|r²Q|/|P| ≤ 0.0028`:
//! `ε ≤ 0.0028 · 10.02 + 3 = 3.03`. The remainder `Σ_{m≥12} |r|^m/m! <
//! 1.01 ρ_E^11/12! · |r| < 2⁻¹¹¹|P|` adds `0.02`: `ε(p) ≤ 3.05`.
//! (iii) *Reconstruction.* `ε(T_j p) ≤ 1 + 3.05 + 4 = 8.05`; in `T_j +
//! T_j p` the weights are `1/|1 + p| ≤ 1.0055` and `|p|/|1 + p| ≤ 0.0055`
//! (`|p| ≤ e^ρ_E − 1 < 0.005 45`): `ε ≤ 1.0055 · 1 + 0.0055 · 8.05 + 3 =
//! 4.05`. With (i), `4.10` (`exp`) and `4.08` (`exp2`).
//! (iv) *Scaling.* `ldexp` is exact while both components stay normal.
//! When the result is at least `2⁻⁹⁶⁸` its leading component is normal and
//! only the trailing one can round, by at most `η/2 = 2⁻¹⁰⁷⁵ ≤ u²|z|/2`: `+
//! 0.5`. Below `2⁻⁹⁶⁸` each component rounds by at most `η/2`. Overflow of
//! `ldexp` returns `+∞` (step E of Chapter 11.2), which happens only when
//! `T_j(1 + p) 2ᵏ`, within `4.6u²` of `eˣ`, exceeds `MAX_dd`. ∎
//!
//! Underflow inside: no product of E2–E4 leaves the range of `two_prod`
//! when `|r| ≥ 2⁻⁴⁰⁰`, and for smaller `r` the only product is `T_j r`,
//! whose absolute error (`≤ η`) is negligible against `T_j ≥ 1`.
//!
//! ## Algorithm M (`exp_m1`)
//! Input `x ∈ dd`, finite; `x_h > 709.79` gives `+∞` and `x_h < −745.2`
//! gives `−1` (then `0 < eˣ < η/2`). `D_N` is the table entry of `e^(N/64) −
//! 1`, `T_j` and `D′_j` those of `2^(j/64)` and `2^(j/64) − 1`.
//!
//! M1. [Small.] If `|x_h| < 1/2`: `N ← trunc(64 x_h) ∈ [−31, 31]`, `r ←
//!     (x_h − N/64) + x_ℓ` — exact, as in `exp2`, and with the sign of `x`
//!     (or `|r| ≤ u|x|`). `p ← P(r)` with the terms up to `m = 13` (Q of 12
//!     coefficients), `|r| ≤ 1/64`. If `N = 0` return `p`; else return `(D_N +
//!     D_N p) + p`.
//! M2. [Large.] Else E1–E3 give `k`, `j`, `r`, `p`. If `|k| > 53` return
//!     `T_j(1 + p)2ᵏ − 1` (E4, then A1 with `−1`).
//! M3. Else return `(2ᵏD′_j + (2ᵏ − 1)) + 2ᵏ(T_j p)` — `2ᵏ − 1` is an exact
//!     `f64` for `|k| ≤ 53`, the first sum is A1, the second A2.
//!
//! **Theorem M.** `ε(exp_m1) ≤ 7.2` for results in `[2⁻⁹⁶⁸, Ω]`.
//!
//! *Proof.* *Small, `N = 0`:* Lemma H with `ρ = (1/64)/3 = 0.0053`: `E =
//! (1.0053 + 0.0212 + 3)/0.9947 = 4.05`, `ε(Q) ≤ 0.0053 · 8.05 + 3 = 3.05`,
//! `ε(r²Q) ≤ 10.05`; weights `|r|/|P| ≤ 1/(1 − 1/128) = 1.008`, `|r²Q|/|P|
//! ≤ 0.008`; remainder `< 2⁻¹¹⁴|P|`: `ε(p) ≤ 0.008 · 10.05 + 3 + 0.01 =
//! 3.09`. *Small, `N ≠ 0`:* `ε(D p) ≤ 1 + 3.09 + 4 = 8.09`; `A = D + Dp =
//! D(1 + p)` has weights `1/|1 + p| ≤ 1.016` and `|p|/|1 + p| ≤ 0.016`, so
//! `ε(A) ≤ 1.016 + 0.13 + 3 = 4.15`. `A` and `p` have the sign of `x` (`N`
//! and `r` do; in the exceptional case `|r| ≤ u|x|` the weights exceed 1
//! by less than `2⁻⁴⁵`), so in `A + p` the weights sum to 1 and `ε ≤
//! max(4.15, 3.09) + 3 = 7.15`. *Large:* count absolute errors in units of
//! `u²`. `S = 2ᵏD′_j + (2ᵏ − 1)` errs by at most `2ᵏD′_j` (table) `+ 2|S|`
//! (A1); `V = 2ᵏT_j p` by at most `2ᵏT_j(0.0055 · 8.05 + 0.05) = 0.094 ·
//! 2ᵏT_j` (Theorem E (i)–(iii)); the last A2 adds `3|z|`. For `k ≥ 1`, `|z|
//! ≥ 0.4946 · 2ᵏT_j`, so `2ᵏT_j/|z| ≤ 2.03`, `2ᵏD′_j/|z| ≤ 0.68` (largest
//! at `k = 1`, `T_j → 2`) and `|S|/|z| ≤ 1.011`: `ε ≤ 0.68 + 2.03 + 0.19 + 3
//! = 5.9`. For `k = 0` (`x ∈ [1/2, 0.688)`, so `j ≥ 46`, `|z| ≥ 0.648`):
//! `D′_j/|z| ≤ 1.02`, `T_j/|z| ≤ 2.6`: `ε ≤ 1.02 + 2.04 + 0.25 + 3 = 6.3`.
//! For `k ≤ −1` (`|z| ≥ 1 − e^(−1/2) = 0.393`): `2ᵏD′_j/|z| ≤ 0.5`,
//! `2ᵏT_j/|z| ≤ 1.55`: `ε ≤ 0.5 + 2.02 + 0.15 + 3 = 5.7`. For `k > 53`, A1
//! on `eˣ ≥ 2⁵³`: `ε ≤ 4.1(1 + 2⁻⁵²) + 2 = 6.1`; for `k < −53`, `eˣ/|z| ≤
//! 2⁻⁵²`: `ε ≤ 2.01`. The maximum is `7.15`. ∎
//!
//! ## Algorithm L (`ln`, and the parts of `ln_1p`, `log2`, `log10`)
//! Input `x ∈ dd`, `0 < x < +∞` (the rest by the table of special cases).
//! `LNC_j`, `K⁺_j`, `K⁻_j` are the table entries of `ln(1 + j/64)`,
//! `ln(2(1 + j/64))` and `ln((1 + j/64)/2)`; `A`, `B`, `C` the parts of `ln 2`
//! (`A` of 42 bits).
//!
//! L1. [Scale.] If `x_h < 2⁻¹⁰²²` (then `x_ℓ = 0`), `x ← x · 2¹⁰⁶` and `e ←
//!     −106`, else `e ← 0`. Add to `e` the exponent of `x_h`, and `y ← x ·
//!     2^−exponent ∈ [1, 2)`; if `y_h ≥ RN(√2)`, `y ← y/2`, `e ← e + 1`. Now
//!     `x = 2ᵉ y` exactly, `y ∈ [0.707, 1.415)` and `|e| ≤ 1074`.
//! L2. [Table index.] `j ← trunc(64(y_h − 1)) ∈ [−18, 26]` (`y_h − 1` is a
//!     Sterbenz difference), `c ← 1 + j/64` — between 1 and `y_h`.
//! L3. [Reduce.] `d ← (y_h − c) + y_ℓ` (a Sterbenz difference, then
//!     `from_sum`: exact), with the sign of `y − 1` or `|d| ≤ u`; `z ← d` if
//!     `j = 0`, else `z ← d/c` (D1). `|z| ≤ ζ = 0.0218` (`0.0157` if `j =
//!     0`).
//! L4. [Series.] `L ← z − z²Q(z)`, `Q(z) = Σ_{m=2}^{20} (−z)^(m−2)/m`
//!     (Horner); `L ← z` if `|z_h| < 2⁻⁴⁰⁰`. This is `ln(1 + z) = ln(y/c)`.
//! L5. [Combine.] If `e = 1` and `y_h < 1`: `K⁺_j + L`. If `e = −1` and
//!     `y_h ≥ 1`: `K⁻_j + L` (these are the two cases in which `e ln 2` and
//!     `ln y` cancel). Otherwise `ln y ← L` (`j = 0`) or `LNC_j + L`, and the
//!     result is `ln y` (`e = 0`) or `e ln 2 + ln y` (A2).
//! L6. [`e ln 2`.] `a ← e·A` (exact: 42 + 11 bits), `(p, q) ←
//!     two_prod(e, B)`, `(s, t) ← Fast2Sum(a, p)`, `v ← t + (q + e·C)`,
//!     result `from_sum(s, v)`.
//!
//! **Lemma L6.** `ε(e ln 2) ≤ 1 + 2⁻⁴⁰`. *Proof.* `a + p + q = e(A + B)`
//! exactly; `|p| ≤ 2⁻⁴²|a|`, so Fast2Sum applies and is exact; `|t| ≤
//! u|s|`, `|q| ≤ u|p|`, `|e C| ≤ 2¹¹ ulp(B)/2 < 2⁻⁹⁴|s|`. Two roundings remain:
//! `q + eC`, at most `u · 2⁻⁹³|s|`, and `t + (…)`, at most `u|v| ≤ u²|s|(1 +
//! 2⁻⁴¹)`; the parts of `ln 2` beyond `C` weigh `2¹¹ · 2⁻¹⁴⁸`. ∎
//!
//! **Theorem L.** `ε(ln) ≤ 6.7` for all `x ∈ dd`, `0 < x < ∞`, with
//! `ln x ≠ 0` (`ln 1 = +0` exactly).
//!
//! *Proof.* (i) *Series.* In Lemma H the ratio of consecutive terms of `Q`
//! is at most `ζ(1 + ζ)` (alternating when `z > 0`, all positive when `z <
//! 0`), so `ρ = 0.0225` covers both signs (`0.0161` when `j = 0`); `σ₀ = 0`
//! (`1/2` is exact), `σ = 1`, `ε_s = ε(z)`. For `j = 0`, `ε(z) = 0`: `E =
//! 4.15`, `ε(Q) ≤ 3.14`, `ε(z²Q) ≤ 3 + 3.14 + 4 = 10.14`; the weights in `z
//! − z²Q` are at most `1 + ζ/2` and `ζ/2(1 − ζ/2)⁻¹ ≤ 0.008`: `ε(L) ≤ 3.09`.
//! For `j ≠ 0`, `ε(z) = 3` (D1): `E = 4.28`, `ε(Q) ≤ 3.26`, `ε(z²) ≤ 9`,
//! `ε(z²Q) ≤ 16.3`, weights `1.011` and `0.011`: `ε(L) ≤ 1.011 · 3 + 0.011 ·
//! 16.3 + 3 = 6.22`. The remainder after `m = 20` is below `ζ²⁰/21 · |z| <
//! 2⁻¹¹⁴|L|`. (ii) *`ln y`.* `LNC_j` and `L` have the same sign (but for
//! `|d| ≤ u`, where the weights exceed 1 by `2⁻⁴⁵`); the weight of `L` is
//! largest for `j = ±1`, `0.496` and `0.504`: `ε(ln y) ≤ 3 + 0.496 + 0.504 ·
//! 6.22 = 6.63`. (iii) *Cancelling cases.* `|ln x| = |ln(2y)|` or `|ln(y/2)|
//! ≥ ln √2 = 0.3466` and `|L| ≤ 0.022`: weights `1.064` (table) and `0.064`:
//! `ε ≤ 3 + 1.064 + 0.064 · 6.22 = 4.46`. (iv) *Others.* With `e ln 2` and
//! `ln y` of the same sign the weight of `ln y` is at most `0.3466/1.0397`
//! and `ε ≤ 3 + 0.67 + 0.333 · 6.63 = 5.88`; of opposite signs `|e| ≥ 2`,
//! weights `1.333` and `0.333`: `ε ≤ 3 + 1.333 + 2.21 = 6.54`. The maximum
//! is `6.63`. No product leaves the range of `two_prod`: `|z| ≥ 2⁻⁴⁰⁰` in
//! L4, and all other operands are near 1. ∎
//!
//! ## Algorithm L′ (`ln_1p`)
//! L′1. If `|x_h| < 1/64`: `z ← x` and step L4 (`ε ≤ 3.09`).
//! L′2. If `x_h ≤ −1/2`: `w ← (1 + x_h) + x_ℓ` (Sterbenz, then `from_sum`:
//!      exact) and `ln w` (`ε ≤ 6.63`).
//! L′3. Else `w ← x + 1` (A1). If `w_h ≥ 2⁵³`, return `ln w` (the rounding of
//!      `w`, `2u²` relative, changes `ln w` by `2u²/ln w < 0.06u²` of it).
//!      Otherwise `δ ← x − ((w_h − 1) + w_ℓ)` (`w_h − 1` exact for `1/2 ≤ w_h
//!      < 2⁵³`; A2), `κ ← RN(δ_h/w_h)` and return `ln w + κ` (A1).
//!
//! **Theorem L′.** `ε(ln_1p) ≤ 7.7`. *Proof.* `1 + x = w + δ*` with `|δ*| ≤
//! 2u²|w|(1 + 2⁻⁵⁰)` (A1), and `ln(1 + x) = ln w + δ*/w − θ`, `0 ≤ θ ≤
//! (δ*/w)²`. The computed `κ` differs from `δ*/w` by at most `4u|δ*/w|`
//! (A2 on exact operands, the dropped `δ_ℓ` and `w_ℓ`, one rounding).
//! `|ln w| ≥ ln(65/64) > 0.0155` in L′3, so `|κ| ≤ 130u²|ln w| < 2⁻¹⁰u|ln
//! w|` and Lemma S applies: the last A1 adds `u²(1 + 2⁻⁹)`. With the
//! `6.63` of `ln w` (weight `1 + 2⁻⁹⁵`), `ε ≤ 6.63 + 1.002 + 2⁻⁴⁰ < 7.64`. ∎
//!
//! **`log2` and `log10`.** `log10 x = ln x · LOG10_E` (M2 and a table
//! constant: `ε ≤ 6.63 + 5 = 11.63`). `log2 x` is `e` exactly when `x = 2ᵉ`
//! (tested first); in the cancelling cases of L5 it is `ln x · LOG2_E`
//! (`4.46 + 5`); otherwise `ln y · LOG2_E` (`ε ≤ 11.63`) plus the exact `e`
//! by A1, with weight at most `0.5/1.5` on `log₂ y` when `e ≠ 0`: `ε ≤ 2 +
//! 0.333 · 11.63 = 5.88`. So `ε(log2), ε(log10) ≤ 11.7`.
//!
//! ## Algorithm P (`powi`: binary powering with a separate exponent)
//! Input `x ∈ dd` finite and nonzero, `n ∉ {−1, 0, 1}` (those and the
//! non-finite `x` by the table of special cases). `N = |n|`.
//!
//! P1. [Split.] `x = ±2^(e_x) y` with `y ∈ [1, 2)` (exact; a subnormal `x_h`
//!     is first scaled by `2¹⁰⁶`). `b ← y`, `e_b ← e_x`, `a ← 1`, `e_a ← 0`
//!     (exponents in `i64`).
//! P2. [Loop.] For each bit of `N` from the least significant: if it is set,
//!     `a ← a·b` (M2), `e_a ← e_a + e_b`, and renormalize `a` into `[1, 2)`
//!     by an exact power of two; then, if bits remain, `b ← b²` (M3), `e_b ←
//!     2e_b`, and renormalize `b`.
//! P3. [Finish.] If `n < 0`, `a ← 1/a` (D2) and `e_a ← −e_a`. Return `±a ·
//!     2^(e_a)` (`ldexp`; `±∞` or `±0` when `e_a` is out of every range), the
//!     sign by the parity of `n`.
//!
//! **Theorem P.** `ε(powi) ≤ 3N + popcount(N) − 4 ≤ 4N − 4` for `n > 0`
//! and `≤ 4N + 11` for `n < 0`; `powi(x, 2)` is one M3 (`3u²`) and
//! `powi(x, −1)` one D2. *Proof.* After `i` squarings `ε(b) ≤ 3(2ⁱ − 1)`
//! (`ε ← 2ε + 3`, Chapter 11.2). The first factor enters `a = 1` exactly
//! (M2 with the operand 1 is exact), each further one adds `4`: `ε(a) ≤
//! Σ_{i∈bits} 3(2ⁱ − 1) + 4(popcount − 1) = 3N + popcount − 4`. Renormalizing
//! is exact and keeps every operand in `[1, 4)`, so no product leaves the
//! range of `two_prod` and no intermediate over- or underflows, whatever
//! `xⁿ` is (in particular `x⁻ⁿ` with `xⁿ > Ω` is finite and accurate). D2
//! adds `15`. The second-order terms are below `(4N u²)² ≤ 2⁻¹⁴⁶`, since `N
//! ≤ 2³¹`. ∎
//!
//! ## `powf` (`exp(y ln x)`)
//! For finite nonzero `x`, finite nonzero `y ≠ 1` (the rest by the table of
//! special cases): `P ← y · ln|x|` (Algorithm L, then M2) and `z ← exp(P)`
//! (Algorithm E), with the sign `−` when `x < 0` and `y` is an odd integer
//! (`x < 0` with `y` not an integer is NaN). "`y` is an integer": both
//! components are; "odd": the trailing component is odd when nonzero (then
//! `|y_h| ≥ 2⁵³` is even), else the leading one.
//!
//! **Theorem F.** `ε(powf) ≤ 4.6 + 10.7 G`, `G = |y ln x|`. *Proof.* `ε(P) ≤
//! 6.63 + 4` (Theorem L, M2), an absolute error at most `10.63u²G` in the
//! argument of `exp`, which multiplies the result by `e^(ΔP) = 1 + ΔP(1 +
//! ΔP)` (C4); Theorem E adds `4.6`. Overflow of `P` (`|y|` huge) gives `exp`
//! of `±∞`, the right limit. ∎
//!
//! The contract proposed the growth coefficient `8`; the proof gives
//! `10.63`, because the logarithm's own bound, `6.63u²`, is multiplied by
//! `G` as well as the product's `4u²`. The declared row therefore has `11`
//! (ratified by the owner, 2026-11-18, D-0122).
//!
//! ## Algorithm R (argument reduction: Payne–Hanek in integers)
//! Input `x ∈ dd`, finite, `x ≥ 0` (the callers use the parity of the
//! functions), `x_h > RN(π/4)`. Output `q ∈ {0, 1, 2, 3}` and `r ∈ dd`,
//! `|r| ≤ π/4 (1 + 2⁻³¹⁰)`, with `x ≡ qπ/2 + r (mod 2π)` up to the error of
//! Theorem R. `b_i` is bit `i` of `2/π = Σ_{i≥1} b_i 2⁻ⁱ`; the table holds
//! `i ≤ 1408`.
//!
//! R1. [Each component.] For `c ∈ {x_h, x_ℓ}` write `|c| = m 2ᵉ` with an
//!     integer `m < 2⁵³`. Let `I = e + 382`; if `I < 1`, skip `c` (`c · 2/π <
//!     2⁻³²⁸`). Else `W ← ⌊2^I · 2/π⌋ mod 2³⁸⁴` — the bits `b_{I−383}, …, b_I`,
//!     six 64-bit words — and `A_c ← ⌊m W / 2⁶⁴⌋ mod 2³²⁰`, negated modulo
//!     `2³²⁰` when `c < 0` (`x_ℓ` may be negative).
//! R2. [Sum.] `A ← (A_{x_h} + A_{x_ℓ}) mod 2³²⁰`. `A · 2⁻³¹⁸` is `x · 2/π`
//!     modulo 4, as two integer bits and a 318-bit fraction `F`.
//! R3. [Quadrant.] If `F < 2³¹⁷`: `q ←` the integer bits, `G ← 2F`, sign
//!     `+`. Else `q ←` the integer bits `+ 1 (mod 4)`, `G ← 2(2³¹⁸ − F)`,
//!     sign `−`. Then `g = G · 2⁻³¹⁸ ∈ [0, 1]` and `r = ±g · π/4`.
//! R4. [Multiply.] `R ← ⌊G · P / 2²⁵⁶⌋` with `P = ⌊2³²⁰ π/4⌋` (five words),
//!     so `|r| ≈ R · 2⁻³⁸²`, `R < 2³⁸³`.
//! R5. [Convert.] With `L` the position of the leading bit of `R`, take the
//!     53-bit chunks `c₀, c₁, c₂` starting at bits `L − 52`, `L − 105`, `L −
//!     158` (each exactly an `f64` `vᵢ` after scaling by a power of two);
//!     `(s, t) ← Fast2Sum(v₀, v₁)`, `r ← ±from_sum(s, t + v₂)`.
//!
//! **Theorem R.** `|r − r*| ≤ (1 + 2⁻⁴⁹)u²|r*| + 2⁻³¹⁶`, where `r* = x −
//! (4k + q)π/2` exactly, for some integer `k`.
//!
//! *Proof.* *R1:* the bits of `2/π` with `i ≤ e − 2` contribute multiples of
//! 4 to `c · 2/π` (`m` is an integer) and are dropped exactly; those with
//! `i > I` contribute less than `m 2^(e−I) < 2^(53−382) = 2⁻³²⁹`; and
//! `A_c 2⁻³¹⁸ = m W 2^(e−I) − θ` with `0 ≤ θ < 2⁻³¹⁸` (the dropped word), all
//! modulo 4 (`2³²⁰ · 2⁻³¹⁸ = 4`); negation keeps `|θ| < 2⁻³¹⁸`. The skipped
//! components weigh less than `2⁻³²⁸`. So `A 2⁻³¹⁸ ≡ x · 2/π − δ (mod 4)`
//! with `|δ| < 2⁻³¹⁶·⁹`. *R3:*
//! the chosen integer is within `1/2 + δ` of `x · 2/π`; `g` errs by at most
//! `2δ`. *R4:* `P` errs by less than `2⁻³²⁰`, and the floor drops less than
//! `2⁻³⁸²`: `|R 2⁻³⁸² − g π/4| < (π/4) 2δ + 2⁻³²⁰ + 2⁻³⁸² < 2⁻³¹⁶`. *R5:*
//! `v₀ + v₁ + v₂` is `R 2⁻³⁸²` truncated after 159 bits (`< 2⁻¹⁵⁷` of it);
//! Fast2Sum is exact (`|v₁| < ulp(v₀)`); `|t| ≤ u|s|`, `|v₂| < 2⁻¹⁰⁵ ·
//! R 2⁻³⁸²`, so the one rounding, of `t + v₂`, is at most `u²(1 + 2⁻⁵¹)` of
//! the value; `from_sum` is exact. ∎
//!
//! The contract planned Cody–Waite (`π/2` in five parts) below `2⁵²` and
//! Payne–Hanek above. Cody–Waite in double-double arithmetic subtracts with
//! A2, whose error is relative to *its own* result: `x − kP₁` is about
//! `kP₂ ≈ k 2⁻⁵³`, so the reduction carries an absolute error up to `3u²k
//! 2⁻⁵³` — `2⁻¹⁰⁵` at `k ≈ 2⁵²`, and `2⁻¹⁵⁷` even at `k = 1`. That cannot be
//! proven below the floor `2⁻²⁰⁰` of the declared bound (the pilot's good
//! measurements near `kπ/2` presumably reflect A2 being exact there in
//! practice; that is not a theorem). The integer reduction is exact by construction, costs about
//! 40 ns, and is used for every `|x| > π/4`.
//!
//! ## Algorithm K (`sin`, `cos`, `sin_cos`, `tan`)
//! S1. [Reduce.] `a ← |x|`. If `a_h ≤ RN(π/4)`, `q ← 0`, `r ← a`; else
//!     Algorithm R.
//! S2. [Kernels.] `sin r ← r − r³Q_s(r²)`, `Q_s = Σ_{m=1}^{14} (−r²)^(m−1)/
//!     (2m+1)!` (Horner); `cos r ← 1 − 2 sin²(r/2)`, with `sin(r/2)` by the
//!     same series to `m = 11` (`|r/2| ≤ π/8`); `sin r ← r` and `cos r ← 1`
//!     when `|r_h| < 2⁻⁴⁰⁰`.
//! S3. [Quadrant.] `sin x = ±(sin r, cos r, −sin r, −cos r)[q]` with the
//!     sign of `x`; `cos x = (cos r, −sin r, −cos r, sin r)[q]`; `tan x =
//!     sin r/cos r` (`q` even) or `−cos r/sin r` (`q` odd), with the sign of
//!     `x` (D2).
//!
//! Each function calls the same kernels on the same `(q, r)`, so `sin_cos`
//! gives the bits of `sin` and of `cos`; the kernels are odd and even bit
//! for bit (M2, M3 and A2 commute with negation), and S1 works on `|x|`, so
//! `sin`, `tan` are odd and `cos` is even bit for bit.
//!
//! **Theorem K.** For every finite `x`, with `f` the function: `|z − f(x)|
//! ≤ 6.9u²|f(x)| + 2⁻³¹⁵` for `sin`, `cos` (and the pair), and `|z −
//! tan x| ≤ 27.4u²|tan x| + 2⁻³¹⁵(1 + tan²x)`.
//!
//! *Proof.* *Sine kernel:* Lemma H with `s = r²` (`ε_s = 3`), ratio `ρ =
//! (π/4)²/20 (1 + 0.04) = 0.032`, `σ = 1`: `ε(Q_s) ≤ (1.032 + 0.032 · 7 +
//! 3)/0.968 = 4.40`; `ε(r³) = 7`, `ε(r³Q_s) ≤ 15.4`; the weight of `r³Q_s`
//! is `(r − sin r)/sin r ≤ 0.111`: `ε ≤ 0.111 · 15.4 + 3 = 4.71`. The first
//! omitted term is below `2⁻¹²²` of `sin r`. *Cosine kernel:* for `h = r/2`,
//! `ρ = 0.0078`, `ε(Q) ≤ 4.10`, weight `0.026`: `ε(sin h) ≤ 3.39`;
//! `ε(sin²h) ≤ 9.78`; in `1 − 2 sin²h` (A1, exact 1) the weight of `2
//! sin²h` is `(1 − cos r)/cos r ≤ 0.415`: `ε ≤ 2 + 0.415 · 9.78 = 6.06`.
//! *Reduction (C4):* by Theorem R the kernels see `r` instead of `r*`;
//! `|sin r − sin r*| ≤ |r − r*|`, which is `≤ 1.111u²|sin r*| + 2⁻³¹⁶`, and
//! `|cos r − cos r*| ≤ sin|r*| |r − r*| ≤ 0.79u²|cos r*| + 2⁻³¹⁶`. Totals
//! `4.71 + 1.12` and `6.06 + 0.8`: `6.9`. For `|x| ≤ π/4`, `r = x` exactly.
//! *Tangent:* D2 adds `15` to `4.71 + 6.06`; `|tan r − tan r*| ≤ (1 +
//! tan²)|r − r*|`, whose relative part is `r/(sin r cos r) u²(1 + 2⁻⁴⁹) ≤
//! 1.58u²` (the same for `cot`): `27.4`. ∎
//!
//! ## Algorithm T (`atan`)
//! Input `a ∈ dd`, `0 ≤ a < ∞` (`atan` is odd: the sign is applied last;
//! `atan(±∞) = ±π/2`). `θ_c` is the table entry of `atan c`.
//!
//! T1. [Tiny.] If `a_h < 2⁻⁶`: `atan a ← a − a³Q_a(a²)`, `Q_a =
//!     Σ_{m=1}^{9} (−a²)^(m−1)/(2m + 1)` (Horner; `a` itself below
//!     `2⁻⁴⁰⁰`).
//! T2. [Table.] If `a_h < 8`: with `e` the exponent of `a_h` (`−6 ≤ e ≤
//!     2`) and `i` the top four bits of its fraction, `c ← 2ᵉ(1 + (2i +
//!     1)/32)` (an `f64` of six bits, `|a − c| ≤ 2ᵉ/32`); `t ← ((a_h − c) +
//!     a_ℓ)/(a·c + 1)` (Sterbenz and `from_sum`: exact numerator; M1, A1,
//!     D2); `atan a ← θ_c + atan t` with `atan t` by the series of T1
//!     (`|t| ≤ 2ᵉ/32/(1 + 2²ᵉ) ≤ 1/64`).
//! T3. [Large.] Else `atan a ← π/2 − atan(1/a)` (D2, then T1 or T2).
//!
//! **Theorem T.** `ε(atan) ≤ 5.8`. *Proof.* *T1:* Lemma H with `s = a²`
//! (`ε_s = 3`), `ρ = 2⁻¹² · 0.61`: `ε(Q_a) ≤ 4.002`, so `ε(a³Q_a) ≤ 15.01`;
//! its weight is at most `a²/3 < 10⁻⁴`: `ε ≤ 3.002`; the omitted terms are
//! below `2⁻¹²⁰` of the value. *T2:* `ε(a·c + 1) ≤ 2 + 2`, so `ε(t) ≤ 4 +
//! 15 = 19` and `ε(atan t) ≤ 3.002 + 19` (C4, the condition number of
//! `atan` near 0 is at most 1). `|atan t|/atan a ≤ 1/(8π) < 0.04` (for `a`
//! at most 1, `atan a ≥ aπ/4 ≥ 2ᵉπ/4`; for `a > 1`, `atan a ≥ π/4`, `|t| ≤ 1/64`):
//! weights `1.04` and `0.04`, `ε ≤ 3 + 1.04 + 0.04 · 22.0 = 4.92`. *T3:*
//! `ε(1/a) ≤ 15`, `ε(atan(1/a)) ≤ 4.92 + 15`; with `atan(1/a) ≤ atan(1/8)`
//! the weights of `π/2` and of `atan(1/a)` are `1.086` and `0.086`, so
//! `ε ≤ 3 + 1.086 + 0.086 · 19.92 = 5.80`. ∎
//!
//! **`atan2(y, x)`.** After the special cases (the table of `f64`), both
//! magnitudes are scaled by the same power of two so that the larger is in
//! `[1, 2)` (exact; the smaller rounds only if the quotient is below
//! `2⁻¹⁰²¹`). If `|y| ≤ |x|`: `z ← atan(|y|/|x|)`, and the result is `z`
//! (`x > 0`) or `π − z`; else `z ← atan(|x|/|y|)` and the result is `π/2 −
//! z` (`x > 0`) or `π/2 + z`; then the sign of `y`. A quotient below
//! `2⁻⁴⁰⁰` is formed scaled by `2⁶⁰⁰` and is its own arctangent. *Bound:*
//! `ε(q) ≤ 15`, `ε(z) ≤ 4.92 + 15κ` with `κ ≤ 1` the condition number of
//! `atan` on `[0, 1]`. `z`: `19.92`. `π − z`, `z ≤ π/4`: weights `4/3`,
//! `1/3`: `3 + 1.33 + 6.64 = 11.0`. `π/2 − z`: weight `≤ 2` on `π/2`, `≤ 1`
//! on `z`, and `w_z κ = q/((1 + q²)(π/2 − atan q)) ≤ 2/π`: `3 + 2 + 4.92 +
//! 15 · 0.637 = 19.5`. `π/2 + z`: `≤ 10.7`. So `ε(atan2) ≤ 20`.
//!
//! **`asin`, `acos`.** With `a = |x| ≤ 1` and `w = √((1 − a)(1 + a))`:
//! for `a ≥ 1/2`, `1 − a` is exact (Sterbenz, `from_sum`), `1 + a` is A1
//! and the product M2: `ε(w) ≤ (2 + 4)/2 + 25/8 = 6.13`; for `a < 1/2`, `w
//! = √(1 − a²)` (M3, A1 with the weight `1/3` on `a²`): `ε(w) ≤ 3/2 + 25/8 =
//! 4.63`. Then `asin x = ±atan(a/w)` (`a ≤ w`) or `±(π/2 − atan(w/a))`, and
//! `acos x` is `atan(w/a)` (`w ≤ a`) or `π/2 − atan(a/w)` for `x ≥ 0`, and `π
//! − atan(w/a)` or `π/2 + atan(a/w)` for `x < 0`. *Bound:* the quotient has
//! `ε ≤ 15 + ε(w) ≤ 21.13`; `atan(q)` then `4.92 + 21.13 = 26.05`; `π/2 −
//! atan q` gives `3 + 2 + 4.92 + 0.637 · 21.13 = 23.4`; the others are
//! smaller (`13.0`, `12.7`). So `ε(asin), ε(acos) ≤ 26.1`. `asin(±1) =
//! ±π/2`, `acos(1) = +0` and `acos(−1) = π` exactly (`w = 0`).
//!
//! ## The hyperbolic functions and their inverses (composition)
//! With `a = |x|` (the odd functions apply the sign of `x` last; `cosh` is
//! even), each below is a short composition whose bound follows from (C1)–
//! (C4) and Theorems E, M, L, L′:
//!
//! * `sinh a = (E + E/(E + 1))/2`, `E = exp_m1(a)`, for `a ≤ 709`;
//!   `exp(a − 1) · e/2` above (`a − 1` is exact there, `e/2` is `E/2` of the
//!   table, exact scaling). *Bound:* `ε(E + 1) ≤ 2 + 7.15 E/(E + 1)`,
//!   `ε(E/(E + 1)) ≤ 24.15 + 7.15 E/(E + 1)`, and the last A2 weighs them
//!   `(E + 1)/(E + 2)` and `1/(E + 2)`: largest as `E → 0`, `3 + (7.15 +
//!   24.15)/2 = 18.7`. Above 709: `4.6 + 1 + 4 = 9.6` (the neglected `e⁻ᵃ` is
//!   `2⁻²⁰⁴⁵` relative).
//! * `cosh a = (E + 1/E)/2`, `E = exp(a)`, for `a ≤ 709`; `exp(a − 1) · e/2`
//!   above. *Bound:* `ε(1/E) ≤ 4.6 + 15`; the weights `E²/(E² + 1)` and
//!   `1/(E² + 1)`, largest at `E = 1`: `3 + (4.6 + 19.6)/2 = 15.1`.
//! * `tanh a = E₂/(E₂ + 2)`, `E₂ = exp_m1(2a)`, for `a < 5/16`; else `(1 −
//!   F)/(1 + F)`, `F = exp(−2a)`. *Bound:* below `5/16`, `E₂ ≤ 0.868`,
//!   `ε(E₂ + 2) ≤ 2 + 0.303 · 7.15`, `ε ≤ 7.15 + 4.17 + 15 = 26.4`; above,
//!   `F ≤ 0.536`, `ε(1 − F) ≤ 2 + 1.152 · 4.6`, `ε(1 + F) ≤ 2 + 0.349 · 4.6`,
//!   `ε ≤ 7.30 + 3.61 + 15 = 25.9`. For large `a`, `F` underflows (an
//!   absolute `η` against 1) and D2 returns the pair `(1, −2F)` until `F <
//!   η/2`, then exactly 1.
//! * `asinh a = ln_1p(a + a²/(1 + √(1 + a²)))` for `a < 2`, `ln(a + √(a² +
//!   1))` for `2 ≤ a < 2⁶⁰`, `ln a + ln 2` above (`asinh a − ln 2a < 2⁻¹²²` of
//!   it). *Bound:* for `a < 2`, `ε(√(1 + a²)) ≤ 2.2 + 25/8`, `ε(1 + √) ≤
//!   5.68`, `ε(a²/(1 + √)) ≤ 23.7`, and in the argument `v` of `ln_1p` its
//!   weight is at most `0.382`: `ε(v) ≤ 12.1`; the condition number of
//!   `ln_1p` is at most 1: `ε ≤ 7.64 + 12.1 = 19.7`. For `2 ≤ a < 2⁶⁰`, `ε(v)
//!   ≤ 3 + 0.528 · 5.63` and the condition number of `ln` at `v ≥ 4.23` is
//!   `1/ln v ≤ 0.693`: `ε ≤ 6.63 + 4.14 = 10.8`. Above: `3 + 6.63 = 9.6`.
//! * `acosh x = ln_1p(d + √(d(d + 2)))`, `d = x − 1` (exact), for `x < 2`;
//!   `ln(x + √(x² − 1))` for `2 ≤ x < 2⁶⁰`; `ln x + ln 2` above. *Bound:*
//!   `ε(√(d(d + 2))) ≤ 3 + 25/8`, `ε(v) ≤ 3 + 6.13`, `ε ≤ 7.64 + 9.13 =
//!   16.8`; `ε(x² − 1) ≤ 2 + (4/3) · 3`, `ε(v) ≤ 3 + 6.13/2`, `ε ≤ 6.63 + 0.76 ·
//!   6.07 = 11.3`; above, `9.6`.
//! * `atanh a = ln_1p(2a/(1 − a))/2`, with `1 − a` exact for `a ≥ 1/2` and A1
//!   below. *Bound:* `ε(2a/(1 − a)) ≤ 2 + 15`, condition number of `ln_1p` at
//!   most 1: `ε ≤ 7.64 + 17 = 24.7`.
//!
//! Below `2⁻⁴⁰⁰` the odd functions return `x` (the cubic term is `2⁻⁸⁰⁰`
//! relative).
//!
//! ## Invariants
//! (F1) *Well formed.* Every output satisfies (D1)/(D2) of Chapter 11.2,
//!      for every input, non-finite and subnormal ones included (a property
//!      test calls `check_invariants` on every result).
//! (F2) *Determinism.* Only `+ − × ÷ √ fma`, `round`, `trunc` of `f64` and
//!      the operations of Chapter 11.2, in a fixed order with branches that
//!      depend only on the bits of the arguments: the same bits on every
//!      platform, in debug and release, with or without the features (a
//!      golden test of 246 values and a digest of 21 000; `clippy.toml`
//!      forbids the transcendental functions of `f64` in the crate).
//! (F3) *Symmetry.* `sin`, `tan`, `asin`, `atan`, `sinh`, `tanh`, `asinh`,
//!      `atanh` are odd and `cos`, `cosh` even, bit for bit; `sin_cos(x) =
//!      (sin x, cos x)` bit for bit.
//! (F4) *Inputs.* Each function `debug_assert!`s (D1) of its arguments; in
//!      release a non-normalized argument gives an unspecified but well
//!      formed result.
//!
//! ## Measured
//! Largest error observed against `dashu-float`, `10⁵` random arguments per
//! family (in `u²`; mixed bounds: the relative part); the survey is the
//! `#[ignore]`d test `survey_of_observed_errors`. `exp` 1.77, `exp2` 1.77,
//! `exp_m1` 2.53, `ln` 1.91, `ln_1p` 2.16, `log2` 3.16, `log10` 3.40, `sin`
//! 2.37, `cos` 1.89, `tan` 5.57, `atan` 1.94, `atan2` 5.21, `asin` 5.87,
//! `acos` 4.41, `sinh` 3.40, `cosh` 2.86, `tanh` 6.23, `asinh` 2.70,
//! `acosh` 3.73, `atanh` 3.46; `powi` `≤ 2.07 max(1, |n|)`, `powf` `≤ 4.30
//! max(1, |y ln x|)`. At the double-doubles nearest to `kπ/2`, `k ≤ 2·10⁶`:
//! `0.50` relative, `4·10⁻⁵⁹` absolute at most.
//!
//! ## Cost
//! Time per call (ns, median of 21 runs over 1 024 independent arguments,
//! so a throughput; release, Apple M2 Pro on AC power, load average 3.0
//! from other sessions): `exp` 232, `exp2` 214, `exp_m1` 241, `ln` 430,
//! `ln_1p` 443, `log2` 439, `log10` 443, `powi(x, 7)` 53, `powi(x, −100)`
//! 138, `powf` 753, `sin` 279 (`|x| ≤ 1`) to 344 (`|x| ≤ 10`) and 352
//! (`|x|` up to `2⁹⁶⁰`), `cos` 340, `tan` 599, `sin_cos` 599, `atan` 190,
//! `atan2` 258, `asin` 266, `acos` 266, `sinh` 298, `cosh` 271, `tanh` 278,
//! `asinh` 538, `acosh` 508, `atanh` 487. For scale: `dd · dd` 1.9, `dd +
//! dd` 2.8, `dd / dd` 3.4, `f64::exp` 2.5.
//!
//! ## The declared bounds
//! The column "proven" is the constant of the proof in this chapter,
//! rounded up; "declared" is the row of [`DoubleDouble::FUNCTION_BOUNDS`].
//!
//! | function | declared (`u²`) | proven (`u²`) | how |
//! |---|---|---|---|
//! | `exp`, `exp2` | `8` | `4.6` | Algorithm E |
//! | `exp_m1` | `8` | `7.2` | Algorithm M |
//! | `ln` | `8` | `6.7` | Algorithm L |
//! | `ln_1p` | `8` | `7.7` | Algorithm L′ |
//! | `log2`, `log10` | `12` | `11.7` | Algorithm L, then a product |
//! | `powi` | `15 + 4\|n\|` | `11 + 4\|n\|` | Algorithm P |
//! | `powf` | `16 + 11\|y ln x\|` | `4.6 + 10.7\|y ln x\|` | `exp(y ln x)` (Theorem F) |
//! | `sin`, `cos`, `sin_cos` | `8\|f\| + 2⁻²⁰⁰` | `6.9\|f\| + 2⁻³¹⁵` | Algorithms R and K |
//! | `tan` | `32\|tan\| + 2⁻²⁰⁰(1 + tan²)` | `27.4\|tan\| + 2⁻³¹⁵(1 + tan²)` | Algorithms R and K, D2 |
//! | `atan` | `8` | `5.8` | Algorithm T |
//! | `atan2`, `asin`, `acos` | `32` | `20`, `26.1`, `26.1` | Algorithm T, composition |
//! | `sinh`, `cosh`, `tanh` | `32` | `18.7`, `15.1`, `26.4` | Algorithms E and M, composition |
//! | `asinh`, `acosh`, `atanh` | `32` | `19.7`, `16.8`, `24.7` | Algorithms L and L′, composition |
//!
//! ## Tables
//! The constants of the algorithms are private tables of literals
//! (`elementary/tables.rs`): parts of `ln 2/64` and `ln 2`, `2^(j/64)`,
//! `2^(j/64) − 1`, `e^(N/64) − 1`, `1/n!`, `1/n`, `ln(1 + j/64)` and its
//! two shifted forms, `atan(c)` at 144 points, and 1 408 bits of `2/π`
//! and 320 of `π/4`. A unit test checks each entry, bit for bit, against
//! `dashu-float` and against the independent enclosures of the tests.
//!
//! ## References
//! Muller, *Elementary Functions: Algorithms and Implementation*, 3rd ed.
//! (2016): Cody–Waite and Payne–Hanek reduction, table-driven methods
//! (Tang) — cited from memory. Tang, "Table-driven implementation of the
//! exponential function in IEEE floating-point arithmetic", ACM TOMS 15
//! (1989) — from memory. Payne & Hanek, "Radian reduction for
//! trigonometric functions", SIGNUM Newsletter 18 (1983) — from memory.
//! Joldes, Muller & Popescu (2017) and the other sources of Chapter 11.2
//! for the bounds of the operations. C99 Annex F and IEEE 754-2019 §9.2 for
//! the special cases (as restated in the documentation of `f64`).

#![allow(clippy::float_cmp)] // exact comparisons, as in Chapter 11.2

use crate::dd::DoubleDouble;

mod tables;

use crate::eft::{fast_two_sum, two_prod};
use tables::{
    ATAN_C, EXP2_64, EXP2_64_M1, EXPM1_64, FRAC_3PI_4, INV_FACT, INV_INT, INV_LN2_64, LN_2C, LN_C,
    LN_C_2, LN2_64, LN2_SPLIT, PI_OVER_4, TWO_OVER_PI,
};

/// The declared error bound of one elementary function of [`DoubleDouble`]:
/// `|z − f(x)| ≤ (units_of_u2 + growth_units_of_u2 · G) u² |f(x)| + absolute`,
/// for results of magnitude in [2⁻⁹⁶⁸, Ω] (below: `+ η`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FunctionBound {
    /// The function, as named in Rust (`"exp"`, `"powi"`, …).
    pub function: &'static str,
    /// The constant term, in units of `u² = 2⁻¹⁰⁶`.
    pub units_of_u2: f64,
    /// The coefficient of the growth term `G`, in units of `u²` (0 if none).
    pub growth_units_of_u2: f64,
    /// What `G` is: `""`, `"|n|"` (powi), `"|y ln x|"` (powf) or
    /// `"|tan x|"` (tan).
    pub growth: &'static str,
    /// The absolute floor (`2⁻²⁰⁰` for the circular functions, else 0).
    pub absolute: f64,
}

/// `2⁻²⁰⁰`, the absolute floor of the circular functions.
const TRIG_FLOOR: f64 = f64::from_bits((1023 - 200) << 52);
/// `2⁻⁹⁴ = 2⁻²⁰⁰/u²`: the growth coefficient of `tan`.
const TAN_GROWTH: f64 = f64::from_bits((1023 - 94) << 52);

const fn plain(function: &'static str, units_of_u2: f64) -> FunctionBound {
    FunctionBound {
        function,
        units_of_u2,
        growth_units_of_u2: 0.0,
        growth: "",
        absolute: 0.0,
    }
}

const fn circular(function: &'static str, units_of_u2: f64) -> FunctionBound {
    FunctionBound {
        function,
        units_of_u2,
        growth_units_of_u2: 0.0,
        growth: "",
        absolute: TRIG_FLOOR,
    }
}

impl DoubleDouble {
    /// The declared error bounds of the elementary functions (Chapter
    /// 11.11), one row per function, each proven in the chapter. The tests
    /// read them.
    pub const FUNCTION_BOUNDS: &'static [FunctionBound] = &[
        plain("exp", 8.0),
        plain("exp2", 8.0),
        plain("exp_m1", 8.0),
        plain("ln", 8.0),
        plain("ln_1p", 8.0),
        plain("log2", 12.0),
        plain("log10", 12.0),
        FunctionBound {
            function: "powi",
            units_of_u2: 15.0,
            growth_units_of_u2: 4.0,
            growth: "|n|",
            absolute: 0.0,
        },
        FunctionBound {
            function: "powf",
            units_of_u2: 16.0,
            // The contract's 8 does not cover the proof (Theorem F).
            growth_units_of_u2: 11.0,
            growth: "|y ln x|",
            absolute: 0.0,
        },
        circular("sin", 8.0),
        circular("cos", 8.0),
        FunctionBound {
            function: "tan",
            units_of_u2: 32.0,
            growth_units_of_u2: TAN_GROWTH,
            growth: "|tan x|",
            absolute: TRIG_FLOOR,
        },
        circular("sin_cos", 8.0),
        plain("asin", 32.0),
        plain("acos", 32.0),
        plain("atan", 8.0),
        plain("atan2", 32.0),
        plain("sinh", 32.0),
        plain("cosh", 32.0),
        plain("tanh", 32.0),
        plain("asinh", 32.0),
        plain("acosh", 32.0),
        plain("atanh", 32.0),
    ];
}

/// A table entry `(hi, lo)` as a double-double.
const fn entry(p: (u64, u64)) -> DoubleDouble {
    DoubleDouble::from_raw(f64::from_bits(p.0), f64::from_bits(p.1))
}

/// `x_h > EXP_OVERFLOW` gives `eˣ = +∞` (`ln MAX_dd ≈ 709.7827`; between
/// the two the computation overflows by itself, Theorem E (iv)).
const EXP_OVERFLOW: f64 = 709.79;
/// `x_h < EXP_UNDERFLOW` gives `eˣ < η/2`: `exp` returns `+0`, `exp_m1`
/// returns `−1`.
const EXP_UNDERFLOW: f64 = -745.2;
/// `2⁻⁴⁰⁰`: below it a series is its first term (to `2⁻⁴⁰⁰` relative), and
/// above it no product of a series underflows.
const TINY: f64 = f64::from_bits((1023 - 400) << 52);
/// The last term `r^m/m!` of `eʳ − 1` in Algorithm E (`|r| ≤ 0.005 43`).
const EXP_TERMS: usize = 11;
/// The last term in step M1 (`|r| ≤ 1/64`).
const EXPM1_TERMS: usize = 13;
/// `|x_h| < EXPM1_SMALL` takes step M1.
const EXPM1_SMALL: f64 = 0.5;
/// `|k| ≤ EXPM1_EXACT_K` makes `2ᵏ − 1` an `f64` (step M3).
const EXPM1_EXACT_K: i32 = 53;

/// `2ᵏ` for `|k| ≤ 1022`.
fn pow2(k: i32) -> f64 {
    f64::from_bits(u64::try_from(k + 1023).expect("|k| ≤ 1022") << 52)
}

/// An integer-valued `f64` of magnitude below `2³¹`, as an `i32`.
#[allow(clippy::cast_possible_truncation)] // the value is an integer below 2³¹
fn to_i32(n: f64) -> i32 {
    debug_assert!(n == n.trunc() && n.abs() < 2_147_483_648.0);
    n as i32
}

/// `P(r) = eʳ − 1` by `r + r² Q(r)`, `Q = Σ_{m=2}^{terms} r^(m−2)/m!` (step E3).
fn expm1_poly(r: DoubleDouble, terms: usize) -> DoubleDouble {
    if r.hi().abs() < TINY {
        return r;
    }
    let mut q = entry(INV_FACT[terms]);
    for m in (2..terms).rev() {
        q = q * r + entry(INV_FACT[m]);
    }
    r + r.square() * q
}

/// Steps E1–E2: `x = (64k + j) ln 2/64 + r`.
fn exp_reduce(x: DoubleDouble) -> (i32, usize, DoubleDouble) {
    let n = (x.hi() * INV_LN2_64).round();
    let r = if n == 0.0 {
        x
    } else {
        let mut r = x;
        for c in LN2_64 {
            r -= DoubleDouble::from_product(n, c);
        }
        r
    };
    let n = to_i32(n);
    (n >> 6, usize::try_from(n & 63).expect("in [0, 64)"), r)
}

/// Steps E3–E4.
fn exp_reconstruct(k: i32, j: usize, r: DoubleDouble) -> DoubleDouble {
    let t = entry(EXP2_64[j]);
    (t + t * expm1_poly(r, EXP_TERMS)).ldexp(k)
}

/// `(x_h − n/64) + x_ℓ` for `n = round(64 x_h)` or `trunc(64 x_h)`: exact
/// (and `x` itself, signed zero included, when `n = 0`).
fn minus_sixty_fourths(x: DoubleDouble, n: f64) -> DoubleDouble {
    if n == 0.0 {
        x
    } else {
        DoubleDouble::from_sum(x.hi() - n * (1.0 / 64.0), x.lo())
    }
}

impl DoubleDouble {
    /// `eˣ` (Algorithm E). Relative error at most `8u²` for results in
    /// `[2⁻⁹⁶⁸, Ω]` (proven: `4.6u²`), plus `η` below. `exp(±0) = 1`,
    /// `exp(+∞) = +∞`, `exp(−∞) = +0`; `+∞` on overflow, `+0` for `x <
    /// −745.2`.
    #[must_use]
    pub fn exp(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let h = self.hi();
        if h.is_nan() {
            return Self::NAN;
        }
        if h > EXP_OVERFLOW {
            return Self::INFINITY;
        }
        if h < EXP_UNDERFLOW {
            return Self::ZERO;
        }
        let (k, j, r) = exp_reduce(self);
        exp_reconstruct(k, j, r)
    }

    /// `2ˣ` (Algorithm E); exact for integer `x` whose power is
    /// representable. Bound as [`DoubleDouble::exp`].
    #[must_use]
    pub fn exp2(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let h = self.hi();
        if h.is_nan() {
            return Self::NAN;
        }
        if h > 1024.0 {
            return Self::INFINITY;
        }
        if h < -1076.0 {
            return Self::ZERO;
        }
        let n = (h * 64.0).round();
        let r = minus_sixty_fourths(self, n) * Self::LN_2;
        let n = to_i32(n);
        exp_reconstruct(n >> 6, usize::try_from(n & 63).expect("in [0, 64)"), r)
    }

    /// `eˣ − 1`, with relative accuracy near zero (Algorithm M): relative
    /// error at most `8u²` for results in `[2⁻⁹⁶⁸, Ω]` (proven: `7.2u²`).
    /// `exp_m1(±0) = ±0`, `exp_m1(+∞) = +∞`, `exp_m1(−∞) = −1`; for very
    /// negative `x` the pair `(−1, eˣ)`, exactly `−1` once `eˣ < η/2`.
    #[must_use]
    pub fn exp_m1(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let h = self.hi();
        if h.is_nan() {
            return Self::NAN;
        }
        if h > EXP_OVERFLOW {
            return Self::INFINITY;
        }
        if h < EXP_UNDERFLOW {
            return -Self::ONE;
        }
        if h.abs() < EXPM1_SMALL {
            // Step M1.
            let n = (h * 64.0).trunc();
            let p = expm1_poly(minus_sixty_fourths(self, n), EXPM1_TERMS);
            if n == 0.0 {
                return p;
            }
            let d = entry(EXPM1_64[usize::try_from(to_i32(n) + 31).expect("|N| ≤ 31")]);
            return (d + d * p) + p;
        }
        let (k, j, r) = exp_reduce(self);
        if k.abs() > EXPM1_EXACT_K {
            // Step M2.
            return exp_reconstruct(k, j, r) - 1.0;
        }
        // Step M3.
        let t = entry(EXP2_64[j]);
        let s = entry(EXP2_64_M1[j]).ldexp(k) + (pow2(k) - 1.0);
        s + (t * expm1_poly(r, EXP_TERMS)).ldexp(k)
    }
}

/// `RN(√2)`: `y_h ≥ SQRT_2_HI` halves `y` in step L1.
const SQRT_2_HI: f64 = core::f64::consts::SQRT_2;
/// The last term `z^m/m` of the series of `ln(1 + z)` (step L4).
const LN_TERMS: usize = 20;
/// `|x_h| < LN1P_SMALL` takes step L′1.
const LN1P_SMALL: f64 = 1.0 / 64.0;
/// `2⁵³`: from here on `1 + x` loses nothing that matters (step L′3).
const TWO_53: f64 = 9_007_199_254_740_992.0;

/// The unbiased exponent of a positive normal `f64`.
fn exponent(h: f64) -> i32 {
    i32::try_from((h.to_bits() >> 52) & 0x7ff).expect("11 bits") - 1023
}

/// A table index `j + offset` for `j ∈ [−offset, …]`.
fn index(j: i32, offset: i32) -> usize {
    usize::try_from(j + offset).expect("an index of the table")
}

/// `ln(1 + z)` by step L4.
fn ln1p_series(z: DoubleDouble) -> DoubleDouble {
    if z.hi().abs() < TINY {
        return z;
    }
    let coefficient = |m: usize| {
        let c = entry(INV_INT[m]);
        if m % 2 == 1 { -c } else { c }
    };
    let mut q = coefficient(LN_TERMS);
    for m in (2..LN_TERMS).rev() {
        q = q * z + coefficient(m);
    }
    z - z.square() * q
}

/// Steps L1–L4 for a finite `x > 0`: `(e, y_h, j, L)`.
fn ln_parts(x: DoubleDouble) -> (i32, f64, i32, DoubleDouble) {
    let (x, mut e) = if x.hi() < f64::MIN_POSITIVE {
        (x.ldexp(106), -106)
    } else {
        (x, 0)
    };
    let ex = exponent(x.hi());
    e += ex;
    let mut y = x.ldexp(-ex);
    if y.hi() >= SQRT_2_HI {
        y = y.ldexp(-1);
        e += 1;
    }
    let j = to_i32((64.0 * (y.hi() - 1.0)).trunc());
    let c = 1.0 + f64::from(j) / 64.0;
    let d = DoubleDouble::from_sum(y.hi() - c, y.lo());
    let z = if j == 0 { d } else { d / c };
    (e, y.hi(), j, ln1p_series(z))
}

/// The two cancelling cases of step L5.
fn ln_cancelling(e: i32, y_hi: f64, j: i32, l: DoubleDouble) -> Option<DoubleDouble> {
    if e == 1 && y_hi < 1.0 {
        Some(entry(LN_2C[index(j, 18)]) + l)
    } else if e == -1 && y_hi >= 1.0 {
        Some(entry(LN_C_2[index(j, 0)]) + l)
    } else {
        None
    }
}

/// `ln y` of step L5.
fn ln_y(j: i32, l: DoubleDouble) -> DoubleDouble {
    if j == 0 {
        l
    } else {
        entry(LN_C[index(j, 18)]) + l
    }
}

/// Step L6: `e ln 2`, `e ≠ 0`, `|e| < 2¹¹`.
fn e_ln2(e: i32) -> DoubleDouble {
    let e = f64::from(e);
    let a = e * LN2_SPLIT[0];
    let (p, q) = two_prod(e, LN2_SPLIT[1]);
    let (s, t) = fast_two_sum(a, p);
    DoubleDouble::from_sum(s, t + (q + e * LN2_SPLIT[2]))
}

/// Algorithm L for a finite `x > 0`.
fn ln_positive(x: DoubleDouble) -> DoubleDouble {
    let (e, y_hi, j, l) = ln_parts(x);
    if let Some(z) = ln_cancelling(e, y_hi, j, l) {
        return z;
    }
    let ly = ln_y(j, l);
    if e == 0 { ly } else { e_ln2(e) + ly }
}

impl DoubleDouble {
    /// Natural logarithm (Algorithm L): relative error at most `8u²`
    /// (proven: `6.7u²`); `ln(1) = +0`. NaN for `x < 0`, `−∞` for `±0`,
    /// `+∞` for `+∞`.
    #[must_use]
    pub fn ln(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let h = self.hi();
        if h.is_nan() || h < 0.0 {
            return Self::NAN;
        }
        if h == 0.0 {
            return -Self::INFINITY;
        }
        if h == f64::INFINITY {
            return Self::INFINITY;
        }
        ln_positive(self)
    }

    /// `ln(1 + x)`, with relative accuracy near zero (Algorithm L′):
    /// relative error at most `8u²` (proven: `7.7u²`). `±0` for `±0`, `−∞`
    /// for `−1`, NaN for `x < −1`.
    #[must_use]
    pub fn ln_1p(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let h = self.hi();
        if h.is_nan() || h < -1.0 || (h == -1.0 && self.lo() < 0.0) {
            return Self::NAN;
        }
        if h == -1.0 && self.lo() == 0.0 {
            return -Self::INFINITY;
        }
        if h == f64::INFINITY {
            return Self::INFINITY;
        }
        if h.abs() < LN1P_SMALL {
            // Step L′1.
            return ln1p_series(self);
        }
        if h <= -0.5 {
            // Step L′2.
            return ln_positive(Self::from_sum(1.0 + h, self.lo()));
        }
        // Step L′3.
        let w = self + 1.0;
        if w.hi() >= TWO_53 {
            return ln_positive(w);
        }
        let delta = self - Self::from_sum(w.hi() - 1.0, w.lo());
        ln_positive(w) + delta.hi() / w.hi()
    }

    /// Base-2 logarithm; exact for powers of two. Relative error at most
    /// `12u²` (proven: `11.7u²`). Special cases as [`DoubleDouble::ln`].
    #[must_use]
    pub fn log2(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let h = self.hi();
        if h.is_nan() || h < 0.0 {
            return Self::NAN;
        }
        if h == 0.0 {
            return -Self::INFINITY;
        }
        if h == f64::INFINITY {
            return Self::INFINITY;
        }
        if self.lo() == 0.0 {
            // A power of two: one bit in the significand (normal or not).
            let bits = h.to_bits();
            let frac = bits & ((1 << 52) - 1);
            if bits >> 52 != 0 && frac == 0 {
                return Self::from_f64(f64::from(exponent(h)));
            }
            if bits >> 52 == 0 && frac.is_power_of_two() {
                let k = i32::try_from(frac.trailing_zeros()).expect("< 52");
                return Self::from_f64(f64::from(k - 1074));
            }
        }
        let (e, y_hi, j, l) = ln_parts(self);
        if let Some(z) = ln_cancelling(e, y_hi, j, l) {
            return z * Self::LOG2_E;
        }
        let t = ln_y(j, l) * Self::LOG2_E;
        if e == 0 { t } else { t + f64::from(e) }
    }

    /// Base-10 logarithm, `ln x · LOG10_E`: relative error at most `12u²`
    /// (proven: `11.7u²`). Special cases as [`DoubleDouble::ln`].
    #[must_use]
    pub fn log10(self) -> Self {
        self.ln() * Self::LOG10_E
    }
}

/// Renormalizes `a > 0` into `[1, 2)`: `(a 2^−k, k)`.
fn renormalize(a: DoubleDouble) -> (DoubleDouble, i64) {
    let k = exponent(a.hi());
    (a.ldexp(-k), i64::from(k))
}

/// `x = 2^e y`, `y ∈ [1, 2)`, for a finite `x > 0`.
fn split(x: DoubleDouble) -> (DoubleDouble, i64) {
    if x.hi() < f64::MIN_POSITIVE {
        let (y, e) = renormalize(x.ldexp(106));
        (y, e - 106)
    } else {
        renormalize(x)
    }
}

/// `y` (finite) is an integer: both components are.
fn is_integer(y: DoubleDouble) -> bool {
    y.hi().trunc() == y.hi() && y.lo().trunc() == y.lo()
}

/// An integer `f64` is odd.
fn is_odd_f64(v: f64) -> bool {
    v.abs() < TWO_53 && (v * 0.5).trunc() != v * 0.5
}

/// An integer double-double is odd (by the trailing component when it is
/// nonzero, since then `|y_h| ≥ 2⁵³` is even).
fn is_odd(y: DoubleDouble) -> bool {
    if y.lo() == 0.0 {
        is_odd_f64(y.hi())
    } else {
        is_odd_f64(y.lo())
    }
}

/// `±∞` (`big`) or `±0`, negative when `neg`.
fn signed_limit(big: bool, neg: bool) -> DoubleDouble {
    let z = if big {
        DoubleDouble::INFINITY
    } else {
        DoubleDouble::ZERO
    };
    if neg { -z } else { z }
}

impl DoubleDouble {
    /// `xⁿ` by binary powering with a separate exponent (Algorithm P): no
    /// spurious overflow or underflow. Relative error at most `(15 +
    /// 4|n|)u²` (proven: `(11 + 4|n|)u²`); exact for `n ∈ {0, 1}`, one
    /// square (`3u²`) for `n = 2`, [`DoubleDouble::recip`] for `n = −1`.
    /// The special cases of IEEE 754 `pown`: `powi(x, 0) = 1` for every `x`,
    /// NaN included; `±0` and `±∞` by the sign and parity of `n`.
    #[must_use]
    pub fn powi(self, n: i32) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        if n == 0 {
            return Self::ONE;
        }
        if n == 1 {
            return self;
        }
        let h = self.hi();
        if h.is_nan() {
            return Self::NAN;
        }
        let neg = h.is_sign_negative() && n & 1 == 1;
        if h == 0.0 {
            return signed_limit(n < 0, neg);
        }
        if h.is_infinite() {
            return signed_limit(n > 0, neg);
        }
        if n == -1 {
            return self.recip();
        }
        // Steps P1–P2.
        let (mut b, mut eb) = split(self.abs());
        let (mut a, mut ea) = (Self::ONE, 0_i64);
        let mut m = n.unsigned_abs();
        loop {
            if m & 1 == 1 {
                let (r, k) = renormalize(a * b);
                a = r;
                ea += eb + k;
            }
            m >>= 1;
            if m == 0 {
                break;
            }
            let (r, k) = renormalize(b.square());
            b = r;
            eb = 2 * eb + k;
        }
        // Step P3.
        if n < 0 {
            a = a.recip();
            ea = -ea;
        }
        let z = match i32::try_from(ea) {
            Ok(e) => a.ldexp(e),
            Err(_) => signed_limit(ea > 0, false),
        };
        if neg { -z } else { z }
    }

    /// `xʸ` as `exp(y ln x)` (Theorem F): relative error at most `(16 +
    /// 11|y ln x|)u²` (proven: `(4.6 + 10.7|y ln x|)u²`); `x` itself for `y =
    /// 1`. The special cases of IEEE 754 `pow` (C99 Annex F), as `f64`:
    /// `pow(x, ±0) = 1` and `pow(1, y) = 1` even for NaN; NaN for `x < 0`
    /// and `y` not an integer; `±0`, `±∞` and `±1` by the usual limits.
    #[must_use]
    pub fn powf(self, y: Self) -> Self {
        debug_assert!(self.check_invariants().is_ok() && y.check_invariants().is_ok());
        let (xh, yh) = (self.hi(), y.hi());
        if yh == 0.0 || (xh == 1.0 && self.lo() == 0.0) {
            return Self::ONE;
        }
        if xh.is_nan() || yh.is_nan() {
            return Self::NAN;
        }
        if yh == 1.0 && y.lo() == 0.0 {
            return self;
        }
        let y_odd = yh.is_finite() && is_integer(y) && is_odd(y);
        if xh == 0.0 {
            return signed_limit(yh < 0.0, xh.is_sign_negative() && y_odd);
        }
        if yh.is_infinite() {
            if xh == -1.0 && self.lo() == 0.0 {
                return Self::ONE;
            }
            return signed_limit((yh > 0.0) == (self.abs() > Self::ONE), false);
        }
        if xh.is_infinite() {
            return signed_limit(yh > 0.0, xh < 0.0 && y_odd);
        }
        if xh < 0.0 && !is_integer(y) {
            return Self::NAN;
        }
        let z = (y * ln_positive(self.abs())).exp();
        if xh < 0.0 && y_odd { -z } else { z }
    }
}

/// `RN(π/4)`: `|x|_h ≤ FRAC_PI_4_HI` needs no reduction (step S1).
const FRAC_PI_4_HI: f64 = core::f64::consts::FRAC_PI_4;
/// The last term `r^(2m+1)/(2m+1)!` of `sin r`, `|r| ≤ π/4` (step S2).
const SIN_TERMS: usize = 14;
/// The last term of `sin h`, `|h| ≤ π/8`, inside `cos r` (step S2).
const SIN_HALF_TERMS: usize = 11;
/// `I = e + WINDOW_SHIFT` in step R1: 318 fraction bits, 53 for `m`, 11
/// guard bits.
const WINDOW_SHIFT: i32 = 382;

/// The 64 bits of `N = ⌊2¹⁴⁰⁸ · 2/π⌋` from bit `t` up (bit 0 the least
/// significant; zeros outside `[0, 1408)`).
fn two_over_pi_bits(t: i32) -> u64 {
    // Little-endian word k of N is the big-endian word 21 − k.
    let word = |k: i32| -> u64 {
        if (0..22).contains(&k) {
            TWO_OVER_PI[usize::try_from(21 - k).expect("in range")]
        } else {
            0
        }
    };
    let k = t.div_euclid(64);
    let off = t.rem_euclid(64);
    if off == 0 {
        word(k)
    } else {
        (word(k) >> off) | (word(k + 1) << (64 - off))
    }
}

/// `x = m 2ᵉ` for a finite `x ≥ 0`, `m < 2⁵³` an integer.
fn integer_significand(x: f64) -> (u64, i32) {
    let bits = x.to_bits();
    let be = i32::try_from(bits >> 52).expect("x ≥ 0");
    let frac = bits & ((1 << 52) - 1);
    if be == 0 {
        (frac, -1074)
    } else {
        (frac | (1 << 52), be - 1075)
    }
}

/// `a · b` for little-endian words, `out.len() = a.len() + b.len()`.
fn mul_words(a: &[u64], b: &[u64], out: &mut [u64]) {
    out.fill(0);
    for (i, &ai) in a.iter().enumerate() {
        let mut carry = 0_u128;
        for (j, &bj) in b.iter().enumerate() {
            let t = u128::from(ai) * u128::from(bj) + u128::from(out[i + j]) + carry;
            out[i + j] = u64::try_from(t & u128::from(u64::MAX)).expect("the low word");
            carry = t >> 64;
        }
        out[i + b.len()] = u64::try_from(carry).expect("a word");
    }
}

/// Step R1 for one component: `A_c` (five little-endian words), negated
/// modulo `2³²⁰` for a negative `c` (a trailing component may be).
fn component(c: f64) -> [u64; 5] {
    let a = magnitude_component(c.abs());
    if c >= 0.0 {
        return a;
    }
    let mut n = [0_u64; 5];
    let mut borrow = false;
    for k in 0..5 {
        let (s1, b1) = 0_u64.overflowing_sub(a[k]);
        let (s2, b2) = s1.overflowing_sub(u64::from(borrow));
        n[k] = s2;
        borrow = b1 || b2;
    }
    n
}

/// Step R1 for `c ≥ 0`.
fn magnitude_component(c: f64) -> [u64; 5] {
    let (m, e) = integer_significand(c);
    let i = e + WINDOW_SHIFT;
    if i < 1 || m == 0 {
        return [0; 5];
    }
    // W = ⌊2^I 2/π⌋ mod 2³⁸⁴ = bits [1408 − I, 1408 − I + 384) of N.
    let base = 1408 - i;
    let w: [u64; 6] =
        core::array::from_fn(|k| two_over_pi_bits(base + 64 * i32::try_from(k).expect("k < 6")));
    let mut p = [0_u64; 7];
    mul_words(&[m], &w, &mut p);
    [p[1], p[2], p[3], p[4], p[5]]
}

/// Bits `[t, t + 53)` of a little-endian number (zeros below bit 0).
fn chunk53(r: &[u64; 6], t: i32) -> u64 {
    let get = |b: i32| -> u64 {
        if (0..384).contains(&b) {
            let b = usize::try_from(b).expect("b ≥ 0");
            (r[b / 64] >> (b % 64)) & 1
        } else {
            0
        }
    };
    // 53 bits, assembled from two shifted words when t ≥ 0.
    if t >= 0 {
        let k = usize::try_from(t / 64).expect("t ≥ 0");
        let off = t % 64;
        let lo = r[k] >> off;
        let hi = if off == 0 || k + 1 >= 6 {
            0
        } else {
            r[k + 1] << (64 - off)
        };
        (lo | hi) & ((1 << 53) - 1)
    } else {
        (0..53).fold(0, |acc, b| acc | (get(t + b) << b))
    }
}

/// Algorithm R for a finite `x > π/4`: `(q, r)` with `x ≡ qπ/2 + r`.
fn reduce(x: DoubleDouble) -> (u32, DoubleDouble) {
    // Steps R1–R2.
    let (a, b) = (component(x.hi()), component(x.lo()));
    let mut acc = [0_u64; 5];
    let mut carry = false;
    for k in 0..5 {
        let (s1, c1) = a[k].overflowing_add(b[k]);
        let (s2, c2) = s1.overflowing_add(u64::from(carry));
        acc[k] = s2;
        carry = c1 || c2;
    }
    // Step R3: two integer bits (62, 63 of word 4) and a 318-bit fraction.
    let int = u32::try_from(acc[4] >> 62).expect("two bits");
    let half = (acc[4] >> 61) & 1 == 1;
    let mut f = acc;
    f[4] &= (1 << 62) - 1;
    let (q, negative, g) = if half {
        // G = 2(2³¹⁸ − F).
        let mut d = [0_u64; 5];
        let mut borrow = false;
        for k in 0..5 {
            let top: u64 = if k == 4 { 1 << 62 } else { 0 };
            let (s1, b1) = top.overflowing_sub(f[k]);
            let (s2, b2) = s1.overflowing_sub(u64::from(borrow));
            d[k] = s2;
            borrow = b1 || b2;
        }
        ((int + 1) % 4, true, d)
    } else {
        (int, false, f)
    };
    // G = 2g', as an integer of 319 bits: one shift.
    let mut gg = [0_u64; 5];
    for k in (0..5).rev() {
        gg[k] = (g[k] << 1) | if k == 0 { 0 } else { g[k - 1] >> 63 };
    }
    // Step R4: R = ⌊G P / 2²⁵⁶⌋.
    let p: [u64; 5] = core::array::from_fn(|k| PI_OVER_4[4 - k]);
    let mut prod = [0_u64; 10];
    mul_words(&gg, &p, &mut prod);
    let rw: [u64; 6] = core::array::from_fn(|k| prod[k + 4]);
    // Step R5.
    let Some(top) = (0..6).rev().find(|&k| rw[k] != 0) else {
        return (q, DoubleDouble::ZERO);
    };
    let l = 64 * i32::try_from(top).expect("k < 6") + 63
        - i32::try_from(rw[top].leading_zeros()).expect("< 64");
    let v = |t: i32| {
        // The chunk at bit t, as the exact value c · 2^(t − 382).
        #[allow(clippy::cast_precision_loss)] // 53 bits: exact
        let c = chunk53(&rw, t) as f64;
        DoubleDouble::from_f64(c).ldexp(t - 382).hi()
    };
    let (s, t) = fast_two_sum(v(l - 52), v(l - 105));
    let r = DoubleDouble::from_sum(s, t + v(l - 158));
    (q, if negative { -r } else { r })
}

/// `sin r` for `|r| ≤ π/4` by step S2, the series to `terms`.
fn sin_kernel(r: DoubleDouble, terms: usize) -> DoubleDouble {
    if r.hi().abs() < TINY {
        return r;
    }
    let r2 = r.square();
    let coefficient = |m: usize| {
        let c = entry(INV_FACT[2 * m + 1]);
        if m.is_multiple_of(2) { -c } else { c }
    };
    let mut q = coefficient(terms);
    for m in (1..terms).rev() {
        q = q * r2 + coefficient(m);
    }
    r - (r * r2) * q
}

/// `cos r = 1 − 2 sin²(r/2)` for `|r| ≤ π/4` (step S2).
fn cos_kernel(r: DoubleDouble) -> DoubleDouble {
    let s = sin_kernel(r.ldexp(-1), SIN_HALF_TERMS);
    -s.square().ldexp(1) + 1.0
}

/// Step S1 on `|x|`: `(q, r)`.
fn quadrant(a: DoubleDouble) -> (u32, DoubleDouble) {
    if a.hi() <= FRAC_PI_4_HI {
        (0, a)
    } else {
        reduce(a)
    }
}

/// `sin x` from `(q, r)` and the sign of `x` (step S3).
fn sin_from(q: u32, r: DoubleDouble, negative: bool) -> DoubleDouble {
    let z = match q {
        0 => sin_kernel(r, SIN_TERMS),
        1 => cos_kernel(r),
        2 => -sin_kernel(r, SIN_TERMS),
        _ => -cos_kernel(r),
    };
    if negative { -z } else { z }
}

/// `cos x` from `(q, r)` (step S3).
fn cos_from(q: u32, r: DoubleDouble) -> DoubleDouble {
    match q {
        0 => cos_kernel(r),
        1 => -sin_kernel(r, SIN_TERMS),
        2 => -cos_kernel(r),
        _ => sin_kernel(r, SIN_TERMS),
    }
}

impl DoubleDouble {
    /// Sine (Algorithms R and K): error at most `8u²|sin x| + 2⁻²⁰⁰` for
    /// every finite `x` (proven: `6.9u²|sin x| + 2⁻³¹⁵`), so relative when
    /// `|sin x| ≥ 2⁻⁹⁴`. `sin(±0) = ±0`; NaN for `±∞`; odd bit for bit.
    #[must_use]
    pub fn sin(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        if !self.is_finite() {
            return Self::NAN;
        }
        let (q, r) = quadrant(self.abs());
        sin_from(q, r, self.hi().is_sign_negative())
    }

    /// Cosine; bound as [`DoubleDouble::sin`]. `cos(±0) = 1`; NaN for `±∞`;
    /// even bit for bit.
    #[must_use]
    pub fn cos(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        if !self.is_finite() {
            return Self::NAN;
        }
        let (q, r) = quadrant(self.abs());
        cos_from(q, r)
    }

    /// Tangent (Algorithms R and K, then D2): error at most `32u²|tan x| +
    /// 2⁻²⁰⁰(1 + tan²x)` (proven: `27.4u²|tan x| + 2⁻³¹⁵(1 + tan²x)`).
    /// `tan(±0) = ±0`; NaN for `±∞`; odd bit for bit.
    #[must_use]
    pub fn tan(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        if !self.is_finite() {
            return Self::NAN;
        }
        let (q, r) = quadrant(self.abs());
        let (s, c) = (sin_kernel(r, SIN_TERMS), cos_kernel(r));
        let z = if q % 2 == 0 { s / c } else { -(c / s) };
        if self.hi().is_sign_negative() { -z } else { z }
    }

    /// `(sin x, cos x)` from one argument reduction; the same bits as
    /// [`DoubleDouble::sin`] and [`DoubleDouble::cos`].
    #[must_use]
    pub fn sin_cos(self) -> (Self, Self) {
        debug_assert!(self.check_invariants().is_ok());
        if !self.is_finite() {
            return (Self::NAN, Self::NAN);
        }
        let (q, r) = quadrant(self.abs());
        (sin_from(q, r, self.hi().is_sign_negative()), cos_from(q, r))
    }
}

/// `|a_h| < ATAN_DIRECT` takes step T1.
const ATAN_DIRECT: f64 = 1.0 / 64.0;
/// `a_h ≥ ATAN_RECIP` takes step T3.
const ATAN_RECIP: f64 = 8.0;
/// The last term `t^(2m+1)/(2m+1)` of the series of `atan t` (`|t| ≤ 1/64`).
const ATAN_TERMS: usize = 9;
/// `2⁶⁰⁰`: the scale of a tiny quotient in `atan2`.
const ATAN2_SCALE: i32 = 600;

/// `atan t` by the series of step T1, `|t| ≤ 1/64`.
fn atan_series(t: DoubleDouble) -> DoubleDouble {
    if t.hi().abs() < TINY {
        return t;
    }
    let t2 = t.square();
    let coefficient = |m: usize| {
        let c = entry(INV_INT[2 * m + 1]);
        if m.is_multiple_of(2) { -c } else { c }
    };
    let mut q = coefficient(ATAN_TERMS);
    for m in (1..ATAN_TERMS).rev() {
        q = q * t2 + coefficient(m);
    }
    t - (t * t2) * q
}

/// Algorithm T for a finite `a ≥ 0`.
fn atan_positive(a: DoubleDouble) -> DoubleDouble {
    let h = a.hi();
    if h < ATAN_DIRECT {
        return atan_series(a);
    }
    if h >= ATAN_RECIP {
        return DoubleDouble::FRAC_PI_2 - atan_positive(a.recip());
    }
    let e = exponent(h);
    let i = i32::try_from((h.to_bits() >> 48) & 15).expect("four bits");
    let c = f64::from(33 + 2 * i) * pow2(e - 5);
    let theta = entry(ATAN_C[index(16 * (e + 6) + i, 0)]);
    let t = DoubleDouble::from_sum(h - c, a.lo()) / (a * c + 1.0);
    theta + atan_series(t)
}

/// `√((1 − a)(1 + a))` for `0 ≤ a ≤ 1` (`asin`, `acos`).
fn cosine_of_arcsine(a: DoubleDouble) -> DoubleDouble {
    if a.hi() >= 0.5 {
        (DoubleDouble::from_sum(1.0 - a.hi(), -a.lo()) * (a + 1.0)).sqrt()
    } else {
        (-a.square() + 1.0).sqrt()
    }
}

/// `|x| > 1` (NaN excluded).
fn beyond_one(a: DoubleDouble) -> bool {
    a.hi() > 1.0 || (a.hi() == 1.0 && a.lo() > 0.0)
}

impl DoubleDouble {
    /// Arctangent in `[−π/2, π/2]` (Algorithm T): relative error at most
    /// `8u²` (proven: `5.8u²`). `atan(±0) = ±0`, `atan(±∞) = ±π/2`; odd bit
    /// for bit.
    #[must_use]
    pub fn atan(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let h = self.hi();
        if h.is_nan() {
            return Self::NAN;
        }
        let z = if h.is_infinite() {
            Self::FRAC_PI_2
        } else {
            atan_positive(self.abs())
        };
        if h.is_sign_negative() { -z } else { z }
    }

    /// Four-quadrant arctangent of `self / other` (`self` is `y`), in `[−π,
    /// π]`: relative error at most `32u²` (proven: `20u²`). The special
    /// cases of `f64::atan2`: `(±0, −0) → ±π`, `(±0, +0) → ±0`, `(±0, x < 0)
    /// → ±π`, `(±0, x > 0) → ±0`, `(y ≠ 0, ±0) → ±π/2` by the sign of `y`,
    /// `(±y, −∞) → ±π` and `(±y, +∞) → ±0` for finite `y`, `(±∞, x finite) →
    /// ±π/2`, `(±∞, −∞) → ±3π/4`, `(±∞, +∞) → ±π/4`; NaN in either → NaN.
    #[must_use]
    pub fn atan2(self, other: Self) -> Self {
        debug_assert!(self.check_invariants().is_ok() && other.check_invariants().is_ok());
        let (y, x) = (self, other);
        let (yh, xh) = (y.hi(), x.hi());
        if yh.is_nan() || xh.is_nan() {
            return Self::NAN;
        }
        let negative = yh.is_sign_negative();
        let sign = |z: Self| if negative { -z } else { z };
        let x_left = xh.is_sign_negative();
        if yh == 0.0 {
            return sign(if x_left { Self::PI } else { Self::ZERO });
        }
        if xh == 0.0 {
            return sign(Self::FRAC_PI_2);
        }
        if yh.is_infinite() {
            return sign(if xh.is_infinite() {
                if x_left {
                    entry(FRAC_3PI_4)
                } else {
                    Self::FRAC_PI_4
                }
            } else {
                Self::FRAC_PI_2
            });
        }
        if xh.is_infinite() {
            return sign(if x_left { Self::PI } else { Self::ZERO });
        }
        let (ay, ax) = (y.abs(), x.abs());
        let s = exponent(ay.hi().max(ax.hi()).max(f64::MIN_POSITIVE));
        let (ay, ax) = (ay.ldexp(-s), ax.ldexp(-s));
        let quotient = |n: Self, d: Self| {
            if n.hi() < TINY {
                (n.ldexp(ATAN2_SCALE) / d).ldexp(-ATAN2_SCALE)
            } else {
                n / d
            }
        };
        let z = if ay <= ax {
            let z = atan_positive(quotient(ay, ax));
            if x_left { Self::PI - z } else { z }
        } else {
            let z = atan_positive(quotient(ax, ay));
            if x_left {
                Self::FRAC_PI_2 + z
            } else {
                Self::FRAC_PI_2 - z
            }
        };
        sign(z)
    }

    /// Arcsine in `[−π/2, π/2]`: relative error at most `32u²` (proven:
    /// `26.1u²`). NaN outside `[−1, 1]`; `asin(±1) = ±π/2`, `asin(±0) = ±0`;
    /// odd bit for bit.
    #[must_use]
    pub fn asin(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let a = self.abs();
        if self.is_nan() || beyond_one(a) {
            return Self::NAN;
        }
        if a.hi() < TINY {
            return self;
        }
        let w = cosine_of_arcsine(a);
        let z = if a <= w {
            atan_positive(a / w)
        } else {
            Self::FRAC_PI_2 - atan_positive(w / a)
        };
        if self.hi().is_sign_negative() { -z } else { z }
    }

    /// Arccosine in `[0, π]`: relative error at most `32u²` (proven:
    /// `26.1u²`). NaN outside `[−1, 1]`; `acos(1) = +0`, `acos(−1) = π`,
    /// `acos(±0) = π/2`.
    #[must_use]
    pub fn acos(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let a = self.abs();
        if self.is_nan() || beyond_one(a) {
            return Self::NAN;
        }
        let w = cosine_of_arcsine(a);
        let left = self.hi() < 0.0;
        if w <= a {
            let z = atan_positive(w / a);
            if left { Self::PI - z } else { z }
        } else {
            let z = atan_positive(a / w);
            if left {
                Self::FRAC_PI_2 + z
            } else {
                Self::FRAC_PI_2 - z
            }
        }
    }
}

/// `sinh`, `cosh` above it use `exp(a − 1) · e/2`.
const HYPER_LARGE: f64 = 709.0;
/// `tanh` below it uses `exp_m1(2a)`, above it `exp(−2a)`.
const TANH_SPLIT: f64 = 0.3125;
/// `2⁶⁰`: `asinh`, `acosh` above it are `ln x + ln 2`.
const TWO_60: f64 = 1_152_921_504_606_846_976.0;
/// `e/2` (the table `E`, scaled exactly).
const E_HALF: DoubleDouble =
    DoubleDouble::from_raw(DoubleDouble::E.hi() * 0.5, DoubleDouble::E.lo() * 0.5);

/// `exp(a − 1) · e/2 = eᵃ/2` for `a ≥ 709` (`a − 1` exact).
fn half_exp_large(a: DoubleDouble) -> DoubleDouble {
    DoubleDouble::from_sum(a.hi() - 1.0, a.lo()).exp() * E_HALF
}

/// `ln x + ln 2` for `x ≥ 2⁶⁰`.
fn ln_twice(x: DoubleDouble) -> DoubleDouble {
    ln_positive(x) + DoubleDouble::LN_2
}

impl DoubleDouble {
    /// Hyperbolic sine: relative error at most `32u²` (proven: `18.7u²`).
    /// `±0`, `±∞` for `±0`, `±∞`; `±∞` on overflow; odd bit for bit.
    #[must_use]
    pub fn sinh(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let a = self.abs();
        if self.is_nan() {
            return Self::NAN;
        }
        if a.hi() < TINY {
            return self;
        }
        let z = if a.hi() > HYPER_LARGE {
            half_exp_large(a)
        } else {
            let e = a.exp_m1();
            (e + e / (e + 1.0)).ldexp(-1)
        };
        if self.hi().is_sign_negative() { -z } else { z }
    }

    /// Hyperbolic cosine: relative error at most `32u²` (proven: `15.1u²`).
    /// `cosh(±0) = 1`, `+∞` for `±∞` and on overflow; even bit for bit.
    #[must_use]
    pub fn cosh(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let a = self.abs();
        if self.is_nan() {
            return Self::NAN;
        }
        if a.hi() > HYPER_LARGE {
            return half_exp_large(a);
        }
        let e = a.exp();
        (e + e.recip()).ldexp(-1)
    }

    /// Hyperbolic tangent: relative error at most `32u²` (proven:
    /// `26.4u²`). `±0` for `±0`, `±1` for `±∞`; for large `|x|` the pair
    /// `(±1, ∓2e^(−2|x|))`, exactly `±1` once that is below `η/2`; odd bit
    /// for bit.
    #[must_use]
    pub fn tanh(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let a = self.abs();
        if self.is_nan() {
            return Self::NAN;
        }
        if a.hi() < TINY {
            return self;
        }
        let z = if a.hi() < TANH_SPLIT {
            let e = a.ldexp(1).exp_m1();
            e / (e + 2.0)
        } else {
            let f = (-a.ldexp(1)).exp();
            (-f + 1.0) / (f + 1.0)
        };
        if self.hi().is_sign_negative() { -z } else { z }
    }

    /// Inverse hyperbolic sine: relative error at most `32u²` (proven:
    /// `19.7u²`). `±0`, `±∞` for `±0`, `±∞`; odd bit for bit.
    #[must_use]
    pub fn asinh(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let a = self.abs();
        if self.is_nan() {
            return Self::NAN;
        }
        if a.hi() < TINY || a.hi().is_infinite() {
            return self;
        }
        let z = if a.hi() >= TWO_60 {
            ln_twice(a)
        } else if a.hi() >= 2.0 {
            ln_positive(a + (a.square() + 1.0).sqrt())
        } else {
            let a2 = a.square();
            (a + a2 / ((a2 + 1.0).sqrt() + 1.0)).ln_1p()
        };
        if self.hi().is_sign_negative() { -z } else { z }
    }

    /// Inverse hyperbolic cosine: relative error at most `32u²` (proven:
    /// `16.8u²`). NaN for `x < 1`, `acosh(1) = +0`, `+∞` for `+∞`.
    #[must_use]
    pub fn acosh(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let h = self.hi();
        if h.is_nan() || h < 1.0 || (h == 1.0 && self.lo() < 0.0) {
            return Self::NAN;
        }
        if h == f64::INFINITY {
            return Self::INFINITY;
        }
        if h >= TWO_60 {
            ln_twice(self)
        } else if h >= 2.0 {
            ln_positive(self + (self.square() - 1.0).sqrt())
        } else {
            let d = Self::from_sum(h - 1.0, self.lo());
            (d + (d * (d + 2.0)).sqrt()).ln_1p()
        }
    }

    /// Inverse hyperbolic tangent: relative error at most `32u²` (proven:
    /// `24.7u²`). NaN outside `[−1, 1]`, `±∞` at `±1`, `±0` for `±0`; odd bit
    /// for bit.
    #[must_use]
    pub fn atanh(self) -> Self {
        debug_assert!(self.check_invariants().is_ok());
        let a = self.abs();
        if self.is_nan() || beyond_one(a) {
            return Self::NAN;
        }
        let negative = self.hi().is_sign_negative();
        if a == Self::ONE {
            return if negative {
                -Self::INFINITY
            } else {
                Self::INFINITY
            };
        }
        if a.hi() < TINY {
            return self;
        }
        let den = if a.hi() >= 0.5 {
            Self::from_sum(1.0 - a.hi(), -a.lo())
        } else {
            -a + 1.0
        };
        let z = (a.ldexp(1) / den).ln_1p().ldexp(-1);
        if negative { -z } else { z }
    }
}

#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod oracle;

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::assign_op_pattern
)]
mod tests {
    use super::oracle::{self, Dy, Iv};
    use super::*;

    /// The first oracle's value of an enclosure: `dashu-float` at 2 400
    /// bits gives `c`; the nearest double-double of `c` and of the
    /// enclosure must agree with the table entry.
    fn check_entry(name: &str, i: usize, e: (u64, u64), f: &oracle::F, iv: &Iv) {
        let want = oracle::nearest_dd(&oracle::dy(f));
        let got = (f64::from_bits(e.0), f64::from_bits(e.1));
        assert_eq!(
            (got.0.to_bits(), got.1.to_bits()),
            (want.0.to_bits(), want.1.to_bits()),
            "{name}[{i}] against dashu-float"
        );
        let second = iv.nearest_dd().expect("a narrow enclosure");
        assert_eq!(
            (got.0.to_bits(), got.1.to_bits()),
            (second.0.to_bits(), second.1.to_bits()),
            "{name}[{i}] against the enclosure"
        );
    }

    fn fz(n: i64, d: i64) -> oracle::F {
        oracle::F::from(n).with_precision(2400).value() / oracle::F::from(d)
    }

    fn q(n: i64, d: i64) -> Iv {
        Iv::int(n).div(&Iv::int(d))
    }

    #[test]
    fn function_bounds_cover_every_function() {
        let names = [
            "exp", "exp2", "exp_m1", "ln", "ln_1p", "log2", "log10", "powi", "powf", "sin", "cos",
            "tan", "sin_cos", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh",
            "acosh", "atanh",
        ];
        assert_eq!(DoubleDouble::FUNCTION_BOUNDS.len(), names.len());
        for (b, n) in DoubleDouble::FUNCTION_BOUNDS.iter().zip(names) {
            assert_eq!(b.function, n);
            assert!(b.units_of_u2 >= 8.0 && b.growth_units_of_u2 >= 0.0);
        }
        assert_eq!(TRIG_FLOOR, oracle::pow2f(-200));
        assert_eq!(TAN_GROWTH, oracle::pow2f(-94));
    }

    #[test]
    fn tables_are_the_nearest_double_doubles() {
        let p = 2400;
        let one = oracle::F::from(1).with_precision(p).value();
        let ln2f = oracle::F::from(2).with_precision(p).value().ln();
        let pif = oracle::F::pi(p);
        let ln2 = oracle::ln2();
        // 2^(j/64) and 2^(j/64) − 1.
        for j in 0..64 {
            let f = (ln2f.clone() * fz(j, 64)).exp();
            let iv = oracle::exp_iv(&ln2.mul(&q(j, 64)));
            check_entry("EXP2_64", j as usize, EXP2_64[j as usize], &f, &iv);
            check_entry(
                "EXP2_64_M1",
                j as usize,
                EXP2_64_M1[j as usize],
                &(f - one.clone()),
                &iv.sub(&Iv::int(1)),
            );
        }
        for n in -31..=31_i64 {
            let i = (n + 31) as usize;
            let f = fz(n, 64).exp() - one.clone();
            let iv = oracle::exp(&Dy::int(n).scale(-6)).sub(&Iv::int(1));
            check_entry("EXPM1_64", i, EXPM1_64[i], &f, &iv);
        }
        let mut factf = one.clone();
        for n in 0..34_usize {
            if n > 0 {
                factf = factf * oracle::F::from(n as i64);
            }
            let f = one.clone() / factf.clone();
            let iv = Iv::int(1).div(&Iv::exact(oracle::dy(&factf)));
            check_entry("INV_FACT", n, INV_FACT[n], &f, &iv);
        }
        for n in 1..42_i64 {
            check_entry(
                "INV_INT",
                n as usize,
                INV_INT[n as usize],
                &fz(1, n),
                &q(1, n),
            );
        }
        for j in -18..=26_i64 {
            let i = (j + 18) as usize;
            let c = Dy::int(64 + j).scale(-6);
            check_entry("LN_C", i, LN_C[i], &fz(64 + j, 64).ln(), &oracle::ln(&c));
        }
        for j in -18..=0_i64 {
            let i = (j + 18) as usize;
            let c = Dy::int(64 + j).scale(-5);
            check_entry("LN_2C", i, LN_2C[i], &fz(64 + j, 32).ln(), &oracle::ln(&c));
        }
        for j in 0..=26_i64 {
            let c = Dy::int(64 + j).scale(-7);
            check_entry(
                "LN_C_2",
                j as usize,
                LN_C_2[j as usize],
                &fz(64 + j, 128).ln(),
                &oracle::ln(&c),
            );
        }
        for e in -6..=2_i64 {
            for i in 0..16_i64 {
                let k = (16 * (e + 6) + i) as usize;
                // c = 2^e (1 + (2i + 1)/32) = (33 + 2i) 2^(e − 5)
                let c = Dy::int(33 + 2 * i).scale((e - 5) as isize);
                let f = oracle::fdd(
                    oracle::pow2f((e - 5) as isize) * (33 + 2 * i) as f64,
                    0.0,
                    p,
                )
                .atan();
                check_entry("ATAN_C", k, ATAN_C[k], &f, &oracle::atan(&c));
            }
        }
        let three = pif.clone() * oracle::F::from(3) / oracle::F::from(4);
        check_entry(
            "FRAC_3PI_4",
            0,
            FRAC_3PI_4,
            &three,
            &oracle::pi().mul(&q(3, 4)),
        );
        // The f64 parts.
        let parts = |v: &oracle::F, n: usize| -> Vec<f64> {
            let mut r = oracle::dy(v);
            (0..n)
                .map(|_| {
                    let h = r.rn();
                    r = r.sub(&Dy::f(h));
                    h
                })
                .collect()
        };
        let l64 = ln2f.clone() / oracle::F::from(64);
        assert_eq!(
            parts(&l64, 3),
            LN2_64.to_vec(),
            "LN2_64 against dashu-float"
        );
        let iv = ln2.scale(-6);
        let mut lo = iv.lo.clone();
        let mut hi = iv.hi.clone();
        for (k, &p) in LN2_64.iter().enumerate() {
            assert_eq!(
                (lo.rn(), hi.rn()),
                (p, p),
                "LN2_64[{k}] against the enclosure"
            );
            lo = lo.sub(&Dy::f(p));
            hi = hi.sub(&Dy::f(p));
        }
        let inv = oracle::F::from(64).with_precision(p).value() / ln2f.clone();
        assert_eq!(oracle::dy(&inv).rn(), INV_LN2_64);
        let inv = Iv::int(64).div(ln2);
        assert_eq!((inv.lo.rn(), inv.hi.rn()), (INV_LN2_64, INV_LN2_64));
        // LN2_SPLIT: A = ln 2 rounded to 42 bits.
        let a = oracle::dy(&ln2f).round(42, false);
        let a_up = oracle::dy(&ln2f).round(42, true);
        let ln2d = oracle::dy(&ln2f);
        let a = if ln2d.sub(&a).abs().le(&a_up.sub(&ln2d).abs()) {
            a
        } else {
            a_up
        };
        assert_eq!(a.rn(), LN2_SPLIT[0]);
        let rest = ln2d.sub(&a);
        assert_eq!(rest.rn(), LN2_SPLIT[1]);
        assert_eq!(rest.sub(&Dy::f(LN2_SPLIT[1])).rn(), LN2_SPLIT[2]);
        for (k, end) in [&ln2.lo, &ln2.hi].into_iter().enumerate() {
            let r = end.sub(&Dy::f(LN2_SPLIT[0]));
            assert_eq!(r.rn(), LN2_SPLIT[1], "LN2_SPLIT[1], end {k}");
            assert_eq!(
                r.sub(&Dy::f(LN2_SPLIT[1])).rn(),
                LN2_SPLIT[2],
                "LN2_SPLIT[2], end {k}"
            );
        }
        // The bits of 2/π and π/4.
        let two_pi = oracle::F::from(2).with_precision(p).value() / pif.clone();
        let words = |v: &oracle::F, n: usize| -> Vec<u64> {
            Iv::exact(oracle::dy(v)).fixed_words(n).expect("exact")
        };
        assert_eq!(
            words(&two_pi, 22),
            TWO_OVER_PI.to_vec(),
            "2/π against dashu-float"
        );
        let pi = oracle::pi();
        assert_eq!(
            Iv::int(2).div(pi).fixed_words(22).expect("narrow"),
            TWO_OVER_PI.to_vec(),
            "2/π against the enclosure"
        );
        assert_eq!(words(&(pif / oracle::F::from(4)), 5), PI_OVER_4.to_vec());
        assert_eq!(
            pi.scale(-2).fixed_words(5).expect("narrow"),
            PI_OVER_4.to_vec()
        );
    }
}
