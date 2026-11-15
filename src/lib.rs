//! # Chapter 11 — `xingo-extprec`: extended precision and rigorous enclosures
//!
//! **Status: the scalar half (L2.2a).** The contract was approved by André
//! on 2026-11-15 (PLAN.md trilha L2.1, D-0081 and D-0109; questions E1–E8
//! below all answered "yes" as proposed) and lives in
//! `contracts/xingo-extprec/`. Implemented here: the error-free
//! transformations (Chapter 11.1), [`DoubleDouble`] with `impl
//! xingo_sparse::Real` (Chapter 11.2), its decimal conversions (Chapters
//! 11.3 and 11.4) and the scalar reference kernels (Chapter 11.5). **Not
//! yet in the crate:** the SIMD kernels and `DdDense` (L2.2b), `Interval`
//! and the module `verify` (L2.3); the parts of this chapter about them
//! describe the contract. The section "Corrections to the approved
//! contract" lists what the implementation found wrong in it.
//!
//! ## Purpose
//! A solver computes in `f64` and must sometimes know *how wrong* it is. A
//! basis solve `B x = b` on an ill-conditioned basis returns an `x` whose
//! residual, recomputed in `f64`, is itself rounding noise; a feasibility
//! claim `|a_iᵀx − b_i| ≤ 10⁻⁹` checked in the arithmetic that produced `x`
//! proves nothing. Between `f64` and exact rational arithmetic (PLAN §8,
//! `--exact-check`) there is a cheap intermediate step with two parts:
//!
//! 1. **more precision** — double-double numbers (an unevaluated sum of two
//!    `f64`, about 32 decimal digits) at 10–25 flops per operation, in
//!    which residuals can be computed and solutions *polished* by iterative
//!    refinement, and in which the LU of `xingo-sparse` can be instantiated
//!    unchanged;
//! 2. **rigour** — an enclosure `[r̲, r̄] ∋ a_iᵀx − b_i` that is a theorem
//!    about the `f64` data, obtained in ordinary round-to-nearest arithmetic
//!    from error-free transformations and a running error bound.
//!
//! This crate provides both, independent of the solver and publishable on
//! its own.
//!
//! ## Notation
//! `F` is the set of `f64` values, `fl(·)` rounding to nearest (ties to
//! even), `u = 2⁻⁵³` the unit roundoff, `ulp(a)` the unit in the last place
//! of `a ∈ F`, `η = 2⁻¹⁰⁷⁴` the smallest positive subnormal. A
//! *double-double* is a pair `(h, ℓ) ∈ F²` with value `h + ℓ` (exact sum).
//! `u_dd = 2⁻¹⁰⁶ = u²`. An *interval* is a pair `[a, b]`, `a ≤ b`, of
//! elements of `F ∪ {±∞}` and stands for `{x ∈ R : a ≤ x ≤ b}`.
//!
//! An **error-free transformation** (EFT) writes the exact result of an
//! `f64` operation as an `f64` pair: `two_sum(a, b) = (s, e)` with `s =
//! fl(a + b)` and `s + e = a + b` exactly (Knuth; 6 flops, no branch);
//! `two_prod(a, b) = (p, e)` with `p = fl(a b)` and `p + e = a b` exactly
//! (one multiplication and one fused multiply-add). They are the only
//! primitives of the crate; everything else is built from them.
//!
//! ## Scope of v0.1
//! 1. The EFTs [`two_sum`], [`fast_two_sum`], [`two_prod`] as public
//!    functions.
//! 2. [`DoubleDouble`]: field operations, `sqrt`, comparisons, conversions,
//!    decimal formatting and parsing; the algorithms of Joldes, Muller &
//!    Popescu (2017), whose relative error bounds are *proven* (and partly
//!    machine-checked), in preference to the QD library's, whose bounds are
//!    stated without proof and are wrong for the "sloppy" addition.
//! 3. `impl xingo_sparse::Real for DoubleDouble` (feature `sparse`), so
//!    that `Lu<DoubleDouble>`, `SupernodalLdl<DoubleDouble>` and the
//!    containers exist without one line of duplicated code.
//! 4. Slice kernels, [`kernels`]: compensated `f64 → DoubleDouble` sums and
//!    dot products (Ogita, Rump & Oishi 2005) and the double-double `dot`,
//!    `axpy` and small-block `gemm`; scalar reference and SIMD (`pulp`)
//!    versions with the same results bit for bit.
//! 5. `Interval` (L2.3): outward-rounded interval arithmetic on `f64` endpoints
//!    *without* changing the rounding mode.
//! 6. `verify` (L2.3; feature `sparse`): rigorous enclosures of residuals
//!    `A x − b` over the containers of `xingo-sparse`, and the polishing
//!    step (iterative refinement with double-double residuals) around a
//!    caller-supplied solve.
//!
//! ## Out of scope (D-0081, restated)
//! Arbitrary precision (MPFR-like); complex scalars; `f32`; elementary
//! functions of double-doubles other than `sqrt` (no `exp`, `log`,
//! trigonometry — a linear solver needs none; question E7); interval
//! *linear algebra* (interval LU, Krawczyk, verified solution of linear
//! systems: Rump's `verifylss`) — v0.1 verifies residuals, not solution
//! sets; `QuadDouble` (question E4).
//!
//! ## Algorithms and their bounds
//! All bounds hold in the absence of overflow and of underflow in the
//! EFTs (precisely: every `two_prod` product `a b` satisfies `|a b| ≥
//! 2⁻⁹⁶⁹` or is zero — Chapter 11.1, invariant T2; Chapter 11.2 states the
//! "safe range" of operands in which this is guaranteed); outside that
//! range the operations still return a normalized pair, with an absolute
//! error of order `η`, and the *interval* functions account for it
//! explicitly (invariant I2). JMP is Joldes, Muller & Popescu (2017), MR
//! Muller & Rideau (2022), LLMPR Lefèvre, Louvet, Muller, Picot & Rideau
//! (2023); the flop counts are those of JMP's Table 1 (FMA available).
//!
//! | operation | algorithm | relative error | flops |
//! |---|---|---|---|
//! | `dd + f64` | DWPlusFP (JMP Alg. 4) | `≤ 2u²/(1 − 2u) < 2u² + 5u³` | 10 |
//! | `dd + dd` | AccurateDWPlusDW (JMP Alg. 6) | `≤ 3u²/(1 − 4u) < 3u² + 13u³` | 20 |
//! | `dd · f64` | DWTimesFP3 (JMP Alg. 9, FMA) | `≤ 2u²` | 6 |
//! | `dd · dd` | DWTimesDW3 (JMP Alg. 12, FMA) | `≤ 4u²` (MR; JMP proved `5u²`) | 9 |
//! | `dd²` | DWPow2 (two-FMA product specialized) | `≤ 3u²` (Hubrecht & Melquiond 2026) | 6 |
//! | `dd / f64` | DWDivFP3 (JMP Alg. 15) | `≤ 3u²` | 10 |
//! | `dd / dd` | DWDivDW2 (JMP Alg. 17) | `≤ 15u² + 56u³` | 18 |
//! | `√dd` | SQRTDWtoDW (LLMPR Alg. 8): a Newton step on `fl(√h)` | `≤ 25/8 u²` (**not checked**) | 8 |
//!
//! The addition is the *accurate* one: the QD library's default "sloppy"
//! addition (11 flops) has unbounded relative error when the operands have
//! opposite signs (JMP §3), and a residual `b − A x` is exactly that case.
//! No sloppy variant is exposed (question E5).
//!
//! **Consequence for the plan.** L2.2's acceptance criterion reads "erro ≤
//! 2⁻¹⁰⁴ relativo nos kernels". `2⁻¹⁰⁴ = 4 u²` holds for `+`, `−`, `·` and
//! `√`, **not for the division**: DWDivDW2 is proven to `15u² + 56u³`
//! (`3,75 · 2⁻¹⁰⁴`), and the best proven bound of a double-word division
//! is `9,8 u²` (DWDivDW3, JMP Alg. 18: 31 flops instead of 18, through a
//! double-word reciprocal). The contract therefore states one bound per
//! operation ([`DoubleDouble::ERROR_BOUNDS`]) and L2.2 is accepted against
//! *those* (question E6).
//!
//! **Why not DWDivDW3.** It was implemented and measured (2026-11-15), and
//! the owner kept DWDivDW2. DWDivDW3 computes `x · (1/y)`, and a rounded
//! reciprocal loses the *exact* quotients: `53/53` came out as `1 −
//! 6,2 · 10⁻³³`; of the 39 601 quotients `(q y)/y` with integers `1 ≤ q, y
//! < 200`, 4 600 were not `q`; `(a b)/b ≠ a` for about 30 % of random
//! `f64` pairs and `x/x ≠ 1` for 45 % of random double-doubles. DWDivDW2,
//! which corrects the `f64` quotient with an exact residual, returns all
//! of these exactly (Chapter 11.2; the tests pin it). The average error
//! was also larger (`0,32u²` against `0,12u²` on `f64`-valued operands,
//! `0,56u²` against `0,48u²` on general ones; the largest observed,
//! `4,4u²` against `5,1u²`, is the only figure in its favour), and the
//! scaled residual of `Lu<DoubleDouble>` rose from about `1,5` to `74`
//! units of `n ε` (order 300). A better worst-case bound did not pay for
//! that.
//!
//! **Intervals without the rounding mode.** Rust offers no sound way to
//! change the floating-point rounding mode (LLVM assumes round-to-nearest;
//! `fesetround` through FFI is undefined behaviour in the presence of
//! constant folding), and this crate has no `unsafe` (D-0006 allows it only
//! in `xingo-sparse`). Directed rounding is therefore *computed*: for `∘ ∈
//! {+, −, ·}` the EFT gives `x ∘ y = s + e` exactly, so `▽(x ∘ y) = s` if
//! `e ≥ 0` and `pred(s)` otherwise, and symmetrically `△`; the result is
//! the *correctly rounded* directed value, as tight as hardware directed
//! rounding, at 2–4 times the cost of a plain operation. Division and
//! square root use the FMA residual (`e = fma(−q, y, x)` has the sign of
//! `x − q y` exactly). When an EFT is not exact (underflow) the endpoint
//! moves one further step outward. This is the design of Rump, Zimmermann,
//! Boldo & Melquiond (2009) and of the IEEE 1788 reference notes; `inari`
//! (read, not used) takes the other route, MXCSR through `unsafe`.
//!
//! **Residual enclosures.** For `f64` vectors `x`, `y` of length `n`,
//! Dot2Err (Ogita, Rump & Oishi 2005, Alg. 5.8) returns in round-to-nearest
//! a double-double `r` and an `f64` bound `β` with `|xᵀy − r| ≤ β`, valid
//! whenever `2 n u < 1` and no overflow occurs, underflow included in `β`
//! (their Corollary 5.9; [`kernels::dot2_err`] is that algorithm over four
//! accumulators, with its own proof in Chapter 11.5 and the hypothesis
//! `2(n + 3)u < 1`). `verify::residual` applies it row by row: the enclosure of `a_iᵀx −
//! b_i` costs about 25 flops per nonzero, against 2 for the plain residual
//! and thousands for a rational one.
//!
//! ## What a certificate means
//! (C1) The enclosures are statements about the **`f64` model**: the
//!      matrix and vectors as stored. That a decimal `0.1` in an MPS file
//!      became the `f64` nearest to it is outside this crate (the rational
//!      check of PLAN §8 starts from the decimal text).
//! (C2) An enclosure is rigorous under IEEE 754 binary64 arithmetic with
//!      round-to-nearest and a correctly rounded `fma` — what Rust
//!      guarantees on every tier-1 target (`f64::mul_add` falls back to a
//!      correct software FMA where the CPU has none; slow, never wrong).
//! (C3) `verify` never claims more than it proved: when a hypothesis of a
//!      bound fails (overflow, `n u ≥ 1`, NaN in the data) the result is
//!      `Interval::ENTIRE` or an error, never a finite interval.
//!
//! ## Invariants (each becomes a `debug_assert!` or `check_invariants()`)
//! (D1) *Normalized.* Every finite [`DoubleDouble`] `(h, ℓ)` satisfies `h =
//!      fl(h + ℓ)` (hence `|ℓ| ≤ ulp(h)/2`); `h = ±0` implies `ℓ = 0`.
//! (D2) *Non-finite.* If `h` is infinite or NaN then `ℓ = 0` when `h` is
//!      infinite and the value is "infinite"/"NaN" as `h` says; operations
//!      follow IEEE on `h`. (`∞ − ∞` is NaN, etc.)
//! (D3) *Order.* The derived lexicographic order on `(h, ℓ)` is the order
//!      of the values `h + ℓ` (true for normalized pairs), total on
//!      non-NaN values: requirement (R2) of `xingo_sparse::Real`.
//! (D4) *Conversions.* `from_f64` is exact; `to_f64` is `h`, the `f64`
//!      nearest to `h + ℓ` by (D1): requirement (R4).
//! (D5) *Determinism.* Every operation is a fixed sequence of `f64`
//!      operations: no data-dependent algorithm choice, no reassociation,
//!      the same bits on every platform (C2).
//! (K1) *Kernels.* The SIMD kernels return, bit for bit, the results of
//!      the scalar reference kernels of the same name: the summation order
//!      is part of the contract of each kernel (stated in its docs), not
//!      an implementation detail.
//! (I1) *Inclusion.* For intervals `X`, `Y` and `∘` an operation: `x ∈ X,
//!      y ∈ Y ⇒ x ∘ y ∈ X ∘ Y` whenever `x ∘ y` is defined.
//! (I2) *Well-formed.* `lo ≤ hi`, neither is NaN, `lo < +∞`, `hi > −∞`; the
//!      empty set is a separate value (`Interval::EMPTY`).
//! (I3) *Tightness.* `+`, `−`, `·`, `/` (divisor not containing zero) and
//!      `sqrt` return the *hull of the exact range rounded outward to
//!      `F`*: the tightest representable interval.
//! (V1) *Residuals.* `verify::residual(A, x, b)[i] ∋ a_iᵀx − b_i` for the
//!      stored `f64` values (C1), for every row.
//!
//! ## How `xingo_sparse::Real` is met, and one amendment to it
//! `Real` (Chapter 5.6) asks (R1) "the field operations are the correctly
//! rounded ones of `T`". Double-double operations are not correctly
//! rounded; they are *faithful with a proven relative bound*. The generic
//! code of `xingo-sparse` uses (R1) only through tolerances scaled by
//! `T::EPSILON`, so the proposal is to **reword (R1)** there as "each
//! field operation is a deterministic function of its operands with
//! relative error at most `c_T · EPSILON`, `c_T ≤ 4`" (true for `f64` with
//! `c = 1/2`) — a documentation change in `xingo-sparse`, no API change
//! (question E2). `Real::EPSILON` for double-double is `2⁻¹⁰⁴`, the QD
//! library's convention. `Real::mul_add(a, b)` is `self · a + b` with two
//! roundings (the trait says "a single rounding where the type allows").
//! `Sum` adds left to right from `ZERO` (R3).
//!
//! ## Divergences from the references (to review)
//! * **QD (Hida, Li & Bailey 2001).** Same representation; the algorithms
//!   are JMP 2017's (proven bounds, FMA-based products) and the addition
//!   is always the accurate one. *Read, not copied* (QD is BSD-like, but
//!   the code here is written from the papers).
//! * **`twofloat`** (BSD-3). Double-double without a shared trait with the
//!   sparse containers and with the sloppy addition as `+`. Not used.
//! * **`inari`** (MIT). IEEE 1788 intervals with decorations, x86-64 and
//!   AArch64 only, rounding mode through `unsafe` intrinsics. Here:
//!   portable, no `unsafe`, no decorations, only the operations a residual
//!   needs. Not used.
//! * **INTLAB / `verifylss` (Rump).** Verified *solutions* of linear
//!   systems are out of scope for v0.1 (above).
//! * **D-0081** recommended "`xingo-sparse` depends optionally on
//!   `xingo-extprec`". The contract proposes the **opposite direction**
//!   (question E1): `verify` needs the containers of `xingo-sparse`, so
//!   the D-0081 direction would be a cycle; with `xingo-extprec →
//!   xingo-sparse` the `impl Real` lives here (orphan rule satisfied),
//!   `xingo-sparse` stays free of this dependency, and its vestigial
//!   contract feature `extprec` is dropped.
//! * **D-0081** names `malachite` or `dashu` as the exact oracle of the
//!   tests. `malachite` is LGPL-3.0, which CLAUDE.md forbids; the oracle is
//!   **`dashu`** (MIT OR Apache-2.0), a dev-dependency (question E8).
//!
//! ## Questions of the review (all approved as proposed, 2026-11-15)
//! * **E1** — dependency direction `xingo-extprec → xingo-sparse` (feature
//!   `sparse`, on by default), against D-0081's recommendation. *Proposed:
//!   yes.*
//! * **E2** — reword requirement (R1) of `xingo_sparse::Real` as above (a
//!   patch release of `xingo-sparse`, documentation only). *Proposed: yes.*
//! * **E3** — intervals by computed directed rounding, no `unsafe`, 2–4×
//!   the cost of hardware rounding modes. *Proposed: yes; the alternative
//!   needs `unsafe` here and an amendment of D-0006.*
//! * **E4** — `QuadDouble`: D-0081 says "if viable", L2.3 "if the cost
//!   pays". It roughly triples the surface (renormalization of four
//!   components, no proven bounds in the literature for the QD
//!   algorithms). *Proposed: not in v0.1; the name is reserved and a
//!   measured need (a basis that double-double does not resolve) reopens
//!   it.*
//! * **E5** — no sloppy addition in the public API. *Proposed: yes.*
//! * **E6** — L2.2's criterion becomes "each operation within its bound of
//!   [`DoubleDouble::ERROR_BOUNDS`], checked against exact rationals",
//!   instead of a single `2⁻¹⁰⁴`. *Proposed: yes.*
//! * **E7** — no elementary functions beyond `sqrt`; `Display`/`FromStr`
//!   are faithful (error `≤ 1` unit of the 32nd digit), not correctly
//!   rounded. *Proposed: yes.*
//! * **E8** — `dashu` as the dev-only exact oracle; `pulp` (MIT) as the
//!   only runtime dependency besides `xingo-sparse`. *Proposed: yes.*
//!
//! ## Corrections to the approved contract (L2.2a)
//! The table of bounds of the contract was written from memory (D-0109,
//! E6); L2.2a checked it against the texts and changed, in this chapter:
//!
//! * `dd + f64`: the bound is `2u²/(1 − 2u)`, not `2u²` (it is
//!   asymptotically optimal; JMP report `2u² − 6u³` attained).
//! * `dd · dd`: the `4u²` of DWTimesDW3 is Muller & Rideau's (2022), who
//!   formalized the proofs in Coq and improved JMP's `5u²`.
//! * `dd / dd`: DWDivDW2 costs 18 flops, not 31; 31 flops and `9,8u²` are
//!   DWDivDW3's. The contract's remark "a cheaper variant is `9,8u²`" was
//!   backwards: the *more expensive* algorithm has the better bound. The
//!   division stays DWDivDW2 (see "Why not DWDivDW3" above).
//! * `√dd`: the algorithm is SQRTDWtoDW of LLMPR; its constant `25/8`
//!   could not be checked against the paper (not accessible). A
//!   first-order analysis reproduces exactly `25/8` (Chapter 11.2), and
//!   the tests check it against exact arithmetic.
//! * `dd²` gained its own row (DWPow2, `3u²`), cited at second hand.
//! * Dot2Err needs `2 n u < 1`, not `n u < 1`; and the bounds of Sum2 and
//!   Dot2 are restated for the four-lane order of the kernels (`γ²_{n+3}`
//!   and `γ²_{n+5}` instead of `γ²_{n−1}` and `γ²_n`; Chapter 11.5).
//! * [`fast_two_sum`]: the requirement that matters is on the *exponents*
//!   (`e_a ≥ e_b`), which `|a| ≥ |b|` implies. The accurate addition calls
//!   it with `|a| < |b|` in one binade under cancellation, so the
//!   `debug_assert!` tests the exponent condition (Chapter 11.1).
//! * [`two_sum`] is not exact "unless `s` overflows" only: with `a = ±Ω`
//!   an intermediate difference can overflow (Chapter 11.1, invariant T1).
//! * Invariant (D1) needs care below the range of [`two_prod`]: a rounded
//!   error term can be exactly half an ulp on the wrong side of the tie.
//!   `from_product` and `from_str` renormalize (one exact Fast2Sum).
//! * `Display`: "the shortest of at most 32 significant digits that parses
//!   back to the same value" does not exist in general — a double-double
//!   is not a 106-bit number (its trailing component may lie arbitrarily
//!   far below the leading one). The default output is the value correctly
//!   rounded to **31** digits (the owner's choice, 2026-11-15, instead of
//!   the contract's 32): a decimal of at most 31 digits that was read
//!   comes back as the same text, which 32 digits cannot give (`10³¹ <
//!   2¹⁰⁶ < 10³²`). `{:.N}` and `{:.Ne}` with `N` large give the exact
//!   expansion (Chapter 11.4, invariant P3). All conversions are correctly
//!   rounded, which is stronger than the "faithful" of question E7.
//! * Added to the contract's surface, ratified by the owner: `impl
//!   LowerExp for DoubleDouble` (`{:e}`, as for `f64`), `impl Display` and
//!   `impl Error` for [`ParseDoubleDoubleError`], and the row `"dd²"` of
//!   `ERROR_BOUNDS`.
//!
//! ## Plan
//! L2.2a (this state): EFTs, `DoubleDouble`, `impl Real`, scalar kernels,
//! proptests against `dashu`, `Lu<DoubleDouble>` and
//! `SupernodalLdl<DoubleDouble>` as integration tests. L2.2b: `pulp`
//! kernels, `DdDense`. L2.3: `Interval`, `verify`, polishing; residual
//! certificates on the Netlib bases; `--exact-check` uses the enclosures
//! before the rational check.
//!
//! ## References
//! Dekker, "A floating-point technique for extending the available
//! precision", Numer. Math. 18 (1971). Knuth, TAOCP vol. 2, §4.2.2.
//! Hida, Li & Bailey, "Algorithms for quad-double precision floating
//! point arithmetic", ARITH-15 (2001). Ogita, Rump & Oishi, "Accurate sum
//! and dot product", SIAM J. Sci. Comput. 26 (2005). Joldes, Muller &
//! Popescu, "Tight and rigorous error bounds for basic building blocks of
//! double-word arithmetic", ACM TOMS 44 (2017). Muller & Rideau,
//! "Formalization of double-word arithmetic, and comments on 'Tight and
//! rigorous error bounds for basic building blocks of double-word
//! arithmetic'", ACM TOMS 48 (2022). Stloukal & Jézéquel, "Lord of the
//! Precision, the Two Floats: double-word arithmetic for complex numbers",
//! arXiv:2611.08371 (2026), §3 (it restates the algorithms and bounds
//! used here, and cites Hubrecht & Melquiond, "Verifying code that uses
//! error-free transformations", ARITH 2026, for DWPow2). Lefèvre, Louvet, Muller,
//! Picot & Rideau, "Accurate calculation of Euclidean norms using
//! double-word arithmetic", ACM TOMS 49 (2023). Rump, "Verification
//! methods: rigorous results using floating-point arithmetic", Acta
//! Numerica 19 (2010). Rump, Zimmermann, Boldo & Melquiond, "Computing
//! predecessor and successor in rounding to nearest", BIT 49 (2009).
//! Higham, *Accuracy and Stability of Numerical Algorithms* (2002), ch.
//! 4 and 12. IEEE Std 1788-2015. Muller et al., *Handbook of
//! Floating-Point Arithmetic*, 2nd ed. (2018), ch. 4 and 14.

#![forbid(unsafe_code)]

mod big;
mod dd;
mod decimal;
mod eft;
pub mod kernels;

pub use dd::{DoubleDouble, ErrorBound};
pub use decimal::ParseDoubleDoubleError;
pub use eft::{fast_two_sum, two_prod, two_sum};
