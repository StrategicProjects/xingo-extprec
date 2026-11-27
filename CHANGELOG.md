# Changelog

All notable changes to `xingo-extprec` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crate
follows [Semantic Versioning](https://semver.org/); before 1.0, a minor
version may change the public API.

## [0.2.0] — 2026-11-27

A rigorous bound of the error of a solution from the triangular factors of
an LU, and the move to `xingo-sparse` 0.6, whose types are in this crate's
public API (`impl xingo_sparse::Real for DoubleDouble`, the containers that
`verify` reads): hence a new minor version.

### Added

* `verify::forward_error_bound_lu(a, r, &factors, &opts)` (issue
  [#4](https://github.com/StrategicProjects/xingo-extprec/issues/4), still
  open for the general case): Algorithm E′ and Theorem E′ of Chapter
  11.10. From an enclosed residual `r ∋ A x − b` and triangular factors
  `P A Q ≈ L U` (`xingo_sparse::TriangularFactors<f64>`, for instance from
  `Lu::triangular_factors` of `xingo-sparse` 0.6), it proves that `A` is
  nonsingular and bounds `‖x − A⁻¹b‖∞` and every `|x_p − (A⁻¹b)_p|`,
  returning the same `ForwardErrorBound` as `forward_error_bound`
  (`contraction = β`, `solves = 0`). The bound is a theorem whatever the
  factors are, as long as they are triangular with valid permutations
  (checked): the defect `L U − P A Q`, rounding and the entries `drop_tol`
  dropped included, is enclosed entry by entry with `dot2_err`, the solves
  with the comparison matrices `⟨L⟩`, `⟨U⟩` round upward, and a
  first-order correction (default on) keeps the cancellations of `A⁻¹r`.
  Cost: one product `L U` (the multiplications of the elimination), a few
  triangular solves and two passes over `A`, instead of the `n` solves of
  `forward_error_bound`. It panics, in release builds too, if `A` violates
  the invariants of its container (a repeated column in a row, a NaN), as
  its documentation states.
* `verify::FactorBoundOptions` (`#[non_exhaustive]`, field
  `first_order_correction`) and the variant `VerifyError::InvalidFactors`
  (factors that fail `TriangularFactors::check_invariants`; `VerifyError`
  was already `#[non_exhaustive]`).

### Changed

* The optional dependency is `xingo-sparse` 0.6 (`>=0.6.0, <0.7.0`), no
  longer 0.4 or 0.5: `TriangularFactors` exists only since 0.6.0, and the
  tests use the `Option` that `KktForm::expected_inertia` returns since
  then. Without the feature `sparse` nothing changes.
* Chapter 11.2 describes the exact shape of the overflow corner of the
  double-double additions (an operand `±Ω` whose 2Sum is a tie in the top
  binade), why operands in the safe range never reach it, that
  `kernels::dot2_err` answers `β = +∞` there, and why it is not repaired:
  every variant measured costs 35–67 % on the loops the compiler
  vectorizes or 15–23 % of latency on every addition (issue
  [#6](https://github.com/StrategicProjects/xingo-extprec/issues/6),
  closed as won't fix). Documentation only; no bit changes.
* No bit of any other result changes: `residual`, `residual_csc`,
  `residual_of_dd`, `activity`, `polish`, `forward_error_bound`, the
  operations, the kernels and the elementary functions are those of
  0.1.3.

Migration: depend on `xingo-sparse` 0.6 together with this version (with
`KktForm::expected_inertia` now returning `Option<Inertia>`; see the
`xingo-sparse` CHANGELOG). Code that does not use the feature `sparse`
needs no change.

### Known limitations

`forward_error_bound_lu` certifies only when `β < 1`, and `β` is bounded
through `⟨U⟩⁻¹⟨L⟩⁻¹`, which grows exponentially with the depth of the
elimination for general sparse factors (up to `10⁴⁵` at `n = 400`).
Measured with the factors of `xingo_sparse::Lu`: 48 % of random sparse
systems certified at `n = 50`, 24 % at 100, 1 % at 200, 0 % at 400
(Algorithm E certified all of them), and 14 of the 83 Netlib KKT matrices
that `Lu` factorizes. Where it certifies, the bound of an `f64` solution
matched the true error to six digits. On `NotContracting`, fall back to
`forward_error_bound`. Its cost follows the product `L U`, not `nnz(L) +
nnz(U)` (9.4 s on maros-r7, 13 ms on fit2p, `n = 16 525`). Theorem E′ was
reviewed independently before release (proof valid, no counterexample);
about 49 000 bounds against exact rationals, 796 singular matrices and
the reviewer's 714 000 calls found no false bound. Not measured: the
optimal LP bases of Netlib, and synthetic systems of `10⁴` to `3·10⁵`.

## [0.1.3] — 2026-11-19

### Changed

* `verify::residual_of_dd` encloses the residual of a double-double
  solution much more tightly: after the Dot2Err enclosure (steps V1–V3,
  unchanged), steps V4–V7 of Chapter 11.10 split every product exactly,
  run two passes of VecSum over the terms and bound the floating-point sum
  of what is left a posteriori — the residual as if summed in three times
  the working precision — and intersect the result with the Dot2Err
  enclosure, which is therefore never widened (invariant V3; exact rows
  keep their bits). The width goes from `≈ k² u² S` to a few ulps of the
  residual plus `≈ 8N³u³S` (`N = 2k + 1` pairs). As a consequence the
  bound of `verify::forward_error_bound` for a polished solution follows
  its true error: on the optimal bases of the 93 Netlib LPs the
  overestimation went from a median of about 230 (largest `7.7 · 10⁴`) to 1.000
  at three digits on every basis with a nonzero error
  ([#5](https://github.com/StrategicProjects/xingo-extprec/issues/5)).
  No signature changes; `residual`, `residual_csc`, `activity` and every
  result for an `f64` solution are unchanged bit for bit. Cost: about 65
  more flops per stored entry in `residual_of_dd`.
* The limitation "for a solution polished in double-double the bound can
  overestimate the true error" is removed from the README and the landing
  page.
* Steps V4–V7 were reviewed independently: no enclosure missed the exact
  residual in about 37 000 adversarial rows checked in exact arithmetic.
  The review built a row on which a tail bound half as large as the proven
  one would miss; it is now a regression test
  (`tests/data/tail_bound_row.txt`). Should a defect ever make the
  intersection empty, a release build now returns the Dot2Err enclosure
  instead of an empty interval.

## [0.1.2] — 2026-11-19

### Changed

* The bound of `tan` proven in Chapter 11.11, `27.4u²|tan x| + 2⁻³¹⁵(1 +
  tan²x)`, now holds for every finite argument: Lemma D shows that no
  double-double above `π/4` lies within `2⁻¹⁴⁰` of a multiple of `π/2` (in
  fact `2⁻¹²⁹`; the closest found lies at `2⁻¹²⁴`), so the case left open
  in 0.1 (an odd quadrant with a reduced argument below `2⁻³⁰⁰`) does not
  exist. The proof is a computation in integers, `tests/reduction_bound.rs`
  (continued fractions, and a lattice of dimension 3 reduced by LLL and
  checked exactly); its full run is `#[ignore]`d, a sample runs by default.
  Step S4 of `tan`, now known to be unreachable, stays as a guard. The
  limitation "One gap in `tan`" is removed from the README and the landing
  page. No change to the bits of any result and none to
  `FUNCTION_BOUNDS`; debug builds assert the lemma in the argument
  reduction.
* The CI of the mirror also runs the suite on a big-endian target
  (`s390x-unknown-linux-gnu` under QEMU) and on WebAssembly
  (`wasm32-wasip1` under wasmtime), and re-runs the whole certificate of
  Lemma D. For those targets the dev-dependency `proptest` drops its
  features `fork` and `timeout` (unused), and the generic SIMD kernels,
  which no backend calls there, are allowed to be dead code.
* The optional dependency accepts `xingo-sparse` 0.4 or 0.5 (`>=0.4.0,
  <0.6.0`): 0.5 changes nothing this crate uses, and the suite passes
  against both.

## [0.1.1] — 2026-11-18

### Changed

* Documentation only: the README (and the landing page) gain the section
  "How it was verified" (exact oracles, independent reviews, subnormal
  regime, targets of the CI), taken out of the list of known limitations;
  each limitation now states only what is still open and links to its
  issue in the repository. The README carries the Zenodo DOI, which did
  not exist when 0.1.0 was packaged. No change to the code.

## [0.1.0] — 2026-11-18

The first release: the extended-precision layer of the Xingó solver,
published as a crate that depends on nothing in the solver except,
optionally, `xingo-sparse` 0.4.

### Added

* The error-free transformations `two_sum`, `fast_two_sum` and `two_prod`
  (Chapter 11.1).
* `DoubleDouble`: the field operations with the algorithms of Joldes,
  Muller & Popescu (2017) — always the accurate addition, the FMA-based
  products, the division DWDivDW2 — and the square root SQRTDWtoDW of
  Lefèvre, Louvet, Muller, Picot & Rideau (2023); comparisons, `min`,
  `max`, `abs`, `ldexp`, `recip`, `square`; the proven relative bound of
  each operation as data, `DoubleDouble::ERROR_BOUNDS` (Chapter 11.2),
  each value an `f64` at least the full proven expression (rounded up
  where the expression has higher-order terms: `2 + 2⁻⁵⁰` for `dd + f64`,
  `3 + 2⁻⁴⁹` for `dd + dd`, `15 + 2⁻⁴⁷` for `dd / dd`), checked in exact
  arithmetic by a test.
* Correctly rounded decimal conversions: `Display` (31 significant digits
  by default; `{:.N}` and `{:.Ne}` up to the exact expansion), `LowerExp`
  and `FromStr`, in exact integer arithmetic (Chapters 11.3 and 11.4);
  `ParseDoubleDoubleError`.
* The elementary functions of `DoubleDouble` with the names and special
  cases of `f64` — `exp`, `exp2`, `exp_m1`, `ln`, `ln_1p`, `log2`,
  `log10`, `powi`, `powf`, `sin`, `cos`, `tan`, `sin_cos`, `asin`, `acos`,
  `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` — and
  the 19 constants of `core::f64::consts` as associated constants. Each
  function has a declared bound, a row of `DoubleDouble::FUNCTION_BOUNDS`
  (`FunctionBound`), proven in Chapter 11.11. No `f64` transcendental
  function of `std` is called (their precision is unspecified): table
  seeds, Payne–Hanek reduction in integer arithmetic for every `|x| > π/4`.
* `kernels`: `sum2`, `dot2` and `dot2_err` (compensated, Ogita, Rump &
  Oishi 2005; `dot2_err` with a rigorous bound on its own error, `+∞`
  also when an intermediate step of its last 2Sum overflows near `Ω` — a
  corner found by the review of X0.3, where the bound was finite and
  false), the
  double-double `dot`, `dot_mixed`, `axpy` and `gemm_sub`, over four
  accumulators in a fixed order (Chapter 11.5); `kernels::backend()`.
* Feature `simd` (default): the same kernels on vector registers through
  `pulp` 0.22 — NEON and AVX2 + FMA, dispatched at run time, with the
  results of the scalar reference bit for bit (Chapter 11.6).
* `Interval`: outward-rounded interval arithmetic on `f64` endpoints with
  directed rounding *computed* from the error-free transformations, exact
  under underflow too; `EMPTY`, `ENTIRE`, hull, intersection, `sqrt`,
  `square`, `abs`, `mag`, `mig`, `mid`, `width`, the verified predicates
  `certainly_le`, `certainly_ge` and `certainly_within`; operators with
  `Interval` and `f64` on either side, their assignment forms,
  `From<f64>`, `Display` and `LowerExp` rounded outward;
  `InvalidInterval` (Chapter 11.9).
* Feature `sparse` (default): `impl xingo_sparse::Real for DoubleDouble`,
  so that the containers and factorizations of `xingo-sparse` run in
  double-double; the dense kernel `DdDense` (Chapter 11.7); and the module
  `verify` — `residual`, `residual_csc`, `residual_of_dd`, `residual_dd`,
  `activity`, `norm_inf_bound` and `polish` with `PolishOptions`,
  `PolishReport`, `PolishStop` and `VerifyError` (Chapter 11.10).
* `verify::forward_error_bound` (Algorithm E, Theorem E of Chapter 11.10):
  from the enclosure of the residual of an approximate solution (`f64` or
  double-double) and the caller's transposed solve, a proof that a square
  `A` is nonsingular and a rigorous bound of `‖x − A⁻¹b‖∞` and of each
  component of the error, by an approximate inverse applied one implicit
  row at a time, each row refined once in double-double by default;
  `ForwardErrorOptions` (`refine_rows`), `ForwardErrorBound`,
  `ForwardErrorOutcome` (`Certified`, `NotContracting`, `NotFinite`). The
  bound holds whatever the solve returns; without a certificate the bounds
  are `+∞` (`Ok`, not an error).
* `ErrorBound`, `FunctionBound`, `PolishReport`, `PolishStop`,
  `ForwardErrorBound`, `ForwardErrorOutcome` and `VerifyError` are
  `#[non_exhaustive]`, so that fields and variants can be added in a minor
  version; `PolishOptions` and `ForwardErrorOptions` are not (they are
  built with `..Default::default()`).
* The documentation, on docs.rs too, includes the private items: most
  chapters of the book, the proofs of the bounds among them, document
  private modules.
* The examples `dd_profile`, `dd_bench` and `ft_limit`, which reproduce
  the measurements behind the `LuOptions` profile for double-double of
  `xingo-sparse` 0.4.0 and the timing tables of the kernels, and
  `elementary_bench`, the timing table of the elementary functions (no
  feature needed).

### Known limitations

See "Known limitations" in the README: the proofs of the bounds are written
analyses, not machine-checked (those of the elementary functions were
re-derived by three independent reviews before this release; for `tan` with
a reduced argument below `2⁻³⁰⁰` in an odd quadrant, never observed, only the
declared bound is proven); the same bits across platforms were observed
on aarch64 (NEON), x86_64 (scalar and AVX2 + FMA) and i686, on Linux,
macOS and Windows, not elsewhere; the
subnormal regime of the elementary functions is tested for all 23 (subnormal
arguments and trailing components, underflowing results), not exhausted;
`forward_error_bound` bounds the error of one solution of a square system
with point data (no solution sets of interval systems, no 2-norm bounds),
costs at most `n` (`2n` with refinement) transposed solves and is
quadratic in `n`, the condition numbers it certified are measurements, not
a guarantee, and its proof is the chapter's own (reviewed before release by
an independent reviewer with 262 000 adversarial cases checked in exact
arithmetic, no false bound found).

[0.2.0]: https://github.com/StrategicProjects/xingo-extprec/releases/tag/v0.2.0
[0.1.3]: https://github.com/StrategicProjects/xingo-extprec/releases/tag/v0.1.3
[0.1.2]: https://github.com/StrategicProjects/xingo-extprec/releases/tag/v0.1.2
[0.1.1]: https://github.com/StrategicProjects/xingo-extprec/releases/tag/v0.1.1
[0.1.0]: https://github.com/StrategicProjects/xingo-extprec/releases/tag/v0.1.0
