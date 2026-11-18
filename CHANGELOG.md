# Changelog

All notable changes to `xingo-extprec` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crate
follows [Semantic Versioning](https://semver.org/); before 1.0, a minor
version may change the public API.

## [0.1.0] — Unreleased

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
  Oishi 2005; `dot2_err` with a rigorous bound on its own error), the
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
* `ErrorBound`, `FunctionBound`, `PolishReport`, `PolishStop` and
  `VerifyError` are `#[non_exhaustive]`, so that fields and variants can be
  added in a minor version; `PolishOptions` is not (it is built with
  `..PolishOptions::default()`).
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
`verify` bounds residuals, not `‖x − x*‖`.

<!-- TODO(owner): replace "Unreleased" in the 0.1.0 heading with the date of publication, and add the link line below. -->
<!-- [0.1.0]: https://github.com/StrategicProjects/xingo-extprec/releases/tag/v0.1.0 -->
