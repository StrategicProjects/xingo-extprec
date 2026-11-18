# xingo-extprec

[![crates.io](https://img.shields.io/crates/v/xingo-extprec.svg)](https://crates.io/crates/xingo-extprec)
[![docs.rs](https://docs.rs/xingo-extprec/badge.svg)](https://docs.rs/xingo-extprec)
![MSRV 1.90](https://img.shields.io/badge/MSRV-1.90-blue.svg)
![license: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)
<!-- TODO(owner): the Zenodo DOI badge, once the first release of the mirror is archived. -->

**Website:** <https://strategicprojects.github.io/xingo-extprec/> (overview, bounds, measured cost, API reference).

Extended precision **for numerical linear algebra**, in pure Rust:
double-double numbers whose every operation has a **proven** error bound,
compensated sums and dot products, and **rigorous** interval enclosures of
residuals `A x − b` computed in ordinary round-to-nearest arithmetic. It is
the extended-precision layer of [Xingó](https://github.com/StrategicProjects/xingo),
a linear and mixed-integer programming solver, and depends on nothing in it
except, optionally, its sparse linear algebra crate
[`xingo-sparse`](https://crates.io/crates/xingo-sparse).

A solver computes in `f64` and must sometimes know *how wrong* it is. A
residual recomputed in the arithmetic that produced the solution is itself
rounding noise, and a feasibility claim `|a_iᵀx − b_i| ≤ 10⁻⁹` checked
that way proves nothing. Between `f64` and exact rational arithmetic there
is a cheap intermediate step, and this crate is that step:

1. **more precision** — `DoubleDouble`, an unevaluated sum of two `f64`
   (about 32 decimal digits, 106 bits of significand), at a few
   nanoseconds per operation, in which residuals are computed, solutions
   are *polished* by iterative refinement, and the LU and LDLᵀ of
   `xingo-sparse` run unchanged;
2. **rigour** — an interval `[r̲, r̄] ∋ a_iᵀx − b_i` that is a theorem about
   the stored `f64` data, obtained from error-free transformations and a
   running error bound, without touching the processor's rounding mode.

It is **not** an arbitrary-precision library (see
[`dashu`](https://crates.io/crates/dashu) for that), has no quad-double, no
complex numbers, and verifies *residuals*, not solution sets of linear
systems (see "Scope" below).

## Four entry points

**1. `DoubleDouble`** — the field operations, `sqrt`, comparisons,
correctly rounded decimal conversions, the elementary functions of `f64`
(same names, same special cases) and the 19 constants of
`core::f64::consts`. The algorithms are those of Joldes, Muller & Popescu
(2017), whose relative error bounds are proven; the addition is always the
*accurate* one (the "sloppy" addition of the QD library has an unbounded
relative error under cancellation, which is exactly the case of a residual).

```rust
use xingo_extprec::DoubleDouble;

// 1 + 10⁻¹⁶ is 1 in f64, not in double-double.
let big = DoubleDouble::from_f64(1e16);
assert_eq!(((big + 1.0) - big).to_f64(), 1.0);
assert_eq!((1e16_f64 + 1.0) - 1e16, 0.0);

// Decimal conversions are correctly rounded; `Display` prints 31 digits,
// so a decimal of at most 31 digits comes back as the same text.
let x: DoubleDouble = "0.1000000000000000000000000000001".parse()?;
assert_eq!(x.to_string(), "0.1000000000000000000000000000001");
let third = DoubleDouble::ONE / 3.0;
assert_eq!(third.to_string(), "0.3333333333333333333333333333333");

// Elementary functions, each with a declared and proven bound: 8u² for exp.
let u2 = DoubleDouble::ONE.ldexp(-106);
let e = DoubleDouble::ONE.exp();
assert!((e - DoubleDouble::E).abs() <= DoubleDouble::E * u2 * 8.0);
# Ok::<(), Box<dyn std::error::Error>>(())
```

**2. Compensated kernels** — `kernels::sum2`, `dot2` and `dot2_err` (Ogita,
Rump & Oishi 2005) take `f64` slices and return a double-double as accurate
as if computed in twice the working precision; `dot2_err` also returns a
rigorous bound `β` on its own error. The double-double `dot`, `dot_mixed`,
`axpy` and a small-block `gemm_sub` complete the set. The SIMD versions
(feature `simd`) return the bits of the scalar reference, input by input.

```rust
use xingo_extprec::kernels::{dot2, dot2_err};

let x = [1e16, 1.0, -1e16];
let y = [1.0, 1.0, 1.0];
let naive: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
assert_eq!(naive, 0.0); // the 1 is lost
assert_eq!(dot2(&x, &y).to_f64(), 1.0);

// |xᵀy − r| ≤ β, a theorem about the f64 data.
let (r, beta) = dot2_err(&x, &y);
assert!((r.to_f64() - 1.0).abs() <= beta);
```

**3. `Interval`** — closed intervals with `f64` endpoints whose `+ − ·`,
`/` (by an interval without zero) and `sqrt` return the tightest
representable enclosure of the exact range, underflow included.
Directed rounding is *computed* from the error-free transformations
instead of set in hardware: Rust has no sound way to change the rounding
mode, and this crate has no `unsafe`.

```rust
use xingo_extprec::Interval;

let tenth = Interval::point(0.1); // the f64 nearest to 0.1, exactly
let s = tenth + tenth + tenth;
assert!(s.contains(0.1 + 0.1 + 0.1));
assert!(s.inf() < s.sup()); // the exact sum is not an f64
assert!((s - Interval::point(0.3)).certainly_within(1e-15));
```

**4. `verify`** (feature `sparse`) — Algorithm V encloses every residual
`a_iᵀx − b_i` of a sparse system (`CsrMatrix` or `CscMatrix` of
`xingo-sparse`) for an `f64` or a double-double `x`, at about 25 flops per
nonzero; Algorithm P *polishes* `x` by iterative refinement with
double-double residuals around an `f64` solve the caller already has (an
LU, say). The polished residual is then certified by Algorithm V.

```rust
use xingo_extprec::verify::{PolishOptions, PolishStop, norm_inf_bound, polish, residual, residual_of_dd};
use xingo_extprec::DoubleDouble;
use xingo_sparse::CsrMatrix;

// A = [1 2; 3 4] by rows, b = A (1/3, 1/3).
let a = CsrMatrix::from_raw_parts(2, vec![0, 2, 4], vec![0, 1, 0, 1], vec![1.0, 2.0, 3.0, 4.0])?;
let b = [1.0, 7.0 / 3.0];
let mut solve = |s: &mut [f64]| {
    // A⁻¹ s in f64 (Cramer's rule; in practice an LU of A).
    let (s0, s1) = (s[0], s[1]);
    s[0] = (4.0 * s0 - 2.0 * s1) / -2.0;
    s[1] = (-3.0 * s0 + s1) / -2.0;
};
let mut x = [0.0; 2];
x.copy_from_slice(&b);
solve(&mut x);

// Certified residual of the f64 solution: every row's interval contains
// the exact a_iᵀx − b_i of the stored data.
let r = residual(&a, &x, &b)?;
assert!(norm_inf_bound(&r) <= 1e-14);

// Polish in double-double around the same f64 solve, then certify.
let mut xd = x.map(DoubleDouble::from_f64);
let report = polish(&a, &b, &mut xd, &mut solve, &PolishOptions::default())?;
assert_eq!(report.stop, PolishStop::Converged);
assert!(norm_inf_bound(&residual_of_dd(&a, &xd, &b)?) <= 1e-25);
# Ok::<(), Box<dyn std::error::Error>>(())
```

With the feature `sparse`, `DoubleDouble` implements `xingo_sparse::Real`,
so `Lu<DoubleDouble>`, `SupernodalLdl<DoubleDouble>` and every container of
`xingo-sparse` exist without one line of duplicated code, and `DdDense` is
the `DenseKernel` of the supernodal LDLᵀ in double-double, with SIMD and the
bits of the generic kernel. The programs in [`examples/`](examples/)
(`dd_profile`, `dd_bench`, `ft_limit`) reproduce the measurements recorded
in the solver's decision log; they print Markdown tables.

## Proven error bounds

Every bound below is a theorem, not a fit to a measurement, and every row
is data the code exposes: `DoubleDouble::ERROR_BOUNDS` (operations) and
`DoubleDouble::FUNCTION_BOUNDS` (elementary functions). `u = 2⁻⁵³`, so
`u² = 2⁻¹⁰⁶`. The bounds are relative and hold barring overflow and
underflow: for the operations, operands in the "safe range" of Chapter
11.2 (every product formed by `two_prod` at least `2⁻⁹⁶⁹` in magnitude, or
zero; outside it the result is still normalized, with an absolute error of
order `2⁻¹⁰⁷⁴`); for the functions, results of magnitude in `[2⁻⁹⁶⁸, Ω]`
(below, plus `2⁻¹⁰⁷⁴`). The tests check every row against exact
arithmetic.

| operation | algorithm | relative error | source of the proof |
|---|---|---|---|
| `dd + f64` | DWPlusFP | `2u²/(1 − 2u)` | Joldes, Muller & Popescu 2017 |
| `dd + dd` | AccurateDWPlusDW | `3u²/(1 − 4u)` | Joldes, Muller & Popescu 2017 |
| `dd · f64` | DWTimesFP3 | `2u²` | Joldes, Muller & Popescu 2017 |
| `dd · dd` | DWTimesDW3 | `4u²` | Muller & Rideau 2022 (Coq) |
| `dd²` | DWPow2 | `3u²` | Hubrecht & Melquiond 2026 (Gappa, binary64) |
| `dd / f64` | DWDivFP3 | `3u²` | Joldes, Muller & Popescu 2017 |
| `dd / dd` | DWDivDW2 | `15u² + 56u³` | Joldes, Muller & Popescu 2017 |
| `√dd` | SQRTDWtoDW | `25/8 u²` | Lefèvre, Louvet, Muller, Picot & Rideau 2023 |

Each `units_of_u2` of `ERROR_BOUNDS` is an `f64` at least the full
expression above, so that `units_of_u2 · u²` is itself a rigorous bound:
`2`, `4`, `3`, `3` and `3.125` exactly, and the three expressions with
higher-order terms rounded up — `2 + 2⁻⁵⁰`, `3 + 2⁻⁴⁹`, `15 + 2⁻⁴⁷` (a test
checks each row in exact rational arithmetic). The division is DWDivDW2, not the DWDivDW3 with the better
bound `9.8u²`: a quotient through a rounded reciprocal loses the exact
quotients (`53/53 ≠ 1`, `(q·y)/y ≠ q` for 4 600 of the 39 601 integer pairs
below 200), which DWDivDW2 keeps.

| function | declared bound (`u²` relative, plus absolute floor) | proven in Chapter 11.11 |
|---|---|---|
| `exp`, `exp2`, `exp_m1`, `ln`, `ln_1p`, `atan` | `8u²` | 4.6–7.7 |
| `log2`, `log10` | `12u²` | 11.7 |
| `powi(x, n)` | `(15 + 4|n|)u²` | `11 + 4|n|` |
| `powf(x, y)` | `(16 + 11|y ln x|)u²` | `4.6 + 10.63|y ln x|` |
| `sin`, `cos`, `sin_cos` | `8u²` + `2⁻²⁰⁰` absolute | `6.9u²` + `2⁻³¹⁵` |
| `tan` | `(32 + 2⁻⁹⁴|tan x|)u²` + `2⁻²⁰⁰` absolute | 27.4 |
| `asin`, `acos`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` | `32u²` | 15.1–26.4 |

The circular functions reduce their argument by Payne–Hanek in integer
arithmetic for every `|x| > π/4`, so the bound holds on the whole real
line (`sin(2⁹⁶⁰)` included). In a validation outside the test suite,
against an oracle the implementation does not use (`mpmath` at 3 000 bits,
256 000 samples over the 23 functions), no bound was violated and the
worst error was 30 % of its declared bound.

## Features

* `sparse` (default) — brings in [`xingo-sparse`](https://crates.io/crates/xingo-sparse)
  0.4 (pure Rust, no dependency of its own with `default-features = false`,
  which is how this crate asks for it): `impl xingo_sparse::Real for
  DoubleDouble`, the dense kernel `DdDense` and the module `verify`.
* `simd` (default) — the kernels on vector registers through
  [`pulp`](https://crates.io/crates/pulp) 0.22 (pure Rust, MIT; safe
  run-time dispatch): NEON on aarch64, AVX2 + FMA on x86_64 CPUs that
  have them, the scalar reference elsewhere. **Same results bit for bit** with or without it
  (invariant K1); only the speed changes. `kernels::backend()` names the
  path taken.
* Without both (`default-features = false`) the crate has no dependency at
  all: `DoubleDouble` with its elementary functions, the error-free
  transformations, the scalar kernels and `Interval` remain.

## Guarantees

* **Deterministic, the same bits everywhere.** Every operation is a fixed
  sequence of `f64` operations: no data-dependent choice of algorithm, no
  reassociation, no threads. The SIMD kernels return the bits of the scalar
  ones (where a result is a NaN both are NaN; its sign and payload are not
  specified). Requires IEEE 754 binary64 with round-to-nearest and a
  correctly rounded `fma`, which Rust guarantees on every tier-1 target
  (`f64::mul_add` falls back to a correct software FMA where the CPU has
  none: slow, never wrong).
* **No libm calls.** The precision of `f64::exp`, `ln`, `sin`, … is
  unspecified by the standard library and may differ between platforms, so
  none of them is called by this crate (a `clippy.toml` rule forbids them):
  the seeds of the elementary functions are tables, and only
  `+ − × ÷ √ fma round trunc` of `f64` are used. (`pulp`, with the `simd`
  feature, depends on the `libm` crate; the kernels use only its
  additions, multiplications and FMAs.)
* **No `unsafe`** (`#![forbid(unsafe_code)]`); `pulp` has its own, behind a
  safe API.
* **A certificate never claims more than it proved.** When a hypothesis of
  a bound fails (overflow, a NaN or an infinity in the data, a length with
  `2(n + 3)u ≥ 1`) the result is `Interval::ENTIRE`, a bound of `+∞` or an
  error, never a finite interval. The `certainly_*` predicates are `false`
  on the empty set.
* **What a certificate means.** The enclosures are statements about the
  `f64` data as stored. That a decimal `0.1` in an input file became the
  `f64` nearest to it is outside this crate.

## Scope

In: the error-free transformations `two_sum`, `fast_two_sum`, `two_prod`;
`DoubleDouble` with its field operations, `sqrt`, comparisons, decimal
conversions, 23 elementary functions and 19 constants; the compensated and
double-double slice kernels; `Interval` with `+ − · /`, `sqrt`, `square`,
`abs`, hull, intersection and the verified predicates; rigorous residual
enclosures and polishing over the containers of `xingo-sparse`; the
`DenseKernel` `DdDense`.

Out, by decision: arbitrary precision; complex scalars; `f32`; a
`QuadDouble` (the name is reserved; a basis that double-double does not
resolve would reopen it — none of the 93 Netlib optimal bases needed it);
elementary functions of intervals (they would need enclosures, not bounds);
correctly rounded elementary functions; interval linear algebra (interval
LU, Krawczyk, verified solution sets à la `verifylss`); the QD library's
"sloppy" addition.

Not in 0.1, and additive when they come: arithmetic operators with an
`f64` on the left (`f64 + DoubleDouble`) and the assignment operators with
an `f64` (`x += 1.0`) for `DoubleDouble` (`Interval` has both); `floor`,
`ceil`, `round`, `trunc`, `signum`, `total_cmp`.

## Known limitations

1. **The proofs are not machine-checked.** The bounds of the operations
   come from the literature (some formalized there: Muller & Rideau in
   Coq, Hubrecht & Melquiond with Gappa); the bounds of the elementary
   functions and of the kernels are first-order analyses written in the
   chapters of the sources, checked by tests against two exact oracles and
   once, outside the suite, against `mpmath`; not by a proof assistant.
   Each proven constant of Chapter 11.11 is honest up to a factor
   `1 + 2⁻⁸⁰` (the neglected products of errors), and every declared
   constant exceeds it by far more (the smallest margin is `log2` and
   `log10`: 12 declared, 11.7 proven). Before 0.1 the proofs of Chapter
   11.11 were re-derived line by line by three independent reviews, each
   with a directed search against `mpmath`: they found three defects
   (`acos` and `acosh` of `1 ∓ δ` and `atanh` of `±(1 − δ)` with a tiny
   trailing component `δ`), fixed before release, and one gap: for `tan`
   in an odd quadrant with a reduced argument below `2⁻³⁰⁰` — never
   observed, the smallest found being `2⁻¹¹⁸`, but not proven impossible
   — only the declared bound is proven, not the sharper one of the chapter.
2. **Same bits across platforms: observed on the targets of the CI.** The
   design gives the same bits on every target; the test suite, with the
   golden bits of the elementary functions and the SIMD kernels compared
   bit for bit with the scalar reference, runs in the CI of the mirror on
   x86_64 Linux, x86_64 Windows, aarch64 Linux and macOS (NEON) and i686
   Linux (32 bits), and was run by hand on x86_64 Linux with AVX2 + FMA (a
   Xeon E5 v4: the AVX2 path) and on x86_64 macOS under emulation (the
   scalar path). Not run: big-endian targets, WebAssembly, and x86 CPUs
   without SSE2.
3. The bounds hold for operands in the "safe range" (every `two_prod`
   product of magnitude at least `2⁻⁹⁶⁹` or zero). Below it the operations
   stay normalized with an absolute error of order `2⁻¹⁰⁷⁴`; `Interval`
   and `verify` account for underflow explicitly. The elementary
   functions state their bounds with that `η = 2⁻¹⁰⁷⁴` added for results
   below `2⁻⁹⁶⁸`, and all 23 are tested there against the oracle, at
   directed points and in property tests: subnormal arguments, normal
   ones with a subnormal trailing component (`1 ± δ` with `δ` down to
   `2⁻¹⁰⁷⁴` included), and results that underflow (`exp`, `exp2`,
   `exp_m1`, `powi`, `powf`, `atan2`). That regime is tested, not
   exhausted: `sin` and `tan` were not driven to subnormal results other
   than from tiny arguments (near nonzero multiples of `π` the absolute
   term `2⁻²⁰⁰` of their bound is far larger than `η`).
4. `verify` bounds residuals, not the error `‖x − x*‖` (that needs a
   rigorous bound on `‖A⁻¹‖`). `polish` stops on a per-row test
   `|r_i| ≤ τ (Σ_j |a_ij x_j| + |b_i|)`; on bases with degenerate rows it
   often ends `Stagnated`, which is harmless (9 of the 93 Netlib bases).
5. Cost: an elementary function takes 190–750 ns per call (`exp` 232,
   `ln` 430, `sin` 344, `atan` 190, `powf` 753) against 1.9 ns for a
   double-double product and 2.5 ns for `f64::exp`. `sin_cos` shares the
   reduction, not the kernels (599 ns). Measured once, on one machine,
   under load from other processes, by a harness since replaced by the
   example `elementary_bench` (`cargo run --release --example
   elementary_bench`, no feature needed), which prints the same rows on
   any machine.

## Measured cost

On an Apple M2 Pro (NEON), release builds, medians recorded in the
solver's benchmark log (x86_64 not measured):

| what | cost |
|---|---|
| `dd · dd`, `dd + dd`, `dd / dd`, `√dd` (throughput) | 1.9, 2.8, 3.4, 1.4 ns |
| SIMD over the scalar reference: `sum2`, `dot2`, `dot`, `dot_mixed` | 1.9–2.1× faster |
| … `axpy`, `gemm_sub`, `dot2_err` | 1.65–1.84×, 1.63–1.66×, 1.3–1.4× |
| `dot2` (SIMD) against the naive `f64` dot product, `n ≥ 1 024` | 0.8–0.9× the time |
| supernodal LDLᵀ factorization in double-double, `DdDense` over the generic kernel | 1.47–1.67× faster (1.00× when the factorization is simplicial; solves unchanged) |
| double-double over `f64`: LDLᵀ factorization / solve | 6–11× / 10–16× |
| double-double over `f64`: LU factorization / FTRAN | 1.7–1.9× / 5–8× |

In the Xingó solver, certifying the residuals of the optimal basis of each
of the 93 Netlib LPs by intervals takes 0.18 % of the solve time in total
(`--exact-check=interval`), and all 93 come out certified.

## Design notes

* **Literate.** Every module is a chapter: notation, the algorithm in
  numbered steps, invariants (each a `debug_assert!` or a
  `check_invariants()`), proofs of the bounds, references. The crate
  documentation *is* the text: start at the crate root on docs.rs
  (Chapter 11, with the sections "Corrections to the approved contract",
  which list what the implementation found wrong in its own contract).
  Most chapters, the proofs of Chapters 11.2 and 11.11 among them, are the
  documentation of private modules, so docs.rs builds with
  `--document-private-items`; private items are marked as such.
* **Read, not copied.** The QD library (Hida, Li & Bailey), `twofloat`
  and `inari` were studied as references, INTLAB's `verifylss` as the
  boundary of the scope; the code is written from the papers. The exact oracles of the
  tests are `dashu` (MIT OR Apache-2.0), a development dependency only.
* **Errors, not panics**, on data: a dimension mismatch is a
  `VerifyError`, an unreadable decimal a `ParseDoubleDoubleError`, an
  inverted interval an `InvalidInterval`. Panics are reserved for contract
  violations (slices of different lengths in a kernel, a NaN endpoint in
  `Interval::point`).

## Minimum supported Rust version

1.90 (edition 2024), tested in CI together with stable.

## Repository, issues and citation

The crate is developed inside the Xingó monorepo
([StrategicProjects/xingo](https://github.com/StrategicProjects/xingo),
directory `crates/extprec`), where every change is validated together with
the solver that uses it. This stand-alone repository,
[StrategicProjects/xingo-extprec](https://github.com/StrategicProjects/xingo-extprec),
is a read-only mirror of that directory (`git subtree split`, full history),
updated at every release; it exists so that the library can be archived and
cited on its own. Issues and pull requests about the crate are welcome in
either repository. Each release of the mirror is archived on Zenodo, and
`CITATION.cff` has the citation metadata.
<!-- TODO(owner): the concept DOI and the sentence "the concept DOI … always resolves to the latest version", once Zenodo has archived the first release. -->

## Authorship, credits and disclaimer

`xingo-extprec` is written and maintained by **André Leite** as the
extended-precision layer of the [Xingó](https://github.com/StrategicProjects/xingo)
solver, developed at [StrategicProjects](https://github.com/StrategicProjects).

The algorithms are those of their authors, implemented here from the
published papers: Dekker and Knuth (error-free transformations), Joldes,
Muller & Popescu and Muller & Rideau (double-word arithmetic and its
bounds), Lefèvre, Louvet, Muller, Picot & Rideau (the double-word square
root), Hubrecht & Melquiond (the double-word square), Ogita, Rump & Oishi
(compensated sums and dot products), Rump, Zimmermann, Boldo & Melquiond
(predecessor and successor in rounding to nearest), Tang (table-driven
exponential), Payne & Hanek (argument reduction), Kahan (the correction of
`ln_1p`). The QD library of Hida, Li & Bailey, `twofloat`, `inari` and
INTLAB were studied as references; `dashu` (MIT OR Apache-2.0) is the exact
oracle of the tests and `pulp` (MIT) the optional SIMD dependency. Each chapter of the sources lists its references.

This is an independent project: it is not affiliated with, maintained by or
endorsed by the authors of the QD library, `twofloat`, `inari`, INTLAB,
`pulp` or `dashu`, and it includes no code from them (`pulp` is linked only
through the optional `simd` feature). All names and trademarks belong to
their owners. The software is provided "as is", without warranty of any
kind, under the licenses below.

## License

Licensed under either of the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
or the [MIT license](https://opensource.org/licenses/MIT), at your option.
