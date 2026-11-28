# Machine-checked proofs for `xingo-extprec`

This directory holds proofs, checked by the Rocq prover, of some error bounds
stated in the chapters of `xingo-extprec` (issue xingo-extprec#2). It is not
part of the published crate (`exclude` in `Cargo.toml`), and none of the
tools below is a dependency of the crate: they are needed only to recheck the
proofs.

## Recheck everything

```sh
crates/extprec/proofs/check.sh
```

The script (about 1.5 min on an Apple M-series laptop) regenerates the Gappa
certificates, checks every Rocq file, checks that the theorems rest on no
assumption besides the standard axioms of Rocq's real numbers, runs every
**mutant** (a wrong constant or a wrong step) and requires each one to be
rejected, and finally evaluates concrete counterexamples to the mutated
statements in exact rational arithmetic. It exits with a non-zero status if a
proof fails, a mutant is accepted or a counterexample does not refute its
mutant. Scratch files go to `proofs/_build/` (git-ignored).

In the mirror, the workflow `.github/workflows/proofs.yml` runs the same
script in a job of its own (Ubuntu 24.04, pinned tool versions, the opam
root cached): on pushes and pull requests that touch `proofs/`,
`src/dd.rs`, `src/eft.rs` or `src/kernels.rs`, weekly, and on demand.

Tools: Gappa 1.8.3, Rocq 9.2.0, Flocq 4.2.2, the Gappa support library for
Rocq (`coq-gappa`) 1.11.0, and Python >= 3.13 (`math.fma`) for the
counterexamples (that step is skipped without it). `check.sh` uses `gappa` and
`rocq` from `PATH`, or else the opam switch named by `XINGO_PROOFS_SWITCH`
(default `xingo-proofs`):

```sh
opam switch create xingo-proofs ocaml-base-compiler.5.2.1
opam repo add rocq-released https://rocq-prover.org/opam/released
opam install --switch=xingo-proofs rocq-core.9.2.0 rocq-stdlib.9.2.0 \
  coq-flocq.4.2.2 coq-gappa.1.11.0
# Gappa 1.8.3 is newer than the opam package (1.6.0): build it from
# https://gappa.gitlabpages.inria.fr/releases/gappa-1.8.3.tar.gz
# (./configure --prefix=$(opam var --switch=xingo-proofs prefix); ./remake; ./remake install)
```

On macOS with a Command Line Tools SDK newer than the running system (SDK
27 on macOS 26), OCaml must be configured with `ac_cv_func_pipe2=no
ac_cv_func_dup3=no`: the SDK declares `pipe2` and `dup3` for macOS 27, they
are weakly linked, and every `Unix.pipe` then crashes (dune included).

## What is proved

The model of binary64 is Flocq's `FLT_exp (-1074) 53` format (subnormals
included) with round-to-nearest-even (`rocq/Model.v`). Flocq's format has no
largest exponent, so each theorem also bounds every rounded intermediate
result by `2^965` or less, which excludes overflow; below `2^1024` an IEEE 754
operation returns exactly the rounded value of the model.

| Bound (chapter) | File | Statement proved | Hypotheses |
|---|---|---|---|
| `dd²`, Algorithm M3 (11.2), `3u²` | `rocq/M3.v`, `gappa/m3_*.g` | `\|z_h + z_l − x²\| ≤ 3·2⁻¹⁰⁶ x²`; the 2Prod error term, `2 x_h` and the final Fast2Sum are exact; `z_h = RN(z_h + z_l)`; all intermediates `≤ 2⁹⁶⁵` | `x_h`, `x_l` binary64, `x_h = RN(x_h + x_l)`, `2⁻⁴⁸⁰ ≤ \|x_h\| ≤ 2⁴⁸⁰` |
| `dd · f64`, Algorithm M1 (11.2) | `rocq/M1.v`, `gappa/m1_*.g` | `\|z_h + z_l − x y\| ≤ 2u²(1 + 2⁻³⁷)\|x y\|`; exactness and magnitudes as for M3 | as for M3, and `y` binary64 with `2⁻⁴⁸⁰ ≤ \|y\| ≤ 2⁴⁸⁰` |
| `dot2_err`, step C5 (11.5) | `rocq/C5.v` | `(n + 3)u / (1 − (2n + 5)u) · e + nη/2 ≤ β` for the `β` computed by `result_with_bound`; `a`, `1 − 2a`, `δ` exact or bounded by `2⁵²` | `n` integer, `1 ≤ n`, `n + 3 < 2⁵²`; `e` real, `e ≥ 0` |

Notes.

* **M1 is not proved with the constant of `ERROR_BOUNDS`.** The table of
  Chapter 11.2 states `2u²` (Joldes, Muller & Popescu). Gappa 1.8.3 reaches
  `2u²(1 + 2⁻³⁷)` with the hints tried, not `2u²`. This is a limit of the
  proof, not a counterexample: the largest error found by search is
  `1.55u²`.
* **C5, overflow.** If `RN(δ e)`, the sum or the product overflows, IEEE 754
  returns `+∞` and `β = +∞`, so the bound holds; that step is argued on paper,
  not in Rocq.
* **The zero operand** of the safe range (`x_h = 0`, hence `x_l = 0`, result
  exactly `0`) is not covered by the theorems; it is immediate.
* **Step E** of Chapter 11.2 (`finish`) returns `(z_h, z_l)` unchanged when
  both are finite, which the magnitude bounds give; that reading of the code
  is not mechanized.

## How the proofs are built

Gappa proves the error of the computation in one binade of each operand, with
an unbounded exponent (`float<53,ne>`), in two cases: step 2 (the FMA that
forms the low part) does not underflow (`*_normal.g`), or it does
(`*_underflow.g`, where the rounding error is absolute, at most `2⁻¹⁰⁷⁵`,
that is `2⁻¹¹⁵` after scaling). Gappa writes a certificate in Rocq; Rocq
checks it. The files `rocq/M*.v` then reduce the binary64 statement over the
whole safe range to these two cases: they scale the operands by powers of two
(rounding to 53 bits commutes with that), prove that every other rounding is
of a normal number, take the exactness of the 2Prod error term from Flocq
(`mult_error_FLT`) and that of the Fast2Sum from Flocq (`Fast2Sum_correct`).
`tools/mkunfold.sh` generates, for each certificate, the tactic that unfolds
its definitions (their names depend on the Gappa version).

Gappa's rewriting hints (`a -> b;`) are **hypotheses** of its certificates
(`Hypothesis a1 : ...` in the generated file): Gappa does not prove them. The
Rocq files discharge each one by `ring` before using the certificate. The
mutant `m3-no-factor-2-stale-hint` shows why it matters: with a false hint,
Gappa alone accepts a wrong algorithm, and only the Rocq side rejects it.

C5 is proved directly in Rocq with Flocq's error bounds; Gappa was not needed
there (the statement is linear in `e` with two unbounded parameters, which
suits Rocq better than interval reasoning).

## Trust base and correspondence with the code

Trusted: the Rocq kernel, the axioms of Rocq's real numbers (plus classical
logic), Flocq's definition of rounding and the Gappa support library's
definitions. The Gappa binary is not trusted: its certificates are checked.

The proofs are about the algorithms **as transcribed** in `m3_steps`,
`m1_steps` and `c5_steps`, whose comments name the lines of `src/dd.rs` and
`src/kernels.rs`. That the Rust code performs these binary64 operations (each
`+`, `-`, `*`, `/` rounded to nearest even, `mul_add` a single rounding, no
contraction by the compiler) is checked by review, and the exact-rational
tests of the crate remain the bridge between the code and the model.

## Mutants

| Mutant | Change | Rejected by | Counterexample |
|---|---|---|---|
| `m3-bound-2.5u2` | M3 claims `2.5u²` | Gappa | error `2.90u²` |
| `m3-no-factor-2` | step 2 of M3 without the factor 2 (hint kept true) | Gappa | error `≈ 9·10¹⁵ u²` |
| `m3-no-factor-2-stale-hint` | same, hint left false | Rocq (`ring` on the hint) | same |
| `m1-bound-1.5u2` | M1 claims `1.5u²` | Gappa | error `1.55u²` |
| `m1-no-cl1` | step 2 of M1 without the 2Prod error term | Gappa | error `≈ 7·10¹⁵ u²` |
| `c5-inflate-1` | `INFLATE = 1` | Rocq (needs `INFLATE ≥ 1 + 8u`) | `n = 527222` |
| `c5-underflow-0` | `UNDERFLOW = 0` | Rocq (needs `UNDERFLOW ≥ 3η/u`) | `n = 1`, `e = 0` |
| `c5-terms-n+2` | `a = (n + 2)u` | Rocq (the proof script no longer applies) | `n = 1`, `e = 1` |

The inputs of the counterexamples are in `mutants/counterexamples.py`.
