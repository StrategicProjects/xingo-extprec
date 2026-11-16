//! # Chapter 11.8 — Thresholds for double-double: the profile of `LuOptions`
//!
//! ## Purpose
//! `xingo_sparse::LuOptions::default()` is calibrated for `f64`: some of
//! its thresholds are multiples of the machine epsilon in disguise. Handed
//! unchanged to `Lu<DoubleDouble>` they cap the accuracy of the factors at
//! that of `f64` or reject matrices that double-double factorizes without
//! trouble. This chapter says which thresholds depend on the epsilon,
//! states the rule that carries them to double-double and records the
//! measurements behind it. [`DoubleDouble::lu_options`] is the result.
//!
//! ## Notation
//! `ε₆₄ = 2⁻⁵²` and `ε_dd = 2⁻¹⁰⁴` are the `Real::EPSILON` of the two
//! scalars; `ρ = ε_dd / ε₆₄ = 2⁻⁵²`. The *scaled residual* of a solve `B x
//! = b` of order `n` is `‖B x − b‖∞ / (n ε_dd (max|b_ij| ‖x‖∞ + ‖b‖∞))`,
//! computed in double-double: `O(1)` for a backward-stable solve, `O(1/ρ)
//! ≈ 10¹⁵` for one that is only as accurate as `f64`.
//!
//! ## The thresholds of `LuOptions`
//! | field | default | depends on `ε`? | double-double |
//! |---|---|---|---|
//! | `drop_tol` | `10⁻¹²` (`4 504 ε₆₄`) | yes: "below rounding noise" | `10⁻¹² ρ ≈ 2.2·10⁻²⁸` |
//! | `abs_pivot_tol` | `10⁻¹⁰` | yes: "too small to trust as a pivot" | `10⁻¹⁰ ρ ≈ 2.2·10⁻²⁶` |
//! | `update_stability_tol` | `10⁻⁸` (`≈ √ε₆₄`) | yes: digits an update may lose | `10⁻⁸ ρ ≈ 2.2·10⁻²⁴` |
//! | `growth_limit` | `10¹⁰` | in principle (`growth · ε` is the backward error) | unchanged |
//! | `rel_pivot_tol` | `10⁻²` | no: sparsity against growth | unchanged |
//! | `search_limit`, `fill_factor`, `refactor_interval`, `hyper_*_density` | | no: structure and cost | unchanged |
//!
//! ## Rule P (the profile)
//! P1. \[Scale.\] A threshold that is a magnitude below which a *computed*
//!     quantity is taken for rounding noise — `drop_tol`, `abs_pivot_tol`,
//!     `update_stability_tol` — is multiplied by `ρ`.
//! P2. \[Keep.\] Every other option keeps its default.
//!
//! For `f64` the rule is the identity (`ρ = 1`), so it could be stated for
//! any scalar `T` as "multiply by `ε_T / ε₆₄`"; that generic form belongs
//! in `xingo-sparse` and is proposed there, not implemented here.
//!
//! ## Measurements (`examples/dd_profile.rs`, 2026-11-15)
//! All in `Lu<DoubleDouble>`, residuals computed in double-double.
//!
//! * **`drop_tol`.** With the `f64` default (`10⁻¹²`) the median scaled
//!   residual over 15 random bases was `2.2·10¹³` (order 300) and
//!   `4.1·10¹⁵` (order 800): the accuracy of `f64`, sixteen digits thrown
//!   away. With `0`, `2⁻¹⁰⁰`, `10⁻²⁹` and `10⁻¹² ρ` the medians were between
//!   `3.2` and `10.6` with no order among them (the differences are the
//!   pivot sequences, which the pattern decides); with `10⁻²⁴` the median
//!   of the order-800 family rose to `231`. Dropping is still worth
//!   having: an entry that cancels exactly in `f64` leaves a residue of a
//!   few `ε_dd` in double-double, and with `drop_tol = 0` those residues
//!   stay as fill (0.6–2 % more entries of `L + U` here). On bases of small
//!   integers the tolerance made no difference at all (nothing tiny
//!   arises).
//! * **`abs_pivot_tol`.** With the `f64` default the Hilbert matrices of
//!   order 16, 20 and 24 are declared singular (deficiency 4, 6, 8), as is
//!   a basis with two columns `10⁻¹¹` apart and a matrix whose columns are
//!   graded over 12 decades. With `10⁻¹⁰ ρ` all of them factorize: Hilbert
//!   16 to an error of `1.3·10⁻¹⁰` in the solution (condition `≈ 10²²`),
//!   Hilbert 20 to `1.2·10⁻⁴`; Hilbert 24 (condition above `1/ε_dd`) is
//!   accepted and its solution is noise — the threshold is not a condition
//!   estimator, in either scalar. Columns `10⁻²⁷` apart, and columns graded
//!   over 30 decades, are singular under the profile too: it is absolute,
//!   in the units of `B`, as the default is (scale first).
//! * **`update_stability_tol`.** 90 Forrest–Tomlin exchanges on a basis of
//!   order 150, every third with a pivot `α_r` of a chosen size. For
//!   `α_r = 10⁻²` none of the three candidates below refused anything and
//!   the updated factor was at most 110 times less accurate than a fresh
//!   one. For `α_r = 10⁻⁶`,
//!   `10⁻¹⁰`, `10⁻¹⁴` the updated factors were up to `3.5·10⁹`,
//!   `3.4·10¹³`, `2.8·10¹³` times less accurate under `10⁻⁸` **and** (the
//!   last one `2.5·10¹³`) under `10⁻⁸ √ρ`, which refused 0, 1, 1 updates, and `2.1·10⁸`,
//!   `7.4·10¹²`, `2.1·10¹¹` under `10⁻⁸ ρ` (which refused 1, 20, 13;
//!   median loss `230`, `160`, `1.4` against `4 400`, `23 000`, `11`).
//!   Hence `ρ`, not `√ρ` — and a caveat: **no tolerance bounds the loss**
//!   (`10⁻²⁸` still let `7.4·10¹²` through); a caller that needs
//!   double-double accuracy after an exchange with a tiny pivot should
//!   refactorize.
//! * **`growth_limit`.** The largest growth seen was `6.5·10⁷`; no
//!   measurement speaks for changing the limit, so it stays.
//!
//! ## What is not in `LuOptions`
//! The same question for the KKT solver was measured and is **proposed,
//! not implemented** (the defaults live in `xingo-sparse`):
//! `Refinement::default()` stops at a residual of `10⁻¹²`, the accuracy of
//! `f64` (`2.2·10⁻²⁸ = 10⁻¹² ρ` was reached in 5–8 steps where it is
//! attainable); `Regularization::default()` (`10⁻⁶`, `10⁻⁸`) leaves a plain
//! solve at `10⁻⁵`–`10⁰` and costs those steps, against 2–4 with the shifts
//! scaled by `√ρ`; the two thresholds of the pivoted path
//! (`SupernodalOptions::zero_pivot_tol`, `pivot_threshold`) made no
//! difference on the matrices tried.
//!
//! ## Invariants
//! (O1) [`DoubleDouble::lu_options`] differs from `LuOptions::default()`
//!      in the three fields of step P1 only, each exactly `ρ` times the
//!      default (a power of two: the product is exact). Tested.
//!
//! ## References
//! Chapter 5.3 of `xingo-sparse` (`LuOptions`: what each threshold tests)
//! and Chapter 5.5 (the stability test of the Forrest–Tomlin update);
//! Higham, *Accuracy and Stability of Numerical Algorithms* (2002), §9.3
//! (growth and backward error of Gaussian elimination); D-0109 (the
//! reservation of L2.2a that this chapter answers).

use xingo_sparse::LuOptions;

use crate::dd::DoubleDouble;

/// `ρ = ε_dd / ε₆₄ = 2⁻⁵²` (the chapter).
const EPSILON_RATIO: f64 = f64::EPSILON;

impl DoubleDouble {
    /// The options of `xingo_sparse::Lu<DoubleDouble>`: the defaults of
    /// `LuOptions` with the three thresholds that are magnitudes of
    /// rounding noise — `drop_tol`, `abs_pivot_tol`,
    /// `update_stability_tol` — multiplied by `ε_dd / ε₆₄ = 2⁻⁵²` (Rule P
    /// of Chapter 11.8, with the measurements behind it). With
    /// `LuOptions::default()` a double-double factorization is only as
    /// accurate as an `f64` one.
    ///
    /// ```
    /// use xingo_extprec::DoubleDouble;
    /// use xingo_sparse::{Lu, LuOptions};
    /// let opts = DoubleDouble::lu_options();
    /// assert_eq!(opts.drop_tol, 1e-12 * f64::EPSILON);
    /// assert_eq!(opts.rel_pivot_tol, LuOptions::default().rel_pivot_tol);
    /// let _lu = Lu::<DoubleDouble>::new(opts);
    /// ```
    ///
    /// *Provisional.* The profile belongs in `xingo-sparse`, as a
    /// constructor generic in the scalar (`LuOptions::for_scalar::<T>()`,
    /// D-0111); this function is hidden from the documentation and leaves
    /// before the crate is published.
    #[doc(hidden)]
    #[must_use]
    pub fn lu_options() -> LuOptions {
        let d = LuOptions::default();
        LuOptions {
            abs_pivot_tol: d.abs_pivot_tol * EPSILON_RATIO,
            update_stability_tol: d.update_stability_tol * EPSILON_RATIO,
            drop_tol: d.drop_tol * EPSILON_RATIO,
            ..d
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the scalings are exact
mod tests {
    use super::*;

    /// Invariant (O1).
    #[test]
    fn profile_scales_three_thresholds_and_nothing_else() {
        let d = LuOptions::default();
        let p = DoubleDouble::lu_options();
        let rho = DoubleDouble::EPSILON.hi() / f64::EPSILON;
        assert_eq!(rho, EPSILON_RATIO);
        assert_eq!(p.drop_tol, d.drop_tol * rho);
        assert_eq!(p.abs_pivot_tol, d.abs_pivot_tol * rho);
        assert_eq!(p.update_stability_tol, d.update_stability_tol * rho);
        assert_eq!(
            LuOptions {
                drop_tol: d.drop_tol,
                abs_pivot_tol: d.abs_pivot_tol,
                update_stability_tol: d.update_stability_tol,
                ..p
            },
            d
        );
    }
}
