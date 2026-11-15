//! # Chapter 11.1 — Error-free transformations
//!
//! ## Purpose
//! The three primitives of the crate. Each rewrites the exact result of one
//! `f64` operation as a pair of `f64` values: the rounded result and the
//! rounding error, which — the fact everything else rests on — is itself an
//! `f64`.
//!
//! ## Notation
//! `F` is the set of `f64` values, `RN(·)` rounding to nearest (ties to
//! even), `u = 2⁻⁵³`, `η = 2⁻¹⁰⁷⁴` the smallest positive subnormal, `Ω` the
//! largest finite `f64`. For binary64 the precision is `p = 53` and the
//! minimal exponent `e_min = −1022`.
//!
//! ## Algorithms
//! **Algorithm S (TwoSum; Knuth, after Møller).** Input `a, b ∈ F`.
//!
//! S1. `s ← RN(a + b)`.
//! S2. `a′ ← RN(s − b)`; `b′ ← RN(s − a′)`.
//! S3. `δ_a ← RN(a − a′)`; `δ_b ← RN(b − b′)`.
//! S4. `e ← RN(δ_a + δ_b)`. Return `(s, e)`.
//!
//! Six operations, no branch, no condition on the operands.
//!
//! **Algorithm F (Fast2Sum; Dekker).** Input `a, b ∈ F` with `e_a ≥ e_b`
//! (the exponents; a zero has the least one), which holds in particular
//! when `|a| ≥ |b|` or `a = 0` — the form in which the requirement is
//! usually stated. The weaker form matters: under cancellation the
//! additions of Chapter 11.2 call F with `|a| < |b|` in the same binade
//! (`a = −2⁻⁵²`, `b ≈ −1.23 · 2⁻⁵²` in a test of this crate), where the
//! proofs of Joldes, Muller & Popescu rely on the exponent condition.
//!
//! F1. `s ← RN(a + b)`.
//! F2. `z ← RN(s − a)`.
//! F3. `e ← RN(b − z)`. Return `(s, e)`.
//!
//! **Algorithm P (2ProdFMA).** Input `a, b ∈ F`.
//!
//! P1. `p ← RN(a b)`.
//! P2. `e ← RN(a b − p)`, one fused multiply-add. Return `(p, e)`.
//!
//! ## Invariants
//! (T1) *Sums.* If `s` is finite, Algorithm F returns `s = RN(a + b)` and
//!      `s + e = a + b` exactly, and so does Algorithm S unless `|a| = Ω`
//!      — also when subnormal numbers occur: an `f64` addition whose
//!      result is subnormal is exact, so the rounding error of a sum is
//!      always representable. The exception: with `a = ±Ω` step S2 can
//!      overflow although S1 did not (`a′ = RN(a + δ)` for the rounding
//!      error `δ` of S1, which reaches `ulp(Ω)/2`: `a = Ω`, `b = −3 ·
//!      2⁹⁷⁰`), and `e` is then not finite. `DoubleDouble::from_sum` and
//!      the additions of Chapter 11.2 detect it (their step E).
//! (T2) *Products.* If `p` is finite and `a b = 0` or `|a b| ≥ 2⁻⁹⁶⁹`,
//!      Algorithm P returns `p = RN(a b)` and `p + e = a b` exactly. The
//!      exact condition is `e_a + e_b ≥ e_min + p − 1 = −970` on the
//!      exponents of `a` and `b` (the error `a b − p` is a multiple of
//!      `2^(e_a + e_b − 2p + 2)`, which must be at least `η`); `|a b| ≥
//!      2⁻⁹⁶⁹` implies it, and it can fail for any smaller bound. Otherwise
//!      `e` is the *rounded* error and `|a b − (p + e)| ≤ η/2`.
//! (T3) *Normalized.* In (T1), and in (T2) when the product is exact, `|e|
//!      ≤ ulp(s)/2` (resp. `ulp(p)/2`) and `RN(s + e) = s`: the pair is a
//!      normalized double-double (invariant D1 of Chapter 11.2). When the
//!      error of a product is *rounded* (below the range of (T2)) it may
//!      become exactly `±ulp(p)/2` with `p` odd, and `(p, e)` is then not
//!      normalized: `DoubleDouble::from_product` adds one Fast2Sum, which
//!      is exact there (T1).
//! (T4) *Non-finite.* If `s` (resp. `p`) is infinite or NaN, `e` is NaN or
//!      infinite and carries no information; the callers of this crate
//!      test the leading component (invariant D2).
//!
//! ## References
//! Møller, "Quasi double-precision in floating point addition", BIT 5
//! (1965); Knuth, TAOCP vol. 2, §4.2.2, Theorem B; Dekker, "A
//! floating-point technique for extending the available precision", Numer.
//! Math. 18 (1971); Ogita, Rump & Oishi, "Accurate sum and dot product",
//! SIAM J. Sci. Comput. 26 (2005), Theorem 3.4 and Algorithm 3.5
//! (TwoProductFMA); Joldes, Muller & Popescu, ACM TOMS 44 (2017),
//! Algorithms 1–3 (the condition of (T2) is that of their Fast2Mult);
//! Muller et al., *Handbook of Floating-Point Arithmetic*, 2nd ed. (2018),
//! §4.3–4.4.

/// `(s, e)` with `s = fl(a + b)` and `s + e = a + b` exactly (Knuth's
/// TwoSum: 6 flops, no branch, no condition on the operands). Exact unless
/// `s` overflows — or `a = ±Ω`, where an intermediate difference can
/// overflow and leave `e` non-finite (invariant T1 of Chapter 11.1).
#[inline]
#[must_use]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b; // S1
    let a1 = s - b; // S2
    let b1 = s - a1;
    let da = a - a1; // S3
    let db = b - b1;
    (s, da + db) // S4
}

/// The binary exponent field of `x`, subnormals and zeros sharing the
/// least one.
#[inline]
fn exponent_field(x: f64) -> u64 {
    ((x.to_bits() >> 52) & 0x7ff).max(1)
}

/// As [`two_sum`] in 3 flops, **requires** `|a| ≥ |b|` or `a = 0` (Dekker's
/// Fast2Sum); the requirement is a `debug_assert!`.
///
/// Precisely, the algorithm is exact whenever the exponent of `a` is at
/// least that of `b` (Algorithm F of Chapter 11.1), and that — the weaker
/// condition, which `|a| ≥ |b|` implies — is what the assertion tests.
#[inline]
#[must_use]
#[allow(clippy::float_cmp)] // the precondition is an exact comparison
pub fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    // A non-finite operand passes: those cases are the callers' business
    // (T4).
    debug_assert!(
        a == 0.0
            || b == 0.0
            || !a.is_finite()
            || !b.is_finite()
            || exponent_field(a) >= exponent_field(b),
        "fast_two_sum requires |a| >= |b| or a = 0 (a = {a:e}, b = {b:e})"
    );
    let s = a + b; // F1
    let z = s - a; // F2
    (s, b - z) // F3
}

/// `(p, e)` with `p = fl(a · b)` and `p + e = a · b` exactly, by one fused
/// multiply-add. Exact when `a b = 0` or `|a b| ≥ 2⁻⁹⁶⁹` and `p` does not
/// overflow; otherwise `|a b − (p + e)| ≤ η`.
#[inline]
#[must_use]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b; // P1
    (p, a.mul_add(b, -p)) // P2
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the tests compare exact values
mod tests {
    use super::*;

    const U: f64 = 1.110_223_024_625_156_5e-16; // 2^-53

    #[test]
    fn two_sum_recovers_the_rounding_error() {
        // 1 + 2^-53 rounds to 1 (tie to even); the error is 2^-53.
        assert_eq!(two_sum(1.0, U), (1.0, U));
        assert_eq!(two_sum(U, 1.0), (1.0, U));
        // No condition on the order or the signs.
        assert_eq!(two_sum(1e16, -1.0), (1e16, -1.0));
        assert_eq!(two_sum(3.0, 4.0), (7.0, 0.0));
        // Subnormal sums are exact (T1).
        let eta = f64::from_bits(1);
        assert_eq!(two_sum(eta, eta), (2.0 * eta, 0.0));
    }

    #[test]
    fn fast_two_sum_agrees_with_two_sum_when_ordered() {
        // The last pair has |a| < |b| in the same binade: still exact.
        for (a, b) in [
            (1.0, U),
            (1e16, -1.0),
            (-8.0, 0.1),
            (0.0, 5.0),
            (0.3, 0.3),
            (1.0, 1.75),
        ] {
            assert_eq!(fast_two_sum(a, b), two_sum(a, b));
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "fast_two_sum requires")]
    fn fast_two_sum_asserts_its_precondition() {
        let _ = fast_two_sum(1.0, 2.0);
    }

    #[test]
    fn two_prod_recovers_the_rounding_error() {
        // (1 + 2^-52)^2 = 1 + 2^-51 + 2^-104.
        let x = 1.0 + f64::EPSILON;
        let (p, e) = two_prod(x, x);
        assert_eq!(p, 1.0 + 2.0 * f64::EPSILON);
        assert_eq!(e, f64::EPSILON * f64::EPSILON);
        assert_eq!(two_prod(3.0, 4.0), (12.0, 0.0));
        assert_eq!(two_prod(0.0, 5.0), (0.0, 0.0));
    }

    #[test]
    fn two_prod_threshold_is_two_to_minus_969() {
        // a = b = (1 + 2^-52) 2^-485: e_a + e_b = -970, the last exact case;
        // a b ≈ 2^-970 (1 + 2^-51), below 2^-969 — the bound of the docs is
        // sufficient, not necessary.
        let x = 1.0 + f64::EPSILON;
        let a = x * 2f64.powi(-485);
        let (p, e) = two_prod(a, a);
        assert_eq!(p, (1.0 + 2.0 * f64::EPSILON) * 2f64.powi(-970));
        assert_eq!(e, f64::from_bits(1)); // 2^-104 · 2^-970 = η: exact
        // One exponent lower the error 2^-1075 is not representable.
        let b = x * 2f64.powi(-486);
        let (p, e) = two_prod(a, b);
        assert_eq!(p, (1.0 + 2.0 * f64::EPSILON) * 2f64.powi(-971));
        assert_eq!(e, 0.0); // η/2 rounds to even: |a b − (p + e)| = η/2
    }

    #[test]
    fn non_finite_sums_have_no_error_term() {
        let (s, e) = two_sum(f64::MAX, f64::MAX);
        assert!(s.is_infinite() && !e.is_finite());
        let (p, e) = two_prod(f64::MAX, 2.0);
        assert!(p.is_infinite() && !e.is_finite());
    }
}
