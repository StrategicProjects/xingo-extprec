//! The acceptance tests of L2.2a (PLAN.md; D-0109, E6): every operation of
//! `DoubleDouble` within its bound of `DoubleDouble::ERROR_BOUNDS`, checked
//! in exact rational arithmetic (`dashu`), the invariants (D1)–(D5) after
//! every operation, the error-free transformations exact, the conversions,
//! the order, the decimal input and output, and the kernels within the
//! bounds of Chapter 11.5.
//!
//! An `f64` is a dyadic rational, so `q(x)` below is exact and every
//! comparison of this file is a comparison of rationals: no tolerance
//! appears. The bound tested for an operation with first-order constant `c`
//! is `c u² (1 + 8u)`, which covers the full expressions of the chapter's
//! table (`2u²/(1 − 2u)`, `3u²/(1 − 4u)`, `15u² + 56u³`).

#![allow(clippy::disallowed_methods)] // L2.4: libm in tests and examples, never as a reference (clippy.toml)
#![allow(clippy::float_cmp)] // exact comparisons are the subject

use core::cmp::Ordering;

use dashu_int::{IBig, UBig};
use dashu_ratio::RBig;
use proptest::prelude::*;
use xingo_extprec::{DoubleDouble, fast_two_sum, kernels, two_prod, two_sum};

type Dd = DoubleDouble;

// ---------------------------------------------------------------------
// Exact arithmetic
// ---------------------------------------------------------------------

/// The value of a finite `f64`, exactly.
fn q(x: f64) -> RBig {
    RBig::try_from(x).expect("a finite f64")
}

/// The value of a finite double-double, exactly.
fn qd(x: Dd) -> RBig {
    q(x.hi()) + q(x.lo())
}

fn zero() -> RBig {
    RBig::from(0)
}

fn abs(x: RBig) -> RBig {
    if x < zero() { -x } else { x }
}

/// `2ᵏ`.
fn pow2(k: i32) -> RBig {
    let p = UBig::ONE << usize::try_from(k.unsigned_abs()).expect("fits usize");
    if k >= 0 {
        RBig::from(p)
    } else {
        RBig::from_parts(IBig::ONE, p)
    }
}

/// `10ᵏ`.
fn pow10(k: i32) -> RBig {
    let p = UBig::from(10_u8).pow(usize::try_from(k.unsigned_abs()).expect("fits usize"));
    if k >= 0 {
        RBig::from(p)
    } else {
        RBig::from_parts(IBig::ONE, p)
    }
}

/// The relative bound of `operation`: `c u² (1 + 8u)`.
fn bound(operation: &str) -> RBig {
    let c = Dd::ERROR_BOUNDS
        .iter()
        .find(|b| b.operation == operation)
        .unwrap_or_else(|| panic!("no bound for {operation:?}"))
        .units_of_u2;
    q(c) * pow2(-106) * (RBig::from(1) + pow2(-50))
}

/// `|z − exact| ≤ bound · |exact|`.
fn within(z: Dd, exact: &RBig, operation: &str) -> bool {
    abs(qd(z) - exact) <= bound(operation) * abs(exact.clone())
}

/// `|z − exact| / (u² |exact|)` as an `f64`, for the survey.
fn error_in_u2(z: Dd, exact: &RBig) -> f64 {
    if *exact == zero() {
        return 0.0;
    }
    (abs(qd(z) - exact) / abs(exact.clone()) * pow2(106))
        .to_f64()
        .value()
}

/// `γ_k = k u / (1 − k u)`.
fn gamma(k: usize) -> RBig {
    let ku = RBig::from(k) * pow2(-53);
    ku.clone() / (RBig::from(1) - ku)
}

/// The value of a decimal string `[-]digits[.digits][e[-]k]`, exactly, and
/// the exponent of its last digit.
fn decimal(s: &str) -> (RBig, i32) {
    let (mantissa, exponent) = match s.split_once('e') {
        Some((m, e)) => (m, e.parse::<i32>().expect("an exponent")),
        None => (s, 0),
    };
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mantissa),
    };
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits: String = [int, frac].concat();
    let n = UBig::from_str_radix(&digits, 10).expect("decimal digits");
    let last = exponent - i32::try_from(frac.len()).expect("short");
    let v = RBig::from(n) * pow10(last);
    (if negative { -v } else { v }, last)
}

// ---------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------

const FRACTION: u64 = (1 << 52) - 1;

/// Significands that exercise the rounding: random, powers of two, all
/// ones, a single bit away from either, short.
fn fraction() -> impl Strategy<Value = u64> {
    prop_oneof![
        6 => any::<u64>().prop_map(|b| b & FRACTION),
        1 => Just(0_u64),
        1 => Just(FRACTION),
        1 => (0_u32..52).prop_map(|k| 1 << k),
        1 => (0_u32..52).prop_map(|k| FRACTION ^ (1 << k)),
        1 => any::<u64>().prop_map(|b| (b & FRACTION) >> 40 << 40),
        1 => any::<u64>().prop_map(|b| (b & FRACTION) >> 40),
    ]
}

/// A normal `f64` with exponent in `emin..=emax`, either sign.
fn normal(emin: i32, emax: i32) -> impl Strategy<Value = f64> {
    (any::<bool>(), fraction(), emin..=emax).prop_map(|(neg, frac, e)| {
        let biased = u64::try_from(e + 1023).expect("a normal exponent");
        f64::from_bits((u64::from(neg) << 63) | (biased << 52) | frac)
    })
}

/// The safe range of Chapter 11.2 (well inside it), zero included.
fn safe_f64() -> impl Strategy<Value = f64> {
    prop_oneof![12 => normal(-300, 300), 1 => Just(0.0)]
}

/// A normalized double-double with leading component `hi`: the trailing
/// one is a random fraction of `ulp(hi)/2`, possibly far below or zero.
fn with_trailing(hi: impl Strategy<Value = f64>) -> impl Strategy<Value = Dd> {
    let r = prop_oneof![
        6 => -1.0..1.0_f64,
        1 => Just(1.0),
        1 => Just(-1.0),
        1 => Just(0.0),
    ];
    let gap = prop_oneof![6 => Just(0_i32), 2 => 0..70_i32];
    (hi, r, gap)
        .prop_map(|(hi, r, gap)| Dd::from_sum(hi, hi * r * 2f64.powi(-53) * 2f64.powi(-gap)))
}

fn safe_dd() -> impl Strategy<Value = Dd> {
    with_trailing(safe_f64())
}

/// A pair whose sum cancels heavily: `y ≈ −x` to a few ulps.
fn cancelling() -> impl Strategy<Value = (Dd, Dd)> {
    (safe_dd(), -4_i32..=4, -1.0..1.0_f64, 0..3_i32).prop_map(|(x, k, r, gap)| {
        let hi = -x.hi() * (1.0 + f64::from(k) * f64::EPSILON);
        let y = Dd::from_sum(hi, hi * r * 2f64.powi(-53) * 2f64.powi(-40 * gap));
        (x, y)
    })
}

/// Any finite `f64`: zeros, subnormals, the largest values.
fn finite_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        8 => any::<u64>().prop_map(f64::from_bits).prop_filter("finite", |x| x.is_finite()),
        2 => normal(-1022, -960),
        2 => normal(960, 1023),
        1 => any::<u64>().prop_map(|b| f64::from_bits(b & FRACTION)), // subnormal
        1 => prop_oneof![Just(0.0), Just(-0.0), Just(f64::MAX), Just(f64::MIN_POSITIVE)],
    ]
}

/// Any `f64` at all.
fn any_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        6 => finite_f64(),
        1 => prop_oneof![Just(f64::INFINITY), Just(f64::NEG_INFINITY), Just(f64::NAN)],
    ]
}

/// Any double-double value, non-finite ones included.
fn any_dd() -> impl Strategy<Value = Dd> {
    prop_oneof![
        8 => with_trailing(finite_f64()),
        1 => prop_oneof![Just(Dd::INFINITY), Just(-Dd::INFINITY), Just(Dd::NAN)],
    ]
}

/// Any finite double-double.
fn finite_dd() -> impl Strategy<Value = Dd> {
    with_trailing(finite_f64()).prop_filter("finite", |x| x.is_finite())
}

/// No regression files: the strategies are small enough to shrink to a
/// case worth a unit test, which is where a failure should be recorded.
fn config(cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

fn invariants(z: Dd) -> Result<(), TestCaseError> {
    prop_assert!(
        z.check_invariants().is_ok(),
        "{z:?}: {:?}",
        z.check_invariants()
    );
    Ok(())
}

/// `hi` is an `f64` nearest to `v` (invariant D4).
fn is_nearest(hi: f64, v: &RBig) -> bool {
    let d = abs(v.clone() - q(hi));
    let up = hi.next_up();
    let down = hi.next_down();
    (!up.is_finite() || d <= abs(v.clone() - q(up)))
        && (!down.is_finite() || d <= abs(v.clone() - q(down)))
}

// ---------------------------------------------------------------------
// Error-free transformations
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(config(12_000))]

    #[test]
    fn two_sum_is_exact(a in finite_f64(), b in finite_f64()) {
        let (s, e) = two_sum(a, b);
        prop_assume!(s.is_finite());
        // T1: e is finite unless a = ±Ω.
        prop_assert!(e.is_finite() || a.abs() == f64::MAX, "two_sum({a:e}, {b:e})");
        invariants(Dd::from_sum(a, b))?;
        prop_assume!(e.is_finite());
        prop_assert_eq!(s.to_bits(), (a + b).to_bits());
        prop_assert_eq!(q(s) + q(e), q(a) + q(b));
        invariants(Dd::from_sum(a, b))?;
    }

    #[test]
    fn fast_two_sum_is_exact(a in finite_f64(), b in finite_f64()) {
        let (a, b) = if a.abs() >= b.abs() { (a, b) } else { (b, a) };
        let (s, e) = fast_two_sum(a, b);
        prop_assume!(s.is_finite());
        prop_assert_eq!(s.to_bits(), two_sum(a, b).0.to_bits());
        prop_assert_eq!(q(s) + q(e), q(a) + q(b));
    }

    #[test]
    fn two_prod_is_exact_in_its_range(a in normal(-480, 480), b in normal(-480, 480)) {
        let (p, e) = two_prod(a, b);
        prop_assert_eq!(p.to_bits(), (a * b).to_bits());
        prop_assert_eq!(q(p) + q(e), q(a) * q(b));
        invariants(Dd::from_product(a, b))?;
    }

    /// At the threshold: exact from `|a b| ≥ 2⁻⁹⁶⁹`, within `η/2` below.
    #[test]
    fn two_prod_at_the_underflow_threshold(a in normal(-520, -450), b in normal(-520, -450)) {
        let (p, e) = two_prod(a, b);
        let exact = q(a) * q(b);
        let defect = abs(q(p) + q(e) - exact.clone());
        if abs(exact) >= pow2(-969) {
            prop_assert_eq!(defect, zero());
        } else {
            prop_assert!(defect <= pow2(-1075));
        }
        invariants(Dd::from_product(a, b))?;
    }
}

// ---------------------------------------------------------------------
// The operations within their bounds
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(config(12_000))]

    #[test]
    fn add_dd_within_bound(x in safe_dd(), y in safe_dd()) {
        let exact = qd(x) + qd(y);
        let z = x + y;
        invariants(z)?;
        prop_assert!(within(z, &exact, "dd + dd"), "{x:?} + {y:?} = {z:?}");
        let w = x - y;
        invariants(w)?;
        prop_assert!(within(w, &(qd(x) - qd(y)), "dd + dd"), "{x:?} - {y:?} = {w:?}");
        let mut a = x;
        a += y;
        prop_assert_eq!(a, z);
        a -= y;
        prop_assert_eq!(a, z - y);
    }

    #[test]
    fn add_dd_within_bound_under_cancellation((x, y) in cancelling()) {
        let exact = qd(x) + qd(y);
        let z = x + y;
        invariants(z)?;
        prop_assert!(within(z, &exact, "dd + dd"), "{x:?} + {y:?} = {z:?}");
        let z = x + y.hi();
        invariants(z)?;
        prop_assert!(within(z, &(qd(x) + q(y.hi())), "dd + f64"), "{x:?} + {:?} = {z:?}", y.hi());
    }

    /// The additions keep their bound over the whole finite range, the
    /// subnormal one included (Chapter 11.2, "Bounds, and where they hold").
    #[test]
    fn add_dd_within_bound_everywhere(
        x in finite_dd(),
        y in finite_dd(),
    ) {
        prop_assume!(x.hi().abs() < 2f64.powi(1021) && y.hi().abs() < 2f64.powi(1021));
        let z = x + y;
        invariants(z)?;
        prop_assert!(within(z, &(qd(x) + qd(y)), "dd + dd"), "{x:?} + {y:?} = {z:?}");
        let z = x + y.hi();
        invariants(z)?;
        prop_assert!(within(z, &(qd(x) + q(y.hi())), "dd + f64"), "{x:?} + {:?} = {z:?}", y.hi());
    }

    #[test]
    fn add_f64_within_bound(x in safe_dd(), y in safe_f64()) {
        let z = x + y;
        invariants(z)?;
        prop_assert!(within(z, &(qd(x) + q(y)), "dd + f64"), "{x:?} + {y:?} = {z:?}");
        let w = x - y;
        invariants(w)?;
        prop_assert!(within(w, &(qd(x) - q(y)), "dd + f64"), "{x:?} - {y:?} = {w:?}");
    }

    #[test]
    fn mul_dd_within_bound(x in safe_dd(), y in safe_dd()) {
        let z = x * y;
        invariants(z)?;
        prop_assert!(within(z, &(qd(x) * qd(y)), "dd · dd"), "{x:?} * {y:?} = {z:?}");
        let mut a = x;
        a *= y;
        prop_assert_eq!(a, z);
    }

    #[test]
    fn mul_f64_within_bound(x in safe_dd(), y in safe_f64()) {
        let z = x * y;
        invariants(z)?;
        prop_assert!(within(z, &(qd(x) * q(y)), "dd · f64"), "{x:?} * {y:?} = {z:?}");
    }

    #[test]
    fn square_within_bound(x in safe_dd()) {
        let z = x.square();
        invariants(z)?;
        prop_assert!(within(z, &(qd(x) * qd(x)), "dd²"), "{x:?}² = {z:?}");
    }

    #[test]
    fn div_dd_within_bound(x in safe_dd(), y in with_trailing(normal(-300, 300))) {
        let z = x / y;
        invariants(z)?;
        prop_assert!(within(z, &(qd(x) / qd(y)), "dd / dd"), "{x:?} / {y:?} = {z:?}");
        let mut a = x;
        a /= y;
        prop_assert_eq!(a, z);
        let r = y.recip();
        invariants(r)?;
        prop_assert!(within(r, &(RBig::from(1) / qd(y)), "dd / dd"), "1 / {y:?} = {r:?}");
    }

    /// Exact quotients stay exact (Chapter 11.2, after Algorithm D2):
    /// `(a b)/b = a` for the exact product of two `f64`, and `x/x = 1`.
    #[test]
    fn div_dd_keeps_exact_quotients(a in normal(-300, 300), b in normal(-300, 300), x in safe_dd()) {
        let z = Dd::from_product(a, b) / Dd::from_f64(b);
        prop_assert_eq!(z, Dd::from_f64(a), "({:e} · {:e}) / second", a, b);
        let z = Dd::from_product(a, b) / Dd::from_f64(a);
        prop_assert_eq!(z, Dd::from_f64(b), "({:e} · {:e}) / first", a, b);
        if x.hi() != 0.0 {
            prop_assert_eq!(x / x, Dd::ONE, "{:?} / itself", x);
            prop_assert_eq!(x / -x, -Dd::ONE, "{:?} / its negation", x);
        }
    }

    #[test]
    fn div_f64_within_bound(x in safe_dd(), y in normal(-300, 300)) {
        let z = x / y;
        invariants(z)?;
        prop_assert!(within(z, &(qd(x) / q(y)), "dd / f64"), "{x:?} / {y:?} = {z:?}");
    }

    /// `(1 − ε)² x ≤ z² ≤ (1 + ε)² x` is `|z − √x| ≤ ε √x` without the
    /// irrational.
    #[test]
    fn sqrt_within_bound(x in safe_dd()) {
        let x = x.abs();
        let z = x.sqrt();
        invariants(z)?;
        let eps = bound("√dd");
        let one = RBig::from(1);
        let (lo, hi) = (one.clone() - eps.clone(), one + eps);
        let z2 = qd(z) * qd(z);
        prop_assert!(z.hi() >= 0.0);
        prop_assert!(lo.clone() * lo * qd(x) <= z2, "√{x:?} = {z:?} is too small");
        prop_assert!(z2 <= hi.clone() * hi * qd(x), "√{x:?} = {z:?} is too large");
    }

    #[test]
    fn ldexp_is_exact(x in safe_dd(), k in -400_i32..=400) {
        let z = x.ldexp(k);
        invariants(z)?;
        prop_assert_eq!(qd(z), qd(x) * pow2(k));
    }
}

// ---------------------------------------------------------------------
// Invariants, order and conversions
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(config(20_000))]

    /// (D1), (D2): whatever the operands — zeros, subnormals, the largest
    /// values, infinities, NaN — every operation returns a well-formed
    /// value (and no `debug_assert!` of the algorithms fires).
    #[test]
    fn invariants_hold_for_any_operands(x in any_dd(), y in any_dd(), k in -2300_i32..2300) {
        for z in [
            x + y, x - y, x * y, x / y,
            x + y.hi(), x - y.hi(), x * y.hi(), x / y.hi(),
            x.sqrt(), x.abs().sqrt(), x.square(), x.recip(), x.abs(), -x,
            x.ldexp(k), x.max(y), x.min(y),
            Dd::from_sum(x.hi(), y.hi()), Dd::from_product(x.hi(), y.hi()),
            [x, y, x].into_iter().sum(),
        ] {
            invariants(z)?;
        }
    }

    /// (D2): the leading component of a non-finite result is the IEEE one.
    #[test]
    fn non_finite_results_follow_ieee(x in any_f64(), y in any_f64()) {
        let (dx, dy) = (Dd::from_f64(x), Dd::from_f64(y));
        let same = |z: Dd, g: f64| {
            g.is_finite() || (z.hi().to_bits() == g.to_bits() || (z.is_nan() && g.is_nan()))
        };
        prop_assert!(same(dx + dy, x + y));
        prop_assert!(same(dx - dy, x - y));
        prop_assert!(same(dx * dy, x * y));
        prop_assert!(same(dx / dy, x / y));
        prop_assert!(same(dx + y, x + y));
        prop_assert!(same(dx * y, x * y));
        prop_assert!(same(dx / y, x / y));
        prop_assert!(same(dx.sqrt(), x.sqrt()));
        // And a finite f64 result is never lost to a NaN.
        if (x / y).is_finite() {
            prop_assert!((dx / dy).is_finite() && (dx / y).is_finite());
        }
    }

    /// (D3): the derived order is the order of the values.
    #[test]
    fn order_is_the_order_of_the_values(x in safe_dd(), y in safe_dd(), near in any::<bool>()) {
        // Half the time compare neighbours: same leading component.
        let y = if near { Dd::from_sum(x.hi(), y.lo() * (x.hi() / y.hi()).abs()) } else { y };
        prop_assume!(y.is_finite());
        let exact = qd(x).cmp(&qd(y));
        prop_assert_eq!(x.partial_cmp(&y), Some(exact));
        prop_assert_eq!(x == y, exact == Ordering::Equal);
        prop_assert_eq!(x.max(y), if exact == Ordering::Less { y } else { x });
        prop_assert_eq!(x.min(y), if exact == Ordering::Greater { y } else { x });
        prop_assert_eq!(qd(x.abs()), abs(qd(x)));
        prop_assert_eq!(qd(-x), -qd(x));
    }

    /// (D4), requirement (R4) of `Real`: `from_f64` is exact and `to_f64`
    /// is the nearest `f64`.
    #[test]
    fn conversions(a in finite_f64(), x in safe_dd(), y in safe_dd()) {
        let d = Dd::from_f64(a);
        prop_assert_eq!(d.to_f64().to_bits(), a.to_bits());
        prop_assert_eq!(d.lo(), 0.0);
        prop_assert_eq!(Dd::from(a), d);
        if a != 0.0 {
            prop_assert_eq!(qd(d), q(a));
        }
        for z in [x, x + y, x * y, x / y.abs().max(Dd::ONE), x.abs().sqrt()] {
            prop_assert!(is_nearest(z.to_f64(), &qd(z)), "{z:?}");
            prop_assert_eq!(z.to_f64().to_bits(), z.hi().to_bits());
            prop_assert_eq!(Dd::from_parts(z.hi(), z.lo()), z);
        }
    }

    /// (D5), requirement (R3): the sum is the left-to-right one from zero.
    #[test]
    fn sum_adds_left_to_right(v in prop::collection::vec(safe_dd(), 0..12)) {
        let s: Dd = v.iter().copied().sum();
        let mut t = Dd::ZERO;
        for &x in &v {
            t += x;
        }
        prop_assert_eq!(s, t);
    }
}

// ---------------------------------------------------------------------
// Decimal input and output
// ---------------------------------------------------------------------

/// A decimal literal with up to 45 digits and an exponent that spans the
/// whole range, the overflow and underflow thresholds included.
fn literal() -> impl Strategy<Value = String> {
    (
        any::<bool>(),
        prop::collection::vec(0_u8..10, 1..45),
        0_usize..46,
        -345_i32..310,
    )
        .prop_map(|(neg, digits, point, e)| {
            let mut s = String::new();
            if neg {
                s.push('-');
            }
            for (i, d) in digits.iter().enumerate() {
                if i == point.min(digits.len() - 1) && i > 0 {
                    s.push('.');
                }
                s.push(char::from(b'0' + d));
            }
            s.push('e');
            s.push_str(&e.to_string());
            s
        })
}

proptest! {
    #![proptest_config(config(3_000))]

    /// (P2): the leading component is the `f64` nearest to the string's
    /// value, the trailing one the `f64` nearest to the rest.
    #[test]
    fn parse_is_correctly_rounded(s in literal()) {
        let z: Dd = s.parse().expect("a literal");
        invariants(z)?;
        let (v, _) = decimal(&s);
        if z.is_finite() {
            let h = z.hi();
            if is_nearest(h, &v) {
                prop_assert!(is_nearest(z.lo(), &(v.clone() - q(h))), "{s} -> {z:?}");
            } else {
                // The tie of invariant P2: a subnormal trailing component
                // of exactly half an ulp, the value within η/2 of v.
                prop_assert!(z.lo().abs() < f64::MIN_POSITIVE, "{s} -> {z:?}");
                prop_assert_eq!(q(z.lo().abs()) * RBig::from(2), q(h.next_up()) - q(h), "{}", s);
                prop_assert!(abs(qd(z) - v.clone()) <= pow2(-1075), "{s} -> {z:?}");
            }
            if v != zero() {
                prop_assert_eq!(h.is_sign_negative(), v < zero());
            }
        } else {
            // Overflow: beyond the rounding boundary Ω + ulp(Ω)/2.
            prop_assert!(abs(v) >= q(f64::MAX) + pow2(970), "{s} -> {z:?}");
        }
    }

    /// (P1), (P3): with enough digits the output is the value, and parses
    /// back to the same bits.
    #[test]
    fn exact_output_round_trips(x in finite_dd()) {
        let s = format!("{x:.1100}");
        let (v, _) = decimal(&s);
        if x.hi() != 0.0 {
            prop_assert_eq!(&v, &qd(x), "{}", s);
        }
        let back: Dd = s.parse().expect("its own output");
        prop_assert_eq!((back.hi().to_bits(), back.lo().to_bits() << 1), (x.hi().to_bits(), x.lo().to_bits() << 1), "{}", s);
    }

    /// (P1): the default output of `{}` and of `{:e}` is the value
    /// correctly rounded to at most 31 significant digits.
    #[test]
    fn default_output_has_31_correct_digits(x in finite_dd()) {
        prop_assume!(x.hi() != 0.0);
        let exact = qd(x);
        // Half a unit of the 31st significant digit of x.
        let half_unit = abs(exact.clone()) * pow10(-30) / RBig::from(2);
        for s in [x.to_string(), format!("{x:e}")] {
            let digits = s.split('e').next().expect("a mantissa").replace(['-', '.'], "");
            let significant = digits.trim_start_matches('0').trim_end_matches('0');
            prop_assert!(significant.len() <= 31, "{}", s);
            prop_assert!(!s.contains('e') || !digits.ends_with('0') || digits.len() == 1, "{}", s);
            let (v, _) = decimal(&s);
            prop_assert!(abs(v.clone() - exact.clone()) <= half_unit, "{}", s);
            // Short values come back exactly (P3).
            let back: Dd = s.parse().expect("its own output");
            invariants(back)?;
            if v == exact {
                prop_assert_eq!(back, x, "{}", s);
            }
        }
        // `{:e}` is always scientific, with the digits of `{}`.
        let (a, b) = (decimal(&x.to_string()).0, decimal(&format!("{x:e}")).0);
        prop_assert_eq!(a, b);
        let scientific = format!("{x:e}");
        prop_assert!(scientific.contains('e'), "{}", scientific);
    }

    /// (P3), text: a decimal of at most 31 digits, read and written back
    /// with `{}` or `{:e}`, is the same decimal — and the output is a fixed
    /// point of read-then-write.
    #[test]
    fn decimals_of_31_digits_round_trip_as_text(
        digits in prop::collection::vec(0_u8..10, 1..=31),
        e in -250_i32..250,
        neg in any::<bool>(),
    ) {
        let body: String = digits.iter().map(|d| char::from(b'0' + d)).collect();
        let literal = format!("{}{body}e{e}", if neg { "-" } else { "" });
        let (want, _) = decimal(&literal);
        let x: Dd = literal.parse().expect("a literal");
        for s in [x.to_string(), format!("{x:e}")] {
            prop_assert_eq!(&decimal(&s).0, &want, "{} -> {}", literal, s);
            let again: Dd = s.parse().expect("its own output");
            prop_assert_eq!(again, x, "{} -> {}", literal, s);
        }
        prop_assert_eq!(format!("{:e}", format!("{x:e}").parse::<Dd>().expect("its own output")), format!("{x:e}"));
        prop_assert_eq!(x.to_string().parse::<Dd>().expect("its own output").to_string(), x.to_string());
    }

    /// `{:.Ne}`: `N` digits after the point, half a unit of the last one;
    /// with enough digits the output is the value and parses back to it.
    #[test]
    fn lower_exp_with_precision_is_correctly_rounded(x in finite_dd(), n in 0_usize..60) {
        let s = format!("{x:.n$e}");
        let (mantissa, _) = s.split_once('e').expect("scientific");
        prop_assert_eq!(mantissa.split_once('.').map_or(0, |p| p.1.len()), n, "{}", s);
        prop_assert_eq!(mantissa.trim_start_matches('-').split('.').next().map(str::len), Some(1), "{}", s);
        let (v, last) = decimal(&s);
        prop_assert!(abs(v - qd_or_zero(x)) * RBig::from(2) <= pow10(last), "{}", s);
        let exact = format!("{x:.1100e}");
        let back: Dd = exact.parse().expect("its own output");
        prop_assert_eq!((back.hi().to_bits(), back.lo().to_bits() << 1), (x.hi().to_bits(), x.lo().to_bits() << 1), "{}", exact);
    }

    /// (P1) with a precision: half a unit of the last digit written.
    #[test]
    fn precision_output_is_correctly_rounded(x in finite_dd(), n in 0_usize..60) {
        let s = format!("{x:.n$}");
        let (v, last) = decimal(&s);
        prop_assert!(abs(v - qd_or_zero(x)) * RBig::from(2) <= pow10(last), "{}", s);
        let frac = s.split('e').next().expect("a mantissa").split_once('.').map_or(0, |p| p.1.len());
        prop_assert_eq!(frac, n, "{}", s);
    }

    /// (P3): a value with at most 31 significant decimal digits (here a
    /// small dyadic rational) is written exactly and read back exactly.
    #[test]
    fn short_values_round_trip(k in 1_u32..65_536, j in -20_i32..=60, neg in any::<bool>()) {
        let x = Dd::from_f64(f64::from(k)).ldexp(j);
        let x = if neg { -x } else { x };
        let s = x.to_string();
        prop_assert_eq!(decimal(&s).0, qd(x), "{}", s);
        prop_assert_eq!(s.parse::<Dd>().expect("its own output"), x, "{}", s);
    }
}

/// `qd` that accepts the zeros (`dashu` rejects nothing here, but `-0.0`
/// and `0.0` are the same rational).
fn qd_or_zero(x: Dd) -> RBig {
    if x.hi() == 0.0 { zero() } else { qd(x) }
}

// ---------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------

/// `f64` data with a wide spread of magnitudes, and its negation appended
/// half the time: sums and dot products that cancel.
fn data(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    (
        prop::collection::vec(normal(-60, 60), 0..max_len),
        any::<bool>(),
        any::<u64>(),
    )
        .prop_map(|(mut v, cancel, salt)| {
            if cancel {
                let n = v.len();
                for i in 0..n {
                    // A negated copy, rotated, some entries off by an ulp.
                    let x = -v[(i + 1) % n];
                    let bump = (salt >> (i % 60)) & 1 == 1;
                    v.push(if bump { x * (1.0 + f64::EPSILON) } else { x });
                }
            }
            v
        })
}

/// Values whose products underflow, overflow or do neither.
fn extreme_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        2 => finite_f64(),
        3 => normal(-540, -430),
        1 => normal(-1022, -1000),
        1 => normal(480, 540),
        1 => normal(-10, 10),
    ]
}

fn dd_data(len: usize) -> impl Strategy<Value = Vec<Dd>> {
    prop::collection::vec(with_trailing(normal(-40, 40)), len)
}

proptest! {
    #![proptest_config(config(3_000))]

    #[test]
    fn sum2_within_bound(x in data(40)) {
        let r = kernels::sum2(&x);
        invariants(r)?;
        let exact = x.iter().fold(zero(), |acc, &v| acc + q(v));
        let mass = x.iter().fold(zero(), |acc, &v| acc + abs(q(v)));
        let g = gamma(x.len() + 3);
        prop_assert!(abs(qd_or_zero(r) - exact) <= g.clone() * g * mass, "{x:?} -> {r:?}");
        // Bit for bit the definition (step C4 keeps the unrounded pair).
        prop_assert_eq!(r, kernels::sum2(&x));
    }

    #[test]
    fn dot2_within_bound_and_dot2_err_is_rigorous(x in data(30), salt in any::<u64>()) {
        // y: x rotated and rescaled, so that the products cancel too.
        let n = x.len();
        let y: Vec<f64> = (0..n)
            .map(|i| if (salt >> (i % 64)) & 1 == 1 { 1.0 / x[i] } else { x[(i + n / 2) % n] })
            .collect();
        let exact = x.iter().zip(&y).fold(zero(), |acc, (&a, &b)| acc + q(a) * q(b));
        let mass = x.iter().zip(&y).fold(zero(), |acc, (&a, &b)| acc + abs(q(a) * q(b)));
        let r = kernels::dot2(&x, &y);
        invariants(r)?;
        let err = abs(qd_or_zero(r) - exact);
        let g = gamma(n + 5);
        prop_assert!(err <= g.clone() * g * mass, "{x:?} · {y:?} -> {r:?}");
        let (r2, beta) = kernels::dot2_err(&x, &y);
        prop_assert_eq!(r2, r);
        prop_assert!(beta.is_finite() && beta >= 0.0);
        prop_assert!(err <= q(beta), "{x:?} · {y:?}: error above β = {beta:e}");
    }

    /// (K2) where products underflow or overflow: the bound still holds,
    /// or is infinite.
    #[test]
    fn dot2_err_is_rigorous_at_the_ends_of_the_range(
        x in prop::collection::vec(extreme_f64(), 0..12),
        y in prop::collection::vec(extreme_f64(), 12),
    ) {
        let y = &y[..x.len()];
        let (r, beta) = kernels::dot2_err(&x, y);
        invariants(r)?;
        prop_assert!(!beta.is_nan() && beta >= 0.0);
        if beta.is_finite() {
            prop_assert!(r.is_finite());
            let exact = x.iter().zip(y).fold(zero(), |acc, (&a, &b)| acc + q(a) * q(b));
            prop_assert!(abs(qd_or_zero(r) - exact) <= q(beta), "{x:?} · {y:?}: β = {beta:e}");
        }
    }

    /// Step B1 and its accuracy: each product within `4u²`, each of the at
    /// most `n/4 + 3` additions on a path within `3u² + 13u³`.
    #[test]
    fn dd_dot_kernels((x, y) in (0_usize..24).prop_flat_map(|n| (dd_data(n), dd_data(n)))) {
        let n = x.len();
        let yf: Vec<f64> = y.iter().map(|v| v.hi()).collect();
        let slack = RBig::from(4 * (n + 4)) * pow2(-106);
        let r = kernels::dot(&x, &y);
        invariants(r)?;
        let exact = x.iter().zip(&y).fold(zero(), |acc, (&a, &b)| acc + qd(a) * qd(b));
        let mass = x.iter().zip(&y).fold(zero(), |acc, (&a, &b)| acc + abs(qd(a) * qd(b)));
        prop_assert!(abs(qd_or_zero(r) - exact) <= slack.clone() * mass);
        let r = kernels::dot_mixed(&x, &yf);
        invariants(r)?;
        let exact = x.iter().zip(&yf).fold(zero(), |acc, (&a, &b)| acc + qd(a) * q(b));
        let mass = x.iter().zip(&yf).fold(zero(), |acc, (&a, &b)| acc + abs(qd(a) * q(b)));
        prop_assert!(abs(qd_or_zero(r) - exact) <= slack * mass);
        // The order of accumulation is the definition (K1).
        let mut lanes = [Dd::ZERO; 4];
        for i in 0..n {
            lanes[i % 4] += x[i] * y[i];
        }
        prop_assert_eq!(kernels::dot(&x, &y), ((lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
    }

    #[test]
    fn dd_axpy_and_gemm(
        alpha in with_trailing(normal(-40, 40)),
        (m, k, n, a, b, c) in (0_usize..5, 0_usize..5, 0_usize..5).prop_flat_map(|(m, k, n)| {
            (Just(m), Just(k), Just(n), dd_data(m * k), dd_data(k * n), dd_data(m * n))
        }),
    ) {
        // axpy on the columns of A against C's first entries.
        let len = a.len().min(c.len());
        let mut y = c[..len].to_vec();
        kernels::axpy(alpha, &a[..len], &mut y);
        for i in 0..len {
            prop_assert_eq!(y[i], c[i] + alpha * a[i]);
            let exact = qd(c[i]) + qd(alpha) * qd(a[i]);
            let mass = abs(qd(c[i])) + abs(qd(alpha) * qd(a[i]));
            prop_assert!(abs(qd_or_zero(y[i]) - exact) <= RBig::from(8) * pow2(-106) * mass);
        }
        // gemm_sub against the exact C − A B.
        let mut out = c.clone();
        kernels::gemm_sub(m, k, n, &a, &b, &mut out);
        for j in 0..n {
            for i in 0..m {
                let mut exact = qd(c[i + j * m]);
                let mut mass = abs(exact.clone());
                let mut want = c[i + j * m];
                for l in 0..k {
                    let t = qd(a[i + l * m]) * qd(b[l + j * k]);
                    mass += abs(t.clone());
                    exact -= t;
                    want -= a[i + l * m] * b[l + j * k];
                }
                let z = out[i + j * m];
                invariants(z)?;
                prop_assert_eq!(z, want);
                let slack = RBig::from(8 * (k + 1)) * pow2(-106);
                prop_assert!(abs(qd_or_zero(z) - exact) <= slack * mass);
            }
        }
    }
}

/// The example with which Joldes, Muller & Popescu show that the bound of
/// DWPlusFP is asymptotically optimal: `x = (1, (2ᵖ − 1) 2⁻²ᵖ)`, `y = −(1 −
/// 2⁻ᵖ)/2`; the error is `2u² − 6u³ + …`, just under the bound.
#[test]
fn dw_plus_fp_worst_case_of_the_literature() {
    let u = 2f64.powi(-53);
    let x = Dd::from_parts(1.0, (2f64.powi(53) - 1.0) * u * u);
    let y = -0.5 * (1.0 - u);
    let z = x + y;
    let exact = qd(x) + q(y);
    let err = error_in_u2(z, &exact);
    assert!(err > 1.999 && err < 2.0, "{err}");
    assert!(within(z, &exact, "dd + f64"));
}

/// The exact integer quotients `(q y)/y = q` on a grid: the property that
/// decided for DWDivDW2 against a division through a reciprocal (Chapter
/// 11, "Why not DWDivDW3").
#[test]
fn exact_integer_quotients() {
    for q in 1..200_u32 {
        for y in 1..200_u32 {
            let (fq, fy) = (f64::from(q), f64::from(y));
            assert_eq!(
                Dd::from_f64(fq * fy) / Dd::from_f64(fy),
                Dd::from_f64(fq),
                "{q} · {y} / {y}"
            );
            assert_eq!(
                Dd::from_f64(-fq * fy) / Dd::from_f64(fy),
                Dd::from_f64(-fq),
                "−{q} · {y} / {y}"
            );
        }
    }
}

// ---------------------------------------------------------------------
// Survey
// ---------------------------------------------------------------------

/// The largest relative errors observed on a fixed pseudo-random sample, in
/// units of `u²`, printed with `--nocapture`: a report, and a second check
/// of the bounds (the square root's constant could not be checked against
/// its source — Chapter 11).
#[test]
#[allow(clippy::cast_precision_loss)] // 53 random bits
fn survey_of_observed_errors() {
    struct Lcg(u64);
    impl Lcg {
        fn bits(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 11
        }
        /// A double-double in `[1, 2)` or `[2, 4)`: the relative error of
        /// the operations does not depend on the binade beyond its parity
        /// (for the square root).
        fn dd(&mut self) -> Dd {
            let hi = f64::from_bits(((1023 + (self.bits() & 1)) << 52) | (self.bits() & FRACTION));
            let r = (self.bits() as f64) / (1_u64 << 52) as f64 - 1.0;
            let sign = if self.bits() & 1 == 1 { -1.0 } else { 1.0 };
            Dd::from_sum(sign * hi, hi * r * 2f64.powi(-53))
        }
    }
    let mut rng = Lcg(2026);
    let mut worst = [0.0_f64; 8];
    let names = [
        "dd + f64",
        "dd + dd",
        "dd · f64",
        "dd · dd",
        "dd²",
        "dd / f64",
        "dd / dd",
        "√dd",
    ];
    for _ in 0..20_000 {
        let (x, y) = (rng.dd(), rng.dd());
        let (ex, ey) = (qd(x), qd(y));
        let root = x.abs().sqrt();
        // |z − √x| / √x = |z² − x| / (√x (z + √x)) ≈ |z² − x| / (2x).
        let root_err = (abs(qd(root) * qd(root) - abs(ex.clone())) / abs(ex.clone()) * pow2(105))
            .to_f64()
            .value();
        let errs = [
            error_in_u2(x + y.hi(), &(ex.clone() + q(y.hi()))),
            error_in_u2(x + y, &(ex.clone() + ey.clone())),
            error_in_u2(x * y.hi(), &(ex.clone() * q(y.hi()))),
            error_in_u2(x * y, &(ex.clone() * ey.clone())),
            error_in_u2(x.square(), &(ex.clone() * ex.clone())),
            error_in_u2(x / y.hi(), &(ex.clone() / q(y.hi()))),
            error_in_u2(x / y, &(ex / ey)),
            root_err,
        ];
        for (w, e) in worst.iter_mut().zip(errs) {
            *w = w.max(e);
        }
    }
    for ((name, w), b) in names.iter().zip(worst).zip(Dd::ERROR_BOUNDS) {
        assert_eq!(*name, b.operation, "the survey follows ERROR_BOUNDS");
        eprintln!(
            "survey: {name:9} largest error {w:6.3} u²  (bound {} u²)",
            b.units_of_u2
        );
        assert!(
            w <= b.units_of_u2 * 1.000_001,
            "{name}: {w} u² above the bound"
        );
    }
}
