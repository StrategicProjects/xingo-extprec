//! The acceptance tests of `Interval` (L2.3; Chapter 11.9): inclusion (I1)
//! in every case and tightness (I3) — each endpoint the directed rounding
//! of the exact one — checked in exact rational arithmetic (`dashu`), over
//! the whole range of `f64`: overflow, subnormals, zeros, infinite
//! endpoints, the empty set.
//!
//! An `f64` is a dyadic rational, so `q(x)` below is exact and every
//! comparison of this file is a comparison of rationals: no tolerance
//! appears. The slack the chapter declares is **one ulp**: an operation on
//! two points returns two adjacent `f64` at most (`one_ulp`).

#![allow(clippy::disallowed_methods)] // L2.4: libm in tests and examples, never as a reference (clippy.toml)
#![allow(clippy::float_cmp)] // exact comparisons are the subject

use dashu_int::{IBig, UBig};
use dashu_ratio::RBig;
use proptest::prelude::*;
use xingo_extprec::{DoubleDouble, Interval};

type Dd = DoubleDouble;

const MAX: f64 = f64::MAX;
const INF: f64 = f64::INFINITY;

// ---------------------------------------------------------------------
// Exact arithmetic
// ---------------------------------------------------------------------

/// The value of a finite `f64`, exactly.
fn q(x: f64) -> RBig {
    RBig::try_from(x).expect("a finite f64")
}

fn zero() -> RBig {
    RBig::from(0)
}

fn abs(x: RBig) -> RBig {
    if x < zero() { -x } else { x }
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

/// A printed endpoint, read as an exact decimal: `None` for `-inf` and
/// `inf`, else the value and the unit of its last digit.
fn decimal(s: &str) -> Option<(RBig, RBig)> {
    if s == "inf" || s == "-inf" {
        return None;
    }
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
    let unit = pow10(exponent - i32::try_from(frac.len()).expect("short"));
    let v = RBig::from(n) * unit.clone();
    Some((if negative { -v } else { v }, unit))
}

/// The two printed endpoints of `[a, b]`.
fn endpoints(s: &str) -> (&str, &str) {
    s.strip_prefix('[')
        .and_then(|t| t.strip_suffix(']'))
        .and_then(|t| t.split_once(", "))
        .unwrap_or_else(|| panic!("not an interval: {s}"))
}

/// Algorithm D for one printed interval: the decimals contain `x`, each
/// within one unit of its last digit; infinite endpoints print as such.
fn printed_contains(x: Interval, s: &str) -> Result<(), TestCaseError> {
    let (lo, hi) = endpoints(s);
    match decimal(lo) {
        None => prop_assert!(lo == "-inf" && x.inf() == -INF, "{s}"),
        Some((v, unit)) => {
            prop_assert!(x.inf() > -INF, "{s}");
            prop_assert!(v <= q(x.inf()) && q(x.inf()) - v < unit, "{s} for {x:?}");
        }
    }
    match decimal(hi) {
        None => prop_assert!(hi == "inf" && x.sup() == INF, "{s}"),
        Some((v, unit)) => {
            prop_assert!(x.sup() < INF, "{s}");
            prop_assert!(v >= q(x.sup()) && v - q(x.sup()) < unit, "{s} for {x:?}");
        }
    }
    Ok(())
}

/// `lo = ▽t`: the largest element of `F ∪ {−∞}` not above `t` (Chapter
/// 11.9, Notation).
fn is_down(lo: f64, t: &RBig) -> bool {
    if lo == -INF {
        return *t < q(-MAX);
    }
    lo < INF && q(lo) <= *t && (lo == MAX || q(lo.next_up()) > *t)
}

/// `hi = △t`.
fn is_up(hi: f64, t: &RBig) -> bool {
    if hi == INF {
        return *t > q(MAX);
    }
    hi > -INF && q(hi) >= *t && (hi == -MAX || q(hi.next_down()) < *t)
}

/// `t ∈ X` for a rational `t`.
fn has(x: Interval, t: &RBig) -> bool {
    !x.is_empty() && (x.inf() == -INF || q(x.inf()) <= *t) && (x.sup() == INF || q(x.sup()) >= *t)
}

/// `X` is the hull of `[lo, hi]` rounded outward: invariant (I3).
fn tight(x: Interval, lo: &RBig, hi: &RBig) -> Result<(), TestCaseError> {
    prop_assert!(x.check_invariants().is_ok() && !x.is_empty(), "{x:?}");
    prop_assert!(
        is_down(x.inf(), lo),
        "inf {:e} is not the rounding down",
        x.inf()
    );
    prop_assert!(
        is_up(x.sup(), hi),
        "sup {:e} is not the rounding up",
        x.sup()
    );
    Ok(())
}

/// At most two adjacent `f64`: the declared slack of an operation on points.
fn one_ulp(x: Interval) -> bool {
    x.sup() == x.inf() || x.sup() == x.inf().next_up()
}

fn min_max(v: &[RBig]) -> (RBig, RBig) {
    let mut lo = v[0].clone();
    let mut hi = v[0].clone();
    for t in &v[1..] {
        if *t < lo {
            lo.clone_from(t);
        }
        if *t > hi {
            hi.clone_from(t);
        }
    }
    (lo, hi)
}

// ---------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------

const FRACTION: u64 = (1 << 52) - 1;

/// A normal `f64` with exponent in `emin..=emax`, either sign; one time in
/// four the significand is short (exact results, ties).
fn normal(emin: i32, emax: i32) -> impl Strategy<Value = f64> {
    let frac = prop_oneof![
        3 => any::<u64>().prop_map(|b| b & FRACTION),
        1 => any::<u64>().prop_map(|b| (b & FRACTION) >> 46 << 46),
    ];
    (any::<bool>(), frac, emin..=emax).prop_map(|(neg, frac, e)| {
        let biased = u64::try_from(e + 1023).expect("a normal exponent");
        f64::from_bits((u64::from(neg) << 63) | (biased << 52) | frac)
    })
}

/// Any finite `f64`: every exponent, subnormals, zeros, the extremes, and
/// the two ranges where products and quotients change regime. (Boxed, like
/// the strategies built on it: the value trees of ten-armed unions nested
/// three deep do not fit the stack of a test thread in a debug build.)
fn finite() -> BoxedStrategy<f64> {
    prop_oneof![
        6 => any::<u64>().prop_map(f64::from_bits).prop_filter("finite", |x| x.is_finite()),
        6 => normal(-40, 40),
        3 => (-20_i32..=20).prop_map(f64::from),
        2 => normal(-540, -430),
        2 => normal(430, 540),
        2 => normal(-1022, -940),
        2 => normal(990, 1023),
        2 => any::<u64>().prop_map(|b| f64::from_bits((b & FRACTION) | (b & (1 << 63)))),
        1 => (0_u64..8, any::<bool>())
            .prop_map(|(k, neg)| if neg { -f64::from_bits(k) } else { f64::from_bits(k) }),
        1 => prop_oneof![Just(MAX), Just(-MAX), Just(f64::MIN_POSITIVE), Just(-0.0)],
    ]
    .boxed()
}

/// A bounded, non-empty interval: wide, a point, or a few ulps wide.
fn bounded() -> BoxedStrategy<Interval> {
    prop_oneof![
        5 => (finite(), finite())
            .prop_map(|(a, b)| Interval::new(a.min(b), a.max(b)).expect("ordered")),
        3 => finite().prop_map(Interval::point),
        2 => (finite(), 1_u32..4).prop_map(|(a, k)| {
            let mut b = a;
            for _ in 0..k {
                b = b.next_up();
            }
            Interval::new(a, if b.is_finite() { b } else { a }).expect("ordered")
        }),
    ]
    .boxed()
}

/// Any non-empty interval, half-lines and the whole line included.
fn nonempty() -> BoxedStrategy<Interval> {
    prop_oneof![
        8 => bounded(),
        1 => finite().prop_map(|a| Interval::new(a, INF).expect("a half-line")),
        1 => finite().prop_map(|b| Interval::new(-INF, b).expect("a half-line")),
        1 => Just(Interval::ENTIRE),
    ]
    .boxed()
}

/// A finite `f64` inside the non-empty `x`, steered by `t`.
fn pick(x: Interval, t: f64) -> f64 {
    t.clamp(x.inf().max(-MAX), x.sup().min(MAX))
}

/// A normalized finite double-double.
fn dd() -> impl Strategy<Value = Dd> {
    (
        finite(),
        -1.0..=1.0_f64,
        prop_oneof![3 => Just(0_i32), 1 => 0..60_i32],
    )
        .prop_map(|(h, r, gap)| Dd::from_sum(h, h * r * 2f64.powi(-53 - gap)))
        .prop_filter("finite", |x| x.is_finite())
}

/// No regression files (as in `exact.rs`): a failure belongs in a unit
/// test of the module.
fn config() -> ProptestConfig {
    ProptestConfig {
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

// ---------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(config())]

    /// (I3) for `+`, `−`, `·`: the exact range over a box is spanned by
    /// its corners; each endpoint must be its directed rounding.
    #[test]
    fn sums_and_products_are_the_rounded_hull(x in bounded(), y in bounded()) {
        let (a, b, c, d) = (q(x.inf()), q(x.sup()), q(y.inf()), q(y.sup()));
        tight(x + y, &(a.clone() + c.clone()), &(b.clone() + d.clone()))?;
        tight(x - y, &(a.clone() - d.clone()), &(b.clone() - c.clone()))?;
        let corners = [a.clone() * c.clone(), a * d.clone(), b.clone() * c, b * d];
        let (lo, hi) = min_max(&corners);
        tight(x * y, &lo, &hi)?;
    }

    /// (I3) for `/` when the divisor does not contain zero, and the shape
    /// of step I4 when it does.
    #[test]
    fn quotients_are_the_rounded_hull(x in bounded(), y in bounded()) {
        let z = x / y;
        prop_assert!(z.check_invariants().is_ok() && !z.is_empty(), "{z:?}");
        let (a, b, c, d) = (q(x.inf()), q(x.sup()), q(y.inf()), q(y.sup()));
        if y.inf() > 0.0 || y.sup() < 0.0 {
            let corners = [
                a.clone() / c.clone(),
                a / d.clone(),
                b.clone() / c,
                b / d,
            ];
            let (lo, hi) = min_max(&corners);
            tight(z, &lo, &hi)?;
        } else if y.inf() == 0.0 && y.sup() == 0.0 {
            prop_assert_eq!(z, Interval::ENTIRE);
        } else if x.inf() == 0.0 && x.sup() == 0.0 {
            prop_assert_eq!(z, Interval::ZERO);
        } else if y.inf() < 0.0 && y.sup() > 0.0 {
            prop_assert_eq!(z, Interval::ENTIRE);
        } else if x.inf() < 0.0 && x.sup() > 0.0 {
            prop_assert_eq!(z, Interval::ENTIRE);
        } else {
            // Zero is an endpoint of Y and X has one sign: a half-line
            // that starts at the quotient by the other endpoint of Y.
            let far = if y.inf() == 0.0 { d } else { c };
            let positive = (x.inf() >= 0.0) == (y.inf() == 0.0);
            if positive {
                let near = if x.inf() >= 0.0 { a } else { b };
                prop_assert_eq!(z.sup(), INF);
                prop_assert!(is_down(z.inf(), &(near / far)), "{z:?}");
            } else {
                let near = if x.inf() >= 0.0 { a } else { b };
                prop_assert_eq!(z.inf(), -INF);
                prop_assert!(is_up(z.sup(), &(near / far)), "{z:?}");
            }
        }
    }

    /// (I1) with unbounded operands too: sampled elements, exact results.
    #[test]
    fn inclusion_of_sampled_elements(
        x in nonempty(), y in nonempty(), s in finite(), t in finite(),
    ) {
        let (xs, ys) = (pick(x, s), pick(y, t));
        prop_assert!(x.contains(xs) && y.contains(ys));
        let (p, r) = (q(xs), q(ys));
        for (name, z, exact) in [
            ("+", x + y, p.clone() + r.clone()),
            ("-", x - y, p.clone() - r.clone()),
            ("*", x * y, p.clone() * r.clone()),
        ] {
            prop_assert!(z.check_invariants().is_ok(), "{name}: {z:?}");
            prop_assert!(has(z, &exact), "{xs:e} {name} {ys:e} is not in {z:?}");
        }
        if ys != 0.0 {
            let z = x / y;
            prop_assert!(z.check_invariants().is_ok(), "/: {z:?}");
            prop_assert!(has(z, &(p.clone() / r.clone())), "{xs:e} / {ys:e} is not in {z:?}");
            let z = x / ys;
            prop_assert!(has(z, &(p.clone() / r)), "{xs:e} / {ys:e} is not in {z:?} (point)");
        }
        prop_assert!(has(x.square(), &(p.clone() * p.clone())));
        prop_assert!(has(x.abs(), &abs(p.clone())));
        prop_assert!(has(-x, &(-p.clone())));
        prop_assert!(x.hull(y).contains(xs) && x.hull(y).contains(ys));
        prop_assert_eq!(x.intersection(y).contains(xs), y.contains(xs));
        if xs >= 0.0 {
            // √xs lies between the square roots of its two neighbours in
            // the result: compare squares.
            let z = x.sqrt();
            prop_assert!(!z.is_empty() && z.inf() >= 0.0, "{z:?}");
            prop_assert!(q(z.inf()) * q(z.inf()) <= p, "{z:?}");
            prop_assert!(z.sup() == INF || q(z.sup()) * q(z.sup()) >= p, "{z:?}");
        }
    }

    /// (I3) for `√`, `x²`, and the exact operations.
    #[test]
    fn roots_squares_and_exact_operations(x in bounded()) {
        let (a, b) = (q(x.inf()), q(x.sup()));
        // x².
        let (lo, hi) = min_max(&[a.clone() * a.clone(), b.clone() * b.clone()]);
        let lo = if x.contains(0.0) { zero() } else { lo };
        tight(x.square(), &lo, &hi)?;
        // |x| and −x are exact.
        let m = x.abs();
        prop_assert_eq!(q(m.sup()), abs(a.clone()).max(abs(b.clone())));
        prop_assert_eq!(
            q(m.inf()),
            if x.contains(0.0) { zero() } else { abs(a.clone()).min(abs(b.clone())) }
        );
        prop_assert_eq!(q(x.mag()), q(m.sup()));
        prop_assert_eq!(q(x.mig()), q(m.inf()));
        prop_assert_eq!((-x).inf(), -x.sup());
        prop_assert_eq!((-x).sup(), -x.inf());
        // √x: s = ▽√t iff s² ≤ t < succ(s)²; s = △√t iff pred(s)² < t ≤ s².
        let z = x.sqrt();
        if x.sup() < 0.0 {
            prop_assert!(z.is_empty());
        } else {
            let t = if x.inf() > 0.0 { a } else { zero() };
            let (s, r) = (z.inf(), z.sup());
            prop_assert!(s >= 0.0 && r.is_finite() && s <= r, "{z:?}");
            prop_assert!(q(s) * q(s) <= t && q(s.next_up()) * q(s.next_up()) > t, "{z:?}");
            prop_assert!(q(r) * q(r) >= b, "{z:?}");
            prop_assert!(r == 0.0 || q(r.next_down()) * q(r.next_down()) < b, "{z:?}");
        }
    }

    /// The declared slack: an operation on points is a point or two
    /// adjacent `f64` (barring overflow, where one endpoint is infinite),
    /// and a point exactly when the result is an `f64`.
    #[test]
    fn point_operations_are_one_ulp_wide(a in finite(), b in finite()) {
        let (x, y) = (Interval::point(a), Interval::point(b));
        let mut results = vec![
            (x + y, q(a) + q(b)),
            (x - y, q(a) - q(b)),
            (x * y, q(a) * q(b)),
            (x.square(), q(a) * q(a)),
        ];
        if b != 0.0 {
            results.push((x / y, q(a) / q(b)));
        }
        for (z, exact) in results {
            prop_assert!(has(z, &exact), "{z:?}");
            prop_assert!(one_ulp(z), "{z:?} is wider than one ulp");
            if z.inf().is_finite() && z.sup().is_finite() {
                prop_assert_eq!(z.inf() == z.sup(), q(z.inf()) == exact, "{:?}", z);
            }
        }
        if a >= 0.0 {
            let z = x.sqrt();
            prop_assert!(one_ulp(z), "{z:?}");
            prop_assert_eq!(z.inf() == z.sup(), q(z.inf()) * q(z.inf()) == q(a), "{:?}", z);
        }
    }

    /// A point operand is the interval `[y, y]`.
    #[test]
    fn mixed_operations_are_point_operations(x in nonempty(), y in finite()) {
        let p = Interval::point(y);
        prop_assert_eq!(x + y, x + p);
        prop_assert_eq!(x - y, x - p);
        prop_assert_eq!(x * y, x * p);
        prop_assert_eq!(x / y, x / p);
    }

    /// Invariant (I4): the assigning forms and the left-scalar forms are
    /// the binary ones, bit for bit; `y − X` and `y / X` are the rounded
    /// hulls of the exact ranges.
    #[test]
    fn operator_forms_are_the_binary_ones(x in nonempty(), z in nonempty(), y in finite()) {
        let same = |p: Interval, r: Interval| {
            p.inf().to_bits() == r.inf().to_bits() && p.sup().to_bits() == r.sup().to_bits()
        };
        let p = Interval::point(y);
        let (mut a, mut s, mut m, mut d) = (x, x, x, x);
        a += z;
        s -= z;
        m *= z;
        d /= z;
        prop_assert!(same(a, x + z) && same(s, x - z) && same(m, x * z) && same(d, x / z));
        let (mut a, mut s, mut m, mut d) = (x, x, x, x);
        a += y;
        s -= y;
        m *= y;
        d /= y;
        prop_assert!(same(a, x + y) && same(s, x - y) && same(m, x * y) && same(d, x / y));
        prop_assert!(same(y + x, p + x) && same(y - x, p - x));
        prop_assert!(same(y * x, p * x) && same(y / x, p / x));
        if x.inf().is_finite() && x.sup().is_finite() {
            let (lo, hi) = (q(x.inf()), q(x.sup()));
            tight(y - x, &(q(y) - hi.clone()), &(q(y) - lo.clone()))?;
            if x.inf() > 0.0 || x.sup() < 0.0 {
                let (c1, c2) = (q(y) / lo, q(y) / hi);
                let (lo, hi) = min_max(&[c1, c2]);
                tight(y / x, &lo, &hi)?;
            }
        }
    }

    /// Algorithm D: with a precision, `{}` and `{:e}` print decimals that
    /// contain the interval, each within one unit of its last digit.
    #[test]
    fn printing_with_a_precision_rounds_outward(x in nonempty(), p in 0_usize..=20) {
        printed_contains(x, &format!("{x:.p$}"))?;
        printed_contains(x, &format!("{x:.p$e}"))?;
    }

    /// Without a precision the text identifies the interval.
    #[test]
    fn printing_without_a_precision_round_trips(x in nonempty()) {
        for s in [format!("{x}"), format!("{x:e}")] {
            let (lo, hi) = endpoints(&s);
            let (lo, hi): (f64, f64) = (lo.parse().expect("an f64"), hi.parse().expect("an f64"));
            prop_assert_eq!(lo.to_bits(), x.inf().to_bits(), "{}", &s);
            prop_assert_eq!(hi.to_bits(), x.sup().to_bits(), "{}", &s);
        }
    }

    /// The queries against their definitions.
    #[test]
    fn queries_follow_their_definitions(x in nonempty(), y in nonempty(), t in finite()) {
        let m = x.mid();
        prop_assert!(x.contains(m), "mid {m:e} is not in {x:?}");
        if x.inf().is_finite() && x.sup().is_finite() {
            prop_assert!(is_up(x.width(), &(q(x.sup()) - q(x.inf()))), "{:e}", x.width());
            // Nearest the midpoint: within one ulp of it (subnormals aside).
            let exact = (q(x.inf()) + q(x.sup())) / RBig::from(2);
            prop_assert!(m == -MAX || q(m.next_down()) <= exact, "{m:e}");
            prop_assert!(m == MAX || exact <= q(m.next_up()), "{m:e}");
        } else {
            prop_assert_eq!(x.width(), INF);
            prop_assert_eq!(x.mag(), INF);
        }
        prop_assert_eq!(x.contains(t), has(x, &q(t)));
        prop_assert_eq!(x.certainly_le(t), x.sup() <= t);
        prop_assert_eq!(x.certainly_ge(t), x.inf() >= t);
        prop_assert_eq!(
            x.certainly_within(t),
            x.certainly_le(t) && x.certainly_ge(-t)
        );
        let (h, i) = (x.hull(y), x.intersection(y));
        prop_assert!(x.is_subset_of(h) && y.is_subset_of(h));
        prop_assert!(i.is_subset_of(x) && i.is_subset_of(y));
        prop_assert_eq!(i.is_empty(), x.sup() < y.inf() || y.sup() < x.inf());
        prop_assert_eq!(x.is_subset_of(y), x.intersection(y) == x);
        prop_assert_eq!(h, y.hull(x));
        prop_assert_eq!(i, y.intersection(x));
        prop_assert_eq!(Interval::new(x.inf(), x.sup()), Ok(x));
        for z in [h, i, x.abs(), -x, x.square(), x.sqrt(), x + y, x - y, x * y, x / y] {
            prop_assert!(z.check_invariants().is_ok(), "{z:?}");
        }
    }

    /// `enclose` is the tightest interval around a double-double, and
    /// `mid_rad` the rounded hull of `m ± r`.
    #[test]
    fn constructors_are_tight(x in dd(), m in finite(), r in finite()) {
        let e = Interval::enclose(x);
        let v = q(x.hi()) + q(x.lo());
        tight(e, &v, &v)?;
        prop_assert!(one_ulp(e));
        prop_assert_eq!(e.inf() == e.sup(), x.lo() == 0.0);
        let r = r.abs();
        tight(Interval::mid_rad(m, r), &(q(m) - q(r)), &(q(m) + q(r)))?;
    }
}

/// The corners of the range of `f64`, crossed with each other: every
/// operation is the rounded hull there too (a deterministic complement to
/// the random cases).
#[test]
fn extreme_operands_exhaustively() {
    let eta = f64::from_bits(1);
    let specials = [
        0.0,
        eta,
        2.0 * eta,
        3.0 * eta,
        f64::MIN_POSITIVE.next_down(),
        f64::MIN_POSITIVE,
        f64::MIN_POSITIVE.next_up(),
        2f64.powi(-969),
        2f64.powi(-968).next_down(),
        2f64.powi(-968),
        2f64.powi(-900).next_down(),
        2f64.powi(-537),
        2f64.powi(-512) * 1.5,
        2f64.powi(-486) * (1.0 + f64::EPSILON),
        1.0_f64.next_down(),
        1.0,
        1.0 + f64::EPSILON,
        3.0,
        2f64.powi(53) + 2.0,
        2f64.powi(511) * 1.75,
        2f64.powi(512),
        2f64.powi(969),
        2f64.powi(970),
        MAX / 2.0,
        MAX.next_down(),
        MAX,
    ];
    let signed: Vec<f64> = specials.iter().flat_map(|&v| [v, -v]).collect();
    let check = |z: Interval, exact: &RBig, what: &str| {
        assert!(z.check_invariants().is_ok(), "{what}: {z:?}");
        assert!(is_down(z.inf(), exact), "{what}: inf of {z:?}");
        assert!(is_up(z.sup(), exact), "{what}: sup of {z:?}");
    };
    for &a in &signed {
        let x = Interval::point(a);
        for &b in &signed {
            let y = Interval::point(b);
            let what = format!("{a:e} ∘ {b:e}");
            check(x + y, &(q(a) + q(b)), &what);
            check(x - b, &(q(a) - q(b)), &what);
            check(x * y, &(q(a) * q(b)), &what);
            check(x * b, &(q(a) * q(b)), &what);
            if b != 0.0 {
                check(x / y, &(q(a) / q(b)), &what);
                check(x / b, &(q(a) / q(b)), &what);
            }
        }
        check(x.square(), &(q(a) * q(a)), &format!("({a:e})²"));
        if a >= 0.0 {
            let z = x.sqrt();
            let (s, r) = (z.inf(), z.sup());
            assert!(
                q(s) * q(s) <= q(a) && q(s.next_up()) * q(s.next_up()) > q(a),
                "√{a:e}"
            );
            assert!(q(r) * q(r) >= q(a), "√{a:e}");
            assert!(
                r == 0.0 || q(r.next_down()) * q(r.next_down()) < q(a),
                "√{a:e}"
            );
        }
    }
}

/// Algorithm D on the endpoints that the random cases seldom draw:
/// negative ones, signed zeros, subnormals, the largest values, infinite
/// ones and the empty set, for every precision `0..=20` and far beyond.
#[test]
fn printing_of_extreme_endpoints() {
    let eta = f64::from_bits(1);
    let ends = [
        -INF,
        -MAX,
        -1e300,
        -1e23,
        -999.5,
        -1.0,
        -0.999_999_999_999_999_9,
        -0.1,
        -f64::MIN_POSITIVE,
        -3.0 * eta,
        -eta,
        -0.0,
        0.0,
        eta,
        f64::MIN_POSITIVE.next_down(),
        f64::MIN_POSITIVE,
        0.05,
        0.1,
        0.5,
        0.95,
        0.999_999_999_999_999_9,
        1.0,
        9.5,
        9.999_999_999_999_998,
        99.95,
        1e21,
        1e23,
        MAX,
        INF,
    ];
    for (i, &a) in ends.iter().enumerate() {
        for &b in &ends[i..] {
            let Ok(x) = Interval::new(a, b) else { continue };
            for p in (0..=20).chain([40, 330, 766, 767, 799, 800, 1073, 1074, 1100]) {
                for s in [format!("{x:.p$}"), format!("{x:.p$e}")] {
                    printed_contains(x, &s).unwrap_or_else(|e| panic!("{s}: {e}"));
                }
            }
        }
    }
    for p in 0..=20 {
        assert_eq!(format!("{:.p$}", Interval::EMPTY), "∅");
        assert_eq!(format!("{:.p$e}", Interval::EMPTY), "∅");
    }
}
