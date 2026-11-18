//! The acceptance tests of L2.4 (Chapter 11.11; the contract's §4): every
//! elementary function of `DoubleDouble` within its row of
//! `DoubleDouble::FUNCTION_BOUNDS` against the primary oracle
//! (`dashu-float`, 256 bits beyond the argument) at the directed points of
//! the contract and in property tests; the special cases, function by
//! function (and their classification against `f64`'s — never `f64`'s
//! values); the invariants (D1)/(D2) of every output; the symmetries bit
//! for bit; determinism (golden bits); and, `#[ignore]`d, the survey of the
//! largest errors observed (`cargo test --release -- --ignored survey`).

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::disallowed_methods,
    clippy::too_many_lines
)]

mod support;

use proptest::prelude::*;
use support::{Dy, F};
use xingo_extprec::{DoubleDouble, FunctionBound};

type Dd = DoubleDouble;

// ---------------------------------------------------------------------
// The functions, their oracles and their bounds
// ---------------------------------------------------------------------

/// One function of one argument under test.
#[derive(Clone, Copy)]
struct Fun {
    name: &'static str,
    dd: fn(Dd) -> Dd,
    oracle: fn(&F) -> F,
    f64: fn(f64) -> f64,
}

fn ln2(p: usize) -> F {
    F::from(2).with_precision(p).value().ln()
}

const EXP: Fun = Fun {
    name: "exp",
    dd: Dd::exp,
    oracle: |x| x.exp(),
    f64: f64::exp,
};
const EXP2: Fun = Fun {
    name: "exp2",
    dd: Dd::exp2,
    oracle: |x| {
        let p = x.precision() + 16;
        (x.clone().with_precision(p).value() * ln2(p)).exp()
    },
    f64: f64::exp2,
};
const EXP_M1: Fun = Fun {
    name: "exp_m1",
    dd: Dd::exp_m1,
    oracle: |x| x.exp_m1(),
    f64: f64::exp_m1,
};

const LN: Fun = Fun {
    name: "ln",
    dd: Dd::ln,
    oracle: |x| x.ln(),
    f64: f64::ln,
};
const LN_1P: Fun = Fun {
    name: "ln_1p",
    dd: Dd::ln_1p,
    oracle: |x| x.ln_1p(),
    f64: f64::ln_1p,
};
const LOG2: Fun = Fun {
    name: "log2",
    dd: Dd::log2,
    oracle: |x| x.log2(),
    f64: f64::log2,
};
const LOG10: Fun = Fun {
    name: "log10",
    dd: Dd::log10,
    oracle: |x| x.log10(),
    f64: f64::log10,
};

const SIN: Fun = Fun {
    name: "sin",
    dd: Dd::sin,
    oracle: |x| x.sin(),
    f64: f64::sin,
};
const COS: Fun = Fun {
    name: "cos",
    dd: Dd::cos,
    oracle: |x| x.cos(),
    f64: f64::cos,
};
const TAN: Fun = Fun {
    name: "tan",
    dd: Dd::tan,
    oracle: |x| x.tan(),
    f64: f64::tan,
};

const ATAN: Fun = Fun {
    name: "atan",
    dd: Dd::atan,
    oracle: |x| x.atan(),
    f64: f64::atan,
};
const ASIN: Fun = Fun {
    name: "asin",
    dd: Dd::asin,
    oracle: |x| x.asin(),
    f64: f64::asin,
};
const ACOS: Fun = Fun {
    name: "acos",
    dd: Dd::acos,
    oracle: |x| x.acos(),
    f64: f64::acos,
};

const SINH: Fun = Fun {
    name: "sinh",
    dd: Dd::sinh,
    oracle: |x| x.sinh(),
    f64: f64::sinh,
};
const COSH: Fun = Fun {
    name: "cosh",
    dd: Dd::cosh,
    oracle: |x| x.cosh(),
    f64: f64::cosh,
};
const TANH: Fun = Fun {
    name: "tanh",
    dd: Dd::tanh,
    oracle: |x| x.tanh(),
    f64: f64::tanh,
};
const ASINH: Fun = Fun {
    name: "asinh",
    dd: Dd::asinh,
    oracle: |x| x.asinh(),
    f64: f64::asinh,
};
const ACOSH: Fun = Fun {
    name: "acosh",
    dd: Dd::acosh,
    oracle: |x| x.acosh(),
    f64: f64::acosh,
};
const ATANH: Fun = Fun {
    name: "atanh",
    dd: Dd::atanh,
    oracle: |x| x.atanh(),
    f64: f64::atanh,
};

fn bound(name: &str) -> FunctionBound {
    *Dd::FUNCTION_BOUNDS
        .iter()
        .find(|b| b.function == name)
        .unwrap_or_else(|| panic!("no bound for {name}"))
}

fn exact(fun: &Fun, x: Dd) -> Dy {
    let p = support::prec(x.hi(), x.lo());
    support::dy(&(fun.oracle)(&support::fdd(x.hi(), x.lo(), p)))
}

/// The value `G` of the growth term (functions without one: 0).
fn check_at(fun: &Fun, x: Dd, growth: f64) -> Result<(), String> {
    let z = (fun.dd)(x);
    if !x.is_finite() {
        return Ok(());
    }
    let f = exact(fun, x);
    let b = bound(fun.name);
    if !z.is_finite() {
        // Overflow is allowed only when the exact value is beyond
        // (1 − c u²) MAX_dd; checked by the threshold tests.
        let max = Dy::dd(f64::MAX, f64::from_bits(0x7c8f_ffff_ffff_ffff));
        return if f.abs().lt(&max.sub(&max.scale(-100))) {
            Err(format!(
                "{}({x:?}) = {z:?}, exact {:e}",
                fun.name,
                f.approx()
            ))
        } else {
            Ok(())
        };
    }
    if support::within(
        (z.hi(), z.lo()),
        &f,
        b.units_of_u2,
        b.growth_units_of_u2,
        growth,
        b.absolute,
    ) {
        Ok(())
    } else {
        Err(format!(
            "{}({:e} + {:e}) = ({:e}, {:e}): error {:.3} u² against {} (exact {:e})",
            fun.name,
            x.hi(),
            x.lo(),
            z.hi(),
            z.lo(),
            support::err_u2((z.hi(), z.lo()), &f),
            b.units_of_u2,
            f.approx()
        ))
    }
}

fn check(fun: &Fun, x: Dd) {
    if let Err(e) = check_at(fun, x, growth(fun, x)) {
        panic!("{e}");
    }
}

/// `G` of a function without a second argument: `|tan x|` for `tan`.
fn growth(fun: &Fun, x: Dd) -> f64 {
    if fun.name == "tan" && x.is_finite() {
        exact(fun, x).approx().abs() * (1.0 + p2(-40))
    } else {
        0.0
    }
}

fn d(x: f64) -> Dd {
    Dd::from_f64(x)
}

fn p2(k: i32) -> f64 {
    support::ldexp_f(1.0, k as isize)
}

/// The double-double nearest to `10ᵏ`.
fn ten_pow(k: i32) -> Dd {
    let mut t = Dy::int(1);
    for _ in 0..k.unsigned_abs() {
        t = t.mul(&Dy::int(10));
    }
    let v = if k >= 0 {
        t
    } else {
        Dy::int(1).div(&t, 300, false)
    };
    let (h, l) = support::nearest_dd(&v);
    Dd::from_parts(h, l)
}

/// The neighbours of `x` in the last place of its trailing component.
fn around(x: Dd, k: i32) -> Vec<Dd> {
    let ulp = if x.lo() == 0.0 {
        p2(-106) * x.hi().abs()
    } else {
        x.lo().abs() * p2(-52)
    };
    (-k..=k).map(|i| x + ulp * f64::from(i)).collect()
}

// ---------------------------------------------------------------------
// Strategies (as in tests/exact.rs)
// ---------------------------------------------------------------------

/// A double-double with leading component `hi` and a random trailing one.
fn with_trailing(hi: impl Strategy<Value = f64>) -> impl Strategy<Value = Dd> {
    let r = prop_oneof![6 => -1.0..1.0_f64, 1 => Just(0.0)];
    let gap = prop_oneof![6 => Just(0_i32), 2 => 0..70_i32];
    (hi, r, gap).prop_map(|(hi, r, gap)| Dd::from_sum(hi, hi * r * p2(-53) * p2(-gap)))
}

fn uniform(a: f64, b: f64) -> impl Strategy<Value = Dd> {
    with_trailing(a..b)
}

/// Magnitude log-uniform in `[2^e0, 2^e1)`, either sign.
fn log_uniform(e0: i32, e1: i32) -> impl Strategy<Value = Dd> {
    (with_trailing(1.0..2.0_f64), e0..e1, any::<bool>()).prop_map(|(x, e, neg)| {
        let x = x.ldexp(e);
        if neg { -x } else { x }
    })
}

/// Any double-double, non-finite and subnormal values included.
fn any_dd() -> impl Strategy<Value = Dd> {
    let finite = any::<u64>()
        .prop_map(f64::from_bits)
        .prop_filter("finite", |x| x.is_finite());
    prop_oneof![
        8 => with_trailing(finite),
        1 => prop_oneof![
            Just(Dd::INFINITY),
            Just(-Dd::INFINITY),
            Just(Dd::NAN),
            Just(d(f64::MAX)),
            Just(d(-f64::MAX)),
            Just(d(f64::from_bits(1))),
            Just(d(-0.0)),
        ],
    ]
}

fn config() -> ProptestConfig {
    ProptestConfig::with_cases(if cfg!(debug_assertions) { 500 } else { 4000 })
}

// ---------------------------------------------------------------------
// Special cases
// ---------------------------------------------------------------------

#[derive(Debug, PartialEq)]
enum Class {
    Nan,
    PosInf,
    NegInf,
    PosZero,
    NegZero,
    Pos,
    Neg,
}

fn class(x: f64) -> Class {
    if x.is_nan() {
        Class::Nan
    } else if x == f64::INFINITY {
        Class::PosInf
    } else if x == f64::NEG_INFINITY {
        Class::NegInf
    } else if x == 0.0 {
        if x.is_sign_negative() {
            Class::NegZero
        } else {
            Class::PosZero
        }
    } else if x > 0.0 {
        Class::Pos
    } else {
        Class::Neg
    }
}

/// The arguments at which the classification of the result must be that
/// of `f64`.
fn special_arguments() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MAX,
        -f64::MAX,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
    ]
}

fn same_class_as_f64(fun: &Fun) {
    for x in special_arguments() {
        let z = (fun.dd)(d(x));
        assert!(z.check_invariants().is_ok(), "{}({x:e}): {z:?}", fun.name);
        assert_eq!(
            class(z.hi()),
            class((fun.f64)(x)),
            "{}({x:e}) = {z:?}, f64 gives {:e}",
            fun.name,
            (fun.f64)(x)
        );
    }
}

fn bits(x: Dd) -> (u64, u64) {
    (x.hi().to_bits(), x.lo().to_bits())
}

/// The same value with the same leading bits: the trailing component of a
/// zero has an unspecified sign (invariant D1).
fn same(a: Dd, b: Dd) -> bool {
    a.hi().to_bits() == b.hi().to_bits() && a.lo() == b.lo()
}

// ---------------------------------------------------------------------
// exp, exp2, exp_m1
// ---------------------------------------------------------------------

#[test]
fn exp_special_cases() {
    for f in [&EXP, &EXP2, &EXP_M1] {
        same_class_as_f64(f);
        assert!((f.dd)(Dd::NAN).is_nan());
    }
    assert_eq!(bits(d(0.0).exp()), bits(Dd::ONE));
    assert_eq!(bits(d(-0.0).exp()), bits(Dd::ONE));
    assert_eq!(Dd::INFINITY.exp(), Dd::INFINITY);
    assert_eq!(bits((-Dd::INFINITY).exp()), bits(Dd::ZERO));
    assert_eq!(bits(d(0.0).exp2()), bits(Dd::ONE));
    assert_eq!(bits(d(-0.0).exp2()), bits(Dd::ONE));
    assert_eq!(bits((-Dd::INFINITY).exp2()), bits(Dd::ZERO));
    assert_eq!(bits(d(0.0).exp_m1()), bits(d(0.0)));
    assert_eq!(bits(d(-0.0).exp_m1()), bits(d(-0.0)));
    assert_eq!(Dd::INFINITY.exp_m1(), Dd::INFINITY);
    assert_eq!(bits((-Dd::INFINITY).exp_m1()), bits(-Dd::ONE));
    assert_eq!(bits(d(-800.0).exp_m1()), bits(-Dd::ONE));
    // (−1, eˣ) for very negative x, before eˣ < η/2.
    let z = d(-700.0).exp_m1();
    assert_eq!(z.hi(), -1.0);
    assert!(z.lo() > 0.0);
    // Overflow and underflow.
    assert_eq!(d(709.79).exp(), Dd::INFINITY);
    assert_eq!(d(1e300).exp(), Dd::INFINITY);
    assert_eq!(bits(d(-746.0).exp()), bits(Dd::ZERO));
    assert_eq!(d(1024.0).exp2(), Dd::INFINITY);
    assert_eq!(bits(d(-1076.0).exp2()), bits(Dd::ZERO));
    assert_eq!(d(710.0).exp_m1(), Dd::INFINITY);
}

#[test]
fn exp2_is_exact_on_integers() {
    for n in -1074..=1023 {
        let z = d(f64::from(n)).exp2();
        assert_eq!(bits(z), bits(d(p2(n))), "exp2({n})");
    }
}

#[test]
fn exp_directed_points() {
    let ln2 = Dd::LN_2;
    let mut xs = vec![
        d(1.0),
        d(-1.0),
        d(f64::from_bits(1)),
        d(-f64::from_bits(1)),
        d(p2(-600)),
        d(-p2(-600)),
        d(p2(-60)),
        d(-p2(-60)),
        d(-708.396_4),
        d(-745.133),
        d(-745.13),
        d(-745.0),
        d(-708.0),
        d(709.0),
        d(709.7),
    ];
    // ln Ω and ln MAX_dd, and a few dd ulps around them.
    let ln_max = Dd::from_parts(709.782_712_893_384, 0.0);
    xs.extend(around(ln_max, 4));
    // The boundaries of n: (n + 1/2) ln2/64, on both sides.
    for n in [
        0.0, 1.0, -1.0, 31.0, 63.0, 64.0, -64.0, 1000.0, -1000.0, 65_000.0, -68_000.0,
    ] {
        let b = ln2 * ((n + 0.5) / 64.0);
        xs.extend(around(b, 2));
    }
    for k in 1..=60 {
        xs.push(d(p2(-k)));
        xs.push(d(-p2(-k)));
    }
    for x in &xs {
        check(&EXP, *x);
        check(&EXP_M1, *x);
        check(&EXP2, *x * Dd::LOG2_E);
    }
    // exp(1) against E: within the bound (not necessarily E).
    check(&EXP, Dd::ONE);
    for k in 1..=1074 {
        check(&EXP_M1, d(p2(-k)));
        check(&EXP_M1, d(-p2(-k)));
    }
    // The boundaries of Algorithm M: |x| = 1/2, N/64, |k| = 53.
    for x in [
        0.5,
        -0.5,
        1.0 / 64.0,
        -1.0 / 64.0,
        31.0 / 64.0,
        37.0,
        -37.0,
        36.5,
        -36.9,
    ] {
        for y in around(d(x), 3) {
            check(&EXP_M1, y);
        }
    }
    // Underflow thresholds of the result.
    for x in [-708.396_418_532_264, -745.133_219_101_941] {
        for y in around(d(x), 3) {
            check(&EXP, y);
        }
    }
}

// ---------------------------------------------------------------------
// ln, ln_1p, log2, log10
// ---------------------------------------------------------------------

#[test]
fn ln_special_cases() {
    for f in [&LN, &LN_1P, &LOG2, &LOG10] {
        same_class_as_f64(f);
    }
    for f in [Dd::ln, Dd::log2, Dd::log10] {
        assert_eq!(bits(f(Dd::ONE)), bits(Dd::ZERO));
        assert_eq!(f(Dd::ZERO), -Dd::INFINITY);
        assert_eq!(f(d(-0.0)), -Dd::INFINITY);
        assert_eq!(f(Dd::INFINITY), Dd::INFINITY);
        assert!(f(d(-1e-300)).is_nan() && f(-Dd::INFINITY).is_nan() && f(Dd::NAN).is_nan());
        assert!(f(-Dd::ONE + 2f64.powi(-80)).is_nan());
    }
    assert_eq!(bits(d(0.0).ln_1p()), bits(d(0.0)));
    assert_eq!(bits(d(-0.0).ln_1p()), bits(d(-0.0)));
    assert_eq!(d(-1.0).ln_1p(), -Dd::INFINITY);
    assert!((-Dd::ONE - 2f64.powi(-80)).ln_1p().is_nan());
    assert!(d(-2.0).ln_1p().is_nan() && (-Dd::INFINITY).ln_1p().is_nan());
    assert_eq!(Dd::INFINITY.ln_1p(), Dd::INFINITY);
    check(&LN_1P, -Dd::ONE + 2f64.powi(-80));
}

#[test]
fn log2_is_exact_on_powers_of_two() {
    for n in -1074..=1023 {
        let z = d(p2(n)).log2();
        assert_eq!(bits(z), bits(d(f64::from(n))), "log2(2^{n})");
    }
}

#[test]
fn ln_directed_points() {
    let mut xs = vec![
        Dd::from_parts(1.0, p2(-80)),
        Dd::from_parts(1.0, -p2(-80)),
        d(1.5),
        d(0.75),
        d(f64::from_bits(1)),
        d(f64::MIN_POSITIVE),
        d(f64::MAX),
        Dd::from_parts(f64::MAX, f64::from_bits(0x7c8f_ffff_ffff_ffff)),
        Dd::E,
        d(core::f64::consts::SQRT_2),
        Dd::SQRT_2,
        Dd::FRAC_1_SQRT_2,
        d(2.0),
        d(0.5),
        d(10.0),
    ];
    for k in 1..=106 {
        xs.push(Dd::ONE + p2(-k));
        xs.push(Dd::ONE - p2(-k));
    }
    // The boundaries of the table: 1 + j/64 (the cut of trunc) and halves.
    for j in -20..=28 {
        for x in [1.0 + f64::from(j) / 64.0, 1.0 + (f64::from(j) + 0.5) / 64.0] {
            xs.extend(around(d(x), 2));
            xs.extend(around(d(2.0 * x), 2));
            xs.extend(around(d(0.5 * x), 2));
        }
    }
    for k in (-1074..=1023).step_by(7) {
        xs.extend(around(d(p2(k)), 1));
    }
    for k in -300..=300 {
        xs.push(ten_pow(k));
    }
    for x in &xs {
        for f in [&LN, &LOG2, &LOG10] {
            check(f, *x);
        }
        check(&LN_1P, *x - 1.0);
        check(&LN_1P, *x);
    }
    for k in 1..=106 {
        check(&LN_1P, -Dd::ONE + p2(-k));
        check(&LN_1P, d(p2(-k)));
        check(&LN_1P, d(-p2(-k)));
    }
    for x in [1.0 / 64.0, -1.0 / 64.0, -0.5, 1.0, p2(53), p2(53) - 1.0] {
        for y in around(d(x), 3) {
            check(&LN_1P, y);
        }
    }
}

// ---------------------------------------------------------------------
// sin, cos, tan, sin_cos
// ---------------------------------------------------------------------

/// The double-double nearest to `k π/2` (for `|k|` up to `2¹⁰²³`).
fn near_half_pi(k: f64) -> Dd {
    let p = 1500;
    let pi = F::pi(p);
    let v = pi * support::fdd(k, 0.0, p) / F::from(2);
    let (h, l) = support::nearest_dd(&support::dy(&v));
    Dd::from_parts(h, l)
}

fn check_trig(x: Dd) {
    for f in [&SIN, &COS, &TAN] {
        check(f, x);
        check(f, -x);
    }
}

#[test]
fn trig_special_cases() {
    for f in [&SIN, &COS, &TAN] {
        same_class_as_f64(f);
        assert!((f.dd)(Dd::INFINITY).is_nan() && (f.dd)(-Dd::INFINITY).is_nan());
        assert!((f.dd)(Dd::NAN).is_nan());
    }
    assert_eq!(bits(d(0.0).sin()), bits(d(0.0)));
    assert!(same(d(-0.0).sin(), d(-0.0)));
    assert_eq!(bits(d(0.0).tan()), bits(d(0.0)));
    assert!(same(d(-0.0).tan(), d(-0.0)));
    assert_eq!(bits(d(0.0).cos()), bits(Dd::ONE));
    assert_eq!(bits(d(-0.0).cos()), bits(Dd::ONE));
    let (s, c) = d(-0.0).sin_cos();
    assert!(same(s, d(-0.0)) && same(c, Dd::ONE));
    let (s, c) = Dd::INFINITY.sin_cos();
    assert!(s.is_nan() && c.is_nan());
    // Tiny arguments: sin x = x, tan x = x exactly below 2⁻⁴⁰⁰ (above,
    // the trailing component carries −x³/6, which the bound checks).
    for k in [60, 200] {
        check_trig(d(p2(-k)));
    }
    for k in [401, 600, 1074] {
        let x = d(p2(-k));
        assert_eq!(bits(x.sin()), bits(x));
        assert_eq!(bits(x.tan()), bits(x));
        check(&COS, x); // the pair (1, −x²/2) while x²/2 is representable
    }
}

#[test]
fn trig_directed_points() {
    let mut xs = vec![
        Dd::FRAC_PI_4,
        d(core::f64::consts::FRAC_PI_4),
        d(f64::MAX),
        Dd::from_parts(f64::MAX, f64::from_bits(0x7c8f_ffff_ffff_ffff)),
        d(1.0),
        d(0.5),
        d(1e-10),
        d(6_381_956_970_095_103.0 * p2(797)),
        d(1e22),
    ];
    xs.extend(around(Dd::FRAC_PI_4, 3));
    // The boundaries of the quadrant: odd multiples of π/4.
    for k in [3.0, 5.0, 7.0, 101.0, 1e6 + 1.0] {
        xs.extend(around(near_half_pi(k) * 0.5, 2));
    }
    // The double-doubles nearest to kπ/2: the hardest relative cases.
    for k in 1..=3000 {
        xs.push(near_half_pi(f64::from(k)));
    }
    let mut rng = Rng(77);
    for _ in 0..2000 {
        xs.push(near_half_pi((rng.next() % 2_000_000) as f64 + 1.0));
    }
    for e in [20, 50, 51, 52, 53, 54, 100, 500, 1000, 1023] {
        for k in [p2(e) - 1.0, p2(e), p2(e) + 2.0] {
            if k.is_finite() {
                xs.push(near_half_pi(k));
            }
        }
        xs.extend(around(d(p2(e)), 2));
    }
    for x in &xs {
        check_trig(*x);
    }
}

#[test]
fn trig_symmetry_and_pairs() {
    let mut rng = Rng(5);
    for _ in 0..3000 {
        let x = rng.dd_log(-30, 70, false);
        let (s, c) = x.sin_cos();
        assert_eq!((bits(s), bits(c)), (bits(x.sin()), bits(x.cos())), "{x:?}");
        assert_eq!(bits((-x).sin()), bits(-x.sin()), "sin odd at {x:?}");
        assert_eq!(bits((-x).cos()), bits(x.cos()), "cos even at {x:?}");
        assert_eq!(bits((-x).tan()), bits(-x.tan()), "tan odd at {x:?}");
        let (s2, c2) = (-x).sin_cos();
        assert_eq!((bits(s2), bits(c2)), (bits(-s), bits(c)));
    }
}

// ---------------------------------------------------------------------
// atan, atan2, asin, acos
// ---------------------------------------------------------------------

/// `atan2(y, x)` within its bound of `dashu-float`'s.
fn check_atan2(y: Dd, x: Dd) -> Result<(), String> {
    let z = y.atan2(x);
    let p = support::prec(x.hi(), x.lo()).max(support::prec(y.hi(), y.lo())) + 32;
    let f = support::dy(&support::fdd(y.hi(), y.lo(), p).atan2(&support::fdd(x.hi(), x.lo(), p)));
    let b = bound("atan2");
    if z.is_finite() && support::within((z.hi(), z.lo()), &f, b.units_of_u2, 0.0, 0.0, 0.0) {
        Ok(())
    } else {
        Err(format!(
            "atan2({y:?}, {x:?}) = {z:?}: {:.2} u², exact {:e}",
            if z.is_finite() {
                support::err_u2((z.hi(), z.lo()), &f)
            } else {
                f64::INFINITY
            },
            f.approx()
        ))
    }
}

#[test]
fn atan_special_cases() {
    for f in [&ATAN, &ASIN, &ACOS] {
        same_class_as_f64(f);
    }
    assert_eq!(bits(Dd::INFINITY.atan()), bits(Dd::FRAC_PI_2));
    assert_eq!(bits((-Dd::INFINITY).atan()), bits(-Dd::FRAC_PI_2));
    assert!(same(d(-0.0).atan(), d(-0.0)) && same(d(0.0).atan(), d(0.0)));
    assert!(same(d(-0.0).asin(), d(-0.0)) && same(d(0.0).asin(), d(0.0)));
    assert_eq!(bits(d(1.0).asin()), bits(Dd::FRAC_PI_2));
    assert_eq!(bits(d(-1.0).asin()), bits(-Dd::FRAC_PI_2));
    assert_eq!(bits(d(1.0).acos()), bits(Dd::ZERO));
    assert_eq!(bits(d(-1.0).acos()), bits(Dd::PI));
    assert_eq!(bits(d(0.0).acos()), bits(Dd::FRAC_PI_2));
    assert_eq!(bits(d(-0.0).acos()), bits(Dd::FRAC_PI_2));
    for f in [Dd::asin, Dd::acos] {
        assert!(f(Dd::ONE + p2(-80)).is_nan() && f(-Dd::ONE - p2(-80)).is_nan());
        assert!(f(d(2.0)).is_nan() && f(Dd::INFINITY).is_nan() && f(Dd::NAN).is_nan());
    }
    // atan2: the table of f64 (C99 F.9.1.4), 24 combinations and more.
    let pi = Dd::PI;
    let h = Dd::FRAC_PI_2;
    let q = Dd::FRAC_PI_4;
    let inf = Dd::INFINITY;
    let cases = [
        (d(0.0), d(-0.0), pi),
        (d(-0.0), d(-0.0), -pi),
        (d(0.0), d(0.0), d(0.0)),
        (d(-0.0), d(0.0), d(-0.0)),
        (d(0.0), d(-3.0), pi),
        (d(-0.0), d(-3.0), -pi),
        (d(0.0), d(3.0), d(0.0)),
        (d(-0.0), d(3.0), d(-0.0)),
        (d(0.0), -inf, pi),
        (d(-0.0), -inf, -pi),
        (d(0.0), inf, d(0.0)),
        (d(-0.0), inf, d(-0.0)),
        (d(-2.0), d(0.0), -h),
        (d(-2.0), d(-0.0), -h),
        (d(2.0), d(0.0), h),
        (d(2.0), d(-0.0), h),
        (d(2.0), -inf, pi),
        (d(-2.0), -inf, -pi),
        (d(2.0), inf, d(0.0)),
        (d(-2.0), inf, d(-0.0)),
        (inf, d(5.0), h),
        (-inf, d(-5.0), -h),
        (inf, -inf, pi - q),
        (-inf, -inf, -(pi - q)),
        (inf, inf, q),
        (-inf, inf, -q),
    ];
    for (y, x, want) in cases {
        let z = y.atan2(x);
        if y.hi().is_infinite() && x.hi() == f64::NEG_INFINITY {
            // ±3π/4: the nearest double-double.
            let three = support::pi().mul(&support::Iv::int(3)).scale(-2);
            let (hh, ll) = three.nearest_dd().expect("narrow");
            let w = Dd::from_parts(hh, ll);
            assert!(
                same(z, if y.hi() > 0.0 { w } else { -w }),
                "atan2({y:?}, {x:?}) = {z:?}"
            );
        } else {
            assert!(same(z, want), "atan2({y:?}, {x:?}) = {z:?}");
        }
        assert_eq!(
            class(z.hi()),
            class(y.hi().atan2(x.hi())),
            "atan2({y:?}, {x:?})"
        );
    }
    for (y, x) in [(Dd::NAN, d(1.0)), (d(1.0), Dd::NAN), (Dd::NAN, Dd::NAN)] {
        assert!(y.atan2(x).is_nan());
    }
}

#[test]
fn atan_directed_points() {
    let mut xs = vec![
        d(1.0),
        d(f64::from_bits(1)),
        d(f64::MAX),
        d(8.0),
        d(1.0 / 64.0),
        d(0.5),
        Dd::SQRT_2,
        Dd::FRAC_1_SQRT_2,
        d(1e300),
        d(1e-300),
    ];
    // The cells of the table: 2^e (1 + i/16) and the binades.
    for e in -7..=3 {
        for i in 0..=16 {
            xs.extend(around(d(p2(e) * (1.0 + f64::from(i) / 16.0)), 2));
        }
    }
    for x in &xs {
        check(&ATAN, *x);
        check(&ATAN, -*x);
        if x.abs() <= Dd::ONE {
            for f in [&ASIN, &ACOS] {
                check(f, *x);
                check(f, -*x);
            }
        }
    }
    for k in 1..=106 {
        let x = Dd::ONE - p2(-k);
        for f in [&ASIN, &ACOS] {
            check(f, x);
            check(f, -x);
        }
        check(&ACOS, d(p2(-k)));
    }
    // asin(1/2) against π/6.
    let z = d(0.5).asin();
    assert!(
        (z - Dd::FRAC_PI_6).abs() <= Dd::FRAC_PI_6 * (33.0 * p2(-106)),
        "{z:?}"
    );
    // atan2 over the quadrants, near the axes and the diagonals, with
    // extreme magnitudes.
    let mut rng = Rng(11);
    let mags = [1.0, 1e-300, 1e300, f64::from_bits(1), f64::MAX];
    for &a in &mags {
        for &b in &mags {
            for (sy, sx) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
                let y = d(a * sy) * rng.dd(0.5, 2.0);
                let x = d(b * sx) * rng.dd(0.5, 2.0);
                if y.is_finite() && x.is_finite() && y.hi() != 0.0 && x.hi() != 0.0 {
                    let r = check_atan2(y, x);
                    // Results below 2⁻⁹⁶⁸ carry the η of the bound.
                    assert!(r.is_ok(), "{r:?}");
                }
            }
        }
    }
    for (y, x) in [
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
        (1.0, 1e-20),
        (1e-20, -1.0),
    ] {
        for yy in around(d(y), 2) {
            assert!(check_atan2(yy, d(x)).is_ok(), "{:?}", check_atan2(yy, d(x)));
        }
    }
}

// ---------------------------------------------------------------------
// The hyperbolic functions and their inverses
// ---------------------------------------------------------------------

#[test]
fn hyperbolic_special_cases() {
    for f in [&SINH, &COSH, &TANH, &ASINH, &ACOSH, &ATANH] {
        same_class_as_f64(f);
    }
    for f in [Dd::sinh, Dd::tanh, Dd::asinh, Dd::atanh] {
        assert!(same(f(d(0.0)), d(0.0)) && same(f(d(-0.0)), d(-0.0)));
    }
    assert_eq!(bits(d(0.0).cosh()), bits(Dd::ONE));
    assert_eq!(bits(d(-0.0).cosh()), bits(Dd::ONE));
    assert_eq!(Dd::INFINITY.sinh(), Dd::INFINITY);
    assert_eq!((-Dd::INFINITY).sinh(), -Dd::INFINITY);
    assert_eq!((-Dd::INFINITY).cosh(), Dd::INFINITY);
    assert_eq!(bits(Dd::INFINITY.tanh()), bits(Dd::ONE));
    assert_eq!(bits((-Dd::INFINITY).tanh()), bits(-Dd::ONE));
    assert_eq!(Dd::INFINITY.asinh(), Dd::INFINITY);
    assert_eq!((-Dd::INFINITY).asinh(), -Dd::INFINITY);
    assert_eq!(Dd::INFINITY.acosh(), Dd::INFINITY);
    assert_eq!(bits(Dd::ONE.acosh()), bits(Dd::ZERO));
    assert!(d(0.5).acosh().is_nan() && (Dd::ONE - p2(-80)).acosh().is_nan());
    assert_eq!(Dd::ONE.atanh(), Dd::INFINITY);
    assert_eq!((-Dd::ONE).atanh(), -Dd::INFINITY);
    assert!((Dd::ONE + p2(-80)).atanh().is_nan() && Dd::INFINITY.atanh().is_nan());
    // Overflow of sinh and cosh at ≈ 710.4758600739439.
    assert_eq!(d(710.48).cosh(), Dd::INFINITY);
    assert_eq!(d(-710.48).sinh(), -Dd::INFINITY);
    assert!(d(710.47).cosh().is_finite() && d(710.47).sinh().is_finite());
    // tanh saturates: the pair (1, −2e^(−2x)), then exactly 1.
    let z = d(30.0).tanh();
    assert!(z.hi() == 1.0 && z.lo() < 0.0, "{z:?}");
    assert_eq!(bits(d(400.0).tanh()), bits(Dd::ONE));
    assert_eq!(bits(d(-400.0).tanh()), bits(-Dd::ONE));
}

#[test]
fn hyperbolic_directed_points() {
    let mut xs = vec![
        d(1.0),
        d(0.3125),
        d(2.0),
        d(709.0),
        d(709.5),
        d(710.0),
        d(710.475_860_073_943),
        d(20.0),
        d(40.0),
        d(354.0),
        d(372.0),
        d(p2(60)),
        d(f64::MAX),
        d(f64::from_bits(1)),
    ];
    for k in 1..=60 {
        xs.push(d(p2(-k)));
    }
    for x in [0.3125, 2.0, 709.0, p2(60)] {
        xs.extend(around(d(x), 3));
    }
    for x in &xs {
        check(&ASINH, *x);
        check(&ASINH, -*x);
        // (The oracle would expand e^Ω: beyond 711 the results are ±∞ and
        // ±1, checked above.)
        if x.hi() < 711.0 {
            for f in [&SINH, &COSH, &TANH] {
                check(f, *x);
                check(f, -*x);
            }
        }
        if *x >= Dd::ONE {
            check(&ACOSH, *x);
        }
    }
    assert_eq!(d(f64::MAX).cosh(), Dd::INFINITY);
    assert_eq!(bits(d(-f64::MAX).tanh()), bits(-Dd::ONE));
    for k in 1..=106 {
        check(&ACOSH, Dd::ONE + p2(-k));
        check(&ATANH, Dd::ONE - p2(-k));
        check(&ATANH, -(Dd::ONE - p2(-k)));
        check(&ATANH, d(p2(-k)));
    }
    for x in [0.5, -0.5] {
        for y in around(d(x), 3) {
            check(&ATANH, y);
        }
    }
}

#[test]
fn odd_and_even_bit_for_bit() {
    let odd = [
        Dd::sin,
        Dd::tan,
        Dd::asin,
        Dd::atan,
        Dd::sinh,
        Dd::tanh,
        Dd::asinh,
        Dd::atanh,
    ];
    let even = [Dd::cos, Dd::cosh];
    let mut rng = Rng(9);
    for _ in 0..2000 {
        let x = rng.dd_log(-40, 12, false);
        let y = rng.dd(0.0, 1.0);
        for f in odd {
            for v in [x, y] {
                let (a, b) = (f(-v), f(v));
                if !a.is_nan() {
                    assert!(same(a, -b), "odd at {v:?}: {a:?} {b:?}");
                }
            }
        }
        for f in even {
            assert_eq!(bits(f(-x)), bits(f(x)), "even at {x:?}");
        }
    }
}

// ---------------------------------------------------------------------
// powi, powf
// ---------------------------------------------------------------------

/// `powi(x, n)` within `(15 + 4|n|)u²` of `dashu-float`'s `xⁿ`.
fn check_powi(x: Dd, n: i32) -> Result<(), String> {
    let z = x.powi(n);
    // xⁿ = ±exp(n ln|x|) at 64 more bits (not `FBig::powi`, which expands
    // exact powers of short operands exactly).
    let p = support::prec(x.hi(), x.lo()) + 64;
    let ax = support::fdd(x.abs().hi(), x.abs().lo(), p);
    let v = (ax.ln() * F::from(n)).exp();
    let f = support::dy(&v);
    let f = if x.hi() < 0.0 && n % 2 != 0 {
        f.neg()
    } else {
        f
    };
    let b = bound("powi");
    if z.is_finite()
        && support::within(
            (z.hi(), z.lo()),
            &f,
            b.units_of_u2,
            b.growth_units_of_u2,
            f64::from(n.unsigned_abs()),
            0.0,
        )
    {
        return Ok(());
    }
    let e = if z.is_finite() {
        support::err_u2((z.hi(), z.lo()), &f)
    } else {
        f64::INFINITY
    };
    Err(format!(
        "powi({x:?}, {n}) = {z:?}: {e:.2} u², exact {:e}",
        f.approx()
    ))
}

/// `powf(x, y)` within `(16 + 11|y ln x|)u²` of `dashu-float`, for `x ≠ 0`
/// finite and `y` finite (the sign of a negative `x` by hand).
fn check_powf(x: Dd, y: Dd) -> Result<(), String> {
    let z = x.powf(y);
    let p = support::prec(x.hi(), x.lo()).max(support::prec(y.hi(), y.lo())) + 64;
    let ax = support::fdd(x.abs().hi(), x.abs().lo(), p);
    let fy = support::fdd(y.hi(), y.lo(), p);
    let lnx = ax.ln();
    let g = (support::dy(&(lnx.clone() * fy.clone())).approx().abs()) * (1.0 + p2(-40));
    let mut f = support::dy(&(lnx * fy).exp());
    if x.hi() < 0.0
        && y.lo() == 0.0
        && (y.hi() * 0.5).trunc() != y.hi() * 0.5
        && y.hi().abs() < p2(53)
        || x.hi() < 0.0 && y.lo() != 0.0 && (y.lo() * 0.5).trunc() != y.lo() * 0.5
    {
        f = f.neg();
    }
    let b = bound("powf");
    if !z.is_finite() {
        let max = Dy::dd(f64::MAX, f64::from_bits(0x7c8f_ffff_ffff_ffff));
        return if max.sub(&max.scale(-40)).lt(&f.abs()) {
            Ok(())
        } else {
            Err(format!(
                "powf({x:?}, {y:?}) = {z:?}, exact {:e}",
                f.approx()
            ))
        };
    }
    if support::within(
        (z.hi(), z.lo()),
        &f,
        b.units_of_u2,
        b.growth_units_of_u2,
        g,
        0.0,
    ) {
        Ok(())
    } else {
        Err(format!(
            "powf({x:?}, {y:?}) = {z:?}: {:.2} u² (G = {g:.3}), exact {:e}",
            support::err_u2((z.hi(), z.lo()), &f),
            f.approx()
        ))
    }
}

#[test]
fn powi_special_cases() {
    let inf = Dd::INFINITY;
    let nan = Dd::NAN;
    for x in [nan, inf, -inf, Dd::ZERO, d(-0.0), d(3.0), d(-3.0)] {
        assert_eq!(bits(x.powi(0)), bits(Dd::ONE), "pown({x:?}, 0)");
        assert_eq!(bits(x.powi(1)), bits(x), "pown({x:?}, 1)");
    }
    assert!(nan.powi(2).is_nan() && nan.powi(-3).is_nan());
    for (x, n, want) in [
        (d(0.0), -3, inf),
        (d(-0.0), -3, -inf),
        (d(0.0), -2, inf),
        (d(-0.0), -2, inf),
        (d(0.0), 3, d(0.0)),
        (d(-0.0), 3, d(-0.0)),
        (d(-0.0), 2, d(0.0)),
        (inf, 3, inf),
        (-inf, 3, -inf),
        (-inf, 2, inf),
        (inf, -3, d(0.0)),
        (-inf, -3, d(-0.0)),
        (-inf, -2, d(0.0)),
        (d(-2.0), 3, d(-8.0)),
        (d(-2.0), 2, d(4.0)),
        (d(-2.0), -1, d(-0.5)),
        (d(2.0), i32::MIN, d(0.0)),
        (d(0.5), i32::MIN, inf),
        (d(-1.0), i32::MIN, d(1.0)),
        (d(-1.0), i32::MAX, d(-1.0)),
    ] {
        let z = x.powi(n);
        assert!(same(z, want), "pown({x:?}, {n}) = {z:?}");
        let f = x.hi().powi(n);
        assert_eq!(class(z.hi()), class(f), "pown({x:?}, {n}) against f64");
    }
    // Exact powers and the square.
    assert_eq!(bits(d(3.0).powi(5)), bits(d(243.0)));
    let x = Dd::ONE / 3.0;
    assert_eq!(bits(x.powi(2)), bits(x.square()));
    assert_eq!(bits(x.powi(-1)), bits(x.recip()));
}

#[test]
fn powi_directed_points() {
    let xs = [
        Dd::ONE / 3.0,
        Dd::E,
        Dd::PI,
        -Dd::SQRT_2,
        d(1.0) + p2(-60),
        d(2.1),
        d(f64::from_bits(1)),
        d(f64::MAX),
        d(1.000_001),
        d(0.999_999),
    ];
    for x in xs {
        for n in [
            2,
            3,
            7,
            16,
            100,
            1000,
            -2,
            -3,
            -7,
            -100,
            -1000,
            12_345,
            i32::MAX,
            i32::MIN,
        ] {
            // Results that over- or underflow are checked by the threshold
            // cases below; here only the finite, normal ones (the oracle
            // would expand x^(2³¹) exactly).
            let exact = x.hi().abs().log2() * f64::from(n);
            if exact.abs() < 1000.0 {
                let r = check_powi(x, n);
                assert!(r.is_ok(), "{r:?}");
            }
        }
    }
    // xⁿ > Ω, x⁻ⁿ finite (subnormal): no spurious underflow.
    let z = d(2.1).powi(-1000);
    assert!(z.hi() > 0.0, "{z:?}");
    assert!(
        check_powi(d(2.1), -1000).is_ok(),
        "{:?}",
        check_powi(d(2.1), -1000)
    );
    let z = d(0.6).powi(1400);
    assert!(z.hi() > 0.0 && check_powi(d(0.6), 1400).is_ok());
    assert_eq!(d(2.1).powi(1000), Dd::INFINITY);
    assert!(same(d(2.1).powi(-2000), Dd::ZERO));
}

#[test]
fn powf_special_cases() {
    let inf = Dd::INFINITY;
    let nan = Dd::NAN;
    let z0 = d(0.0);
    let nz = d(-0.0);
    let cases: Vec<(Dd, Dd, Dd)> = vec![
        (nan, z0, Dd::ONE),
        (nan, nz, Dd::ONE),
        (inf, z0, Dd::ONE),
        (Dd::ONE, nan, Dd::ONE),
        (Dd::ONE, inf, Dd::ONE),
        (Dd::ONE, -inf, Dd::ONE),
        (d(-1.0), inf, Dd::ONE),
        (d(-1.0), -inf, Dd::ONE),
        (d(-2.0), d(0.5), nan),
        (z0, d(-3.0), inf),
        (nz, d(-3.0), -inf),
        (z0, d(-2.0), inf),
        (nz, d(-2.0), inf),
        (nz, d(-0.5), inf),
        (z0, d(3.0), z0),
        (nz, d(3.0), nz),
        (nz, d(2.0), z0),
        (nz, d(0.5), z0),
        (d(0.5), -inf, inf),
        (d(-0.5), -inf, inf),
        (d(2.0), -inf, z0),
        (d(0.5), inf, z0),
        (d(2.0), inf, inf),
        (d(-2.0), inf, inf),
        (-inf, d(-3.0), nz),
        (-inf, d(-2.0), z0),
        (-inf, d(-0.5), z0),
        (-inf, d(3.0), -inf),
        (-inf, d(2.0), inf),
        (-inf, d(0.5), inf),
        (inf, d(-1.0), z0),
        (inf, d(0.5), inf),
        (nan, d(1.0), nan),
        (d(2.0), nan, nan),
        (d(3.0), d(1.0), d(3.0)),
        (d(-0.0), d(1.0), nz),
    ];
    for (x, y, want) in cases {
        let z = x.powf(y);
        if want.is_nan() {
            assert!(z.is_nan(), "pow({x:?}, {y:?}) = {z:?}");
        } else {
            assert!(same(z, want), "pow({x:?}, {y:?}) = {z:?}");
        }
        assert_eq!(
            class(z.hi()),
            class(x.hi().powf(y.hi())),
            "pow({x:?}, {y:?}) against f64"
        );
    }
    // An integer y only by its trailing component: odd, so the sign is kept.
    let y = Dd::from_parts(p2(60), 1.0);
    assert_eq!(d(-1.0).powf(y), d(-1.0));
    assert_eq!(d(-1.0).powf(Dd::from_parts(p2(60), 2.0)), Dd::ONE);
    assert!(d(-1.0).powf(Dd::from_parts(p2(60), 0.5)).is_nan());
    let x = Dd::ONE / 3.0;
    assert_eq!(bits(x.powf(Dd::ONE)), bits(x));
    let z = (-Dd::ONE / 3.0).powf(y);
    assert!(z.hi() == 0.0 && z.hi().is_sign_negative(), "{z:?}");
    // A negative base with an integer exponent: the sign by the parity.
    assert!(d(-2.0).powf(d(3.0)).hi() == -8.0 && check_powf(d(-2.0), d(3.0)).is_ok());
    assert!(d(-2.0).powf(d(2.0)).hi() == 4.0 && check_powf(d(-2.0), d(2.0)).is_ok());
}

#[test]
fn powf_directed_points() {
    let xs = [
        Dd::ONE / 3.0,
        Dd::E,
        Dd::PI,
        d(1.0) + p2(-60),
        d(0.5),
        d(1e-300),
        d(1e300),
        d(10.0),
    ];
    let ys = [
        d(0.5),
        d(-0.5),
        d(2.0),
        d(1.0) / 3.0,
        d(-7.25),
        d(100.0),
        d(1e-20),
        Dd::PI,
    ];
    for x in xs {
        for y in ys {
            let r = check_powf(x, y);
            assert!(r.is_ok(), "{r:?}");
            assert!(check_powf(-x, d(3.0)).is_ok());
        }
        // x^(1/2) against the square root (in the safe range of √, Chapter
        // 11.2): within the two bounds.
        if x.hi() < p2(-480) || x.hi() > p2(480) {
            continue;
        }
        let s = x.powf(d(0.5));
        let q = x.sqrt();
        let g = 0.5 * x.hi().ln().abs();
        let tol = p2(-106) * (16.0 + 11.0 * g + 4.0);
        assert!((s - q).abs() <= q * tol, "{s:?} {q:?}");
    }
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn powi_within_bound((x, n) in prop_oneof![
        (uniform(0.5, 2.0), -1000_i32..1000),
        (log_uniform(-20, 20), -45_i32..45),
    ]) {
        prop_assert!(check_powi(x, n).is_ok(), "{:?}", check_powi(x, n));
    }

    #[test]
    fn powf_within_bound(x in prop_oneof![uniform(1e-3, 1e3), log_uniform(-60, 60).prop_map(Dd::abs)],
                         y in prop_oneof![uniform(-10.0, 10.0), uniform(-100.0, 100.0)]) {
        let lg = x.hi().ln() * y.hi();
        prop_assume!(lg.abs() < 700.0);
        prop_assert!(check_powf(x, y).is_ok(), "{:?}", check_powf(x, y));
    }

    #[test]
    fn trig_within_bound(x in prop_oneof![
        uniform(-4.0, 4.0),
        log_uniform(-60, 3),
        log_uniform(3, 60),
        log_uniform(60, 1024),
    ]) {
        for f in [&SIN, &COS, &TAN] {
            prop_assert!(check_at(f, x, growth(f, x)).is_ok(), "{:?}", check_at(f, x, growth(f, x)));
        }
    }

    #[test]
    fn atan_within_bound(x in prop_oneof![log_uniform(-60, 60), uniform(-3.0, 3.0)]) {
        prop_assert!(check_at(&ATAN, x, 0.0).is_ok(), "{:?}", check_at(&ATAN, x, 0.0));
    }

    #[test]
    fn asin_acos_within_bound(x in prop_oneof![
        uniform(-1.0, 1.0),
        log_uniform(-100, -1).prop_map(|d| Dd::ONE - d.abs()),
        log_uniform(-60, -1),
    ]) {
        for f in [&ASIN, &ACOS] {
            prop_assert!(check_at(f, x, 0.0).is_ok(), "{:?}", check_at(f, x, 0.0));
        }
    }

    #[test]
    fn atan2_within_bound(y in log_uniform(-40, 40), x in log_uniform(-40, 40)) {
        prop_assert!(check_atan2(y, x).is_ok(), "{:?}", check_atan2(y, x));
    }

    #[test]
    fn hyperbolic_within_bound(x in prop_oneof![uniform(-5.0, 5.0), log_uniform(-60, 9)]) {
        for f in [&SINH, &COSH, &TANH, &ASINH] {
            prop_assert!(check_at(f, x, 0.0).is_ok(), "{:?}", check_at(f, x, 0.0));
        }
    }

    #[test]
    fn inverse_hyperbolic_within_bound(x in prop_oneof![
        log_uniform(-60, 1000),
        uniform(-1.0, 1.0),
        log_uniform(-100, -1).prop_map(|d| Dd::ONE - d.abs()),
    ]) {
        prop_assert!(check_at(&ASINH, x, 0.0).is_ok(), "{:?}", check_at(&ASINH, x, 0.0));
        let a = x.abs();
        if a < Dd::ONE {
            prop_assert!(check_at(&ATANH, x, 0.0).is_ok(), "{:?}", check_at(&ATANH, x, 0.0));
            let c = Dd::ONE + a;
            prop_assert!(check_at(&ACOSH, c, 0.0).is_ok(), "{:?}", check_at(&ACOSH, c, 0.0));
        } else {
            prop_assert!(check_at(&ACOSH, a, 0.0).is_ok(), "{:?}", check_at(&ACOSH, a, 0.0));
        }
    }

    #[test]
    fn ln_within_bound(x in prop_oneof![
        log_uniform(-1074, 1024).prop_map(Dd::abs),
        uniform(0.5, 2.0),
        log_uniform(-100, -2).prop_map(|d| Dd::ONE + d),
    ]) {
        for f in [&LN, &LOG2, &LOG10] {
            prop_assert!(check_at(f, x, 0.0).is_ok(), "{:?}", check_at(f, x, 0.0));
        }
    }

    #[test]
    fn ln_1p_within_bound(x in prop_oneof![
        log_uniform(-1000, 60).prop_map(|x| x.max(Dd::from_f64(-0.999_999))),
        uniform(-1.0, 4.0).prop_map(|x| x.max(Dd::from_f64(-0.999_999))),
    ]) {
        prop_assert!(check_at(&LN_1P, x, 0.0).is_ok(), "{:?}", check_at(&LN_1P, x, 0.0));
    }

    #[test]
    fn exp_within_bound(x in prop_oneof![uniform(-745.0, 709.7), log_uniform(-60, 3)]) {
        prop_assert!(check_at(&EXP, x, 0.0).is_ok(), "{:?}", check_at(&EXP, x, 0.0));
    }

    #[test]
    fn exp2_within_bound(x in prop_oneof![uniform(-1074.0, 1023.9), log_uniform(-60, 3)]) {
        prop_assert!(check_at(&EXP2, x, 0.0).is_ok(), "{:?}", check_at(&EXP2, x, 0.0));
    }

    #[test]
    fn exp_m1_within_bound(x in prop_oneof![uniform(-50.0, 709.7), log_uniform(-1000, 3)]) {
        prop_assert!(check_at(&EXP_M1, x, 0.0).is_ok(), "{:?}", check_at(&EXP_M1, x, 0.0));
    }

    #[test]
    fn outputs_are_well_formed(x in any_dd(), y in any_dd(), n in any::<i32>()) {
        for f in all_functions() {
            let z = (f.dd)(x);
            prop_assert!(z.check_invariants().is_ok(), "{}({:?}) = {:?}", f.name, x, z);
        }
        let (s, c) = x.sin_cos();
        for z in [x.powi(n), x.powf(y), x.atan2(y), s, c] {
            prop_assert!(z.check_invariants().is_ok(), "{:?} {:?} {}: {:?}", x, y, n, z);
        }
    }
}

/// Every function of one argument (the list grows with the tasks).
fn all_functions() -> Vec<Fun> {
    vec![
        EXP, EXP2, EXP_M1, LN, LN_1P, LOG2, LOG10, SIN, COS, TAN, ATAN, ASIN, ACOS, SINH, COSH,
        TANH, ASINH, ACOSH, ATANH,
    ]
}

// ---------------------------------------------------------------------
// Survey (L2.4 acceptance: ≥ 10⁵ samples per function)
// ---------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }
    fn dd(&mut self, a: f64, b: f64) -> Dd {
        let h = a + (b - a) * self.unit();
        let ulp = f64::from_bits(h.abs().to_bits() + 1) - h.abs();
        Dd::from_sum(h, (self.unit() - 0.5) * ulp)
    }
    fn dd_log(&mut self, e0: i32, e1: i32, signed: bool) -> Dd {
        let e = e0 + (self.next() % (e1 - e0) as u64) as i32;
        let x = self.dd(1.0, 2.0).ldexp(e);
        if signed && self.next() & 1 == 1 {
            -x
        } else {
            x
        }
    }
}

/// A family of arguments for the survey.
struct Family {
    fun: Fun,
    label: &'static str,
    sample: fn(&mut Rng) -> Dd,
}

fn fam(fun: Fun, label: &'static str, sample: fn(&mut Rng) -> Dd) -> Family {
    Family { fun, label, sample }
}

fn families() -> Vec<Family> {
    vec![
        fam(EXP, "[-745, 709.7]", |r| r.dd(-745.0, 709.7)),
        fam(EXP, "[-1, 1]", |r| r.dd(-1.0, 1.0)),
        fam(EXP2, "[-1074, 1023.9]", |r| r.dd(-1074.0, 1023.9)),
        fam(EXP_M1, "[-5, 5]", |r| r.dd(-5.0, 5.0)),
        fam(EXP_M1, "|x| in 2^[-60, -1]", |r| r.dd_log(-60, -1, true)),
        fam(LN, "2^[-1000, 1000]", |r| r.dd_log(-1000, 1000, false)),
        fam(LN, "[0.5, 2]", |r| r.dd(0.5, 2.0)),
        fam(LN, "1 + d, |d| in 2^[-100, -2]", |r| {
            Dd::ONE + r.dd_log(-100, -2, true)
        }),
        fam(LN_1P, "|x| in 2^[-60, 4], x > -1", |r| {
            r.dd_log(-60, 4, true).max(Dd::from_f64(-0.999))
        }),
        fam(LN_1P, "[-0.9, 3]", |r| r.dd(-0.9, 3.0)),
        fam(LOG2, "2^[-1000, 1000]", |r| r.dd_log(-1000, 1000, false)),
        fam(LOG2, "[0.5, 2]", |r| r.dd(0.5, 2.0)),
        fam(LOG10, "2^[-1000, 1000]", |r| r.dd_log(-1000, 1000, false)),
        fam(LOG10, "[0.5, 2]", |r| r.dd(0.5, 2.0)),
        fam(SIN, "[-π/4, π/4]", |r| r.dd(-0.785, 0.785)),
        fam(SIN, "[-10, 10]", |r| r.dd(-10.0, 10.0)),
        fam(SIN, "[-1e15, 1e15]", |r| r.dd(-1e15, 1e15)),
        fam(SIN, "|x| in 2^[50, 1024]", |r| r.dd_log(50, 1024, true)),
        fam(COS, "[-π/4, π/4]", |r| r.dd(-0.785, 0.785)),
        fam(COS, "[-10, 10]", |r| r.dd(-10.0, 10.0)),
        fam(COS, "[-1e15, 1e15]", |r| r.dd(-1e15, 1e15)),
        fam(COS, "|x| in 2^[50, 1024]", |r| r.dd_log(50, 1024, true)),
        fam(TAN, "[-10, 10]", |r| r.dd(-10.0, 10.0)),
        fam(TAN, "|x| in 2^[-30, 60]", |r| r.dd_log(-30, 60, true)),
        fam(ATAN, "|x| in 2^[-30, 30]", |r| r.dd_log(-30, 30, true)),
        fam(ATAN, "[-3, 3]", |r| r.dd(-3.0, 3.0)),
        fam(ASIN, "[-1, 1]", |r| r.dd(-1.0, 1.0)),
        fam(ASIN, "1 - 2^[-100, -1]", |r| {
            Dd::ONE - r.dd_log(-100, -1, false)
        }),
        fam(ACOS, "[-1, 1]", |r| r.dd(-1.0, 1.0)),
        fam(ACOS, "±(1 - 2^[-100, -1])", |r| {
            let x = Dd::ONE - r.dd_log(-100, -1, false);
            if r.next() & 1 == 1 { -x } else { x }
        }),
        fam(SINH, "[-5, 5]", |r| r.dd(-5.0, 5.0)),
        fam(SINH, "|x| in 2^[-60, 9.47]", |r| r.dd_log(-60, 9, true)),
        fam(COSH, "[-5, 5]", |r| r.dd(-5.0, 5.0)),
        fam(COSH, "[-710, 710]", |r| r.dd(-710.0, 710.0)),
        fam(TANH, "[-3, 3]", |r| r.dd(-3.0, 3.0)),
        fam(TANH, "|x| in 2^[-60, 5]", |r| r.dd_log(-60, 5, true)),
        fam(ASINH, "|x| in 2^[-60, 1000]", |r| r.dd_log(-60, 1000, true)),
        fam(ASINH, "[-3, 3]", |r| r.dd(-3.0, 3.0)),
        fam(ACOSH, "1 + 2^[-100, 4]", |r| {
            Dd::ONE + r.dd_log(-100, 4, false)
        }),
        fam(ACOSH, "2^[0, 1000]", |r| {
            r.dd_log(0, 1000, false).max(Dd::ONE)
        }),
        fam(ATANH, "[-1, 1]", |r| r.dd(-0.999_999, 0.999_999)),
        fam(ATANH, "±(1 - 2^[-100, -1])", |r| {
            let x = Dd::ONE - r.dd_log(-100, -1, false);
            if r.next() & 1 == 1 { -x } else { x }
        }),
    ]
}

#[test]
#[ignore = "survey: run in release with --ignored"]
fn survey_of_observed_errors() {
    let n: usize = std::env::var("L24_SURVEY_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    println!("| function | family | samples | declared | max (u²) | mean (u²) | worst x |");
    println!("|---|---|---|---|---|---|---|");
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let only_two = std::env::var("L24_SURVEY_ONLY").is_ok_and(|v| v == "two");
    for fam in families().into_iter().filter(|_| !only_two) {
        let b = bound(fam.fun.name);
        let (mut max, mut sum, mut worst, mut tiny) = (0.0_f64, 0.0, 0.0, 0.0_f64);
        let mut count = 0_usize;
        for _ in 0..n {
            let x = (fam.sample)(&mut rng);
            let z = (fam.fun.dd)(x);
            let f = exact(&fam.fun, x);
            let g = growth(&fam.fun, x);
            assert!(
                support::within(
                    (z.hi(), z.lo()),
                    &f,
                    b.units_of_u2,
                    b.growth_units_of_u2,
                    g,
                    b.absolute
                ),
                "{}({x:?}) = {z:?} beyond the bound",
                fam.fun.name
            );
            if f.abs().lt(&Dy::pow2(-968)) {
                // Below the relative range: the absolute error, in η.
                let a = Dy::dd(z.hi(), z.lo()).sub(&f).abs().scale(1074).approx();
                tiny = tiny.max(a);
                continue;
            }
            let e = support::err_u2_mixed((z.hi(), z.lo()), &f, b.absolute);
            sum += e;
            count += 1;
            if e > max {
                max = e;
                worst = x.hi();
            }
        }
        let below = if tiny > 0.0 {
            format!(" (results < 2⁻⁹⁶⁸: ≤ {tiny:.2} η)")
        } else {
            String::new()
        };
        println!(
            "| `{}` | {}{below} | {n} | {} | {max:.2} | {:.3} | {worst:e} |",
            fam.fun.name,
            fam.label,
            b.units_of_u2,
            sum / count.max(1) as f64
        );
    }
    // The functions of two arguments.
    let pb = bound("powi");
    let (mut max, mut maxg) = (0.0_f64, 0.0_f64);
    for _ in 0..n {
        let x = rng.dd(0.5, 2.0);
        let k = (rng.next() % 2001) as i32 - 1000;
        let r = check_powi(x, k);
        assert!(r.is_ok(), "{r:?}");
        let z = x.powi(k);
        let f = support::dy(&(support::fdd(x.hi(), x.lo(), 400).ln() * F::from(k)).exp());
        if f.abs().lt(&Dy::pow2(-968)) {
            continue; // below the relative range (the η of the bound)
        }
        let e = support::err_u2((z.hi(), z.lo()), &f);
        max = max.max(e);
        maxg = maxg.max(e / f64::from(k.unsigned_abs().max(1)));
    }
    println!(
        "| `powi` | x in [0.5, 2], n in [-1000, 1000] | {n} | {} + {}|n| | {max:.2} | max/max(1,|n|) = {maxg:.2} | — |",
        pb.units_of_u2, pb.growth_units_of_u2
    );
    let fb = bound("powf");
    let (mut max, mut maxg) = (0.0_f64, 0.0_f64);
    for _ in 0..n {
        let x = rng.dd_log(-10, 10, false);
        let y = rng.dd(-40.0, 40.0);
        let r = check_powf(x, y);
        assert!(r.is_ok(), "{r:?}");
        let z = x.powf(y);
        let p = 400;
        let lnxy = support::fdd(x.hi(), x.lo(), p).ln() * support::fdd(y.hi(), y.lo(), p);
        let g = support::dy(&lnxy).approx().abs();
        let f = support::dy(&lnxy.exp());
        let e = support::err_u2((z.hi(), z.lo()), &f);
        max = max.max(e);
        maxg = maxg.max(e / g.max(1.0));
    }
    println!(
        "| `powf` | x in 2^[-10, 10], y in [-40, 40] | {n} | {} + {}G | {max:.2} | max/max(1,G) = {maxg:.2} | — |",
        fb.units_of_u2, fb.growth_units_of_u2
    );
    let ab = bound("atan2");
    let mut max = 0.0_f64;
    for _ in 0..n {
        let y = rng.dd_log(-40, 40, true);
        let x = rng.dd_log(-40, 40, true);
        let r = check_atan2(y, x);
        assert!(r.is_ok(), "{r:?}");
        let z = y.atan2(x);
        let f = support::dy(&support::fdd(y.hi(), y.lo(), 400).atan2(&support::fdd(
            x.hi(),
            x.lo(),
            400,
        )));
        max = max.max(support::err_u2((z.hi(), z.lo()), &f));
    }
    println!(
        "| `atan2` | y, x in ±2^[-40, 40] | {n} | {} | {max:.2} | — | — |",
        ab.units_of_u2
    );
    // sin_cos is sin and cos bit for bit (tested); the pair is surveyed as
    // those two.
    // The contract's directed sweep: the double-doubles nearest to kπ/2,
    // k = 1, …, K (2·10⁶ by default).
    let kmax: u32 = std::env::var("L24_SWEEP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2_000_000);
    for fun in [SIN, COS].into_iter().filter(|_| !only_two) {
        let b = bound(fun.name);
        let (mut max, mut maxabs, mut worst) = (0.0_f64, 0.0_f64, 0.0);
        for k in 1..=kmax {
            let x = near_half_pi(f64::from(k));
            let z = (fun.dd)(x);
            let f = exact(&fun, x);
            assert!(
                support::within((z.hi(), z.lo()), &f, b.units_of_u2, 0.0, 0.0, b.absolute),
                "{}({x:?}) beyond the bound",
                fun.name
            );
            let e = support::err_u2_mixed((z.hi(), z.lo()), &f, 0.0);
            maxabs = maxabs.max(Dy::dd(z.hi(), z.lo()).sub(&f).abs().approx());
            if e > max {
                max = e;
                worst = x.hi();
            }
        }
        println!(
            "| `{}` | nearest kπ/2, k ≤ {kmax} (max abs {maxabs:e}) | {kmax} | {} | {max:.2} | — | {worst:e} |",
            fun.name, b.units_of_u2
        );
    }
}

// ---------------------------------------------------------------------
// Determinism: golden bits
// ---------------------------------------------------------------------

/// The arguments of the golden test: a few per regime of every algorithm.
fn golden_arguments() -> Vec<Dd> {
    let mut v = vec![
        Dd::ONE / 3.0,
        -Dd::ONE / 7.0,
        Dd::E,
        Dd::PI,
        Dd::from_parts(1.0, p2(-80)),
        d(0.3),
        d(-0.75),
        d(1e-10),
        d(-37.5),
        d(123.456),
        d(700.0),
        d(1e22),
        d(p2(100)),
        d(6_381_956_970_095_103.0 * p2(797)),
    ];
    v.extend([1.0, 1000.0, 1e6].map(near_half_pi));
    v
}

/// The digest of every function at many arguments (FNV-1a over the bits).
fn digest() -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |z: Dd| {
        for b in [z.hi().to_bits(), z.lo().to_bits()] {
            for byte in b.to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
    };
    let mut rng = Rng(2026);
    for _ in 0..500 {
        let x = rng.dd_log(-60, 12, true);
        let y = rng.dd(-1.0, 1.0);
        for f in all_functions() {
            for v in [x, y] {
                let z = (f.dd)(v);
                // NaN bits are not specified: hash a fixed word instead.
                eat(if z.is_nan() { Dd::ZERO } else { z });
            }
        }
        for z in [x.powi(7), x.abs().powf(y), y.atan2(x), x.sin_cos().0] {
            eat(if z.is_nan() { Dd::ZERO } else { z });
        }
    }
    h
}

#[test]
#[ignore = "prints the golden table"]
fn print_golden() {
    for f in all_functions() {
        for x in golden_arguments() {
            let z = (f.dd)(x);
            if z.is_finite() {
                println!(
                    "    (\"{}\", 0x{:016x}, 0x{:016x}, 0x{:016x}, 0x{:016x}),",
                    f.name,
                    x.hi().to_bits(),
                    x.lo().to_bits(),
                    z.hi().to_bits(),
                    z.lo().to_bits()
                );
            }
        }
    }
    println!("digest 0x{:016x}", digest());
}

/// The bits of every function at the golden arguments, recorded on
/// aarch64 (Apple M2 Pro) in release; the test runs on every platform of the
/// CI, in debug and release, with and without the features.
const GOLDEN: [(&str, u64, u64, u64, u64); 246] = [
    (
        "exp",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3ff6_546d_b1ba_2d13,
        0x3c70_a7f6_c6f2_7f68,
    ),
    (
        "exp",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0x3feb_bd76_b8a1_ab98,
        0x3c85_23a5_2703_fdf1,
    ),
    (
        "exp",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x402e_4efb_75e4_527b,
        0xbc94_b19f_fdc1_b060,
    ),
    (
        "exp",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x4037_2404_6eb0_933a,
        0xbcd8_4c96_2dd8_1952,
    ),
    (
        "exp",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e820_c3dd,
    ),
    (
        "exp",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3ff5_9905_8c8c_1a96,
        0xbc9b_3ae3_4963_b3d0,
    ),
    (
        "exp",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x3fde_3b40_ebef_cd7e,
        0x3c74_bbf0_f4cf_fc58,
    ),
    (
        "exp",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ff0_0000_0006_df38,
        0xbc63_112d_8e5e_6d4c,
    ),
    (
        "exp",
        0xc042_c000_0000_0000,
        0x0000_0000_0000_0000,
        0x3c8d_d5c5_6630_1ec8,
        0xb928_8483_5ad0_1ca7,
    ),
    (
        "exp",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x4b11_428e_1cbc_0143,
        0x4794_2bbf_04c6_0d97,
    ),
    (
        "exp",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x7f0d_945d_f4f8_ec8e,
        0x7b91_8339_2684_a46e,
    ),
    (
        "exp",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x4013_3ded_c855_935f,
        0x3cb3_e45a_768f_b73c,
    ),
    (
        "exp2",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3ff4_28a2_f98d_728b,
        0xbc7d_dc22_548e_a41e,
    ),
    (
        "exp2",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0x3fec_fbb0_31a7_41a5,
        0xbc88_4289_7f78_3d43,
    ),
    (
        "exp2",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x401a_52d3_c6f8_818f,
        0xbcbd_4485_fc0a_7607,
    ),
    (
        "exp2",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x4021_a663_7e66_6f83,
        0x3cc3_6d4a_9f91_870c,
    ),
    (
        "exp2",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x4000_0000_0000_0000,
        0x3af6_2e42_fefa_39ef,
    ),
    (
        "exp2",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3ff3_b2c4_7bff_8329,
        0xbc84_97d7_d4b1_4dde,
    ),
    (
        "exp2",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x3fe3_06fe_0a31_b715,
        0x3c76_f46a_d231_82e4,
    ),
    (
        "exp2",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ff0_0000_0004_c366,
        0xbc90_bdbd_ded9_a5e3,
    ),
    (
        "exp2",
        0xc042_c000_0000_0000,
        0x0000_0000_0000_0000,
        0x3d96_a09e_667f_3bcd,
        0xba3b_dd34_13b2_6456,
    ),
    (
        "exp2",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x47a5_f29e_9ad5_4c6d,
        0xc439_4703_e6c0_1699,
    ),
    (
        "exp2",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x6bb0_0000_0000_0000,
        0x0000_0000_0000_0000,
    ),
    (
        "exp2",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x4007_c3f7_3e5e_9df5,
        0xbc90_f96a_5f9e_caa5,
    ),
    (
        "exp_m1",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3fd9_51b6_c6e8_b44c,
        0x3c70_a7f6_c6f2_7f6a,
    ),
    (
        "exp_m1",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0xbfc1_0a25_1d79_519f,
        0x3c64_8e94_9c0f_f7c5,
    ),
    (
        "exp_m1",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x402c_4efb_75e4_527b,
        0xbc94_b19f_fdc1_b060,
    ),
    (
        "exp_m1",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x4036_2404_6eb0_933a,
        0xbcd8_4c96_2dd8_1952,
    ),
    (
        "exp_m1",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3ffb_7e15_1628_aed3,
        0xbc96_5502_2fbe_7846,
    ),
    (
        "exp_m1",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3fd6_6416_3230_6a56,
        0x3c73_1472_da71_30c0,
    ),
    (
        "exp_m1",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0xbfe0_e25f_8a08_1941,
        0x3c74_bbf0_f4cf_fc58,
    ),
    (
        "exp_m1",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ddb_7cdf_d9dd_a4e3,
        0x3a70_c95a_385d_91c6,
    ),
    (
        "exp_m1",
        0xc042_c000_0000_0000,
        0x0000_0000_0000_0000,
        0xbff0_0000_0000_0000,
        0x3c8d_d5c5_6630_1ec8,
    ),
    (
        "exp_m1",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x4b11_428e_1cbc_0143,
        0x4794_2bbf_04c6_0d97,
    ),
    (
        "exp_m1",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x7f0d_945d_f4f8_ec8e,
        0x7b91_8339_2684_a46e,
    ),
    (
        "exp_m1",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x400e_7bdb_90ab_26bf,
        0xbca8_374b_12e0_9188,
    ),
    (
        "ln",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0xbff1_93ea_7aad_030b,
        0x3c9a_256f_99ca_abec,
    ),
    (
        "ln",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x3ff0_0000_0000_0000,
        0x0000_0000_0000_0000,
    ),
    (
        "ln",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x3ff2_50d0_48e7_a1bd,
        0x3c67_abf2_ad8d_5088,
    ),
    (
        "ln",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0xb5e0_0000_0000_0000,
    ),
    (
        "ln",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0xbff3_4378_fcbd_a721,
        0x3c99_c140_4e27_f13c,
    ),
    (
        "ln",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0xc037_069e_2aa2_aa5b,
        0x3cbf_0b70_9e89_338d,
    ),
    (
        "ln",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x4013_4377_4f3e_2362,
        0x3ca1_9e20_7fbb_ab3d,
    ),
    (
        "ln",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x401a_344e_6741_ebab,
        0x3cb5_fe49_4b48_2b3c,
    ),
    (
        "ln",
        0x4480_f0cf_064d_d592,
        0x0000_0000_0000_0000,
        0x4049_5414_6219_54fe,
        0xbceb_03ee_44cc_2161,
    ),
    (
        "ln",
        0x4630_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x4051_5424_5733_7d43,
        0xbcde_3948_c376_279d,
    ),
    (
        "ln",
        0x7506_ac5b_262c_a1ff,
        0x0000_0000_0000_0000,
        0x4082_66a4_f98a_7d3f,
        0xbd10_a1f7_9202_1bf9,
    ),
    (
        "ln",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x3fdc_e6bb_25aa_1316,
        0xbc6d_cd49_c8e5_aff6,
    ),
    (
        "ln",
        0x4098_8b2f_704a_940a,
        0xbd32_c3be_b21e_1e21,
        0x401d_6ff6_4bea_a0d2,
        0xbcb7_db6e_3c3d_235e,
    ),
    (
        "ln",
        0x4137_f7ec_53a8_d491,
        0x3dde_acd7_c60e_9693,
        0x402c_88c0_72bd_5039,
        0xbc9b_21c8_a8cc_64f0,
    ),
    (
        "ln_1p",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3fd2_6962_1134_db92,
        0x3c7e_0efa_dd9d_b02b,
    ),
    (
        "ln_1p",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0xbfc3_bb35_a041_d2a9,
        0xbc42_444b_aaf1_0e76,
    ),
    (
        "ln_1p",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x3ff5_031e_afef_b049,
        0x3c7d_0189_442c_189e,
    ),
    (
        "ln_1p",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x3ff6_bcbe_d09f_00af,
        0xbc6f_73f4_bf6c_2b63,
    ),
    (
        "ln_1p",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3fe6_2e42_fefa_39ef,
        0x3c7a_bc9e_4339_803f,
    ),
    (
        "ln_1p",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3fd0_ca93_7be1_b9dc,
        0xbc72_8637_a172_3644,
    ),
    (
        "ln_1p",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0xbff6_2e42_fefa_39ef,
        0xbc8a_bc9e_3b39_803f,
    ),
    (
        "ln_1p",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ddb_7cdf_d9d1_d693,
        0xba70_c8b7_f5fd_9a85,
    ),
    (
        "ln_1p",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x4013_4bba_22cd_5df5,
        0xbcb5_446e_5536_63bc,
    ),
    (
        "ln_1p",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x401a_35c4_a0a6_61ec,
        0x3c5e_8147_4423_e4b8,
    ),
    (
        "ln_1p",
        0x4480_f0cf_064d_d592,
        0x0000_0000_0000_0000,
        0x4049_5414_6219_54fe,
        0xbceb_03ee_35af_9159,
    ),
    (
        "ln_1p",
        0x4630_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x4051_5424_5733_7d43,
        0xbcde_3948_c376_2799,
    ),
    (
        "ln_1p",
        0x7506_ac5b_262c_a1ff,
        0x0000_0000_0000_0000,
        0x4082_66a4_f98a_7d3f,
        0xbd10_a1f7_9202_1bf9,
    ),
    (
        "ln_1p",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x3fee_3703_db0a_b11a,
        0xbc54_8059_92a3_f110,
    ),
    (
        "ln_1p",
        0x4098_8b2f_704a_940a,
        0xbd32_c3be_b21e_1e21,
        0x401d_709d_2127_2518,
        0xbca9_9953_03b1_90a8,
    ),
    (
        "ln_1p",
        0x4137_f7ec_53a8_d491,
        0x3dde_acd7_c60e_9693,
        0x402c_88c0_8819_d576,
        0x3ca2_2a83_eb1f_7aa2,
    ),
    (
        "log2",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0xbff9_5c01_a39f_bd68,
        0xbc9e_7e80_2c48_281a,
    ),
    (
        "log2",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x3ff7_1547_652b_82fe,
        0x3c77_77d0_ffda_0d24,
    ),
    (
        "log2",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x3ffa_6c87_3498_ddf7,
        0x3c96_c36c_b93c_202a,
    ),
    (
        "log2",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3af7_1547_652b_82fe,
        0x3777_77d0_f44f_6971,
    ),
    (
        "log2",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0xbffb_ca9c_6f53_897b,
        0x3c9f_3668_7ebb_3140,
    ),
    (
        "log2",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0xc040_9c11_65ec_0627,
        0x3cd3_0388_d3f7_e384,
    ),
    (
        "log2",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x401b_ca9a_03b1_083a,
        0x3ca9_2ce9_9d32_715d,
    ),
    (
        "log2",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x4022_e705_2491_d2c4,
        0xbca9_b517_ae88_c2fd,
    ),
    (
        "log2",
        0x4480_f0cf_064d_d592,
        0x0000_0000_0000_0000,
        0x4052_4546_5683_a05e,
        0xbccc_4b6c_718c_1ea6,
    ),
    (
        "log2",
        0x4630_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x4059_0000_0000_0000,
        0x0000_0000_0000_0000,
    ),
    (
        "log2",
        0x7506_ac5b_262c_a1ff,
        0x0000_0000_0000_0000,
        0x408a_8c05_fb14_a90e,
        0x3d0c_3850_8a58_b6de,
    ),
    (
        "log2",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x3fe4_d90e_6931_bbef,
        0xbc82_7926_8d87_bfac,
    ),
    (
        "log2",
        0x4098_8b2f_704a_940a,
        0xbd32_c3be_b21e_1e21,
        0x4025_3c0c_2dae_5654,
        0xbc99_8edb_0874_0145,
    ),
    (
        "log2",
        0x4137_f7ec_53a8_d491,
        0x3dde_acd7_c60e_9693,
        0x4034_9543_ba64_c874,
        0x3cdf_61ed_d35d_bdd5,
    ),
    (
        "log10",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0xbfde_8927_964f_d5fd,
        0xbc41_8606_8750_4340,
    ),
    (
        "log10",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x3fdb_cb7b_1526_e50e,
        0x3c69_5355_baaa_fad3,
    ),
    (
        "log10",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x3fdf_d14d_b31b_a3bb,
        0x3c09_7f6d_771a_6500,
    ),
    (
        "log10",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3adb_cb7b_1526_e50e,
        0x3769_5355_b3b8_1c0e,
    ),
    (
        "log10",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0xbfe0_bb6c_34d8_1502,
        0x3c87_d48c_b6bd_6284,
    ),
    (
        "log10",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0xc024_0000_0000_0000,
        0x3c72_3deb_5b9a_1544,
    ),
    (
        "log10",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x4000_bb6a_bfc9_68ef,
        0x3c92_7ece_f2b7_38d3,
    ),
    (
        "log10",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x4006_c2c2_c2de_3310,
        0xbc9a_d25b_401d_5fe8,
    ),
    (
        "log10",
        0x4480_f0cf_064d_d592,
        0x0000_0000_0000_0000,
        0x4036_0000_0000_0000,
        0xb980_0000_0000_0000,
    ),
    (
        "log10",
        0x4630_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x403e_1a5e_2df9_2e9e,
        0x3cd6_f302_34e0_e690,
    ),
    (
        "log10",
        0x7506_ac5b_262c_a1ff,
        0x0000_0000_0000_0000,
        0x406f_f73a_3f9f_7285,
        0xbd0c_fcee_c494_b5f0,
    ),
    (
        "log10",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x3fc9_1a74_c4f8_5378,
        0x3c4b_7414_8106_a372,
    ),
    (
        "log10",
        0x4098_8b2f_704a_940a,
        0xbd32_c3be_b21e_1e21,
        0x4009_91a7_4c4f_8538,
        0xbcaf_922f_adfb_e572,
    ),
    (
        "log10",
        0x4137_f7ec_53a8_d491,
        0x3dde_acd7_c60e_9693,
        0x4018_c8d3_a627_c29c,
        0xbcaf_922f_adfb_e573,
    ),
    (
        "sin",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3fd4_f0c2_068a_80c7,
        0xbc74_ebb3_b7b3_86e4,
    ),
    (
        "sin",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0xbfc2_393c_a1aa_969d,
        0x3c5e_6136_82c5_fdac,
    ),
    (
        "sin",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x3fda_4a3d_9c21_31dc,
        0x3c71_807f_9131_ee3c,
    ),
    (
        "sin",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0xb92f_1976_b7ed_8fbc,
        0x35c4_cf98_e804_177d,
    ),
    (
        "sin",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3fea_ed54_8f09_0cee,
        0x3c40_6375_39ad_82d0,
    ),
    (
        "sin",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3fd2_e9cd_95ba_ba33,
        0x3c75_1dbd_44eb_0887,
    ),
    (
        "sin",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0xbfe5_cffc_16bf_8f0d,
        0xbc89_6cb3_70eb_578a,
    ),
    (
        "sin",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ddb_7cdf_d9d7_bdbb,
        0xb98b_0b0f_fe8f_ae2b,
    ),
    (
        "sin",
        0xc042_c000_0000_0000,
        0x0000_0000_0000_0000,
        0x3fc9_5178_97d0_3677,
        0xbc6a_30e3_d40c_bdae,
    ),
    (
        "sin",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0xbfe9_b9da_dc41_aeb6,
        0x3c7a_57a7_849f_0735,
    ),
    (
        "sin",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x3fe1_6834_defa_adb0,
        0x3c8b_2a2f_ade5_1bab,
    ),
    (
        "sin",
        0x4480_f0cf_064d_d592,
        0x0000_0000_0000_0000,
        0xbfeb_453a_b76b_f397,
        0xbc5f_4537_9077_2648,
    ),
    (
        "sin",
        0x4630_0000_0000_0000,
        0x0000_0000_0000_0000,
        0xbfeb_e8ed_97ac_1f59,
        0x3c80_5598_cf86_dd85,
    ),
    (
        "sin",
        0x7506_ac5b_262c_a1ff,
        0x0000_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0xb842_b089_ea1e_692b,
    ),
    (
        "sin",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x3ff0_0000_0000_0000,
        0xb23e_396a_4794_1a45,
    ),
    (
        "sin",
        0x4098_8b2f_704a_940a,
        0xbd32_c3be_b21e_1e21,
        0x39d2_17b7_7be7_7f97,
        0x3674_6b51_2b5b_0044,
    ),
    (
        "sin",
        0x4137_f7ec_53a8_d491,
        0x3dde_acd7_c60e_9693,
        0xba7f_d4d6_d0ff_ed66,
        0xb70d_1e65_7752_437b,
    ),
    (
        "cos",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3fee_3d16_5991_982d,
        0xbc84_a4b3_3934_d978,
    ),
    (
        "cos",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0x3fef_ac8c_dbd8_7daf,
        0xbc7b_3e1d_0489_be86,
    ),
    (
        "cos",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0xbfed_2cec_9a55_4007,
        0xbc71_5d3f_0bc8_9b55,
    ),
    (
        "cos",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0xbff0_0000_0000_0000,
        0x325e_396a_4794_1a45,
    ),
    (
        "cos",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3fe1_4a28_0fb5_068c,
        0xbc8b_71ed_d34e_99e0,
    ),
    (
        "cos",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3fee_921d_d42f_09ba,
        0x3c88_2c9a_2fb0_7ec2,
    ),
    (
        "cos",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x3fe7_69fe_c655_211f,
        0xbc68_27d5_cf8c_68c5,
    ),
    (
        "cos",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0xbbb7_9ca1_0c92_4224,
    ),
    (
        "cos",
        0xc042_c000_0000_0000,
        0x0000_0000_0000_0000,
        0x3fef_5e25_d08d_9224,
        0xbc75_e0ca_d626_5ba4,
    ),
    (
        "cos",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0xbfe3_07e5_980a_1558,
        0xbc84_45cf_27ff_9e26,
    ),
    (
        "cos",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0xbfea_d9f1_50b2_e096,
        0x3c80_17b2_8370_159d,
    ),
    (
        "cos",
        0x4480_f0cf_064d_d592,
        0x0000_0000_0000_0000,
        0x3fe0_be2c_ef01_c8f4,
        0xbc8b_2d1b_c801_8c50,
    ),
    (
        "cos",
        0x4630_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x3fdf_4eb3_ff66_e36d,
        0xbc66_91c4_f270_021a,
    ),
    (
        "cos",
        0x7506_ac5b_262c_a1ff,
        0x0000_0000_0000_0000,
        0xbc21_4ae7_2e6b_a22f,
        0x3897_3eef_1477_d90e,
    ),
    (
        "cos",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0xb91f_1976_b7ed_8fbc,
        0x35b4_cf98_e804_177d,
    ),
    (
        "cos",
        0x4098_8b2f_704a_940a,
        0xbd32_c3be_b21e_1e21,
        0x3ff0_0000_0000_0000,
        0xb3a4_757f_fe85_3eb2,
    ),
    (
        "cos",
        0x4137_f7ec_53a8_d491,
        0x3dde_acd7_c60e_9693,
        0x3ff0_0000_0000_0000,
        0xb4ff_a9e7_d8e3_2bb4,
    ),
    (
        "tan",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3fd6_2904_a5d1_65cd,
        0x3c7e_7bc3_b816_9f0c,
    ),
    (
        "tan",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0xbfc2_693f_de5e_04b5,
        0xbc54_053a_af0f_a2dd,
    ),
    (
        "tan",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0xbfdc_d5cd_b683_b3ff,
        0x3c7e_c42f_8aeb_93fc,
    ),
    (
        "tan",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x392f_1976_b7ed_8fbc,
        0xb5c4_cf98_e804_177d,
    ),
    (
        "tan",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3ff8_eb24_5cbe_e3a6,
        0xbc91_d4cd_fd47_7c08,
    ),
    (
        "tan",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3fd3_cc2a_44e2_9998,
        0xbc7f_ad79_4f03_f7ba,
    ),
    (
        "tan",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0xbfed_cfa3_6110_eeec,
        0x3c6f_3cf6_6512_7fd0,
    ),
    (
        "tan",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ddb_7cdf_d9d7_bdbb,
        0x399b_0b0f_fe8f_ae2a,
    ),
    (
        "tan",
        0xc042_c000_0000_0000,
        0x0000_0000_0000_0000,
        0x3fc9_d41b_df55_a40f,
        0xbc33_d7bd_0294_aa4a,
    ),
    (
        "tan",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x3ff5_a0fe_5da9_4891,
        0xbc98_e2bf_cfcb_1d88,
    ),
    (
        "tan",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0xbfe4_beab_a102_0051,
        0xbc8f_dcea_a365_0406,
    ),
    (
        "tan",
        0x4480_f0cf_064d_d592,
        0x0000_0000_0000_0000,
        0xbffa_0f79_c1b6_b257,
        0xbc9d_2781_0f57_37de,
    ),
    (
        "tan",
        0x4630_0000_0000_0000,
        0x0000_0000_0000_0000,
        0xbffc_86fb_fa8c_ecc3,
        0xbc7e_fc7a_1211_5f43,
    ),
    (
        "tan",
        0x7506_ac5b_262c_a1ff,
        0x0000_0000_0000_0000,
        0xc3bd_9ba9_a797_5636,
        0x4057_14cf_36c6_5449,
    ),
    (
        "tan",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0xc6c0_769b_1bb3_8249,
        0x435a_e969_10fa_0b1c,
    ),
    (
        "tan",
        0x4098_8b2f_704a_940a,
        0xbd32_c3be_b21e_1e21,
        0x39d2_17b7_7be7_7f97,
        0x3674_6b51_2b5b_0044,
    ),
    (
        "tan",
        0x4137_f7ec_53a8_d491,
        0x3dde_acd7_c60e_9693,
        0xba7f_d4d6_d0ff_ed66,
        0xb70d_1e65_7752_437b,
    ),
    (
        "atan",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3fd4_978f_a326_9ee1,
        0x3c62_419a_87f2_a457,
    ),
    (
        "atan",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0xbfc2_29ae_c476_38dd,
        0x3c6d_ea9c_4393_d894,
    ),
    (
        "atan",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x3ff3_7e16_3725_3389,
        0xbc80_c18e_4cae_396f,
    ),
    (
        "atan",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x3ff4_33b8_a322_ddd3,
        0xbc97_3aa4_3b49_c930,
    ),
    (
        "atan",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3fe9_21fb_5444_2d18,
        0x3c81_a626_3714_5c06,
    ),
    (
        "atan",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3fd2_a73a_661e_af06,
        0xbc72_f6c1_b5c5_f02c,
    ),
    (
        "atan",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0xbfe4_978f_a326_9ee1,
        0xbc72_419a_87f2_a459,
    ),
    (
        "atan",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ddb_7cdf_d9d7_bdbb,
        0xb99b_0b0f_fe8f_ae2b,
    ),
    (
        "atan",
        0xc042_c000_0000_0000,
        0x0000_0000_0000_0000,
        0xbff8_b4c7_ed7e_4639,
        0xbc94_ac95_44e5_3ac7,
    ),
    (
        "atan",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x3ff9_00cd_feb5_60b4,
        0xbc87_aac6_a071_76a0,
    ),
    (
        "atan",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x3ff9_1c21_5d4d_e84d,
        0xbc9e_f2c1_98ab_c560,
    ),
    (
        "atan",
        0x4480_f0cf_064d_d592,
        0x0000_0000_0000_0000,
        0x3ff9_21fb_5444_2d18,
        0x3c91_a624_4f82_5b06,
    ),
    (
        "atan",
        0x4630_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5bc7,
    ),
    (
        "atan",
        0x7506_ac5b_262c_a1ff,
        0x0000_0000_0000_0000,
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
    ),
    (
        "atan",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x3ff0_0fe9_87ed_02ff,
        0x3c8b_c9d1_5829_7730,
    ),
    (
        "atan",
        0x4098_8b2f_704a_940a,
        0xbd32_c3be_b21e_1e21,
        0x3ff9_1f5f_c8f8_730d,
        0xbc83_7039_d641_ad8d,
    ),
    (
        "atan",
        0x4137_f7ec_53a8_d491,
        0x3dde_acd7_c60e_9693,
        0x3ff9_21fa_a95f_ff9f,
        0xbc95_7386_40b9_e90e,
    ),
    (
        "asin",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3fd5_bfe3_4f05_1112,
        0x3c67_1a9d_e748_fff3,
    ),
    (
        "asin",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0xbfc2_5936_9034_6f6f,
        0xbc68_e90d_ce70_7aa9,
    ),
    (
        "asin",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3fd3_8015_9e14_f6ff,
        0xbc7f_aba2_e018_5aa0,
    ),
    (
        "asin",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0xbfeb_2353_15c6_80dc,
        0xbc50_2b07_b66c_1ac0,
    ),
    (
        "asin",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ddb_7cdf_d9d7_bdbb,
        0x398b_0b0f_fe8f_ae2a,
    ),
    (
        "acos",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3ff3_b202_8082_e8d4,
        0xbc91_3d2d_89d4_c3f8,
    ),
    (
        "acos",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0x3ffb_6d22_264a_bb06,
        0x3c89_868f_d9c4_d6b8,
    ),
    (
        "acos",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3ff4_41f5_ecbe_ef59,
        0xbc96_6ef1_14e5_8d50,
    ),
    (
        "acos",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x4003_59d2_6f93_b6c3,
        0x3c92_a8d6_ae7b_1db3,
    ),
    (
        "acos",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ff9_21fb_543d_4de0,
        0x3c94_08aa_5764_5bf9,
    ),
    (
        "sinh",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3fd5_bb08_5145_2b81,
        0x3c74_05ce_a114_5d2e,
    ),
    (
        "sinh",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0xbfc2_5914_d475_4aed,
        0x3c40_93ad_a4f7_95db,
    ),
    (
        "sinh",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x401e_2d32_46d4_8e70,
        0x3cb9_9cfc_0e0e_906d,
    ),
    (
        "sinh",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x4027_18f4_5d72_e672,
        0x3cb1_327b_d9ce_40b2,
    ),
    (
        "sinh",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3ff2_cd9f_c44e_b982,
        0x3c96_a009_2b4e_196e,
    ),
    (
        "sinh",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3fd3_7d42_af54_b926,
        0x3c65_2958_e197_1270,
    ),
    (
        "sinh",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0xbfea_506b_2dd3_c690,
        0x3c6a_2386_1708_1f6c,
    ),
    (
        "sinh",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ddb_7cdf_d9d7_bdbb,
        0x398b_0b0f_fe8f_8000,
    ),
    (
        "sinh",
        0xc042_c000_0000_0000,
        0x0000_0000_0000_0000,
        0xc341_2939_19c2_01e5,
        0xbfba_ae3c_8b17_927e,
    ),
    (
        "sinh",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x4b01_428e_1cbc_0143,
        0x4784_2bbf_04c6_0d97,
    ),
    (
        "sinh",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x7efd_945d_f4f8_ec8e,
        0x7b81_8339_2684_a46e,
    ),
    (
        "sinh",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x4002_690f_661d_d820,
        0x3ca5_6d7f_60d8_4afa,
    ),
    (
        "cosh",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3ff0_e5ab_9d68_e233,
        0xbc90_d775_f688_7771,
    ),
    (
        "cosh",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0x3ff0_29dd_f6df_7f2a,
        0xbc8b_e595_b34b_7b6e,
    ),
    (
        "cosh",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x401e_70c4_a4f4_1686,
        0xbcbe_c964_0d7e_fc85,
    ),
    (
        "cosh",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x4027_2f14_7fee_4001,
        0x3cab_3656_dda2_b40d,
    ),
    (
        "cosh",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3ff8_b075_51d9_f550,
        0x3c93_0af4_a4f3_6e4c,
    ),
    (
        "cosh",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3ff0_b9b4_e0b6_ec4c,
        0x3c60_ff8c_d34b_4f0b,
    ),
    (
        "cosh",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x3ff4_b705_d1e5_d6a8,
        0xbc9e_1574_85ad_04d7,
    ),
    (
        "cosh",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x3bb7_9ca1_0c92_4400,
    ),
    (
        "cosh",
        0xc042_c000_0000_0000,
        0x0000_0000_0000_0000,
        0x4341_2939_19c2_01e5,
        0x3fba_ae3c_8b17_9283,
    ),
    (
        "cosh",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x4b01_428e_1cbc_0143,
        0x4784_2bbf_04c6_0d97,
    ),
    (
        "cosh",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x7efd_945d_f4f8_ec8e,
        0x7b81_8339_2684_a46e,
    ),
    (
        "cosh",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x4004_12cc_2a8d_4e9e,
        0x3ca2_5b35_8c47_237e,
    ),
    (
        "tanh",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3fd4_93aa_293c_8802,
        0x3c53_3be9_126a_623c,
    ),
    (
        "tanh",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0xbfc2_298e_6160_81f8,
        0x3bd4_a6da_7e59_af87,
    ),
    (
        "tanh",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x3fef_b8f7_6b1e_2ab6,
        0x3c8d_ef26_847c_0e90,
    ),
    (
        "tanh",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x3fef_e175_fa29_2810,
        0xbc84_c2e5_a4a0_1a80,
    ),
    (
        "tanh",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3fe8_5efa_b514_f394,
        0x3c85_618c_b2e6_6a90,
    ),
    (
        "tanh",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3fd2_a4dd_a7d9_14fa,
        0xbc5d_caef_7d34_2698,
    ),
    (
        "tanh",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0xbfe4_5323_e552_f228,
        0xbc73_9d58_32bf_78f9,
    ),
    (
        "tanh",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ddb_7cdf_d9d7_bdbb,
        0xb99b_0b0f_fe8f_a291,
    ),
    (
        "tanh",
        0xc042_c000_0000_0000,
        0x0000_0000_0000_0000,
        0xbff0_0000_0000_0000,
        0x393b_d109_d9d9_4bda,
    ),
    (
        "tanh",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0xa9bb_7f9f_0061_a2d8,
    ),
    (
        "tanh",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x0000_0000_0000_0000,
    ),
    (
        "tanh",
        0x4480_f0cf_064d_d592,
        0x0000_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x0000_0000_0000_0000,
    ),
    (
        "tanh",
        0x4630_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x0000_0000_0000_0000,
    ),
    (
        "tanh",
        0x7506_ac5b_262c_a1ff,
        0x0000_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x0000_0000_0000_0000,
    ),
    (
        "tanh",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x3fed_594f_dae4_82ba,
        0xbc87_c8b0_c360_c30e,
    ),
    (
        "tanh",
        0x4098_8b2f_704a_940a,
        0xbd32_c3be_b21e_1e21,
        0x3ff0_0000_0000_0000,
        0x0000_0000_0000_0000,
    ),
    (
        "tanh",
        0x4137_f7ec_53a8_d491,
        0x3dde_acd7_c60e_9693,
        0x3ff0_0000_0000_0000,
        0x0000_0000_0000_0000,
    ),
    (
        "asinh",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3fd4_f4f1_7995_b748,
        0x3c6d_8c76_9098_fd54,
    ),
    (
        "asinh",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0xbfc2_395d_7593_8b35,
        0x3c63_f6be_04f1_f690,
    ),
    (
        "asinh",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x3ffb_9b2a_bdf5_c077,
        0x3c82_234b_e095_6f56,
    ),
    (
        "asinh",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x3ffd_cbf6_9f10_006d,
        0xbc76_9091_f788_ebd4,
    ),
    (
        "asinh",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3fec_3436_6179_d427,
        0xbc79_f270_5ac6_d3a8,
    ),
    (
        "asinh",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3fd2_ec4e_a565_fcc1,
        0xbc6f_2afd_5259_90ed,
    ),
    (
        "asinh",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0xbfe6_2e42_fefa_39ef,
        0xbc7a_bc9e_3b39_803f,
    ),
    (
        "asinh",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ddb_7cdf_d9d7_bdbb,
        0xb98b_0b0f_fe8f_c8c6,
    ),
    (
        "asinh",
        0xc042_c000_0000_0000,
        0x0000_0000_0000_0000,
        0xc011_454a_31ea_dd04,
        0xbcbc_2095_d5d2_6800,
    ),
    (
        "asinh",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x4016_0943_fbda_e50b,
        0xbcb9_2fe6_59ba_192c,
    ),
    (
        "asinh",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x401c_fa16_e95e_6e7a,
        0xbcb9_4c72_3072_a0e5,
    ),
    (
        "asinh",
        0x4480_f0cf_064d_d592,
        0x0000_0000_0000_0000,
        0x4049_accd_6e15_3de5,
        0x3ce4_318a_f7aa_51a0,
    ),
    (
        "asinh",
        0x4630_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x4051_8080_dd31_71b7,
        0xbcef_e72b_2544_a0ce,
    ),
    (
        "asinh",
        0x7506_ac5b_262c_a1ff,
        0x0000_0000_0000_0000,
        0x4082_6c30_8a4a_3bcd,
        0x3d26_a25b_cac6_5934,
    ),
    (
        "asinh",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x3ff3_bc04_e847_ec05,
        0x3c8d_2e50_aa15_a063,
    ),
    (
        "asinh",
        0x4098_8b2f_704a_940a,
        0xbd32_c3be_b21e_1e21,
        0x4020_1adf_594b_4ba3,
        0xbcc2_ba7b_d891_52b8,
    ),
    (
        "asinh",
        0x4137_f7ec_53a8_d491,
        0x3dde_acd7_c60e_9693,
        0x402d_eba4_a2ac_f411,
        0xbca0_4d6d_46c2_1db1,
    ),
    (
        "acosh",
        0x4005_bf0a_8b14_5769,
        0x3ca4_d57e_e2b1_013a,
        0x3ffa_84ee_f633_26e8,
        0xbc96_52b6_5817_b85f,
    ),
    (
        "acosh",
        0x4009_21fb_5444_2d18,
        0x3ca1_a626_3314_5c07,
        0x3ffc_fc02_f901_06c2,
        0xbc95_95e1_3d90_4b01,
    ),
    (
        "acosh",
        0x3ff0_0000_0000_0000,
        0x3af0_0000_0000_0000,
        0x3d76_a09e_667f_3bcd,
        0xba1b_dd34_142b_124d,
    ),
    (
        "acosh",
        0x405e_dd2f_1a9f_be77,
        0x0000_0000_0000_0000,
        0x4016_093b_6252_1257,
        0x3c98_9a27_4676_02c4,
    ),
    (
        "acosh",
        0x4085_e000_0000_0000,
        0x0000_0000_0000_0000,
        0x401c_fa16_a4e3_f3ea,
        0xbcb6_ebe9_c379_72ca,
    ),
    (
        "acosh",
        0x4480_f0cf_064d_d592,
        0x0000_0000_0000_0000,
        0x4049_accd_6e15_3de5,
        0x3ce4_318a_f7aa_51a0,
    ),
    (
        "acosh",
        0x4630_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x4051_8080_dd31_71b7,
        0xbcef_e72b_2544_a0ce,
    ),
    (
        "acosh",
        0x7506_ac5b_262c_a1ff,
        0x0000_0000_0000_0000,
        0x4082_6c30_8a4a_3bcd,
        0x3d26_a25b_cac6_5934,
    ),
    (
        "acosh",
        0x3ff9_21fb_5444_2d18,
        0x3c91_a626_3314_5c07,
        0x3ff0_5f23_c6cb_af31,
        0xbc55_d791_6a01_3fd4,
    ),
    (
        "acosh",
        0x4098_8b2f_704a_940a,
        0xbd32_c3be_b21e_1e21,
        0x4020_1adf_527e_9c5b,
        0x3cce_edeb_baf0_63d0,
    ),
    (
        "acosh",
        0x4137_f7ec_53a8_d491,
        0x3dde_acd7_c60e_9693,
        0x402d_eba4_a2ac_f39f,
        0xbcb2_1299_e9a7_7ba8,
    ),
    (
        "atanh",
        0x3fd5_5555_5555_5555,
        0x3c75_5555_5555_5555,
        0x3fd6_2e42_fefa_39ef,
        0x3c6a_bc9e_3b39_803e,
    ),
    (
        "atanh",
        0xbfc2_4924_9249_2492,
        0xbc62_4924_9249_2492,
        0xbfc2_6962_1134_db92,
        0xbc6e_0efa_dd9d_b02b,
    ),
    (
        "atanh",
        0x3fd3_3333_3333_3333,
        0x0000_0000_0000_0000,
        0x3fd3_cf2b_5061_7c95,
        0x3c58_f7a0_70c5_26f7,
    ),
    (
        "atanh",
        0xbfe8_0000_0000_0000,
        0x0000_0000_0000_0000,
        0xbfef_2272_ae32_5a57,
        0xbc85_1bda_525b_3c97,
    ),
    (
        "atanh",
        0x3ddb_7cdf_d9d7_bdbb,
        0x0000_0000_0000_0000,
        0x3ddb_7cdf_d9d7_bdbb,
        0x399b_0b0f_fe8f_bab0,
    ),
];

/// The digest of [`digest`] on the same machine.
const GOLDEN_DIGEST: u64 = 0x0c46_3638_dd42_abc9;

#[test]
fn golden_bits() {
    let funs = all_functions();
    for &(name, xh, xl, zh, zl) in &GOLDEN {
        let f = funs.iter().find(|f| f.name == name).expect("a function");
        let x = Dd::from_parts(f64::from_bits(xh), f64::from_bits(xl));
        let z = (f.dd)(x);
        assert_eq!(
            (z.hi().to_bits(), z.lo().to_bits()),
            (zh, zl),
            "{name}({x:?}) = {z:?}"
        );
    }
    assert_eq!(digest(), GOLDEN_DIGEST);
}
