//! The oracles of Chapter 11.11 (L2.4; the contract's §4.1) checked
//! against each other, and the 19 constants of `DoubleDouble` checked bit
//! for bit against both.
//!
//! The primary oracle (`dashu-float`) computes with Ziv's loop and is young
//! code; its changelog lists recent fixes in exactly these functions. The
//! secondary one (`tests/support`) is written here from series with
//! explicit remainders, in exact dyadic arithmetic rounded outward. Here
//! the value of the first must fall inside the enclosure of the second
//! (widened by the first's own rounding, `2^(1 − P)` relative) at about 300
//! points: the directed points of the tests of the functions and a random
//! sample. The secondary oracle is slow and is not used elsewhere.

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::disallowed_methods
)]

mod support;

use support::{Dy, F, Iv};
use xingo_extprec::DoubleDouble;

type Dd = DoubleDouble;

const P: usize = 640;

fn f(x: Dd) -> F {
    support::fdd(x.hi(), x.lo(), P)
}

/// The first oracle's value lies in the second's enclosure, widened by
/// `2^(8 − P)|v|`.
fn agrees(name: &str, x: Dd, first: &F, second: &Iv) {
    let v = support::dy(first);
    let slack = v.abs().scale(8 - P as isize);
    assert!(
        second.widen(&slack).contains(&v),
        "{name}({:e} + {:e}): dashu-float {:e} outside [{:e}, {:e}]",
        x.hi(),
        x.lo(),
        v.approx(),
        second.lo.approx(),
        second.hi.approx()
    );
}

/// A deterministic generator (xorshift), as in the other tests.
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
    /// In `[a, b)`, with a random trailing component.
    fn dd(&mut self, a: f64, b: f64) -> Dd {
        let h = a + (b - a) * self.unit();
        let ulp = f64::from_bits(h.abs().to_bits() + 1) - h.abs();
        Dd::from_sum(h, (self.unit() - 0.5) * ulp)
    }
    /// Magnitude log-uniform in `[2^e0, 2^e1)`, random sign.
    fn dd_log(&mut self, e0: i32, e1: i32) -> Dd {
        let e = e0 + (self.next() % (e1 - e0) as u64) as i32;
        let x = self.dd(1.0, 2.0).ldexp(e);
        if self.next() & 1 == 1 { -x } else { x }
    }
}

#[test]
fn the_constants_are_the_nearest_double_doubles() {
    let p = 2400;
    let one = F::from(1).with_precision(p).value();
    let two = F::from(2).with_precision(p).value();
    let ten = F::from(10).with_precision(p).value();
    let pi = F::pi(p);
    let ln2 = two.ln();
    let ln10 = ten.ln();
    let sqrt2 = two.sqrt();
    let first: [(&str, Dd, F); 19] = [
        ("PI", Dd::PI, pi.clone()),
        ("TAU", Dd::TAU, pi.clone() * F::from(2)),
        ("FRAC_PI_2", Dd::FRAC_PI_2, pi.clone() / F::from(2)),
        ("FRAC_PI_3", Dd::FRAC_PI_3, pi.clone() / F::from(3)),
        ("FRAC_PI_4", Dd::FRAC_PI_4, pi.clone() / F::from(4)),
        ("FRAC_PI_6", Dd::FRAC_PI_6, pi.clone() / F::from(6)),
        ("FRAC_PI_8", Dd::FRAC_PI_8, pi.clone() / F::from(8)),
        ("FRAC_1_PI", Dd::FRAC_1_PI, one.clone() / pi.clone()),
        ("FRAC_2_PI", Dd::FRAC_2_PI, two.clone() / pi.clone()),
        (
            "FRAC_2_SQRT_PI",
            Dd::FRAC_2_SQRT_PI,
            two.clone() / pi.sqrt(),
        ),
        ("SQRT_2", Dd::SQRT_2, sqrt2.clone()),
        ("FRAC_1_SQRT_2", Dd::FRAC_1_SQRT_2, one.clone() / sqrt2),
        ("E", Dd::E, one.exp()),
        ("LOG2_E", Dd::LOG2_E, one.clone() / ln2.clone()),
        ("LOG2_10", Dd::LOG2_10, ln10.clone() / ln2.clone()),
        ("LOG10_E", Dd::LOG10_E, one / ln10.clone()),
        ("LOG10_2", Dd::LOG10_2, ln2.clone() / ln10.clone()),
        ("LN_2", Dd::LN_2, ln2),
        ("LN_10", Dd::LN_10, ln10),
    ];
    let spi = support::pi();
    let sln2 = support::ln2();
    let sln10 = support::ln(&Dy::int(10));
    let ssqrt2 = support::sqrt(&Iv::int(2));
    let s1 = Iv::int(1);
    let s2 = Iv::int(2);
    let second: [Iv; 19] = [
        spi.clone(),
        spi.scale(1),
        spi.scale(-1),
        spi.div(&Iv::int(3)),
        spi.scale(-2),
        spi.div(&Iv::int(6)),
        spi.scale(-3),
        s1.div(spi),
        s2.div(spi),
        s2.div(&support::sqrt(spi)),
        ssqrt2.clone(),
        s1.div(&ssqrt2),
        support::exp(&Dy::int(1)),
        s1.div(sln2),
        sln10.div(sln2),
        s1.div(&sln10),
        sln2.div(&sln10),
        sln2.clone(),
        sln10.clone(),
    ];
    for ((name, c, v), iv) in first.iter().zip(&second) {
        let got = (c.hi().to_bits(), c.lo().to_bits());
        let a = support::nearest_dd(&support::dy(v));
        assert_eq!(
            got,
            (a.0.to_bits(), a.1.to_bits()),
            "{name} against dashu-float"
        );
        let b = iv.nearest_dd().expect("a narrow enclosure");
        assert_eq!(
            got,
            (b.0.to_bits(), b.1.to_bits()),
            "{name} against the enclosure"
        );
        assert!(c.check_invariants().is_ok(), "{name}");
    }
}

/// The arguments of the cross-check: directed points of each family and a
/// random sample.
fn points(seed: u64, directed: &[Dd], fam: impl Fn(&mut Rng) -> Dd, n: usize) -> Vec<Dd> {
    let mut rng = Rng(seed);
    let mut v = directed.to_vec();
    v.extend((0..n).map(|_| fam(&mut rng)));
    v
}

#[test]
#[allow(clippy::too_many_lines)]
fn the_oracles_agree() {
    let d = Dd::from_f64;
    let ln2 = Dd::LN_2;
    // exp: zeros of k, the overflow and underflow thresholds, tiny values.
    let exp_pts = points(
        1,
        &[
            d(1.0),
            d(-1.0),
            d(2f64.powi(-60)),
            d(-2f64.powi(-600)),
            ln2 * 0.5,
            ln2 * 1.5,
            ln2 * 100.5,
            d(709.782_712_893_384),
            d(709.78),
            d(-708.396_4),
            d(-745.133),
            d(1.0 / 128.0),
            d(-0.5),
            d(0.75),
        ],
        |r| r.dd(-745.0, 709.0),
        55,
    );
    for x in &exp_pts {
        agrees(
            "exp",
            *x,
            &f(*x).exp(),
            &support::exp(&Dy::dd(x.hi(), x.lo())),
        );
    }
    // ln: near 1, the table boundaries, powers of two, extremes.
    let ln_pts = points(
        2,
        &[
            d(1.0) + 2f64.powi(-80),
            d(1.0) - 2f64.powi(-30),
            d(1.0 + 1.0 / 128.0),
            d(1.5),
            d(0.75),
            d(f64::from_bits(1)),
            d(f64::MIN_POSITIVE),
            d(f64::MAX),
            Dd::E,
            d(10.0),
            d(core::f64::consts::SQRT_2),
            d(0.7),
        ],
        |r| {
            if r.next() & 1 == 0 {
                r.dd_log(-1000, 1000).abs()
            } else {
                r.dd(0.5, 2.0)
            }
        },
        55,
    );
    for x in &ln_pts {
        agrees("ln", *x, &f(*x).ln(), &support::ln(&Dy::dd(x.hi(), x.lo())));
    }
    // sin and cos: near multiples of π/2, the CW/PH boundary, huge.
    let mut trig = vec![
        Dd::FRAC_PI_2,
        Dd::PI,
        Dd::FRAC_PI_4,
        d(1e-10),
        d(0.5),
        d(1e6),
        d(2f64.powi(52)),
        d(2f64.powi(53)),
        d(2f64.powi(100)),
        d(2f64.powi(500)),
        d(f64::MAX),
        d(6_381_956_970_095_103.0 * 2f64.powi(797)),
    ];
    for k in [3.0, 1000.0, 1e6, 2f64.powi(20)] {
        trig.push(Dd::FRAC_PI_2 * k);
    }
    let trig_pts = points(3, &trig, |r| r.dd_log(-20, 60), 45);
    for x in &trig_pts {
        let (s, c) = support::sin_cos(&Dy::dd(x.hi(), x.lo()));
        let (fs, fc) = f(*x).sin_cos();
        agrees("sin", *x, &fs, &s);
        agrees("cos", *x, &fc, &c);
    }
    // atan: ±1, the table boundaries, tiny and huge.
    let atan_pts = points(
        4,
        &[
            d(1.0),
            d(-1.0),
            d(1.0 / 64.0),
            d(8.0),
            d(1e-300),
            d(1e300),
            d(0.5),
            d(3.0),
        ],
        |r| r.dd_log(-30, 30),
        45,
    );
    for x in &atan_pts {
        agrees(
            "atan",
            *x,
            &f(*x).atan(),
            &support::atan(&Dy::dd(x.hi(), x.lo())),
        );
    }
    let n = exp_pts.len() + ln_pts.len() + 2 * trig_pts.len() + atan_pts.len();
    assert!(n >= 300, "{n} points");
}
