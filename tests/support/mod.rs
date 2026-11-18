//! The two oracles of the elementary functions (Chapter 11.11, L2.4; the
//! contract's §4.1), shared by `tests/elementary.rs`,
//! `tests/elementary_oracle.rs` and the unit test of the private tables.
//!
//! * **Primary:** `dashu-float`, evaluated at `P` bits with `P` at least
//!   256 more than the argument spans, so that the argument enters exactly
//!   and the oracle's own error is far below the bounds tested.
//! * **Secondary, independent:** enclosures `[lo, hi]` with dyadic
//!   endpoints (exact rationals `m·2^e` over `dashu-int`), rounded outward
//!   after every operation, built from series whose remainders are bounded
//!   explicitly: `eˣ` (Taylor after halving, then squaring), `ln` (`2 atanh`),
//!   `π` (Machin), `sin`/`cos` (Taylor after reduction by that `π`) and
//!   `atan` (halving, then Taylor). Nothing here calls `dashu-float`.
//!
//! The module works on `f64` pairs, not on `DoubleDouble`, so that the
//! crate's own unit tests can include it too.

#![allow(dead_code)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::needless_pass_by_value,
    clippy::return_self_not_must_use,
    clippy::disallowed_methods
)]

use core::cmp::Ordering;
use std::sync::OnceLock;

use dashu_float::FBig;
use dashu_float::round::mode::HalfEven;
use dashu_int::ops::BitTest;
use dashu_int::{IBig, UBig};

// ---------------------------------------------------------------------
// Exact dyadic rationals
// ---------------------------------------------------------------------

/// The exact value `m · 2^e`.
#[derive(Clone, Debug)]
pub struct Dy {
    pub m: IBig,
    pub e: isize,
}

fn mag(m: &IBig) -> (bool, UBig) {
    let neg = *m < IBig::ZERO;
    let u: UBig = if neg { -m.clone() } else { m.clone() }
        .try_into()
        .expect("non-negative");
    (neg, u)
}

fn signed(neg: bool, u: UBig) -> IBig {
    let i = IBig::from(u);
    if neg { -i } else { i }
}

/// `⌊m / 2^s⌋` (`up = false`) or `⌈m / 2^s⌉` (`up = true`).
fn shr_round(m: &IBig, s: usize, up: bool) -> IBig {
    let (neg, u) = mag(m);
    let q = &u >> s;
    let exact = (&q << s) == u;
    // For a positive m, floor is the truncation; for a negative one, ceil is.
    let away = !exact && (up != neg);
    let q = if away { q + UBig::ONE } else { q };
    signed(neg, q)
}

impl Dy {
    pub fn zero() -> Self {
        Dy {
            m: IBig::ZERO,
            e: 0,
        }
    }

    pub fn int(n: i64) -> Self {
        Dy {
            m: IBig::from(n),
            e: 0,
        }
    }

    pub fn pow2(k: isize) -> Self {
        Dy { m: IBig::ONE, e: k }
    }

    /// The exact value of a finite `f64`.
    pub fn f(x: f64) -> Self {
        assert!(x.is_finite(), "a finite f64");
        if x == 0.0 {
            return Dy::zero();
        }
        let bits = x.to_bits();
        let be = ((bits >> 52) & 0x7ff) as isize;
        let frac = bits & ((1 << 52) - 1);
        let (m, e) = if be == 0 {
            (frac, -1074)
        } else {
            (frac | (1 << 52), be - 1075)
        };
        let m = IBig::from(m);
        Dy {
            m: if x < 0.0 { -m } else { m },
            e,
        }
    }

    /// The exact value of the pair `hi + lo`.
    pub fn dd(hi: f64, lo: f64) -> Self {
        Dy::f(hi).add(&Dy::f(lo))
    }

    pub fn is_zero(&self) -> bool {
        self.m == IBig::ZERO
    }

    pub fn is_neg(&self) -> bool {
        self.m < IBig::ZERO
    }

    pub fn neg(&self) -> Self {
        Dy {
            m: -self.m.clone(),
            e: self.e,
        }
    }

    pub fn abs(&self) -> Self {
        if self.is_neg() {
            self.neg()
        } else {
            self.clone()
        }
    }

    pub fn add(&self, o: &Dy) -> Self {
        if self.is_zero() {
            return o.clone();
        }
        if o.is_zero() {
            return self.clone();
        }
        let e = self.e.min(o.e);
        let a = self.m.clone() << (self.e - e) as usize;
        let b = o.m.clone() << (o.e - e) as usize;
        Dy { m: a + b, e }
    }

    pub fn sub(&self, o: &Dy) -> Self {
        self.add(&o.neg())
    }

    pub fn mul(&self, o: &Dy) -> Self {
        Dy {
            m: self.m.clone() * o.m.clone(),
            e: self.e + o.e,
        }
    }

    pub fn scale(&self, k: isize) -> Self {
        Dy {
            m: self.m.clone(),
            e: self.e + k,
        }
    }

    pub fn cmp(&self, o: &Dy) -> Ordering {
        let d = self.sub(o);
        d.m.cmp(&IBig::ZERO)
    }

    pub fn lt(&self, o: &Dy) -> bool {
        self.cmp(o) == Ordering::Less
    }

    pub fn le(&self, o: &Dy) -> bool {
        self.cmp(o) != Ordering::Greater
    }

    /// `⌊log₂|x|⌋` for `x ≠ 0`.
    pub fn ilog2(&self) -> isize {
        let (_, u) = mag(&self.m);
        u.bit_len() as isize - 1 + self.e
    }

    /// Rounded to `bits` significant bits, down (`up = false`) or up.
    pub fn round(&self, bits: usize, up: bool) -> Self {
        let (_, u) = mag(&self.m);
        let len = u.bit_len();
        if len <= bits {
            return self.clone();
        }
        let s = len - bits;
        Dy {
            m: shr_round(&self.m, s, up),
            e: self.e + s as isize,
        }
    }

    /// `self / o` rounded to `bits` significant bits, down or up.
    pub fn div(&self, o: &Dy, bits: usize, up: bool) -> Self {
        assert!(!o.is_zero(), "division by zero");
        if self.is_zero() {
            return Dy::zero();
        }
        let (na, ua) = mag(&self.m);
        let (nb, ub) = mag(&o.m);
        let neg = na != nb;
        let k = (bits + 2 + ub.bit_len()).saturating_sub(ua.bit_len());
        let num = &ua << k;
        let q = &num / &ub;
        let exact = &q * &ub == num;
        // Round the magnitude toward zero or away, as the sign requires.
        let away = !exact && (up != neg);
        let q = if away { q + UBig::ONE } else { q };
        Dy {
            m: signed(neg, q),
            e: self.e - o.e - k as isize,
        }
        .round(bits, up)
    }

    /// `RN(x)` as an `f64` (ties to even; subnormal results included; a
    /// zero result is `+0`). Panics on overflow.
    pub fn rn(&self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        let (neg, u) = mag(&self.m);
        // The quantum of the result: ulp of a 53-bit significand, or η.
        let qe = (self.ilog2() - 52).max(-1074);
        let (q, e) = if self.e < qe {
            let s = (qe - self.e) as usize;
            let q = &u >> s;
            let rem = &u - (&q << s);
            let half = UBig::ONE << (s - 1);
            let up = rem > half || (rem == half && (&q & UBig::ONE) == UBig::ONE);
            (if up { q + UBig::ONE } else { q }, qe)
        } else {
            (u, self.e)
        };
        let q = u64::try_from(q).expect("54 bits") as f64; // exact
        let v = ldexp_f(q, e);
        assert!(v.is_finite(), "rn overflows: {self:?}");
        if v == 0.0 {
            0.0
        } else if neg {
            -v
        } else {
            v
        }
    }

    /// An approximation, for reports.
    pub fn approx(&self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        let r = self.round(60, false);
        let (neg, u) = mag(&r.m);
        let v = ldexp_f(u64::try_from(u).expect("60 bits") as f64, r.e);
        if neg { -v } else { v }
    }
}

/// `2^k` as an `f64` (an exact product of powers of two; 0 or ∞ outside).
pub fn pow2f(k: isize) -> f64 {
    let mut v = 1.0;
    let mut k = k;
    while k > 1000 {
        v *= f64::from_bits((1023 + 1000) << 52);
        k -= 1000;
    }
    while k < -1000 {
        v *= f64::from_bits((1023 - 1000) << 52);
        k += 1000;
    }
    v * f64::from_bits(((1023 + k) as u64) << 52)
}

/// `x · 2^k`, by steps that stay exact while the result is normal.
pub fn ldexp_f(x: f64, k: isize) -> f64 {
    let mut v = x;
    let mut k = k;
    while k > 1000 {
        v *= f64::from_bits((1023 + 1000) << 52);
        k -= 1000;
    }
    while k < -1000 {
        v *= f64::from_bits((1023 - 1000) << 52);
        k += 1000;
    }
    v * f64::from_bits(((1023 + k) as u64) << 52)
}

/// The nearest double-double `(RN(c), RN(c − RN(c)))` of an exact value.
pub fn nearest_dd(c: &Dy) -> (f64, f64) {
    let h = c.rn();
    let rest = c.sub(&Dy::f(h));
    (h, rest.rn())
}

// ---------------------------------------------------------------------
// Enclosures
// ---------------------------------------------------------------------

/// A closed interval `[lo, hi]` with dyadic endpoints.
#[derive(Clone, Debug)]
pub struct Iv {
    pub lo: Dy,
    pub hi: Dy,
}

/// The working precision of the enclosures, in significant bits.
pub const BITS: usize = 1600;

impl Iv {
    pub fn exact(x: Dy) -> Self {
        Iv {
            lo: x.clone(),
            hi: x,
        }
    }

    pub fn int(n: i64) -> Self {
        Iv::exact(Dy::int(n))
    }

    pub fn round(&self) -> Self {
        Iv {
            lo: self.lo.round(BITS, false),
            hi: self.hi.round(BITS, true),
        }
    }

    pub fn add(&self, o: &Iv) -> Self {
        Iv {
            lo: self.lo.add(&o.lo),
            hi: self.hi.add(&o.hi),
        }
        .round()
    }

    pub fn sub(&self, o: &Iv) -> Self {
        Iv {
            lo: self.lo.sub(&o.hi),
            hi: self.hi.sub(&o.lo),
        }
        .round()
    }

    pub fn neg(&self) -> Self {
        Iv {
            lo: self.hi.neg(),
            hi: self.lo.neg(),
        }
    }

    pub fn mul(&self, o: &Iv) -> Self {
        let c = [
            self.lo.mul(&o.lo),
            self.lo.mul(&o.hi),
            self.hi.mul(&o.lo),
            self.hi.mul(&o.hi),
        ];
        let mut lo = c[0].clone();
        let mut hi = c[0].clone();
        for v in &c[1..] {
            if v.lt(&lo) {
                lo = v.clone();
            }
            if hi.lt(v) {
                hi = v.clone();
            }
        }
        Iv { lo, hi }.round()
    }

    pub fn scale(&self, k: isize) -> Self {
        Iv {
            lo: self.lo.scale(k),
            hi: self.hi.scale(k),
        }
    }

    /// Division by an interval that does not contain zero.
    pub fn div(&self, o: &Iv) -> Self {
        assert!(
            !o.lo.is_neg() && !o.lo.is_zero() || o.hi.is_neg(),
            "divisor contains zero"
        );
        let c = [
            (&self.lo, &o.lo),
            (&self.lo, &o.hi),
            (&self.hi, &o.lo),
            (&self.hi, &o.hi),
        ];
        let downs: Vec<Dy> = c.iter().map(|(a, b)| a.div(b, BITS, false)).collect();
        let ups: Vec<Dy> = c.iter().map(|(a, b)| a.div(b, BITS, true)).collect();
        let mut lo = downs[0].clone();
        for v in &downs[1..] {
            if v.lt(&lo) {
                lo = v.clone();
            }
        }
        let mut hi = ups[0].clone();
        for v in &ups[1..] {
            if hi.lt(v) {
                hi = v.clone();
            }
        }
        Iv { lo, hi }
    }

    /// Widened by `±r` (`r ≥ 0`).
    pub fn widen(&self, r: &Dy) -> Self {
        Iv {
            lo: self.lo.sub(r),
            hi: self.hi.add(r),
        }
        .round()
    }

    pub fn mag_hi(&self) -> Dy {
        let a = self.lo.abs();
        let b = self.hi.abs();
        if a.lt(&b) { b } else { a }
    }

    pub fn contains(&self, x: &Dy) -> bool {
        self.lo.le(x) && x.le(&self.hi)
    }

    /// The nearest double-double, if the enclosure is narrow enough to
    /// decide it.
    pub fn nearest_dd(&self) -> Option<(f64, f64)> {
        let a = nearest_dd(&self.lo);
        let b = nearest_dd(&self.hi);
        (a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()).then_some(a)
    }

    /// The bits `[1, 64n]` of the fraction of a value in `[0, 1)`, as
    /// big-endian words, if decided by the enclosure.
    pub fn fixed_words(&self, n: usize) -> Option<Vec<u64>> {
        let f = |d: &Dy| -> Vec<u64> {
            // ⌊d · 2^(64n)⌋
            let s = d.scale(64 * n as isize);
            let i = if s.e >= 0 {
                s.m.clone() << s.e as usize
            } else {
                shr_round(&s.m, (-s.e) as usize, false)
            };
            let (_, mut u) = mag(&i);
            let mut w = vec![0u64; n];
            for k in (0..n).rev() {
                w[k] = u64::try_from(&u & UBig::from(u64::MAX)).expect("a word");
                u >>= 64;
            }
            w
        };
        let a = f(&self.lo);
        let b = f(&self.hi);
        (a == b).then_some(a)
    }
}

/// `Σ_{n ≥ 0} terms` for a series whose terms, from `first`, satisfy
/// `|t_{n+1}| ≤ ρ|t_n|` with `ρ ≤ 1/2` once `next` stops shrinking them:
/// the sum of the computed terms plus `±2|last term|`.
fn sum_series(mut t: Iv, mut next: impl FnMut(usize, &Iv) -> Iv, tiny: &Dy) -> Iv {
    let mut s = t.clone();
    let mut n = 0;
    loop {
        n += 1;
        t = next(n, &t);
        s = s.add(&t);
        if t.mag_hi().lt(tiny) {
            return s.widen(&t.mag_hi().scale(1));
        }
    }
}

/// `2^-(BITS + 40)`, the size below which a series term is dropped.
fn tiny() -> Dy {
    Dy::pow2(-(BITS as isize) - 40)
}

/// `π` by Machin's formula, `16 atan(1/5) − 4 atan(1/239)`.
pub fn pi() -> &'static Iv {
    static PI: OnceLock<Iv> = OnceLock::new();
    PI.get_or_init(|| {
        let atan_inv = |m: i64| -> Iv {
            // Σ (−1)^n / ((2n + 1) m^(2n+1)); alternating, decreasing.
            let m2 = Iv::int(m * m);
            let p = Iv::int(1).div(&Iv::int(m));
            let mut pow = p.clone();
            let mut s = p;
            let mut n: i64 = 0;
            loop {
                n += 1;
                pow = pow.div(&m2);
                let t = pow.div(&Iv::int(2 * n + 1));
                s = if n % 2 == 1 { s.sub(&t) } else { s.add(&t) };
                if t.mag_hi().lt(&tiny()) {
                    return s.widen(&t.mag_hi());
                }
            }
        };
        atan_inv(5).scale(2).sub(&atan_inv(239)).scale(2)
    })
}

/// `e^x` for an exact `x`, `|x| ≤ 1100`.
pub fn exp(x: &Dy) -> Iv {
    if x.is_zero() {
        return Iv::int(1);
    }
    if x.is_neg() {
        return Iv::int(1).div(&exp(&x.neg()));
    }
    // y = x / 2^s with |y| ≤ 2^-8.
    let s = (x.ilog2() + 9).max(0);
    let y = Iv::exact(x.scale(-s));
    // Taylor; the terms decrease by at least 2^-8 from the first.
    let mut v = sum_series(
        Iv::int(1),
        |n, t| t.mul(&y).div(&Iv::int(n as i64)),
        &tiny(),
    );
    for _ in 0..s {
        v = v.mul(&v);
    }
    v
}

/// `e^x` for an enclosure `x` (monotone).
pub fn exp_iv(x: &Iv) -> Iv {
    Iv {
        lo: exp(&x.lo).lo,
        hi: exp(&x.hi).hi,
    }
}

/// `2 atanh(t) = Σ 2 t^(2n+1)/(2n+1)` for an enclosure `t`, `|t| ≤ 1/3`.
fn two_atanh(t: &Iv) -> Iv {
    let t2 = t.mul(t);
    let mut pow = t.clone();
    let mut s = t.clone();
    let mut n: i64 = 0;
    loop {
        n += 1;
        pow = pow.mul(&t2);
        let term = pow.div(&Iv::int(2 * n + 1));
        s = s.add(&term);
        if term.mag_hi().lt(&tiny()) {
            // Remainder ≤ |term| · t²/(1 − t²) ≤ |term|/8.
            return s.widen(&term.mag_hi()).scale(1);
        }
    }
}

/// `ln 2 = 2 atanh(1/3)`.
pub fn ln2() -> &'static Iv {
    static LN2: OnceLock<Iv> = OnceLock::new();
    LN2.get_or_init(|| two_atanh(&Iv::int(1).div(&Iv::int(3))))
}

/// `ln x` for an exact `x > 0`.
pub fn ln(x: &Dy) -> Iv {
    assert!(!x.is_neg() && !x.is_zero(), "ln of a non-positive value");
    // x = 2^k y with y ∈ [2/3, 4/3).
    let mut k = x.ilog2();
    let mut y = x.scale(-k); // [1, 2)
    if Dy::int(4).lt(&y.scale(2).add(&y)) {
        // 3y > 4
        y = y.scale(-1);
        k += 1;
    }
    let yi = Iv::exact(y.clone());
    let t = yi.sub(&Iv::int(1)).div(&yi.add(&Iv::int(1)));
    let ly = if y.cmp(&Dy::int(1)) == Ordering::Equal {
        Iv::int(0)
    } else {
        two_atanh(&t)
    };
    ln2().mul(&Iv::int(k as i64)).add(&ly)
}

/// `(sin r, cos r)` by Taylor, `|r| ≤ 1`.
fn sin_cos_taylor(r: &Iv) -> (Iv, Iv) {
    let r2 = r.mul(r);
    // sin: t_{n+1} = −t_n r²/((2n)(2n+1)); cos: −t_n r²/((2n−1)(2n)).
    let s = sum_series(
        r.clone(),
        |n, t| t.mul(&r2).div(&Iv::int((2 * n * (2 * n + 1)) as i64)).neg(),
        &tiny(),
    );
    let c = sum_series(
        Iv::int(1),
        |n, t| {
            t.mul(&r2)
                .div(&Iv::int(((2 * n - 1) * (2 * n)) as i64))
                .neg()
        },
        &tiny(),
    );
    (s, c)
}

/// `(sin x, cos x)` for an exact finite `x`.
pub fn sin_cos(x: &Dy) -> (Iv, Iv) {
    let pio2 = pi().scale(-1);
    // k = nearest integer to x / (π/2), from the enclosure's midpoint.
    let q = Iv::exact(x.clone()).div(&pio2);
    let mid = q.lo.add(&q.hi).scale(-1);
    // k = ⌊mid + 1/2⌋, through the exponent.
    let kmid = mid.add(&Dy::pow2(-1));
    let k = if kmid.e >= 0 {
        kmid.m.clone() << kmid.e as usize
    } else {
        shr_round(&kmid.m, (-kmid.e) as usize, false)
    };
    let kd = Dy { m: k.clone(), e: 0 };
    let r = Iv::exact(x.clone()).sub(&pio2.mul(&Iv::exact(kd)));
    let (s, c) = sin_cos_taylor(&r);
    let quadrant = {
        let four = IBig::from(4);
        let m = k % four.clone();
        let m = if m < IBig::ZERO { m + four } else { m };
        i64::try_from(m).expect("small")
    };
    match quadrant {
        0 => (s, c),
        1 => (c, s.neg()),
        2 => (s.neg(), c.neg()),
        _ => (c.neg(), s),
    }
}

/// `√x` for an enclosure with `lo > 0`.
pub fn sqrt(x: &Iv) -> Iv {
    let f = |d: &Dy, up: bool| -> Dy {
        // ⌊√(m 2^e)⌋ at BITS bits: scale to an even exponent.
        let mut e = d.e - 2 * BITS as isize - 4;
        if e % 2 != 0 {
            e -= 1;
        }
        let i = d.m.clone() << (d.e - e) as usize;
        let (_, u) = mag(&i);
        let r = dashu_int::ops::SquareRoot::sqrt(&u);
        let exact = &r * &r == u;
        let r = if up && !exact { r + UBig::ONE } else { r };
        Dy {
            m: IBig::from(r),
            e: e / 2,
        }
    };
    Iv {
        lo: f(&x.lo, false),
        hi: f(&x.hi, true),
    }
    .round()
}

/// `atan x` for an exact finite `x`.
pub fn atan(x: &Dy) -> Iv {
    if x.is_neg() {
        return atan(&x.neg()).neg();
    }
    if x.is_zero() {
        return Iv::int(0);
    }
    let xi = Iv::exact(x.clone());
    if Dy::int(1).lt(x) {
        return pi().scale(-1).sub(&atan_iv(&Iv::int(1).div(&xi)));
    }
    atan_iv(&xi)
}

/// `atan` of an enclosure in `[0, 1]`: three halvings
/// `atan t = 2 atan(t / (1 + √(1 + t²)))`, then the alternating Taylor series.
fn atan_iv(t: &Iv) -> Iv {
    let mut t = t.clone();
    let halvings = 3;
    for _ in 0..halvings {
        let s = sqrt(&Iv::int(1).add(&t.mul(&t)));
        t = t.div(&Iv::int(1).add(&s));
    }
    // |t| ≤ tan(π/32) < 0.1: alternating, decreasing terms.
    let t2 = t.mul(&t);
    let mut pow = t.clone();
    let mut s = t.clone();
    let mut n: i64 = 0;
    loop {
        n += 1;
        pow = pow.mul(&t2);
        let term = pow.div(&Iv::int(2 * n + 1));
        s = if n % 2 == 1 {
            s.sub(&term)
        } else {
            s.add(&term)
        };
        if term.mag_hi().lt(&tiny()) {
            return s.widen(&term.mag_hi()).scale(halvings);
        }
    }
}

// ---------------------------------------------------------------------
// The primary oracle (dashu-float)
// ---------------------------------------------------------------------

pub type F = FBig<HalfEven, 2>;

/// The precision for an argument `(hi, lo)`: 256 bits beyond its span.
pub fn prec(hi: f64, lo: f64) -> usize {
    let span = |x: f64| -> isize {
        if x == 0.0 || !x.is_finite() {
            return 0;
        }
        Dy::f(x).ilog2()
    };
    let top = span(hi);
    // The last bit of lo (or of hi): its exponent.
    let last = |x: f64| -> isize {
        if x == 0.0 {
            return isize::MAX;
        }
        let d = Dy::f(x);
        let (_, u) = mag(&d.m);
        d.e + u.trailing_zeros().unwrap_or(0) as isize
    };
    let bottom = last(hi).min(last(lo));
    let s = if hi == 0.0 { 0 } else { top - bottom + 1 };
    (s.max(0) as usize + 256).max(320)
}

/// The value of the pair `(hi, lo)` exactly, at precision `p`.
pub fn fdd(hi: f64, lo: f64, p: usize) -> F {
    let d = Dy::dd(hi, lo);
    F::from_parts(d.m, d.e).with_precision(p).value()
}

/// The exact dyadic value of an `F`.
pub fn dy(f: &F) -> Dy {
    let r = f.repr();
    Dy {
        m: r.significand().clone(),
        e: r.exponent(),
    }
}

// ---------------------------------------------------------------------
// Checking a result against a bound
// ---------------------------------------------------------------------

/// `|z − f| ≤ (c + g G) u² (1 + 2⁻⁴⁰) |f| + A + 2⁻²⁰⁰ |f| (+ η below 2⁻⁹⁶⁸)`:
/// the criterion of the contract (§4.1), with the oracle's slack explicit.
pub fn within(z: (f64, f64), f: &Dy, c: f64, g: f64, growth: f64, absolute: f64) -> bool {
    let err = Dy::dd(z.0, z.1).sub(f).abs();
    let af = f.abs();
    let rel = Dy::f(c).add(&Dy::f(g).mul(&Dy::f(growth)));
    let mut b = rel
        .mul(&Dy::pow2(-106))
        .mul(&Dy::int(1).add(&Dy::pow2(-40)))
        .mul(&af)
        .add(&Dy::f(absolute))
        .add(&Dy::pow2(-200).mul(&af));
    if af.lt(&Dy::pow2(-968)) {
        b = b.add(&Dy::pow2(-1074));
    }
    err.le(&b)
}

/// `|z − f| / (u² |f|)`, approximately, for reports (0 when `f = 0`).
pub fn err_u2(z: (f64, f64), f: &Dy) -> f64 {
    if f.is_zero() {
        return 0.0;
    }
    let err = Dy::dd(z.0, z.1).sub(f).abs();
    if err.is_zero() {
        return 0.0;
    }
    let q = err.div(&f.abs(), 64, true).scale(106);
    q.approx()
}

/// `max(0, |z − f| − A) / (u² |f|)`: the relative part of a mixed bound.
pub fn err_u2_mixed(z: (f64, f64), f: &Dy, absolute: f64) -> f64 {
    if f.is_zero() {
        return 0.0;
    }
    let err = Dy::dd(z.0, z.1).sub(f).abs().sub(&Dy::f(absolute));
    if err.is_neg() || err.is_zero() {
        return 0.0;
    }
    err.div(&f.abs(), 64, true).scale(106).approx()
}
