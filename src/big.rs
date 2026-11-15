//! # Chapter 11.3 — Natural numbers for the decimal conversions
//!
//! ## Purpose
//! The decimal conversions of Chapter 11.4 are *exact*: a double-double is
//! a dyadic rational `M · 2ᴱ` whose decimal expansion is finite, and a
//! decimal string is a rational `D · 10^q`. Both need integers of a few
//! thousand bits and five operations on them. This module is that and no
//! more — it is not a general big-integer library (no dependency is
//! allowed at run time, and none is needed).
//!
//! ## Notation
//! A natural number `N = Σ dᵢ βⁱ` is held as its digits `dᵢ` in base `β =
//! 2³²`, least significant first. `bits(N)` is the position of its leading
//! one bit plus one (`bits(0) = 0`).
//!
//! ## Algorithms
//! All are the schoolbook ones (Knuth, TAOCP vol. 2, §4.3.1), linear or
//! quadratic in the number of digits.
//!
//! **Algorithm Q (short quotient).** Input `N`, `D > 0` with `⌊N/D⌋ <
//! 2¹²⁸`. Output `(⌊N/D⌋, N mod D ≠ 0)`.
//!
//! Q1. `R ← N`, `q ← 0`, `k ← max(bits(N) − bits(D), 0)`.
//! Q2. For `i = k, k−1, …, 0`: if `D · 2ⁱ ≤ R`, set `R ← R − D · 2ⁱ` and
//!     bit `i` of `q`.
//! Q3. Return `(q, R ≠ 0)`.
//!
//! Binary long division: `k + 1` comparisons and at most as many
//! subtractions of numbers of `bits(N)` bits. The conversions call it with
//! `k ≤ 66`.
//!
//! ## Invariants
//! (N1) *Normalized.* The most significant digit stored is nonzero; zero
//!      is the empty digit string. `Nat::check_invariants` (test builds).
//!
//! ## References
//! Knuth, TAOCP vol. 2, §4.3.1 (Algorithms A, S, M) and §4.4 (radix
//! conversion).

use core::cmp::Ordering;

/// `10⁹`, the largest power of ten below `2³²`.
const BILLION: u32 = 1_000_000_000;

/// A natural number (the chapter).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Nat {
    /// Base-`2³²` digits, least significant first, no leading zero (N1).
    limbs: Vec<u32>,
}

/// The low 32 bits of `x`.
#[inline]
#[allow(clippy::cast_possible_truncation)] // the truncation is the operation
fn low32(x: u64) -> u32 {
    x as u32
}

impl Nat {
    /// Zero.
    pub(crate) fn zero() -> Self {
        Self { limbs: Vec::new() }
    }

    /// The value of a `u64`.
    pub(crate) fn from_u64(x: u64) -> Self {
        let mut n = Self {
            limbs: vec![low32(x), low32(x >> 32)],
        };
        n.trim();
        n
    }

    /// The value of a string of decimal digits (bytes `0..=9`, most
    /// significant first).
    pub(crate) fn from_decimal_digits(digits: &[u8]) -> Self {
        let mut n = Self::zero();
        for chunk in digits.chunks(9) {
            let mut scale = 1_u32;
            let mut value = 0_u32;
            for &d in chunk {
                debug_assert!(d <= 9, "not a decimal digit");
                scale *= 10;
                value = value * 10 + u32::from(d);
            }
            n.mul_small(scale);
            n.add_small(value);
        }
        n
    }

    /// `10ᵏ`.
    pub(crate) fn pow10(k: u32) -> Self {
        let mut n = Self::from_u64(1);
        for _ in 0..k / 9 {
            n.mul_small(BILLION);
        }
        n.mul_small(10_u32.pow(k % 9));
        n
    }

    /// Restores (N1).
    fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    /// `N = 0`.
    pub(crate) fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// `bits(N)`.
    pub(crate) fn bits(&self) -> u64 {
        match self.limbs.last() {
            None => 0,
            Some(&top) => 32 * self.limbs.len() as u64 - u64::from(top.leading_zeros()),
        }
    }

    /// `N ← N · m`.
    pub(crate) fn mul_small(&mut self, m: u32) {
        let mut carry = 0_u64;
        for limb in &mut self.limbs {
            let t = u64::from(*limb) * u64::from(m) + carry;
            *limb = low32(t);
            carry = t >> 32;
        }
        if carry != 0 {
            self.limbs.push(low32(carry));
        }
        self.trim();
    }

    /// `N ← N + a`.
    pub(crate) fn add_small(&mut self, a: u32) {
        let mut carry = u64::from(a);
        for limb in &mut self.limbs {
            if carry == 0 {
                return;
            }
            let t = u64::from(*limb) + carry;
            *limb = low32(t);
            carry = t >> 32;
        }
        if carry != 0 {
            self.limbs.push(low32(carry));
        }
    }

    /// `N ← ⌊N / d⌋`, returning `N mod d`; `d > 0`.
    pub(crate) fn div_rem_small(&mut self, d: u32) -> u32 {
        debug_assert!(d != 0, "division by zero");
        let mut rem = 0_u64;
        for limb in self.limbs.iter_mut().rev() {
            let t = (rem << 32) | u64::from(*limb);
            *limb = low32(t / u64::from(d));
            rem = t % u64::from(d);
        }
        self.trim();
        low32(rem)
    }

    /// `N · 2ˢ`.
    pub(crate) fn shl(&self, s: u64) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let words = usize::try_from(s / 32).expect("shift fits memory");
        let bits = s % 32;
        let mut limbs = vec![0_u32; words];
        limbs.reserve(self.limbs.len() + 1);
        let mut carry = 0_u64;
        for &limb in &self.limbs {
            let t = (u64::from(limb) << bits) | carry;
            limbs.push(low32(t));
            carry = t >> 32;
        }
        if carry != 0 {
            limbs.push(low32(carry));
        }
        Self { limbs }
    }

    /// `(⌊N / 2ˢ⌋, N mod 2ˢ)`.
    pub(crate) fn split_at_bit(&self, s: u64) -> (Self, Self) {
        let words = usize::try_from(s / 32).unwrap_or(usize::MAX);
        if words >= self.limbs.len() {
            return (Self::zero(), self.clone());
        }
        let bits = s % 32;
        let mut low = Self {
            limbs: self.limbs[..=words].to_vec(),
        };
        low.limbs[words] &= (1_u32 << bits) - 1;
        low.trim();
        let mut high = Vec::with_capacity(self.limbs.len() - words);
        for i in words..self.limbs.len() {
            let here = u64::from(self.limbs[i]) >> bits;
            let above = self.limbs.get(i + 1).map_or(0, |&l| u64::from(l));
            high.push(low32(here | (above << (32 - bits))));
        }
        let mut high = Self { limbs: high };
        high.trim();
        (high, low)
    }

    /// `N + M`.
    pub(crate) fn add(&self, other: &Self) -> Self {
        let (long, short) = if self.limbs.len() >= other.limbs.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut limbs = Vec::with_capacity(long.limbs.len() + 1);
        let mut carry = 0_u64;
        for (i, &l) in long.limbs.iter().enumerate() {
            let t = u64::from(l) + short.limbs.get(i).map_or(0, |&s| u64::from(s)) + carry;
            limbs.push(low32(t));
            carry = t >> 32;
        }
        if carry != 0 {
            limbs.push(low32(carry));
        }
        Self { limbs }
    }

    /// `N − M`; requires `N ≥ M`.
    pub(crate) fn sub(&self, other: &Self) -> Self {
        debug_assert!(*self >= *other, "Nat::sub would be negative");
        let mut limbs = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0_i64;
        for (i, &l) in self.limbs.iter().enumerate() {
            let t = i64::from(l) - other.limbs.get(i).map_or(0, |&s| i64::from(s)) - borrow;
            borrow = i64::from(t < 0);
            limbs.push(low32(t.rem_euclid(1_i64 << 32).unsigned_abs()));
        }
        debug_assert_eq!(borrow, 0, "Nat::sub underflowed");
        let mut n = Self { limbs };
        n.trim();
        n
    }

    /// `N · M`.
    pub(crate) fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut limbs = vec![0_u32; self.limbs.len() + other.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0_u64;
            for (j, &b) in other.limbs.iter().enumerate() {
                let t = u64::from(a) * u64::from(b) + u64::from(limbs[i + j]) + carry;
                limbs[i + j] = low32(t);
                carry = t >> 32;
            }
            limbs[i + other.limbs.len()] = low32(carry);
        }
        let mut n = Self { limbs };
        n.trim();
        n
    }

    /// Algorithm Q: `(⌊N / D⌋, N mod D ≠ 0)`; requires `D > 0` and a
    /// quotient below `2¹²⁸`.
    pub(crate) fn short_quotient(&self, den: &Self) -> (u128, bool) {
        debug_assert!(!den.is_zero(), "division by zero");
        let mut rem = self.clone(); // Q1
        let mut q = 0_u128;
        let k = self.bits().saturating_sub(den.bits());
        debug_assert!(k < 128, "short_quotient: the quotient does not fit");
        for i in (0..=k).rev() {
            // Q2
            let t = den.shl(i);
            if t <= rem {
                rem = rem.sub(&t);
                q |= 1_u128 << i;
            }
        }
        (q, !rem.is_zero()) // Q3
    }

    /// The decimal digits (bytes `0..=9`), most significant first; empty
    /// for zero.
    pub(crate) fn to_decimal_digits(&self) -> Vec<u8> {
        let mut n = self.clone();
        let mut out = Vec::new();
        while !n.is_zero() {
            let mut chunk = n.div_rem_small(BILLION);
            for _ in 0..9 {
                out.push(u8::try_from(chunk % 10).expect("a digit"));
                chunk /= 10;
            }
        }
        while out.last() == Some(&0) {
            out.pop();
        }
        out.reverse();
        out
    }

    /// Invariant (N1).
    #[cfg(test)]
    pub(crate) fn check_invariants(&self) -> Result<(), &'static str> {
        if self.limbs.last() == Some(&0) {
            return Err("N1: a leading zero digit");
        }
        Ok(())
    }
}

impl PartialOrd for Nat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Nat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.limbs
            .len()
            .cmp(&other.limbs.len())
            .then_with(|| self.limbs.iter().rev().cmp(other.limbs.iter().rev()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nat(s: &str) -> Nat {
        let digits: Vec<u8> = s.bytes().map(|b| b - b'0').collect();
        let n = Nat::from_decimal_digits(&digits);
        n.check_invariants().unwrap();
        n
    }

    fn dec(n: &Nat) -> String {
        n.check_invariants().unwrap();
        let d = n.to_decimal_digits();
        if d.is_empty() {
            "0".into()
        } else {
            d.iter().map(|&b| char::from(b + b'0')).collect()
        }
    }

    #[test]
    fn decimal_round_trip() {
        for s in [
            "0",
            "7",
            "4294967296",
            "999999999",
            "1000000000",
            "340282366920938463463374607431768211456",
            "123456789012345678901234567890123456789012345678901234567890",
        ] {
            assert_eq!(dec(&nat(s)), s);
        }
        assert_eq!(dec(&nat("000123")), "123");
        assert_eq!(dec(&Nat::from_u64(u64::MAX)), "18446744073709551615");
        assert_eq!(dec(&Nat::pow10(0)), "1");
        assert_eq!(dec(&Nat::pow10(25)), "10000000000000000000000000");
    }

    #[test]
    fn arithmetic() {
        let a = nat("340282366920938463463374607431768211456"); // 2^128
        let b = nat("18446744073709551615"); // 2^64 − 1
        assert_eq!(a.bits(), 129);
        assert_eq!(b.bits(), 64);
        assert_eq!(Nat::zero().bits(), 0);
        assert_eq!(dec(&a.add(&b)), "340282366920938463481821351505477763071");
        assert_eq!(dec(&b.add(&a)), "340282366920938463481821351505477763071");
        assert_eq!(dec(&a.sub(&b)), "340282366920938463444927863358058659841");
        assert_eq!(dec(&a.sub(&a)), "0");
        assert_eq!(
            dec(&a.mul(&b)),
            "6277101735386680763495507056286727952638980837032266301440"
        );
        assert_eq!(dec(&a.mul(&Nat::zero())), "0");
        assert!(b < a);
        assert!(a > b);
        assert_eq!(a, a.clone());
        assert!(nat("4294967296") > nat("4294967295"));
    }

    #[test]
    fn shifts() {
        let one = Nat::from_u64(1);
        assert_eq!(
            dec(&one.shl(128)),
            "340282366920938463463374607431768211456"
        );
        assert_eq!(dec(&Nat::from_u64(5).shl(33)), "42949672960");
        assert_eq!(dec(&Nat::zero().shl(77)), "0");
        let n = nat("123456789012345678901234567890");
        for s in [0, 1, 31, 32, 33, 64, 70, 96, 97, 200] {
            let (high, low) = n.split_at_bit(s);
            high.check_invariants().unwrap();
            low.check_invariants().unwrap();
            assert_eq!(high.shl(s).add(&low), n, "s = {s}");
            assert!(low.bits() <= s);
        }
    }

    #[test]
    fn small_operations() {
        let mut n = nat("99999999999999999999");
        n.add_small(1);
        assert_eq!(dec(&n), "100000000000000000000");
        n.mul_small(7);
        assert_eq!(dec(&n), "700000000000000000000");
        assert_eq!(n.div_rem_small(9), 7);
        assert_eq!(dec(&n), "77777777777777777777");
        n.mul_small(0);
        assert!(n.is_zero());
        n.add_small(3);
        assert_eq!(dec(&n), "3");
    }

    #[test]
    fn short_quotients() {
        let n = nat("1000000000000000000000000000000");
        assert_eq!(
            n.short_quotient(&nat("7")),
            (142_857_142_857_142_857_142_857_142_857, true)
        );
        assert_eq!(
            n.short_quotient(&nat("1000000000000")),
            (1_000_000_000_000_000_000, false)
        );
        assert_eq!(nat("5").short_quotient(&nat("7")), (0, true));
        assert_eq!(Nat::zero().short_quotient(&nat("7")), (0, false));
        assert_eq!(nat("49").short_quotient(&nat("7")), (7, false));
    }
}
