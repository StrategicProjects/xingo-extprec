//! # Chapter 11.4 — Decimal formatting and parsing
//!
//! ## Purpose
//! `Display`, `LowerExp` and `FromStr` for [`DoubleDouble`]. The contract
//! asks for *faithful* conversions (question E7: an error of at most one
//! unit of the 32nd digit). Both are in fact **correctly rounded**: they are computed
//! in exact integer arithmetic (Chapter 11.3), which is simpler to state
//! and to test than a conversion carried out in double-double arithmetic
//! itself, whose error would have to be bounded.
//!
//! ## Notation
//! A finite nonzero double-double is `x = ± M · 2ᴱ` with `M` a natural
//! number (`M = m_h 2^(e_h − E) ± m_ℓ 2^(e_ℓ − E)` from the two
//! significands). Its *decimal expansion* `d₁.d₂d₃… × 10ᵉ`, `d₁ ≠ 0`, is
//! finite, because `2` divides `10`. A decimal string is `v = ± D · 10^q`
//! with `D` a natural number. `RN` rounds to the nearest `f64`, ties to
//! even.
//!
//! ## Algorithms
//! **Algorithm X (exact expansion).** Input `M`, `E`. Output the digits
//! `d₁ d₂ … d_n` and `e`.
//!
//! X1. \[Integer.\] If `E ≥ 0`, the digits are those of `M · 2ᴱ` (radix
//!     conversion by repeated division by `10⁹`) and `e = n − 1`. Stop.
//! X2. \[Split.\] `I ← ⌊M / 2ˢ⌋`, `F ← M mod 2ˢ`, `s = −E`. The integer
//!     digits are those of `I`.
//! X3. \[Fraction.\] While `F ≠ 0`: `F ← 10⁹ F`; emit the nine digits of
//!     `⌊F / 2ˢ⌋`; `F ← F mod 2ˢ`. (Each pass removes nine factors of two
//!     from the denominator: at most `⌈s/9⌉` passes.)
//! X4. \[Normalize.\] Drop the leading and trailing zeros; `e` is the
//!     position of `d₁` relative to the point.
//!
//! **Algorithm O (output).** Input `x` and an optional precision `N` (the
//! `{:.N}` of the format string).
//!
//! O1. \[Special.\] NaN is `NaN`, the infinities `inf` and `-inf`, zero `0`
//!     (`-0` for the negative zero).
//! O2. \[Expand.\] Algorithm X gives the digits and `e`.
//! O3. \[Notation.\] For `{}`: *fixed* if `−5 ≤ e ≤ 31`, else *scientific*
//!     (`d₁.d₂…e±k`, as Rust writes `{:e}` for an `f64`: no `+`, no
//!     leading zero in the exponent). With a precision the choice uses the
//!     `e` of the exact value; without, that of the rounded one. For
//!     `{:e}`: always scientific (zero is `0e0`).
//! O4. \[Round.\] Without a precision: to 31 significant digits, half to
//!     even, trailing zeros dropped. With precision `N`: to `N` digits
//!     after the point of the chosen notation, zero-padded.
//! O5. \[Pad.\] Sign, width, fill and `+` as for the integers of `core`.
//!
//! **Algorithm I (input).** Input a string. Output a double-double or an
//! error.
//!
//! I1. \[Scan.\] `[+-]` then `inf`, `infinity` or `nan` (any case), or
//!     digits with at most one point and an optional exponent `e[+-]k`; at
//!     least one digit; nothing else (no blanks). This gives `D` and `q`.
//! I2. \[Range.\] If `v ≥ 10³¹⁰` return `±∞`; if `v < 10⁻³³⁰` return `±0`.
//! I3. \[Leading.\] `h ← RN(D · 10^q)` by Algorithm N. If `h` is infinite
//!     return `±∞`.
//! I4. \[Trailing.\] `ℓ ← RN(D · 10^q − h)`, the difference formed exactly
//!     as a fraction of naturals, rounded by Algorithm N. Return `±2Sum(h,
//!     ℓ)` (the pair itself, but for the tie of invariant P2).
//!
//! **Algorithm N (nearest `f64` of a fraction).** Input naturals `A`, `B >
//! 0`. Output `RN(A/B)`.
//!
//! N1. \[Scale.\] `t ← 65 − (bits(A) − bits(B))`; `(Q, sticky) ←` the
//!     short quotient (Algorithm Q of Chapter 11.3) of `A · 2ᵗ` by `B`, so
//!     that `2⁶⁴ ≤ Q < 2⁶⁶`.
//! N2. \[Position.\] The unit of the result is `2ᶜ`, `c = max(msb(Q) − t −
//!     52, −1074)` (53 significant bits, fewer in the subnormal range).
//! N3. \[Round.\] Drop the `c + t` low bits of `Q`, rounding half to even
//!     with `sticky` breaking the tie upward.
//! N4. \[Encode.\] Assemble the bits; a significand of `2⁵³` moves to the
//!     next binade, an exponent beyond the range is `+∞`.
//!
//! ## Invariants
//! (P1) *Exact output.* The digits written are those of the decimal
//!      expansion of `x` correctly rounded (half to even) at the stated
//!      position; with a precision large enough (`{:.1100}` always is) the
//!      output *is* `x`.
//! (P2) *Correctly rounded input.* `from_str` returns the normalized pair
//!      (D1) of value `RN(v) + RN(v − RN(v))`, within `ulp(ℓ)/2` of the
//!      string's value `v`. Its components are `h = RN(v)` and `ℓ = RN(v −
//!      h)` themselves, except when `ℓ` is subnormal and was rounded to
//!      exactly half an ulp of an odd `h` (then normalizing moves the tie
//!      to the even neighbour; the value is the same).
//! (P3) *Round trip.* Two statements, both consequences of (P1) and (P2).
//!      *Text.* A decimal `v` of at most 31 significant digits with `|v| ≥
//!      2⁻⁹⁶⁸` (below, the trailing component is subnormal and short),
//!      read by `from_str` and written with `{}` or `{:e}`, comes back as
//!      the same decimal: the double-double read is within `2⁻¹⁰⁶ |v| <
//!      1,24 · 10⁻³² |v|` of `v`, less than half a unit of the 31st digit
//!      (`≥ 5 · 10⁻³² |v|`). This is why the default is 31 digits and not
//!      32 (`10³¹ < 2¹⁰⁶ < 10³²`): with 32 a short decimal can come back
//!      as `…29799999999999e74`. *Value.* `x.to_string().parse()` is `x`
//!      whenever `x` has at most 31 significant decimal digits, and
//!      `format!("{x:.1100e}").parse()` is `x` for every finite `x`. In
//!      general the default output does **not** parse back to the same
//!      bits: a double-double is not a 106-bit number — `ℓ` may lie
//!      arbitrarily far below `h` (`1 + 2⁻⁹⁰⁰` is one) — so no fixed number
//!      of digits identifies it (a divergence from the contract's wording,
//!      reported in Chapter 11).
//!
//! ## References
//! Knuth, TAOCP vol. 2, §4.4 (radix conversion). Clinger, "How to read
//! floating point numbers accurately", PLDI 1990 (the exact-arithmetic
//! reader, Algorithm N here). Steele & White, "How to print floating-point
//! numbers accurately", PLDI 1990 (not used: no shortest-digits search is
//! possible for double-doubles, see (P3)).

use core::cmp::Ordering;
use core::fmt;
use core::str::FromStr;

use crate::big::Nat;
use crate::dd::DoubleDouble;

/// Significant digits of the default output (step O4).
const DEFAULT_DIGITS: i64 = 31;
/// The exponents `e` written in fixed notation (step O3).
const FIXED_RANGE: core::ops::RangeInclusive<i32> = -5..=31;
/// Step I2: decimal magnitudes beyond these are `∞` and `0`.
const MAX_MAGNITUDE: i64 = 310;
const MIN_MAGNITUDE: i64 = -330;

/// `|x| = m · 2ᵉ` for a finite `f64`.
fn decompose(x: f64) -> (u64, i32) {
    let bits = x.to_bits();
    let frac = bits & ((1_u64 << 52) - 1);
    let biased = i32::try_from((bits >> 52) & 0x7ff).expect("11 bits");
    if biased == 0 {
        (frac, -1074)
    } else {
        (frac | (1_u64 << 52), biased - 1075)
    }
}

/// `|x| = M · 2ᴱ` for a finite nonzero double-double (the notation).
fn magnitude(x: DoubleDouble) -> (Nat, i32) {
    let (mh, eh) = decompose(x.hi());
    let (ml, el) = decompose(x.lo());
    if ml == 0 {
        return (Nat::from_u64(mh), eh);
    }
    let e = eh.min(el);
    let h = Nat::from_u64(mh).shl(u64::from((eh - e).unsigned_abs()));
    let l = Nat::from_u64(ml).shl(u64::from((el - e).unsigned_abs()));
    let same_sign = x.hi().is_sign_negative() == x.lo().is_sign_negative();
    (if same_sign { h.add(&l) } else { h.sub(&l) }, e)
}

/// Algorithm X: the digits `d₁…d_n` (bytes `0..=9`) and `e` of a finite
/// nonzero double-double.
fn exact_expansion(x: DoubleDouble) -> (Vec<u8>, i32) {
    let (m, e2) = magnitude(x);
    let s = u64::from(e2.unsigned_abs());
    let (int_digits, mut digits) = if e2 >= 0 {
        (0, m.shl(s).to_decimal_digits()) // X1
    } else {
        let (int, mut frac) = m.split_at_bit(s); // X2
        let mut digits = int.to_decimal_digits();
        let int_digits = digits.len();
        while !frac.is_zero() {
            // X3
            frac.mul_small(1_000_000_000);
            let (top, rest) = frac.split_at_bit(s);
            let chunk = top.to_decimal_digits();
            digits.extend(core::iter::repeat_n(0, 9 - chunk.len()));
            digits.extend(chunk);
            frac = rest;
        }
        (int_digits, digits)
    };
    // X4. For an integer (X1) the point follows the last digit.
    let point = if e2 >= 0 { digits.len() } else { int_digits };
    let leading = digits.iter().take_while(|&&d| d == 0).count();
    digits.drain(..leading);
    while digits.last() == Some(&0) {
        digits.pop();
    }
    let e = i32::try_from(point).expect("a few thousand digits")
        - 1
        - i32::try_from(leading).expect("a few thousand digits");
    (digits, e)
}

/// Rounds `d₁.d₂… × 10ᵉ` to `keep` significant digits, half to even, and
/// drops trailing zeros. `keep ≤ 0` rounds at or above the first digit;
/// the result may be zero (no digits).
fn round_digits(digits: &mut Vec<u8>, e: &mut i32, keep: i64) {
    let Ok(k) = usize::try_from(keep) else {
        digits.clear();
        return;
    };
    if k >= digits.len() {
        return;
    }
    let next = digits[k];
    let rest_nonzero = digits[k + 1..].iter().any(|&d| d != 0);
    let odd = k > 0 && digits[k - 1] % 2 == 1;
    let up = next > 5 || (next == 5 && (rest_nonzero || odd));
    digits.truncate(k);
    if up {
        let mut i = k;
        loop {
            if i == 0 {
                // All nines (or no digit kept): 10^(e+1).
                digits.insert(0, 1);
                digits.truncate(k.max(1));
                *e += 1;
                break;
            }
            i -= 1;
            if digits[i] == 9 {
                digits[i] = 0;
            } else {
                digits[i] += 1;
                break;
            }
        }
    }
    while digits.last() == Some(&0) {
        digits.pop();
    }
}

fn push_digits(out: &mut String, digits: impl IntoIterator<Item = u8>) {
    out.extend(digits.into_iter().map(|d| char::from(b'0' + d)));
}

/// `d₁.d₂… × 10ᵉ` in fixed notation, with exactly `frac` digits after the
/// point when given (the digits were rounded to fit).
fn fixed(digits: &[u8], e: i32, frac: Option<usize>) -> String {
    let mut out = String::new();
    let mut tail: Vec<u8> = Vec::new();
    match usize::try_from(e) {
        Ok(e) if !digits.is_empty() => {
            let int = e + 1;
            push_digits(&mut out, digits.iter().copied().take(int));
            push_digits(
                &mut out,
                core::iter::repeat_n(0, int.saturating_sub(digits.len())),
            );
            tail.extend(digits.iter().skip(int));
        }
        _ => {
            out.push('0');
            if !digits.is_empty() {
                let zeros = usize::try_from(-(e + 1)).expect("e < 0");
                tail.extend(core::iter::repeat_n(0, zeros));
                tail.extend(digits);
            }
        }
    }
    if let Some(n) = frac {
        debug_assert!(
            tail.len() <= n,
            "the digits were not rounded to the precision"
        );
        tail.resize(n, 0);
    }
    if !tail.is_empty() {
        out.push('.');
        push_digits(&mut out, tail);
    }
    out
}

/// `d₁.d₂… × 10ᵉ` in scientific notation (`digits` nonempty).
fn scientific(digits: &[u8], e: i32, frac: Option<usize>) -> String {
    let mut out = String::new();
    push_digits(&mut out, digits.iter().copied().take(1));
    let mut tail: Vec<u8> = digits[1..].to_vec();
    if let Some(n) = frac {
        debug_assert!(
            tail.len() <= n,
            "the digits were not rounded to the precision"
        );
        tail.resize(n, 0);
    }
    if !tail.is_empty() {
        out.push('.');
        push_digits(&mut out, tail);
    }
    out.push('e');
    out.push_str(&e.to_string());
    out
}

/// Algorithm O. `always_scientific` is the `{:e}` of step O3.
#[allow(clippy::float_cmp)] // the zero test is exact
fn write_decimal(
    x: DoubleDouble,
    f: &mut fmt::Formatter<'_>,
    always_scientific: bool,
) -> fmt::Result {
    let negative = x.hi().is_sign_negative();
    // O1. `pad_integral` and not `pad`: a precision must not truncate.
    if x.is_nan() {
        return f.pad_integral(true, "", "NaN");
    }
    if !x.is_finite() {
        return f.pad_integral(!negative, "", "inf");
    }
    let precision = f.precision();
    let body = if x.hi() == 0.0 {
        if always_scientific {
            scientific(&[0], 0, precision)
        } else {
            fixed(&[], 0, precision)
        }
    } else {
        let (mut digits, mut e) = exact_expansion(x); // O2
        let is_fixed = |e: i32| !always_scientific && FIXED_RANGE.contains(&e); // O3
        match precision {
            None => {
                round_digits(&mut digits, &mut e, DEFAULT_DIGITS); // O4
                if is_fixed(e) {
                    fixed(&digits, e, None)
                } else {
                    scientific(&digits, e, None)
                }
            }
            Some(n) => {
                let n_i64 = i64::try_from(n).unwrap_or(i64::MAX);
                if is_fixed(e) {
                    let keep = n_i64.saturating_add(i64::from(e) + 1);
                    round_digits(&mut digits, &mut e, keep);
                    fixed(&digits, e, Some(n))
                } else {
                    round_digits(&mut digits, &mut e, n_i64.saturating_add(1));
                    scientific(&digits, e, Some(n))
                }
            }
        }
    };
    f.pad_integral(!negative, "", &body) // O5
}

/// Decimal, scientific notation when the exponent calls for it; `{:.N}`
/// gives `N` digits after the point, the default is the value correctly
/// rounded to 31 significant digits, trailing zeros dropped (Algorithm O
/// of Chapter 11.4). A decimal of at most 31 digits that was read by
/// `from_str` is written back as the same decimal; the output parses back
/// to the same *bits* whenever the value has at most 31 significant
/// decimal digits — see invariant P3 for why no fixed number of digits can
/// do so in general (`{:.1100e}` always does).
impl fmt::Display for DoubleDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_decimal(*self, f, false)
    }
}

/// Scientific notation as for `f64`: `d.ddd` then `e` and the decimal
/// exponent, without `+` or leading zeros (`1.5e-7`, `1e0`, `0e0`).
/// `{:e}` gives the value correctly rounded to the 31 significant digits
/// of `Display`, trailing zeros dropped; `{:.Ne}` gives `N` digits after
/// the point, correctly rounded (half to even), zero-padded.
impl fmt::LowerExp for DoubleDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_decimal(*self, f, true)
    }
}

/// A decimal string that is not a number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseDoubleDoubleError;

impl fmt::Display for ParseDoubleDoubleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid double-double literal")
    }
}

impl core::error::Error for ParseDoubleDoubleError {}

/// Algorithm N: `RN(num / den)` for naturals, `den > 0`.
fn nearest_f64(num: &Nat, den: &Nat) -> f64 {
    if num.is_zero() {
        return 0.0;
    }
    // N1.
    let d = i64::try_from(num.bits()).expect("bits fit i64")
        - i64::try_from(den.bits()).expect("bits fit i64");
    let t = 65 - d;
    let (q, sticky) = if t >= 0 {
        num.shl(t.unsigned_abs()).short_quotient(den)
    } else {
        num.short_quotient(&den.shl(t.unsigned_abs()))
    };
    debug_assert!(q >> 64 != 0 && q >> 66 == 0, "N1: 2^64 <= Q < 2^66");
    // N2.
    let msb = i64::from(q.ilog2());
    let unit = (msb - t - 52).max(-1074);
    let drop = unit + t;
    debug_assert!(drop >= 12, "N2: at most 53 bits are kept");
    let Ok(drop) = u32::try_from(drop) else {
        return 0.0;
    };
    if drop > 127 {
        return 0.0; // Q < 2^66 is below half a unit
    }
    // N3.
    let mut m = u64::try_from(q >> drop).expect("at most 54 bits");
    let rem = q & ((1_u128 << drop) - 1);
    let half = 1_u128 << (drop - 1);
    let up = match rem.cmp(&half) {
        Ordering::Greater => true,
        Ordering::Equal => sticky || m % 2 == 1,
        Ordering::Less => false,
    };
    m += u64::from(up);
    // N4.
    let mut unit = unit;
    if m == 1_u64 << 53 {
        m >>= 1;
        unit += 1;
    }
    if m < 1_u64 << 52 {
        debug_assert_eq!(unit, -1074, "N4: a short significand is subnormal");
        return f64::from_bits(m);
    }
    let biased = unit + 52 + 1023;
    if biased >= 2047 {
        return f64::INFINITY;
    }
    let biased = u64::try_from(biased).expect("N2: unit >= -1074");
    f64::from_bits((biased << 52) | (m - (1_u64 << 52)))
}

/// Step I1 for a numeric literal: the significant digits (no leading or
/// trailing zeros) and the exponent `q` of `D · 10^q`. `None` if the text
/// is not a number.
fn scan(text: &[u8]) -> Option<(Vec<u8>, i64)> {
    /// Exponents saturate here: far beyond any representable magnitude,
    /// yet small enough that sums with a digit count cannot overflow.
    const EXPONENT_CAP: i64 = 1_000_000_000_000;
    let mut digits = Vec::new();
    let mut i = 0;
    let mut frac_len = 0_i64;
    let mut seen_point = false;
    while i < text.len() {
        match text[i] {
            c @ b'0'..=b'9' => {
                digits.push(c - b'0');
                frac_len += i64::from(seen_point);
            }
            b'.' if !seen_point => seen_point = true,
            _ => break,
        }
        i += 1;
    }
    if digits.is_empty() {
        return None;
    }
    let mut exponent = 0_i64;
    if i < text.len() {
        if !matches!(text[i], b'e' | b'E') {
            return None;
        }
        i += 1;
        let negative = match text.get(i) {
            Some(b'-') => {
                i += 1;
                true
            }
            Some(b'+') => {
                i += 1;
                false
            }
            _ => false,
        };
        if i == text.len() {
            return None;
        }
        for &c in &text[i..] {
            if !c.is_ascii_digit() {
                return None;
            }
            exponent = (exponent * 10 + i64::from(c - b'0')).min(EXPONENT_CAP);
        }
        if negative {
            exponent = -exponent;
        }
    }
    let leading = digits.iter().take_while(|&&d| d == 0).count();
    digits.drain(..leading);
    let mut q = exponent - frac_len;
    while digits.last() == Some(&0) {
        digits.pop();
        q += 1;
    }
    Some((digits, q))
}

/// Steps I2–I4: the double-double nearest to `D · 10^q`, `D ≠ 0` given by
/// its digits.
fn nearest_double_double(digits: &[u8], q: i64) -> DoubleDouble {
    // I2.
    let magnitude = i64::try_from(digits.len())
        .unwrap_or(i64::MAX)
        .saturating_add(q);
    if magnitude > MAX_MAGNITUDE {
        return DoubleDouble::INFINITY;
    }
    if magnitude < MIN_MAGNITUDE {
        return DoubleDouble::ZERO;
    }
    let d = Nat::from_decimal_digits(digits);
    let scale = Nat::pow10(u32::try_from(q.unsigned_abs()).expect("bounded by I2"));
    let (num, den) = if q >= 0 {
        (d.mul(&scale), Nat::from_u64(1))
    } else {
        (d, scale)
    };
    // I3.
    let hi = nearest_f64(&num, &den);
    if !hi.is_finite() || hi.to_bits() == 0 {
        return DoubleDouble::from_f64(hi);
    }
    // I4: v − hi = (a − b) / c with hi = m 2^k.
    let (m, k) = decompose(hi);
    let m = Nat::from_u64(m);
    let shift = u64::from(k.unsigned_abs());
    let (a, b, c) = if k >= 0 {
        (num, m.shl(shift).mul(&den), den)
    } else {
        (num.shl(shift), m.mul(&den), den.shl(shift))
    };
    let lo = match a.cmp(&b) {
        Ordering::Equal => 0.0,
        Ordering::Greater => nearest_f64(&a.sub(&b), &c),
        Ordering::Less => -nearest_f64(&b.sub(&a), &c),
    };
    // Normalized already, except when `lo` was rounded in the subnormal
    // range to exactly half an ulp of an odd `hi`; the sum is exact (P2).
    DoubleDouble::from_sum(hi, lo)
}

/// Decimal or scientific notation, `inf`, `nan`; faithful (question E7) —
/// in fact correctly rounded: the leading component is the `f64` nearest
/// to the string's value and the trailing one the `f64` nearest to the
/// rest (Algorithm I of Chapter 11.4).
impl FromStr for DoubleDouble {
    type Err = ParseDoubleDoubleError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // I1.
        let (negative, rest) = match s.as_bytes().first() {
            Some(b'-') => (true, &s[1..]),
            Some(b'+') => (false, &s[1..]),
            _ => (false, s),
        };
        let magnitude = if rest.eq_ignore_ascii_case("inf") || rest.eq_ignore_ascii_case("infinity")
        {
            Self::INFINITY
        } else if rest.eq_ignore_ascii_case("nan") {
            Self::NAN
        } else {
            let (digits, q) = scan(rest.as_bytes()).ok_or(ParseDoubleDoubleError)?;
            if digits.is_empty() {
                Self::ZERO
            } else {
                nearest_double_double(&digits, q)
            }
        };
        Ok(if negative { -magnitude } else { magnitude })
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // L2.4 (Chapter 11.11): the tests may use the libm of `f64` (clippy.toml)
#[allow(clippy::float_cmp)] // the tests compare exact values
mod tests {
    use super::*;

    type Dd = DoubleDouble;

    fn parse(s: &str) -> Dd {
        s.parse().unwrap_or_else(|_| panic!("{s:?} does not parse"))
    }

    #[test]
    fn displays_short_values_exactly() {
        for (x, s) in [
            (0.0, "0"),
            (-0.0, "-0"),
            (1.0, "1"),
            (-1.5, "-1.5"),
            (0.5, "0.5"),
            (1024.0, "1024"),
            (0.001_953_125, "0.001953125"),
            (1e15, "1000000000000000"),
            (9_007_199_254_740_992.0, "9007199254740992"),
            (0.000_030_517_578_125, "0.000030517578125"),
        ] {
            assert_eq!(Dd::from_f64(x).to_string(), s);
            assert_eq!(parse(s), Dd::from_f64(x));
            assert_eq!(parse(s).hi().to_bits(), x.to_bits());
        }
        assert_eq!(Dd::NAN.to_string(), "NaN");
        assert_eq!(Dd::INFINITY.to_string(), "inf");
        assert_eq!((-Dd::INFINITY).to_string(), "-inf");
    }

    #[test]
    fn displays_thirty_two_digits() {
        let third = Dd::ONE / 3.0;
        assert_eq!(third.to_string(), "0.3333333333333333333333333333333");
        // The f64 nearest to 0.1 is not 0.1.
        assert_eq!(
            Dd::from_f64(0.1).to_string(),
            "0.1000000000000000055511151231258"
        );
        assert_eq!(
            Dd::from_f64(2.0).sqrt().to_string(),
            "1.41421356237309504880168872421"
        );
        // 1 + 2^-900 is 1 to 31 digits (invariant P3).
        let x = Dd::ONE + 2f64.powi(-900);
        assert_eq!(x.to_string(), "1");
        assert_ne!(parse(&x.to_string()), x);
        assert_eq!(parse(&format!("{x:.1100}")), x);
    }

    #[test]
    fn notation_follows_the_exponent() {
        assert_eq!(Dd::from_f64(1e-5).to_string()[..7].to_string(), "0.00001");
        assert_eq!(
            Dd::from_f64(2f64.powi(-20)).to_string(),
            "9.5367431640625e-7"
        );
        assert_eq!(
            parse("1e31").to_string(),
            "10000000000000000000000000000000"
        );
        assert_eq!(parse("1e32").to_string(), "1e32");
        assert_eq!(parse("-1.25e-200").to_string(), "-1.25e-200");
        assert_eq!(parse("123456789e40").to_string(), "1.23456789e48");
        // Rounding may carry into the next decade.
        let nines = parse("9.99999999999999999999999999999999999");
        assert_eq!(nines.to_string(), "10");
    }

    #[test]
    fn precision_counts_digits_after_the_point() {
        let third = Dd::ONE / 3.0;
        assert_eq!(format!("{third:.3}"), "0.333");
        assert_eq!(format!("{third:.0}"), "0");
        assert_eq!(format!("{:.2}", Dd::from_f64(2.675)), "2.67"); // 2.67499999…
        assert_eq!(format!("{:.1}", Dd::from_f64(0.25)), "0.2"); // tie to even
        assert_eq!(format!("{:.1}", Dd::from_f64(0.75)), "0.8");
        assert_eq!(format!("{:.0}", Dd::from_f64(0.5)), "0");
        assert_eq!(format!("{:.0}", Dd::from_f64(1.5)), "2");
        assert_eq!(format!("{:.0}", Dd::from_f64(0.75)), "1");
        assert_eq!(format!("{:.2}", Dd::from_f64(9.999)), "10.00");
        assert_eq!(format!("{:.2}", Dd::from_f64(-0.001)), "-0.00");
        assert_eq!(format!("{:.3}", Dd::ZERO), "0.000");
        assert_eq!(format!("{:.4}", Dd::from_f64(1.0)), "1.0000");
        assert_eq!(format!("{:.2}", parse("1.2345e40")), "1.23e40");
        assert_eq!(format!("{:.0}", parse("9.5e40")), "1e41");
        assert_eq!(format!("{:.3}", parse("1e-7")), "1.000e-7");
        assert_eq!(format!("{third:.40}").len(), 42);
    }

    #[test]
    fn lower_exp_follows_f64() {
        // The same strings as f64. Without a precision f64 writes the
        // shortest digits that identify the f64, we write the value: they
        // agree when the value has few digits.
        for x in [
            1.0,
            -1.5,
            0.0,
            -0.0,
            1024.0,
            0.001_953_125,
            1e15,
            2f64.powi(-20),
        ] {
            assert_eq!(format!("{:e}", Dd::from_f64(x)), format!("{x:e}"));
            assert_eq!(format!("{:12e}|", Dd::from_f64(x)), format!("{x:12e}|"));
            assert_eq!(format!("{:+e}", Dd::from_f64(x)), format!("{x:+e}"));
        }
        // With a precision both are the exact value correctly rounded.
        for x in [
            1.0,
            -1.5,
            0.0,
            -0.0,
            0.1,
            2.5e-7,
            6.25e100,
            0.125,
            9.5,
            f64::MAX,
            5e-324,
        ] {
            assert_eq!(format!("{:.3e}", Dd::from_f64(x)), format!("{x:.3e}"));
            assert_eq!(format!("{:+.0e}", Dd::from_f64(x)), format!("{x:+.0e}"));
            assert_eq!(
                format!("{:<12.1e}|", Dd::from_f64(x)),
                format!("{x:<12.1e}|")
            );
            assert_eq!(format!("{:012.2e}", Dd::from_f64(x)), format!("{x:012.2e}"));
            assert_eq!(format!("{:.25e}", Dd::from_f64(x)), format!("{x:.25e}"));
        }
        assert_eq!(format!("{:e}", Dd::NAN), "NaN");
        assert_eq!(format!("{:e}", Dd::INFINITY), "inf");
        assert_eq!(format!("{:.2e}", -Dd::INFINITY), "-inf");
        // 31 digits without a precision, trailing zeros dropped.
        let third = Dd::ONE / 3.0;
        assert_eq!(format!("{third:e}"), "3.333333333333333333333333333333e-1");
        assert_eq!(format!("{:e}", parse("1e31")), "1e31");
        assert_eq!(format!("{:e}", parse("123.5")), "1.235e2");
        // With a precision: digits after the point, half to even, carries.
        assert_eq!(format!("{third:.5e}"), "3.33333e-1");
        assert_eq!(format!("{:.0e}", Dd::from_f64(2.5)), "2e0");
        assert_eq!(format!("{:.0e}", Dd::from_f64(3.5)), "4e0");
        assert_eq!(format!("{:.1e}", Dd::from_f64(9.96)), "1.0e1");
        assert_eq!(format!("{:.2e}", parse("9.995e-10")), "9.99e-10"); // below the tie in dd
        assert_eq!(format!("{:.3e}", Dd::ZERO), "0.000e0");
        assert_eq!(format!("{third:.40e}").len(), 2 + 40 + 3);
        // Round trips.
        for s in [
            "1e0",
            "-2.5e-7",
            "1.2345678901234567890123456789e-200",
            "9.99e307",
        ] {
            assert_eq!(format!("{:e}", parse(s)), s);
        }
        let x = Dd::ONE + 2f64.powi(-900);
        assert_eq!(parse(&format!("{x:.1100e}")), x);
    }

    #[test]
    fn width_fill_and_sign() {
        let x = Dd::from_f64(1.5);
        assert_eq!(format!("{x:8}"), "     1.5");
        assert_eq!(format!("{x:<8}|"), "1.5     |");
        assert_eq!(format!("{x:+}"), "+1.5");
        assert_eq!(format!("{x:08.2}"), "00001.50");
        assert_eq!(format!("{:+}", Dd::INFINITY), "+inf");
        assert_eq!(format!("{:>5}", Dd::NAN), "  NaN");
        assert_eq!(format!("{:.1}", Dd::NAN), "NaN");
        assert_eq!(format!("{:6.1}", -Dd::INFINITY), "  -inf");
    }

    #[test]
    fn parses_the_grammar() {
        assert_eq!(parse("3"), Dd::from_f64(3.0));
        assert_eq!(parse("+3."), Dd::from_f64(3.0));
        assert_eq!(parse(".5"), Dd::from_f64(0.5));
        assert_eq!(parse("-00012.50"), Dd::from_f64(-12.5));
        assert_eq!(parse("1e3"), Dd::from_f64(1000.0));
        assert_eq!(parse("1E+3"), Dd::from_f64(1000.0));
        assert_eq!(parse("2500e-2"), Dd::from_f64(25.0));
        assert_eq!(parse("0e999999999999999999999"), Dd::ZERO);
        assert_eq!(parse("-0").hi().to_bits(), (-0.0_f64).to_bits());
        assert_eq!(parse("inf"), Dd::INFINITY);
        assert_eq!(parse("-Infinity"), -Dd::INFINITY);
        assert!(parse("NaN").is_nan() && parse("-nan").is_nan());
        for bad in [
            "", "+", "-", ".", "e5", "1e", "1e+", "1.2.3", " 1", "1 ", "1_0", "0x10", "1e5.0",
            "infinit", "--1", "1f",
        ] {
            assert_eq!(bad.parse::<Dd>(), Err(ParseDoubleDoubleError), "{bad:?}");
        }
        assert_eq!(
            ParseDoubleDoubleError.to_string(),
            "invalid double-double literal"
        );
    }

    #[test]
    fn parses_correctly_rounded() {
        // 0.1 = hi + lo with hi the nearest f64 and lo the rest.
        let tenth = parse("0.1");
        assert_eq!(tenth.hi().to_bits(), 0.1_f64.to_bits());
        assert_eq!(tenth, Dd::ONE / 10.0);
        assert_eq!(tenth.to_string(), "0.1");
        // Halfway cases of the leading component go to even.
        assert_eq!(
            parse("9007199254740993"),
            Dd::from_parts(9_007_199_254_740_992.0, 1.0)
        );
        assert_eq!(
            parse("9007199254740995"),
            Dd::from_parts(9_007_199_254_740_996.0, -1.0)
        );
        // π to 40 digits.
        let pi = parse("3.141592653589793238462643383279502884197");
        assert_eq!(pi.hi().to_bits(), core::f64::consts::PI.to_bits());
        assert_eq!(pi.lo(), 1.224_646_799_147_353_2e-16);
        assert_eq!(pi.to_string(), "3.14159265358979323846264338328");
    }

    #[test]
    fn parses_the_ends_of_the_range() {
        assert_eq!(parse("1e400"), Dd::INFINITY);
        assert_eq!(parse("-1e309"), -Dd::INFINITY);
        assert_eq!(parse("1.7976931348623157e308").hi(), f64::MAX);
        assert_eq!(parse("1.7976931348623159e308"), Dd::INFINITY);
        assert_eq!(parse("1e-400"), Dd::ZERO);
        assert_eq!(parse("1e-99999999999999999999"), Dd::ZERO);
        assert_eq!(parse("1e99999999999999999999"), Dd::INFINITY);
        let eta = f64::from_bits(1);
        assert_eq!(parse("5e-324"), Dd::from_f64(eta));
        assert_eq!(parse("2.4703282292062327e-324"), Dd::ZERO); // just below η/2
        assert_eq!(parse("2.4703282292062328e-324"), Dd::from_f64(eta));
        assert_eq!(parse("2.2250738585072014e-308").hi(), f64::MIN_POSITIVE);
        // The exact expansion of η has 751 significant digits.
        let s = format!("{:.1100}", Dd::from_f64(eta));
        assert!(s.starts_with("4.940656458412465441765687928682213723650598"));
        assert!(s.ends_with("e-324") && s.len() == 2 + 1100 + 5);
        assert_eq!(parse(&s), Dd::from_f64(eta));
        assert_eq!(
            parse(&format!("{:.1100}", Dd::from_f64(f64::MAX))).hi(),
            f64::MAX
        );
    }

    #[test]
    fn round_digits_cases() {
        let round = |s: &str, e: i32, keep: i64| {
            let mut d: Vec<u8> = s.bytes().map(|b| b - b'0').collect();
            let mut e = e;
            round_digits(&mut d, &mut e, keep);
            (
                d.iter().map(|&b| char::from(b + b'0')).collect::<String>(),
                e,
            )
        };
        assert_eq!(round("12345", 0, 3), ("123".into(), 0));
        assert_eq!(round("1235", 0, 3), ("124".into(), 0)); // tie, odd
        assert_eq!(round("1225", 0, 3), ("122".into(), 0)); // tie, even
        assert_eq!(round("12251", 0, 3), ("123".into(), 0));
        assert_eq!(round("9996", 2, 3), ("1".into(), 3));
        assert_eq!(round("5", 0, 0), (String::new(), 0)); // tie to zero
        assert_eq!(round("51", 0, 0), ("1".into(), 1));
        assert_eq!(round("9", 0, 0), ("1".into(), 1));
        assert_eq!(round("9", 0, -1), (String::new(), 0));
        assert_eq!(round("12", 0, 5), ("12".into(), 0));
    }
}
