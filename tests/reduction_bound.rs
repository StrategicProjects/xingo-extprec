//! The computation of Lemma D of Chapter 11.11: no double-double `x` with
//! `x_h > RN(π/4)` lies within `2⁻¹⁴⁰` of a multiple of `π/2`.
//!
//! Integers only (`dashu-int`): no floating point enters a bound.
//!
//! 1. `⌊2ᴾ · 2/π⌋`, `P = 1700`, from Machin's formula `π = 16 atan(1/5) −
//!    4 atan(1/239)` in fixed point, every floored term and the omitted
//!    tail bounded; the floor is accepted only if both ends of the
//!    enclosure of `π` give it. `β_j = frac(2ʲ T)` with `T = ⌊2ᴾ 2/π⌋ 2⁻ᴾ`
//!    is then an exact dyadic rational within `2^(j−P)` of `α_j`.
//! 2. For each `E ∈ [−53, 971]`: steps D2 and D3 of the lemma (continued
//!    fractions of `β_E`, `Q = 2⁵³`), then for every `e` from the cut of D3
//!    up to `E − 53`, step D4 (`s = E − e ≤ 60`: continued fractions of
//!    `β_e`, `Q = 2^(54+s)`) or step D5 (a lattice of dimension 3, reduced
//!    by the integral LLL, Cohen's Algorithm 2.6.7, and then verified
//!    exactly: its rows lie in the lattice, its determinant is `±Z`, and the
//!    Gram determinants give `min ‖b_i*‖`).
//! 3. Every case must give `‖x · 2/π‖ ≥ 2⁻¹⁴⁰`.
//!
//! The LLL is only a search: a wrong reduction can make a case fail, never
//! pass, because the bound is taken from the exact Gram determinants of
//! the basis it returns, after checking that the basis spans the lattice.
//! The full run (all 1 025 exponents, 527 301 cases, about 20 s with
//! `--release`) is `#[ignore]`d:
//! `cargo test --release -p xingo-extprec --test reduction_bound --
//! --ignored`. A sample of exponents (the extreme ones and those of the
//! closest arguments known) runs by default.

use dashu_int::ops::{BitTest, SquareRoot, UnsignedAbs};
use dashu_int::{IBig, UBig};

/// Bits of `2/π` kept: `T = ⌊2ᴾ 2/π⌋ 2⁻ᴾ`.
const P: usize = 1700;
/// The bound of the lemma on `‖x · 2/π‖`: `2⁻¹⁴⁰`.
const TARGET: usize = 140;
/// Step D4 up to this gap `s = E − e`; step D5 above.
const S_CF: i32 = 60;
/// The scales `Z = 2ᶻ` of step D5, tried in order.
const SCALES: [usize; 5] = [170, 185, 200, 230, 260];

/// `Σ_k (−1)ᵏ ⌊2ʷ / ((2k + 1) n^(2k+1))⌋` and the number of terms.
fn atan_inv(n: u32, w: usize) -> (IBig, usize) {
    let one = UBig::ONE << w;
    let n2 = UBig::from(n) * UBig::from(n);
    let mut p = UBig::from(n);
    let (mut s, mut k) = (IBig::ZERO, 0_usize);
    loop {
        let term = &one / (&p * UBig::from(2 * k + 1));
        if term == UBig::ZERO {
            return (s, k);
        }
        if k % 2 == 0 {
            s += IBig::from(term);
        } else {
            s -= IBig::from(term);
        }
        k += 1;
        p *= &n2;
    }
}

/// `⌊2ᴾ · 2/π⌋`, certified.
fn two_over_pi() -> UBig {
    let w = P + 64;
    let (a, ta) = atan_inv(5, w);
    let (b, tb) = atan_inv(239, w);
    let s = a * 16 - b * 4;
    // Each floored term errs by less than 1, the omitted tail (alternating,
    // decreasing) by less than its first term, which floored to 0.
    let err = IBig::from(16 * (ta + 1) + 4 * (tb + 1));
    let lo = UBig::try_from(&s - &err).expect("positive");
    let hi = UBig::try_from(&s + &err).expect("positive");
    // 2ᴾ 2/π = 2^(P+1+w) / (π 2ʷ), with lo < π 2ʷ < hi.
    let num = UBig::ONE << (P + 1 + w);
    let (f_lo, f_hi) = (&num / &hi, &num / &lo);
    assert_eq!(f_lo, f_hi, "the floor of 2^P 2/π is not determined");
    f_lo
}

/// `β_j = frac(2ʲ T)` as `num / 2^log_den`.
fn frac(t: &UBig, j: i32) -> (UBig, usize) {
    if j >= 0 {
        let mask = (UBig::ONE << P) - UBig::ONE;
        ((t << usize::try_from(j).expect("j ≥ 0")) & mask, P)
    } else {
        (t.clone(), P + usize::try_from(-j).expect("j < 0"))
    }
}

/// A lower bound `num / 2^log_den`.
#[derive(Clone, Debug)]
struct Bound {
    num: UBig,
    log_den: usize,
}

impl Bound {
    fn at_least_target(&self) -> bool {
        self.log_den >= TARGET && self.num >= (UBig::ONE << (self.log_den - TARGET))
    }

    fn lt(&self, o: &Self) -> bool {
        // a / 2^p < b / 2^q  ⇔  a 2^q < b 2^p
        (&self.num << o.log_den) < (&o.num << self.log_den)
    }

    /// `log₂` of the bound, to 1/64, for the report (only `+ × ÷` of `f64`).
    fn log2(&self) -> f64 {
        let bits = self.num.bit_len();
        let top = if bits > 64 {
            &self.num >> (bits - 64)
        } else {
            self.num.clone() << (64 - bits)
        };
        let top = u64::try_from(top).expect("64 bits");
        #[allow(clippy::cast_precision_loss)] // a report
        let mut y = top as f64 / 9_223_372_036_854_775_808.0; // [1, 2)
        let mut f = 0.0;
        let mut w = 0.5;
        for _ in 0..6 {
            y *= y;
            if y >= 2.0 {
                f += w;
                y /= 2.0;
            }
            w /= 2.0;
        }
        #[allow(clippy::cast_precision_loss)] // small integers
        let int = bits as f64 - 1.0 - self.log_den as f64;
        int + f
    }
}

/// Step D1: a lower bound on `‖qα‖` for `0 < q < 2^log_q`, where `α` is
/// within `ϑ < 2^(log_err − log_den)` of `β = num / 2^log_den`.
fn cf_bound(num: &UBig, log_den: usize, log_q: usize, log_err: usize) -> Option<Bound> {
    let den = UBig::ONE << log_den;
    let q_max = UBig::ONE << log_q;
    let (mut a, mut b) = (num.clone(), den.clone());
    // Convergents p_{−2}/q_{−2} = 0/1 and p_{−1}/q_{−1} = 1/0.
    let (mut p0, mut q0, mut p1, mut q1) = (UBig::ZERO, UBig::ONE, UBig::ONE, UBig::ZERO);
    let mut best: Option<UBig> = None;
    while b != UBig::ZERO {
        let (c, r) = (&a / &b, &a % &b);
        a = b;
        b = r;
        let p2 = &c * &p1 + &p0;
        let q2 = &c * &q1 + &q0;
        if q2 >= q_max {
            break;
        }
        // |q β − p| · 2^log_den = |q num − p den|.
        let (x, y) = (&q2 * num, &p2 * &den);
        let d = if x >= y { x - y } else { y - x };
        best = Some(match best {
            Some(v) if v <= d => v,
            _ => d,
        });
        (p0, q0, p1, q1) = (p1, q1, p2, q2);
    }
    // Every q < Q is below the next convergent: Lagrange's lemma applies.
    // ‖qα‖ ≥ ‖qβ‖ − Qϑ.
    let best = best?;
    let slack = UBig::ONE << (log_q + log_err);
    (best > slack).then(|| Bound {
        num: best - slack,
        log_den,
    })
}

fn dot(a: &[IBig; 3], b: &[IBig; 3]) -> IBig {
    &a[0] * &b[0] + &a[1] * &b[1] + &a[2] * &b[2]
}

/// `⌊a / b⌋` for `b > 0`.
fn floor_div(a: &IBig, b: &IBig) -> IBig {
    let (q, r) = (a / b, a % b);
    if r < IBig::ZERO { q - IBig::ONE } else { q }
}

/// The integral LLL (Cohen, Algorithm 2.6.7) with `δ = 99/100`, in place;
/// a search only (see the module documentation). 1-based as in Cohen.
#[allow(clippy::needless_range_loop, clippy::assigning_clones)] // Cohen's indices, kept
fn lll(basis: &mut [[IBig; 3]; 3]) {
    const N: usize = 3;
    let mut b: Vec<[IBig; 3]> = Vec::with_capacity(N + 1);
    b.push([IBig::ZERO, IBig::ZERO, IBig::ZERO]);
    b.extend(basis.iter().cloned());
    let mut d = vec![IBig::ONE; N + 1];
    let mut lam = vec![vec![IBig::ZERO; N + 1]; N + 1];
    d[1] = dot(&b[1], &b[1]);
    let (mut k, mut k_max) = (2_usize, 1_usize);
    let red = |b: &mut Vec<[IBig; 3]>, lam: &mut Vec<Vec<IBig>>, d: &[IBig], k: usize, l: usize| {
        if (&lam[k][l] * IBig::from(2u8)).unsigned_abs() > d[l].clone().unsigned_abs() {
            let q = floor_div(
                &(&lam[k][l] * IBig::from(2u8) + &d[l]),
                &(&d[l] * IBig::from(2u8)),
            );
            let bl = b[l].clone();
            for (x, y) in b[k].iter_mut().zip(bl.iter()) {
                *x -= &q * y;
            }
            lam[k][l] -= &q * &d[l];
            for i in 1..l {
                let v = &q * &lam[l][i];
                lam[k][i] -= v;
            }
        }
    };
    let mut steps = 0;
    while k <= N {
        steps += 1;
        assert!(steps < 100_000, "LLL does not terminate");
        if k > k_max {
            k_max = k;
            for j in 1..=k {
                let mut u = dot(&b[k], &b[j]);
                for i in 1..j {
                    u = (&d[i] * &u - &lam[k][i] * &lam[j][i]) / &d[i - 1];
                }
                if j < k {
                    lam[k][j] = u;
                } else {
                    assert!(u != IBig::ZERO, "dependent vectors");
                    d[k] = u;
                }
            }
        }
        red(&mut b, &mut lam, &d, k, k - 1);
        let lhs = IBig::from(100u8) * &d[k] * &d[k - 2];
        let rhs = IBig::from(99u8) * &d[k - 1] * &d[k - 1]
            - IBig::from(100u8) * &lam[k][k - 1] * &lam[k][k - 1];
        if lhs < rhs {
            // SWAPI(k).
            b.swap(k, k - 1);
            for j in 1..k - 1 {
                let t = lam[k][j].clone();
                lam[k][j] = lam[k - 1][j].clone();
                lam[k - 1][j] = t;
            }
            let l = lam[k][k - 1].clone();
            let bb = (&d[k - 2] * &d[k] + &l * &l) / &d[k - 1];
            for i in k + 1..=k_max {
                let t = lam[i][k].clone();
                lam[i][k] = (&d[k] * &lam[i][k - 1] - &l * &t) / &d[k - 1];
                lam[i][k - 1] = (&bb * &t + &l * &lam[i][k]) / &d[k];
            }
            d[k - 1] = bb;
            k = (k - 1).max(2);
        } else {
            for l in (1..k - 1).rev() {
                red(&mut b, &mut lam, &d, k, l);
            }
            k += 1;
        }
    }
    for (dst, src) in basis.iter_mut().zip(b.into_iter().skip(1)) {
        *dst = src;
    }
}

fn det3(m: &[[IBig; 3]; 3]) -> IBig {
    &m[0][0] * (&m[1][1] * &m[2][2] - &m[1][2] * &m[2][1])
        - &m[0][1] * (&m[1][0] * &m[2][2] - &m[1][2] * &m[2][0])
        + &m[0][2] * (&m[1][0] * &m[2][1] - &m[1][1] * &m[2][0])
}

/// Step D5 with `Z = 2ᶻ`: a lower bound on `‖Mα_E + mα_e‖` for `|M|, |m| <
/// 2⁵³`, not both zero, or `None` if this `Z` proves none.
fn lattice_bound(t: &UBig, big_e: i32, e: i32, z: usize) -> Option<Bound> {
    // The truncation of 2/π moves each coordinate by at most 2⁵³ Zϑ ≤ 1/2.
    assert!(
        i64::try_from(z + 54).expect("small") + i64::from(big_e.max(e))
            < i64::try_from(P).expect("small")
    );
    let zz = IBig::ONE << z;
    let (na, la) = frac(t, big_e);
    let (nb, lb) = frac(t, e);
    let a = IBig::from((na << z) >> la);
    let b = IBig::from((nb << z) >> lb);
    let mut m = [
        [IBig::ONE, IBig::ZERO, a.clone()],
        [IBig::ZERO, IBig::ONE, b.clone()],
        [IBig::ZERO, IBig::ZERO, zz.clone()],
    ];
    lll(&mut m);
    // The reduced rows span the same lattice: each lies in it, and the
    // determinant is ±Z, the covolume.
    for row in &m {
        let w = &row[2] - &row[0] * &a - &row[1] * &b;
        assert!(&w % &zz == IBig::ZERO, "a row outside the lattice");
    }
    assert!(
        det3(&m).unsigned_abs() == zz.clone().unsigned_abs(),
        "not a basis"
    );
    // ‖b₁*‖² = Δ₁, ‖b₂*‖² = Δ₂/Δ₁, ‖b₃*‖² = Δ₃/Δ₂ = Z²/Δ₂ (floors).
    let g = |i: usize, j: usize| dot(&m[i], &m[j]);
    let d1 = g(0, 0);
    let d2 = &d1 * g(1, 1) - g(0, 1) * g(0, 1);
    let d3 = &zz * &zz;
    let mu2 = d1.clone().min(&d2 / &d1).min(&d3 / &d2);
    // ‖v‖² < 2¹⁰⁷ + (Z|L| + 2⁵⁴ + 1)² for v = (M, m, MA + mB − kZ).
    let rest = mu2 - (IBig::ONE << 107);
    if rest <= IBig::ZERO {
        return None;
    }
    let s = UBig::try_from(rest).expect("positive").sqrt();
    let slack = (UBig::ONE << 54) + UBig::ONE;
    (s > slack).then(|| Bound {
        num: s - slack,
        log_den: z,
    })
}

/// The cases of one exponent `E`: `(count of D4, count of D5, the smallest
/// bound and where)`.
fn exponent(t: &UBig, big_e: i32) -> (usize, usize, Bound, (i32, i32, usize)) {
    // Step D2: d_E, with ϑ < 2^(E−P), so log_err = max(E, 0).
    let (na, la) = frac(t, big_e);
    let err_e = usize::try_from(big_e.max(0)).expect("≥ 0");
    let d = cf_bound(&na, la, 53, err_e).expect("step D2");
    // D3 gives d_E/2.
    let half = Bound {
        num: d.num.clone(),
        log_den: d.log_den + 1,
    };
    assert!(half.at_least_target(), "step D2 at E = {big_e}");
    // The cut of D3: the largest e with 2^(e+53) · 6367/10⁴ ≤ d_E/2
    // (6367/10⁴ > 2/π), i.e. 6367 · 2 · 2^(log_den + e + 53) ≤ 10⁴ d_num.
    let fits = |e: i32| {
        let sh = usize::try_from(i32::try_from(d.log_den).expect("small") + e + 53)
            .expect("log_den + e + 53 ≥ 0");
        (UBig::from(6367u32 * 2) << sh) <= UBig::from(10_000u32) * &d.num
    };
    let mut cut = big_e - 53;
    while !fits(cut) {
        cut -= 1;
    }
    let (mut n_cf, mut n_lat) = (0, 0);
    let (mut worst, mut at) = (half, (big_e, cut, 0));
    for e in (cut + 1).max(-1074)..=big_e - 53 {
        let s = big_e - e;
        let err = usize::try_from(e.max(0)).expect("≥ 0");
        let (b, z) = if s <= S_CF {
            // Step D4.
            let (nb, lb) = frac(t, e);
            let log_q = usize::try_from(54 + s).expect("> 0");
            n_cf += 1;
            (cf_bound(&nb, lb, log_q, err), 0)
        } else {
            // Step D5.
            n_lat += 1;
            SCALES
                .iter()
                .map(|&z| (lattice_bound(t, big_e, e, z), z))
                .find(|(b, _)| b.as_ref().is_some_and(Bound::at_least_target))
                .unwrap_or((None, 0))
        };
        let b = b.unwrap_or_else(|| panic!("no bound at E = {big_e}, e = {e}"));
        assert!(
            b.at_least_target(),
            "E = {big_e}, e = {e}: 2^{:.2}",
            b.log2()
        );
        if b.lt(&worst) {
            worst = b;
            at = (big_e, e, z);
        }
    }
    (n_cf, n_lat, worst, at)
}

fn run(exponents: &[i32]) -> Case {
    let t = two_over_pi();
    // The table of the crate starts 0xa2f9836e4e441529 (2/π = 0.a2f98…₁₆).
    assert_eq!(
        u64::try_from(&t >> (P - 64)).expect("64 bits"),
        0xa2f9_836e_4e44_1529
    );
    // No threads on WebAssembly (`wasm32-wasip1`): one chunk, run in place.
    let threads = if cfg!(target_family = "wasm") {
        1
    } else {
        std::thread::available_parallelism().map_or(1, usize::from)
    };
    if threads == 1 {
        return collect(exponents.iter().map(|&e| exponent(&t, e)).collect());
    }
    let chunks: Vec<Vec<i32>> = (0..threads)
        .map(|k| exponents.iter().copied().skip(k).step_by(threads).collect())
        .collect();
    let results: Vec<_> = std::thread::scope(|sc| {
        let handles: Vec<_> = chunks
            .iter()
            .map(|c| {
                let t = &t;
                sc.spawn(move || c.iter().map(|&e| exponent(t, e)).collect::<Vec<_>>())
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("a thread"))
            .collect()
    });
    collect(results)
}

type Case = (usize, usize, Bound, (i32, i32, usize));

/// Adds the counts and keeps the smallest bound.
fn collect(results: Vec<Case>) -> Case {
    let mut total = (0, 0);
    let mut worst: Option<(Bound, (i32, i32, usize))> = None;
    for (c, l, b, at) in results {
        total.0 += c;
        total.1 += l;
        if worst.as_ref().is_none_or(|(w, _)| b.lt(w)) {
            worst = Some((b, at));
        }
    }
    let (b, at) = worst.expect("one exponent");
    (total.0, total.1, b, at)
}

/// The exponents of the extreme `x_h`, of the closest arguments known
/// (Chapter 11.11, after Lemma D) and of the smallest bounds.
#[test]
fn lemma_d_on_a_sample_of_exponents() {
    let sample = [-53, -47, -29, 0, 1, 52, 53, 338, 671, 797, 971];
    let (c, l, b, at) = run(&sample);
    println!(
        "{} exponents: {c} cases by D4, {l} by D5; smallest 2^{:.2} at {at:?}",
        sample.len(),
        b.log2()
    );
    assert!(b.at_least_target());
}

/// Lemma D in full: every exponent of `x_h`.
#[test]
#[ignore = "about 20 s with --release: cargo test --release --test reduction_bound -- --ignored"]
fn lemma_d_in_full() {
    let all: Vec<i32> = (-53..=971).collect();
    let (c, l, b, at) = run(&all);
    println!(
        "all {} exponents: {c} cases by D4, {l} by D5; smallest 2^{:.2} at {at:?}",
        all.len(),
        b.log2()
    );
    assert!(b.at_least_target());
}
