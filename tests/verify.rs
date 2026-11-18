//! The acceptance tests of `verify` (L2.3; Chapter 11.10): the enclosures
//! of Algorithm V **contain** the exact residual — computed in rational
//! arithmetic (`dashu`) — on small and medium sparse systems, whatever the
//! data (overflow and underflow included); they are as narrow as the
//! chapter says on ordinary data; the double-double residual of step P1 is
//! within its stated accuracy; and Algorithm P reaches double-double
//! accuracy on an ill-conditioned system, with its result certified.

#![allow(clippy::disallowed_methods)] // L2.4: libm in tests and examples, never as a reference (clippy.toml)
#![allow(clippy::float_cmp)] // exact comparisons are the subject

use dashu_int::{IBig, UBig};
use dashu_ratio::RBig;
use proptest::prelude::*;
use xingo_extprec::verify::{
    PolishOptions, PolishStop, activity, norm_inf_bound, polish, residual, residual_csc,
    residual_dd, residual_of_dd,
};
use xingo_extprec::{DoubleDouble, Interval};
use xingo_sparse::CsrMatrix;

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

/// `2⁻ᵏ`.
fn pow2_neg(k: usize) -> RBig {
    RBig::from_parts(IBig::ONE, UBig::ONE << k)
}

/// `t ∈ X` for a rational `t`.
fn has(x: Interval, t: &RBig) -> bool {
    !x.is_empty()
        && (x.inf() == f64::NEG_INFINITY || q(x.inf()) <= *t)
        && (x.sup() == f64::INFINITY || q(x.sup()) >= *t)
}

/// A sparse matrix as rows of `(column, value)` pairs, columns increasing.
type Rows = Vec<Vec<(u32, f64)>>;

fn csr(n: usize, rows: &Rows) -> CsrMatrix<f64> {
    let mut ptr = vec![0];
    let (mut idx, mut val) = (Vec::new(), Vec::new());
    for r in rows {
        for &(j, v) in r {
            idx.push(j);
            val.push(v);
        }
        ptr.push(idx.len());
    }
    CsrMatrix::from_raw_parts(n, ptr, idx, val).expect("a valid matrix")
}

/// For each row: the exact `a_iᵀx − b_i` and `Σ|a_ij x_j| + |b_i|`.
fn exact_rows(rows: &Rows, x: &[RBig], b: &[f64]) -> Vec<(RBig, RBig)> {
    rows.iter()
        .zip(b)
        .map(|(r, &bi)| {
            let (mut sum, mut mass) = (-q(bi), abs(q(bi)));
            for &(j, v) in r {
                let t = q(v) * x[j as usize].clone();
                mass += abs(t.clone());
                sum += t;
            }
            (sum, mass)
        })
        .collect()
}

// ---------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------

const FRACTION: u64 = (1 << 52) - 1;

/// A normal `f64` with exponent in `emin..=emax`, either sign.
fn normal(emin: i32, emax: i32) -> impl Strategy<Value = f64> {
    (any::<bool>(), any::<u64>(), emin..=emax).prop_map(|(neg, bits, e)| {
        let biased = u64::try_from(e + 1023).expect("a normal exponent");
        f64::from_bits((u64::from(neg) << 63) | (biased << 52) | (bits & FRACTION))
    })
}

/// Ordinary data: moderate magnitudes and the small integers of LP models.
fn moderate() -> BoxedStrategy<f64> {
    prop_oneof![
        3 => normal(-30, 30),
        1 => (1_i32..=9, any::<bool>()).prop_map(|(k, neg)| f64::from(if neg { -k } else { k })),
    ]
    .boxed()
}

/// Data whose products overflow and underflow.
fn wild() -> BoxedStrategy<f64> {
    prop_oneof![
        2 => normal(-30, 30),
        2 => normal(-540, 540),
        1 => normal(-1022, -1000),
        1 => normal(1000, 1023),
    ]
    .boxed()
}

/// A normalized double-double around `h`.
fn with_trailing(h: impl Strategy<Value = f64>) -> impl Strategy<Value = Dd> {
    (
        h,
        -1.0..=1.0_f64,
        prop_oneof![3 => Just(0_i32), 1 => 0..40_i32],
    )
        .prop_map(|(h, r, gap)| Dd::from_sum(h, h * r * 2f64.powi(-53 - gap)))
}

/// A system `(n, rows, x, b)` with at most `max` rows and columns, about
/// `density` of the entries stored, the right-hand side either arbitrary
/// or near `A x` (the case of a residual: heavy cancellation).
fn system(
    max: usize,
    density: f64,
    value: fn() -> BoxedStrategy<f64>,
) -> impl Strategy<Value = (usize, Rows, Vec<Dd>, Vec<f64>)> {
    (1..=max, 1..=max).prop_flat_map(move |(m, n)| {
        let entry = prop::option::weighted(density, value());
        let rows = prop::collection::vec(prop::collection::vec(entry, n), m);
        let x = prop::collection::vec(with_trailing(value()), n);
        let b = prop::collection::vec((value(), any::<bool>()), m);
        (rows, x, b).prop_map(move |(dense, x, b)| {
            let rows: Rows = dense
                .iter()
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .filter_map(|(j, v)| v.map(|v| (u32::try_from(j).expect("small"), v)))
                        .collect()
                })
                .collect();
            // Half of the right-hand sides are the rounded activities.
            let b = rows
                .iter()
                .zip(b)
                .map(|(r, (free, near))| {
                    let act: f64 = r.iter().map(|&(j, v)| v * x[j as usize].hi()).sum();
                    if near && act.is_finite() { act } else { free }
                })
                .collect();
            (n, rows, x, b)
        })
    })
}

fn config(cases: u32) -> ProptestConfig {
    ProptestConfig {
        cases,
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

/// Invariant (V1) for the four enclosures of one system; with `sharp`,
/// also the width of the chapter and the accuracy of `residual_dd`.
fn check_system(
    n: usize,
    rows: &Rows,
    xd: &[Dd],
    b: &[f64],
    sharp: bool,
) -> Result<(), TestCaseError> {
    let a = csr(n, rows);
    let x: Vec<f64> = xd.iter().map(|v| v.hi()).collect();
    let k_max = rows.iter().map(Vec::len).max().unwrap_or(0);
    // 2⁻⁸⁵ covers `2β` for rows of up to 64 entries (Chapter 11.10,
    // "Width": β ≲ 2 k² u² S).
    let beta_slack = pow2_neg(85);
    let u = pow2_neg(53);
    let floor = pow2_neg(1000);

    // f64 solution.
    let xq: Vec<RBig> = x.iter().map(|&v| q(v)).collect();
    let exact = exact_rows(rows, &xq, b);
    let r = residual(&a, &x, b).expect("dimensions");
    let s = activity(&a, &x).expect("dimensions");
    prop_assert_eq!(&residual_csc(&a.to_csc(), &x, b).expect("dimensions"), &r);
    let mut norm = zero();
    for (i, (res, mass)) in exact.iter().enumerate() {
        prop_assert!(r[i].check_invariants().is_ok() && !r[i].is_empty());
        prop_assert!(has(r[i], res), "row {i}: {res} is not in {:?}", r[i]);
        prop_assert!(
            has(s[i], &(res.clone() + q(b[i]))),
            "activity of row {i}: {:?}",
            s[i]
        );
        norm = norm.max(abs(res.clone()));
        if sharp {
            prop_assert!(r[i].width().is_finite(), "row {i}: {:?}", r[i]);
            let allowed = RBig::from(4) * u.clone() * abs(res.clone())
                + beta_slack.clone() * mass.clone()
                + floor.clone();
            prop_assert!(
                q(r[i].width()) <= allowed,
                "row {i} is too wide: {:?}",
                r[i]
            );
        }
    }
    let bound = norm_inf_bound(&r);
    prop_assert!(
        bound == f64::INFINITY || q(bound) >= norm,
        "‖R‖∞ ≤ {bound:e}"
    );
    prop_assert_eq!(
        bound.is_finite(),
        r.iter().all(|v| v.inf().is_finite() && v.sup().is_finite())
    );

    // Double-double solution.
    let xq: Vec<RBig> = xd.iter().map(|&v| qd(v)).collect();
    let exact = exact_rows(rows, &xq, b);
    let r = residual_of_dd(&a, xd, b).expect("dimensions");
    let rd = residual_dd(&a, xd, b).expect("dimensions");
    for (i, (res, mass)) in exact.iter().enumerate() {
        prop_assert!(has(r[i], res), "dd, row {i}: {res} is not in {:?}", r[i]);
        if sharp {
            let allowed = RBig::from(4) * u.clone() * abs(res.clone())
                + beta_slack.clone() * mass.clone()
                + floor.clone();
            prop_assert!(
                q(r[i].width()) <= allowed,
                "dd, row {i} is too wide: {:?}",
                r[i]
            );
            // P1: b − A x within (4k + 20) u² (Σ|a x| + |b|).
            prop_assert!(rd[i].check_invariants().is_ok());
            let err = abs(qd(rd[i]) + res.clone());
            let allowed = RBig::from(4 * k_max + 20) * u.clone() * u.clone() * mass.clone();
            prop_assert!(err <= allowed, "residual_dd, row {i}: {:?}", rd[i]);
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(config(256))]

    /// (V1), width and P1 on small systems of ordinary data.
    #[test]
    fn small_systems_are_enclosed_and_sharp((n, rows, x, b) in system(8, 0.6, moderate)) {
        check_system(n, &rows, &x, &b, true)?;
    }

    /// (V1) on small systems whose terms overflow and underflow: the
    /// enclosure contains the exact residual or is unbounded.
    #[test]
    fn small_wild_systems_are_enclosed((n, rows, x, b) in system(8, 0.6, wild)) {
        check_system(n, &rows, &x, &b, false)?;
    }
}

proptest! {
    #![proptest_config(config(48))]

    /// The same on medium systems (up to 60 × 60, rows of up to 60
    /// entries: every lane of the kernels, the SIMD chunks and the tails).
    #[test]
    fn medium_systems_are_enclosed_and_sharp((n, rows, x, b) in system(60, 0.15, moderate)) {
        check_system(n, &rows, &x, &b, true)?;
    }

    #[test]
    fn medium_dense_wild_systems_are_enclosed((n, rows, x, b) in system(40, 0.9, wild)) {
        check_system(n, &rows, &x, &b, false)?;
    }
}

// ---------------------------------------------------------------------
// Polishing
// ---------------------------------------------------------------------

/// A dense LU with partial pivoting in `f64`: the caller's solve of
/// Algorithm P.
struct DenseLu {
    n: usize,
    lu: Vec<f64>,
    perm: Vec<usize>,
}

impl DenseLu {
    fn new(n: usize, a: &[f64]) -> Self {
        let mut lu = a.to_vec();
        let mut perm: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let p = (k..n)
                .max_by(|&i, &j| lu[i * n + k].abs().total_cmp(&lu[j * n + k].abs()))
                .expect("a row");
            perm.swap(k, p);
            for j in 0..n {
                lu.swap(k * n + j, p * n + j);
            }
            for i in k + 1..n {
                lu[i * n + k] /= lu[k * n + k];
                for j in k + 1..n {
                    lu[i * n + j] -= lu[i * n + k] * lu[k * n + j];
                }
            }
        }
        Self { n, lu, perm }
    }

    fn solve(&self, s: &mut [f64]) {
        let n = self.n;
        let mut y: Vec<f64> = self.perm.iter().map(|&p| s[p]).collect();
        for i in 0..n {
            for j in 0..i {
                y[i] -= self.lu[i * n + j] * y[j];
            }
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                y[i] -= self.lu[i * n + j] * y[j];
            }
            y[i] /= self.lu[i * n + i];
        }
        s.copy_from_slice(&y);
    }
}

/// The exact solution of the dense system, by rational elimination.
fn exact_solve(n: usize, a: &[f64], b: &[f64]) -> Vec<RBig> {
    let mut m: Vec<Vec<RBig>> = (0..n)
        .map(|i| {
            let mut row: Vec<RBig> = a[i * n..(i + 1) * n].iter().map(|&v| q(v)).collect();
            row.push(q(b[i]));
            row
        })
        .collect();
    for k in 0..n {
        let p = (k..n).find(|&i| m[i][k] != zero()).expect("nonsingular");
        m.swap(k, p);
        for i in 0..n {
            if i != k && m[i][k] != zero() {
                let f = m[i][k].clone() / m[k][k].clone();
                for j in k..=n {
                    m[i][j] = m[i][j].clone() - f.clone() * m[k][j].clone();
                }
            }
        }
    }
    (0..n).map(|i| m[i][n].clone() / m[i][i].clone()).collect()
}

fn dense_rows(n: usize, a: &[f64]) -> Rows {
    (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| a[i * n + j] != 0.0)
                .map(|j| (u32::try_from(j).expect("small"), a[i * n + j]))
                .collect()
        })
        .collect()
}

/// `max_i |x_i − x*_i| / max_i |x*_i|` as an `f64`.
fn relative_error(x: &[Dd], exact: &[RBig]) -> f64 {
    let scale = exact.iter().map(|v| abs(v.clone())).max().expect("n > 0");
    let err = x
        .iter()
        .zip(exact)
        .map(|(&v, e)| abs(qd(v) - e.clone()))
        .max()
        .expect("n > 0");
    (err / scale).to_f64().value()
}

/// The Hilbert matrix of order 8 as stored (`κ ≈ 1.5 · 10¹⁰`): the `f64`
/// solve leaves about six wrong digits out of sixteen; Algorithm P brings
/// the solution to double-double accuracy, and Algorithm V certifies a
/// residual fourteen orders of magnitude below the `f64` one.
#[test]
fn polish_reaches_double_double_accuracy_on_hilbert_8() {
    let n = 8;
    let a: Vec<f64> = (0..n * n)
        .map(|p| 1.0 / f64::from(u32::try_from(p / n + p % n + 1).expect("small")))
        .collect();
    let b = vec![1.0; n];
    let rows = dense_rows(n, &a);
    let m = csr(n, &rows);
    let lu = DenseLu::new(n, &a);
    let exact = exact_solve(n, &a, &b);

    let mut x0 = b.clone();
    lu.solve(&mut x0);
    let mut x: Vec<Dd> = x0.iter().map(|&v| Dd::from_f64(v)).collect();
    let before = relative_error(&x, &exact);
    let cert_before = norm_inf_bound(&residual(&m, &x0, &b).expect("dimensions"));

    let rep = polish(
        &m,
        &b,
        &mut x,
        &mut |s| lu.solve(s),
        &PolishOptions::default(),
    )
    .expect("dimensions");
    assert_eq!(rep.stop, PolishStop::Converged, "{rep:?}");
    assert!((1..=8).contains(&rep.iterations), "{rep:?}");
    assert!(rep.residual_after <= rep.residual_before, "{rep:?}");

    let after = relative_error(&x, &exact);
    assert!(
        before > 1e-13,
        "the f64 solve is unexpectedly accurate: {before:e}"
    );
    assert!(
        after < 1e-18 && after < 1e-6 * before,
        "{before:e} -> {after:e}"
    );

    // The certificate: it contains the exact residual of the polished x…
    let r = residual_of_dd(&m, &x, &b).expect("dimensions");
    let xq: Vec<RBig> = x.iter().map(|&v| qd(v)).collect();
    for (i, (res, _)) in exact_rows(&rows, &xq, &b).iter().enumerate() {
        assert!(has(r[i], res), "row {i}");
    }
    // …and it is small: ‖A‖∞‖x‖∞ is of order 10⁹ here.
    let cert_after = norm_inf_bound(&r);
    let scale = x.iter().map(|v| v.hi().abs()).fold(0.0, f64::max);
    assert!(cert_before.is_finite() && cert_before > 1e-12 * scale * f64::EPSILON);
    assert!(
        cert_after < 1e-27 * scale,
        "{cert_after:e} (scale {scale:e})"
    );
    assert!(
        cert_after < 1e-10 * cert_before,
        "{cert_before:e} -> {cert_after:e}"
    );
    // The f64 residual, recomputed in f64, cannot see any of this.
    assert!(rep.residual_before >= cert_after);
}

proptest! {
    #![proptest_config(config(64))]

    /// Diagonally dominant random systems: P converges from the zero
    /// vector, never increases the residual, and its result is certified
    /// at double-double level.
    #[test]
    fn polish_converges_on_dominant_systems(
        (n, off, diag, b) in (1_usize..=10).prop_flat_map(|n| (
            Just(n),
            prop::collection::vec(prop::option::weighted(0.4, -1.0..1.0_f64), n * n),
            prop::collection::vec(1.0..2.0_f64, n),
            prop::collection::vec(moderate(), n),
        )),
    ) {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j {
                    // Strictly dominant: the row sum of the rest is < n.
                    diag[i] * f64::from(u32::try_from(n).expect("small"))
                } else {
                    off[i * n + j].unwrap_or(0.0)
                };
            }
        }
        let rows = dense_rows(n, &a);
        let m = csr(n, &rows);
        let lu = DenseLu::new(n, &a);
        let mut x = vec![Dd::ZERO; n];
        let rep = polish(&m, &b, &mut x, &mut |s| lu.solve(s), &PolishOptions::default())
            .expect("dimensions");
        prop_assert_eq!(rep.stop, PolishStop::Converged, "{:?}", rep);
        prop_assert!(rep.residual_after <= rep.residual_before, "{rep:?}");
        for v in &x {
            prop_assert!(v.check_invariants().is_ok());
        }
        let r = residual_of_dd(&m, &x, &b).expect("dimensions");
        let xq: Vec<RBig> = x.iter().map(|&v| qd(v)).collect();
        let mut mass_max = zero();
        for (i, (res, mass)) in exact_rows(&rows, &xq, &b).iter().enumerate() {
            prop_assert!(has(r[i], res), "row {i}: {:?}", r[i]);
            mass_max = mass_max.max(mass.clone());
        }
        // 2⁻⁸⁰ of the size of the terms: certified far below f64.
        let cert = norm_inf_bound(&r);
        prop_assert!(cert.is_finite() && q(cert) <= pow2_neg(80) * mass_max + pow2_neg(1000));
    }
}

// ---------------------------------------------------------------------
// Near Ω (review X0.3i: the overflow of an intermediate step of the last
// 2Sum of Dot2Err)
// ---------------------------------------------------------------------

/// Values whose partial sums touch `±Ω` and come back (as in
/// `tests/exact.rs`).
fn near_omega() -> BoxedStrategy<f64> {
    let top = (0_u64..8, any::<bool>()).prop_map(|(k, neg)| {
        let v = f64::from_bits(f64::MAX.to_bits() - k);
        if neg { -v } else { v }
    });
    let half = (0_i32..5, any::<bool>(), any::<bool>()).prop_map(|(j, down, neg)| {
        let step = f64::from(j) * 2f64.powi(918);
        let v = if down {
            2f64.powi(970) - step
        } else {
            2f64.powi(970) + step
        };
        if neg { -v } else { v }
    });
    prop_oneof![3 => top, 3 => half, 2 => normal(900, 1023), 1 => normal(-10, 10), 1 => Just(0.0)]
        .boxed()
}

/// The minimal case of the review as one row of a residual: before the
/// correction the enclosure was finite (and, by the margin of V3, still
/// contained the residual); now the row is unbounded, as (K2) requires.
#[test]
fn residual_row_at_the_overflow_corner() {
    let lane0 = [
        0x7fef_ffff_ffff_fffe_u64,
        0x7c90_0000_0000_0001,
        0xfc8f_ffff_ffff_fffe,
        0xfc8f_ffff_ffff_fffc,
        0xf970_0000_0000_0000,
    ];
    // 16 entries and the pair (−1, b) as the 17th: lane 0 is 0, 4, 8, 12,
    // 16 — the last is −b.
    let mut row: Vec<(u32, f64)> = Vec::new();
    for (k, &bits) in lane0[..4].iter().enumerate() {
        row.push((u32::try_from(4 * k).expect("small"), f64::from_bits(bits)));
    }
    let b = -f64::from_bits(lane0[4]);
    let a = csr(16, &vec![row.clone()]);
    let x = vec![1.0; 16];
    let r = residual(&a, &x, &[b]).expect("dimensions");
    let exact = row.iter().fold(-q(b), |acc, &(_, v)| acc + q(v));
    assert!(has(r[0], &exact));
    assert_eq!(r[0], Interval::ENTIRE);
}

proptest! {
    #![proptest_config(config(1_000))]

    /// (V1) with entries, solutions and right-hand sides near `Ω`.
    #[test]
    fn residuals_near_omega_are_enclosed(
        (n, rows, x, b) in (1_usize..=16, 1_usize..=4).prop_flat_map(|(n, m)| (
            Just(n),
            prop::collection::vec(prop::collection::vec(prop::option::weighted(0.7, near_omega()), n), m),
            prop::collection::vec(prop_oneof![4 => Just(1.0), 1 => Just(-1.0), 1 => Just(0.5)], n),
            prop::collection::vec(near_omega(), m),
        )),
    ) {
        let rows: Rows = rows
            .iter()
            .map(|r| r.iter().enumerate()
                .filter_map(|(j, v)| v.filter(|&v| v != 0.0).map(|v| (u32::try_from(j).expect("small"), v)))
                .collect())
            .collect();
        let a = csr(n, &rows);
        let r = residual(&a, &x, &b).expect("dimensions");
        let xq: Vec<RBig> = x.iter().map(|&v| q(v)).collect();
        for (i, (res, _)) in exact_rows(&rows, &xq, &b).iter().enumerate() {
            prop_assert!(has(r[i], res), "row {i}: {res} is not in {:?}", r[i]);
        }
    }
}
