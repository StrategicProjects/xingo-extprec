//! The acceptance tests of Algorithm E (`verify::forward_error_bound`,
//! Chapter 11.10, Theorem E; contract X0.3, decision log D-0125): the
//! bound **never** lies below the true error `x̃ − A⁻¹b`, computed in
//! exact rational arithmetic (`dashu`), on random and directed systems,
//! with `x̃` in `f64` and in double-double, whatever the closure — the
//! theorem holds for every `R`, so a wrong, noisy or adversarial solve may
//! only prevent the certificate or loosen the bound; exactly singular
//! matrices are never certified; the special cases of the contract (§6)
//! give the outcomes it states; and the result is deterministic.

#![allow(clippy::disallowed_methods)] // L2.4: libm in tests and examples, never as a reference (clippy.toml)
#![allow(clippy::float_cmp)] // exact comparisons are the subject
#![allow(clippy::cast_precision_loss)] // small indices as f64

use dashu_int::{IBig, UBig};
use dashu_ratio::RBig;
use proptest::prelude::*;
use xingo_extprec::verify::{
    ForwardErrorBound, ForwardErrorOptions, ForwardErrorOutcome, PolishOptions, VerifyError,
    forward_error_bound, polish, residual, residual_of_dd,
};
use xingo_extprec::{DoubleDouble, Interval};
use xingo_sparse::CsrMatrix;

type Dd = DoubleDouble;

const REFINE: ForwardErrorOptions = ForwardErrorOptions { refine_rows: true };
const PLAIN: ForwardErrorOptions = ForwardErrorOptions { refine_rows: false };

// ---------------------------------------------------------------------
// Exact arithmetic
// ---------------------------------------------------------------------

fn q(x: f64) -> RBig {
    RBig::try_from(x).expect("a finite f64")
}

fn qd(x: Dd) -> RBig {
    q(x.hi()) + q(x.lo())
}

fn zero() -> RBig {
    RBig::from(0)
}

fn abs(x: RBig) -> RBig {
    if x < zero() { -x } else { x }
}

/// `2ᵏ` for any integer `k`.
fn pow2(k: i32) -> RBig {
    let m = UBig::ONE << usize::try_from(k.unsigned_abs()).expect("small");
    if k >= 0 {
        RBig::from_parts(IBig::from(m), UBig::ONE)
    } else {
        RBig::from_parts(IBig::ONE, m)
    }
}

/// `2ᵏ` as an `f64`, subnormal included (`powi` would underflow through
/// `1/2¹⁰⁶⁰`).
fn pow2f(k: i32) -> f64 {
    assert!((-1074..=1023).contains(&k));
    if k >= -1022 {
        f64::from_bits(u64::try_from(k + 1023).expect("normal") << 52)
    } else {
        f64::from_bits(1_u64 << (k + 1074))
    }
}

/// A dense matrix, by rows.
#[derive(Clone, Debug)]
struct Dense {
    n: usize,
    a: Vec<f64>,
}

impl Dense {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.a[i * self.n + j]
    }

    fn transpose(&self) -> Self {
        let n = self.n;
        Self {
            n,
            a: (0..n * n).map(|k| self.at(k % n, k / n)).collect(),
        }
    }

    /// The stored entries are the nonzero ones.
    fn csr(&self) -> CsrMatrix<f64> {
        let n = self.n;
        let mut ptr = vec![0];
        let (mut idx, mut val) = (Vec::new(), Vec::new());
        for i in 0..n {
            for j in 0..n {
                let v = self.at(i, j);
                if v != 0.0 {
                    idx.push(u32::try_from(j).expect("small"));
                    val.push(v);
                }
            }
            ptr.push(idx.len());
        }
        CsrMatrix::from_raw_parts(n, ptr, idx, val).expect("a valid matrix")
    }

    /// `fl(A 1)`.
    fn row_sums(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| (0..self.n).map(|j| self.at(i, j)).sum())
            .collect()
    }

    /// `A⁻¹ b` in rational arithmetic; `None` if `A` is singular.
    fn exact_solve(&self, b: &[RBig]) -> Option<Vec<RBig>> {
        let n = self.n;
        let mut m: Vec<Vec<RBig>> = (0..n)
            .map(|i| {
                let mut row: Vec<RBig> = (0..n).map(|j| q(self.at(i, j))).collect();
                row.push(b[i].clone());
                row
            })
            .collect();
        for k in 0..n {
            let p = (k..n).find(|&i| m[i][k] != zero())?;
            m.swap(k, p);
            for i in 0..n {
                if i != k && m[i][k] != zero() {
                    let f = m[i][k].clone() / m[k][k].clone();
                    for j in k..=n {
                        let t = f.clone() * m[k][j].clone();
                        m[i][j] -= t;
                    }
                }
            }
        }
        Some((0..n).map(|i| m[i][n].clone() / m[i][i].clone()).collect())
    }
}

/// The errors `x̃ − A⁻¹b`, exactly; `None` if `A` is singular.
fn exact_errors(a: &Dense, b: &[f64], x: &[RBig]) -> Option<Vec<RBig>> {
    let bq: Vec<RBig> = b.iter().map(|&v| q(v)).collect();
    let xs = a.exact_solve(&bq)?;
    Some(x.iter().zip(xs).map(|(xt, xs)| xt.clone() - xs).collect())
}

/// Theorem E against the exact errors, and the shape of invariant E3.
fn check(bound: &ForwardErrorBound, e: Option<&[RBig]>, what: &str) -> Result<(), String> {
    if bound.is_certified() {
        let Some(e) = e else {
            return Err(format!(
                "{what}: certified without a solution (singular A or b not real): {bound:?}"
            ));
        };
        if !(bound.norm_inf.is_finite() && (0.0..1.0).contains(&bound.contraction)) {
            return Err(format!("{what}: E3: {bound:?}"));
        }
        let mut largest = zero();
        for (p, (ep, &bp)) in e.iter().zip(&bound.componentwise).enumerate() {
            let ep = abs(ep.clone());
            if !(0.0..=bound.norm_inf).contains(&bp) {
                return Err(format!(
                    "{what}: E3, ε_{p} = {bp:e}, ε = {:e}",
                    bound.norm_inf
                ));
            }
            if ep > q(bp) {
                return Err(format!(
                    "{what}: |e_{p}| = {:e} > ε_{p} = {bp:e}",
                    ep.to_f64().value()
                ));
            }
            largest = largest.max(ep);
        }
        if largest > q(bound.norm_inf) {
            return Err(format!(
                "{what}: ‖e‖∞ = {:e} > ε = {:e}",
                largest.to_f64().value(),
                bound.norm_inf
            ));
        }
    } else if bound.norm_inf != f64::INFINITY
        || bound.componentwise.iter().any(|&c| c != f64::INFINITY)
    {
        return Err(format!("{what}: E3, finite bounds without a certificate"));
    }
    Ok(())
}

/// `‖e‖∞` as an `f64` (rounded), for the measurements of tightness.
fn norm_f64(e: &[RBig]) -> f64 {
    e.iter()
        .map(|v| abs(v.clone()))
        .max()
        .unwrap_or_else(zero)
        .to_f64()
        .value()
}

// ---------------------------------------------------------------------
// The caller's solves
// ---------------------------------------------------------------------

/// A dense LU with partial pivoting in `f64` (no safeguard: a zero pivot
/// gives infinities and NaNs, which Algorithm E must refuse).
struct DenseLu {
    n: usize,
    lu: Vec<f64>,
    perm: Vec<usize>,
}

impl DenseLu {
    fn new(a: &Dense) -> Self {
        let n = a.n;
        let mut lu = a.a.clone();
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

/// A deterministic xorshift generator (no state shared between tests).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn sym(&mut self) -> f64 {
        2.0 * ((self.next() >> 11) as f64 / (1_u64 << 53) as f64) - 1.0
    }
}

/// The transposed solves of the tests: correct, and every kind of wrong.
#[derive(Clone, Copy, Debug)]
enum Closure {
    /// `≈ A⁻ᵀ c` by the `f64` LU of `Aᵀ`.
    Lu,
    /// `s · A⁻ᵀ c`.
    Scaled(f64),
    /// `A⁻ᵀ c` with a relative noise of the given size in each component.
    Noisy(f64),
    /// `c` unchanged.
    Identity,
    /// Zero.
    Zero,
    /// NaN.
    Nan,
    /// Random values, independent of `c`.
    Garbage,
    /// The solve of `Aᵀ` with its first entry changed by the given amount
    /// (another matrix's inverse).
    Perturbed(f64),
    /// Right on the first call of each row, NaN on the second (the
    /// refinement of step E2 must be discarded).
    NanOnRefine,
}

impl Closure {
    const ALL: [Self; 9] = [
        Self::Lu,
        Self::Scaled(1.5),
        Self::Noisy(1e-3),
        Self::Identity,
        Self::Zero,
        Self::Nan,
        Self::Garbage,
        Self::Perturbed(0.25),
        Self::NanOnRefine,
    ];

    /// The bound of `x̃` with this closure.
    fn bound(self, a: &Dense, r: &[Interval], opts: ForwardErrorOptions) -> ForwardErrorBound {
        let at = a.transpose();
        let lu = DenseLu::new(&at);
        let perturbed = |delta: f64| {
            let mut m = at.clone();
            m.a[0] += delta;
            DenseLu::new(&m)
        };
        let other = match self {
            Self::Perturbed(delta) => Some(perturbed(delta)),
            _ => None,
        };
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut calls = 0_usize;
        let mut f = |c: &mut [f64]| {
            calls += 1;
            match self {
                Self::Lu => lu.solve(c),
                Self::Scaled(s) => {
                    lu.solve(c);
                    for v in c.iter_mut() {
                        *v *= s;
                    }
                }
                Self::Noisy(t) => {
                    lu.solve(c);
                    for v in c.iter_mut() {
                        *v *= 1.0 + t * rng.sym();
                    }
                }
                Self::Identity => {}
                Self::Zero => c.fill(0.0),
                Self::Nan => c.fill(f64::NAN),
                Self::Garbage => {
                    for v in c.iter_mut() {
                        *v = rng.sym();
                    }
                }
                Self::Perturbed(_) => other.as_ref().expect("built").solve(c),
                Self::NanOnRefine => {
                    if calls.is_multiple_of(2) && opts.refine_rows {
                        c.fill(f64::NAN);
                    } else {
                        lu.solve(c);
                    }
                }
            }
        };
        forward_error_bound(&a.csr(), r, &mut f, &opts).expect("dimensions")
    }
}

// ---------------------------------------------------------------------
// One system: x̃ in f64 and polished, every closure, both options
// ---------------------------------------------------------------------

/// `(label, [r], the exact value of x̃)`.
type Solution = (&'static str, Vec<Interval>, Option<Vec<RBig>>);

/// The `f64` solution by the LU of `A` (with `fl(A 1)`'s right-hand side
/// when `b` is `None`), its polished version, their enclosures.
struct Case {
    a: Dense,
    b: Vec<f64>,
    x: Vec<f64>,
    xd: Vec<Dd>,
}

impl Case {
    fn new(a: Dense, b: Option<Vec<f64>>) -> Self {
        let b = b.unwrap_or_else(|| a.row_sums());
        let lu = DenseLu::new(&a);
        let mut x = b.clone();
        lu.solve(&mut x);
        let mut xd: Vec<Dd> = x.iter().map(|&v| Dd::from_f64(v)).collect();
        if x.iter().all(|v| v.is_finite()) {
            let _ = polish(
                &a.csr(),
                &b,
                &mut xd,
                &mut |s| lu.solve(s),
                &PolishOptions::default(),
            )
            .expect("dimensions");
        }
        Self { a, b, x, xd }
    }

    /// The two solutions as `(label, [r], exact value)`; the exact value is
    /// `None` when `x̃` is not finite.
    fn solutions(&self) -> Vec<Solution> {
        let m = self.a.csr();
        let finite = self.x.iter().all(|v| v.is_finite());
        let finite_dd = self
            .xd
            .iter()
            .all(|v| v.hi().is_finite() && v.lo().is_finite());
        vec![
            (
                "f64",
                residual(&m, &self.x, &self.b).expect("dimensions"),
                finite.then(|| self.x.iter().map(|&v| q(v)).collect()),
            ),
            (
                "dd",
                residual_of_dd(&m, &self.xd, &self.b).expect("dimensions"),
                finite_dd.then(|| self.xd.iter().map(|&v| qd(v)).collect()),
            ),
        ]
    }

    /// Theorem E for every closure, solution and option; returns the
    /// bounds of the LU closure with refinement, `(f64, dd)`.
    fn check_all(&self, closures: &[Closure]) -> Result<Vec<ForwardErrorBound>, String> {
        let mut lu_bounds = Vec::new();
        // No oracle (and no certificate allowed) for a singular matrix or a
        // right-hand side that is not a real number.
        let singular = !self.b.iter().all(|v| v.is_finite())
            || self.a.exact_solve(&vec![zero(); self.a.n]).is_none();
        for (label, r, x) in self.solutions() {
            let e = match &x {
                Some(x) if !singular => exact_errors(&self.a, &self.b, x),
                _ => None,
            };
            for &c in closures {
                for opts in [REFINE, PLAIN] {
                    let bound = c.bound(&self.a, &r, opts);
                    let what = format!("{label}, {c:?}, refine {}", opts.refine_rows);
                    if x.is_none() && bound.is_certified() {
                        return Err(format!("{what}: certified a non-finite x̃"));
                    }
                    check(&bound, e.as_deref(), &what)?;
                    if matches!(c, Closure::Lu) && opts.refine_rows {
                        lu_bounds.push(bound);
                    }
                }
            }
        }
        Ok(lu_bounds)
    }
}

// ---------------------------------------------------------------------
// Directed families
// ---------------------------------------------------------------------

/// Dense `U Σ Vᵀ` with three random reflectors on each side, singular
/// values geometric from 1 to `1/κ` (the pilot's family).
fn dense_kappa(n: usize, kappa: f64, seed: u64) -> Dense {
    let mut rng = Rng(seed);
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        let t = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };
        a[i * n + i] = kappa.powf(-t);
    }
    for _side in 0..2 {
        for _ in 0..3 {
            let v: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
            let vv: f64 = v.iter().map(|x| x * x).sum();
            for j in 0..n {
                let dot: f64 = (0..n).map(|i| v[i] * a[i * n + j]).sum();
                let f = 2.0 * dot / vv;
                for i in 0..n {
                    a[i * n + j] -= f * v[i];
                }
            }
            let mut t = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    t[j * n + i] = a[i * n + j];
                }
            }
            a = t;
        }
    }
    Dense { n, a }
}

/// Integers in `−9..=9` whose last row is the sum of the first two
/// (exactly singular), its first entry then changed by `delta`.
fn integers_singular(n: usize, delta: f64, seed: u64) -> Dense {
    let mut rng = Rng(seed);
    let mut a: Vec<f64> = (0..n * n)
        .map(|_| f64::from(u8::try_from(rng.next() % 19).expect("small")) - 9.0)
        .collect();
    for j in 0..n {
        a[(n - 1) * n + j] = a[j] + a[n + j];
    }
    a[(n - 1) * n] += delta;
    Dense { n, a }
}

/// Condition numbers 10² to 10¹⁶, `x̃` in `f64` and polished, every
/// closure: never a false bound; the LU closure with refinement certifies
/// all of them, and for an `f64` `x̃` up to `κ = 10¹⁴` the bound is within
/// 10 % of the true error.
#[test]
fn condition_numbers_from_1e2_to_1e16() {
    let n = 12;
    for kappa in [1e2, 1e6, 1e10, 1e14, 1e16] {
        for seed in [1_u64, 2] {
            let a = dense_kappa(n, kappa, 0x9e37_79b9_7f4a_7c15 ^ (seed * 31 + 7));
            let case = Case::new(a, None);
            let bounds = case
                .check_all(&Closure::ALL)
                .unwrap_or_else(|e| panic!("κ = {kappa:e}, seed {seed}: {e}"));
            for (bound, (label, _, x)) in bounds.iter().zip(case.solutions()) {
                assert!(
                    bound.is_certified(),
                    "κ = {kappa:e}, seed {seed}, {label}: {bound:?}"
                );
                if label == "f64" && kappa <= 1e14 {
                    let e =
                        exact_errors(&case.a, &case.b, &x.expect("finite")).expect("nonsingular");
                    let truth = norm_f64(&e);
                    assert!(
                        truth > 0.0 && bound.norm_inf <= 1.1 * truth,
                        "κ = {kappa:e}: ε = {:e}, ‖e‖∞ = {truth:e}",
                        bound.norm_inf
                    );
                }
            }
        }
    }
}

/// The refinement of step E2 lowers `α` by orders of magnitude at `κ ≈
/// 10¹⁶`, order 60 (the pilot's dense60-k1e16 cases): on the first
/// matrix, one solve per row does not certify at all.
#[test]
fn refinement_certifies_what_one_solve_per_row_cannot() {
    let n = 60;
    for seed in [1_u64, 2, 3, 4] {
        let a = dense_kappa(n, 1e16, 0x9e37_79b9_7f4a_7c15 ^ (seed * 31 + 60));
        let case = Case::new(a, None);
        let r = residual(&case.a.csr(), &case.x, &case.b).expect("dimensions");
        let refined = Closure::Lu.bound(&case.a, &r, REFINE);
        let plain = Closure::Lu.bound(&case.a, &r, PLAIN);
        assert!(
            refined.is_certified() && refined.contraction < 0.01,
            "seed {seed}: {refined:?}"
        );
        assert_eq!(refined.solves, 2 * n);
        assert!(
            refined.contraction < plain.contraction / 50.0,
            "seed {seed}: α = {:e} refined, {:e} plain",
            refined.contraction,
            plain.contraction
        );
        if seed == 1 {
            assert_eq!(
                plain.outcome,
                ForwardErrorOutcome::NotContracting,
                "{plain:?}"
            );
        }
    }
}

/// Exactly singular matrices (a row the sum of two others; a zero column),
/// with every closure — the identity and the inverse of a perturbation
/// among them: never certified (invariant E2).
#[test]
fn singular_matrices_are_never_certified() {
    let mut cases = Vec::new();
    for (n, seed) in [(3, 1_u64), (6, 2), (12, 3)] {
        cases.push(integers_singular(n, 0.0, seed));
        let mut zero_column = dense_kappa(n, 10.0, seed);
        for i in 0..n {
            zero_column.a[i * n + n / 2] = 0.0;
        }
        cases.push(zero_column);
    }
    for a in cases {
        assert!(a.exact_solve(&vec![zero(); a.n]).is_none());
        for b in [None, Some(vec![1.0; a.n])] {
            let case = Case::new(a.clone(), b);
            let mut closures = Closure::ALL.to_vec();
            closures.extend([Closure::Perturbed(1e-8), Closure::Perturbed(2f64.powi(-40))]);
            for (label, r, _) in case.solutions() {
                for &c in &closures {
                    for opts in [REFINE, PLAIN] {
                        let bound = c.bound(&case.a, &r, opts);
                        assert!(
                            !bound.is_certified(),
                            "n = {}, {label}, {c:?}: {bound:?}",
                            a.n
                        );
                        check(&bound, None, "singular").expect("E3");
                    }
                }
            }
        }
    }
}

/// Integer matrices `2⁻²⁰` and `2⁻⁴⁰` away from singular: never a false
/// bound, and certified with the LU closure.
#[test]
fn nearly_singular_matrices() {
    for (n, seed) in [(6, 4_u64), (12, 5)] {
        for delta in [2f64.powi(-20), 2f64.powi(-40)] {
            let case = Case::new(integers_singular(n, delta, seed), None);
            let bounds = case
                .check_all(&Closure::ALL)
                .unwrap_or_else(|e| panic!("n = {n}, δ = {delta:e}: {e}"));
            assert!(
                bounds.iter().all(ForwardErrorBound::is_certified),
                "n = {n}, δ = {delta:e}: {bounds:?}"
            );
        }
    }
}

/// Data scaled into the subnormal range and up to near `Ω`: the theorem
/// holds through underflow (it is in `β` and in the interval operations)
/// and overflow (refused, never false).
#[test]
fn subnormal_and_huge_data() {
    let base = integers_singular(5, 1.0, 9);
    assert!(base.exact_solve(&vec![zero(); 5]).is_some());
    for scale in [-1060, -1040, -900, 900, 1000, 1015] {
        let s = pow2f(scale);
        let a = Dense {
            n: 5,
            a: base.a.iter().map(|v| v * s).collect(),
        };
        // Entries exact multiples of 2^scale (|v| ≤ 18 needs 5 bits).
        for (v, w) in a.a.iter().zip(&base.a) {
            assert_eq!(q(*v), q(*w) * pow2(scale));
        }
        for b in [None, Some(vec![s; 5]), Some(vec![1.0; 5])] {
            let case = Case::new(a.clone(), b);
            case.check_all(&Closure::ALL)
                .unwrap_or_else(|e| panic!("2^{scale}: {e}"));
        }
    }
    // Entries at Ω: Dot2Err overflows, nothing is certified.
    let big = Dense {
        n: 2,
        a: vec![f64::MAX, f64::MAX, -f64::MAX, f64::MAX],
    };
    let case = Case::new(big, Some(vec![1.0, 0.0]));
    case.check_all(&Closure::ALL).expect("never false");
}

/// The underflow of a product of `C` is in `β`: here `c_01 = −2⁻¹²⁰⁰`,
/// below the smallest subnormal, and the true error `e_0 = −2⁻¹¹⁰⁰` is
/// nonzero although `(R r)_0 = 0` exactly — only the term `ρ_0 ε` of step
/// E7, with `ρ_0` made of `β`, covers it.
#[test]
fn underflow_in_c_is_bounded() {
    let (big, tiny) = (2f64.powi(600), 2f64.powi(-600));
    let a = CsrMatrix::from_raw_parts(2, vec![0, 2, 3], vec![0, 1, 1], vec![big, tiny, big])
        .expect("a valid matrix");
    // A e = r for e = (−2⁻¹¹⁰⁰, 2¹⁰⁰): r = (0, 2⁷⁰⁰), both exact.
    let r = [Interval::ZERO, Interval::point(2f64.powi(700))];
    let e = [-pow2(-1100), pow2(100)];
    for opts in [REFINE, PLAIN] {
        // R = diag(2⁻⁶⁰⁰): the solve of the diagonal.
        let bound = forward_error_bound(
            &a,
            &r,
            &mut |c| c.iter_mut().for_each(|v| *v *= tiny),
            &opts,
        )
        .expect("dimensions");
        assert!(bound.is_certified(), "{bound:?}");
        check(&bound, Some(&e), "underflow").expect("Theorem E");
        assert!(bound.componentwise[0] > 0.0 && bound.componentwise[0] < 1e-250);
    }
}

/// A damped solve, `R ≈ A⁻¹/2`: `C ≈ I/2`, the error is `2 R r`, and only
/// the factor `1/(1 − α)` of step E6 and the term `ρ_p ε` of step E7 cover
/// it.
#[test]
fn a_contraction_far_from_zero() {
    let a = Dense {
        n: 3,
        a: vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0],
    };
    let x = [1.0, 1.0, 1.0];
    let b = [1.0, 2.0, 3.0];
    let r = residual(&a.csr(), &x, &b).expect("dimensions");
    let e = exact_errors(&a, &b, &x.map(q)).expect("nonsingular");
    for s in [0.5, 0.3, 1.5, 1.7] {
        let bound = Closure::Scaled(s).bound(&a, &r, PLAIN);
        assert!(bound.is_certified(), "s = {s}: {bound:?}");
        assert!((bound.contraction - (1.0 - s).abs()).abs() < 1e-12);
        check(&bound, Some(&e), &format!("scale {s}")).expect("Theorem E");
        let bound = Closure::Scaled(s).bound(&a, &r, REFINE);
        check(&bound, Some(&e), &format!("scale {s}, refined")).expect("Theorem E");
    }
}

// ---------------------------------------------------------------------
// Special cases (contract §6)
// ---------------------------------------------------------------------

fn small() -> Dense {
    Dense {
        n: 2,
        a: vec![2.0, 1.0, 1.0, 3.0],
    }
}

#[test]
fn the_empty_system_is_certified_without_a_solve() {
    let a = CsrMatrix::from_raw_parts(0, vec![0], vec![], vec![]).expect("a valid matrix");
    for opts in [REFINE, PLAIN] {
        let bound =
            forward_error_bound(&a, &[], &mut |_| panic!("no solve"), &opts).expect("dimensions");
        assert_eq!(bound.outcome, ForwardErrorOutcome::Certified);
        assert_eq!(
            (bound.norm_inf, bound.contraction, bound.solves),
            (0.0, 0.0, 0)
        );
        assert_eq!(bound.componentwise, Vec::<f64>::new());
    }
}

#[test]
fn dimensions_are_checked_before_any_solve() {
    let err = |expected, got| Err(VerifyError::Dimension { expected, got });
    let a = small().csr();
    let mut never = |_: &mut [f64]| panic!("no solve");
    assert_eq!(
        forward_error_bound(&a, &[Interval::ZERO], &mut never, &REFINE),
        err(2, 1)
    );
    assert_eq!(
        forward_error_bound(&a, &[Interval::ZERO; 3], &mut never, &REFINE),
        err(2, 3)
    );
    let wide = CsrMatrix::from_raw_parts(2, vec![0, 2], vec![0, 1], vec![1.0, 1.0])
        .expect("a valid matrix");
    assert_eq!(
        forward_error_bound(&wide, &[Interval::ZERO], &mut never, &REFINE),
        err(1, 2)
    );
}

#[test]
fn unbounded_or_empty_residuals_are_not_finite() {
    let a = small();
    let m = a.csr();
    let lu = DenseLu::new(&a.transpose());
    for opts in [REFINE, PLAIN] {
        let bound = forward_error_bound(
            &m,
            &[Interval::ZERO, Interval::EMPTY],
            &mut |_| panic!("no solve"),
            &opts,
        )
        .expect("dimensions");
        assert_eq!(
            (bound.outcome, bound.solves),
            (ForwardErrorOutcome::NotFinite, 0)
        );
        check(&bound, None, "empty").expect("E3");
        let bound = forward_error_bound(
            &m,
            &[Interval::ZERO, Interval::ENTIRE],
            &mut |c| lu.solve(c),
            &opts,
        )
        .expect("dimensions");
        assert_eq!(bound.outcome, ForwardErrorOutcome::NotFinite, "{bound:?}");
        assert!(bound.contraction < 1e-15);
        check(&bound, None, "entire").expect("E3");
        // A residual that is unbounded on one side only.
        let half = Interval::new(0.0, f64::INFINITY).expect("an interval");
        let bound = forward_error_bound(&m, &[Interval::ZERO, half], &mut |c| lu.solve(c), &opts)
            .expect("dimensions");
        assert_eq!(bound.outcome, ForwardErrorOutcome::NotFinite);
    }
}

#[test]
fn non_finite_data_and_solves() {
    let a = small();
    let r = residual(&a.csr(), &[0.4, 0.2], &[1.0, 1.0]).expect("dimensions");
    let lu = DenseLu::new(&a.transpose());
    // A NaN or infinite entry cannot be stored (`CsrMatrix` refuses it);
    // the failed hypothesis of Dot2Err is reached by overflow instead: β =
    // +∞, not contracting (with a finite solve, that of the clean matrix).
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(CsrMatrix::from_raw_parts(1, vec![0, 1], vec![0], vec![bad]).is_err());
    }
    let m = CsrMatrix::from_raw_parts(
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![2.0, f64::MAX, 1.0, 3.0],
    )
    .expect("a valid matrix");
    for opts in [REFINE, PLAIN] {
        let bound = forward_error_bound(&m, &r, &mut |c| lu.solve(c), &opts).expect("dimensions");
        assert_eq!(
            bound.outcome,
            ForwardErrorOutcome::NotContracting,
            "{bound:?}"
        );
        assert!(bound.contraction >= 1.0);
        // A product that overflows: Ω · 2.
        let omega = CsrMatrix::from_raw_parts(1, vec![0, 1], vec![0], vec![f64::MAX])
            .expect("a valid matrix");
        let bound = forward_error_bound(&omega, &[Interval::ZERO], &mut |c| c[0] *= 2.0, &opts)
            .expect("dimensions");
        assert_eq!(
            (bound.outcome, bound.contraction),
            (ForwardErrorOutcome::NotContracting, f64::INFINITY)
        );
    }
    // A NaN from the solve of step E1: not finite, at once.
    for opts in [REFINE, PLAIN] {
        let bound = Closure::Nan.bound(&a, &r, opts);
        assert_eq!(
            (bound.outcome, bound.solves),
            (ForwardErrorOutcome::NotFinite, 1)
        );
        assert_eq!(bound.contraction, 0.0);
    }
    // In step E2 the refinement is discarded: the result of E1 alone, with
    // twice the solves.
    let plain = Closure::Lu.bound(&a, &r, PLAIN);
    let discarded = Closure::NanOnRefine.bound(&a, &r, REFINE);
    assert!(plain.is_certified());
    assert_eq!(discarded.solves, 4);
    assert_eq!(
        (
            discarded.norm_inf,
            &discarded.componentwise,
            discarded.contraction
        ),
        (plain.norm_inf, &plain.componentwise, plain.contraction)
    );
    // A refinement whose sum overflows is discarded too: A = [2⁻¹⁰⁰⁰],
    // h = 2¹⁰⁰⁰, and a "correction" of Ω.
    let tiny = CsrMatrix::from_raw_parts(1, vec![0, 1], vec![0], vec![pow2f(-1000)])
        .expect("a valid matrix");
    let r1 = [Interval::point(1.0)];
    let mut calls = 0_u32;
    let mut solve = |c: &mut [f64]| {
        calls += 1;
        c[0] = if calls.is_multiple_of(2) {
            f64::MAX
        } else {
            c[0] * pow2f(1000)
        };
    };
    let refined = forward_error_bound(&tiny, &r1, &mut solve, &REFINE).expect("dimensions");
    let plain =
        forward_error_bound(&tiny, &r1, &mut |c| c[0] *= pow2f(1000), &PLAIN).expect("dimensions");
    assert!(plain.is_certified() && plain.contraction < 1e-300);
    assert_eq!(refined.solves, 2);
    assert_eq!(
        (
            refined.norm_inf,
            &refined.componentwise,
            refined.contraction
        ),
        (plain.norm_inf, &plain.componentwise, plain.contraction)
    );
}

/// Subnormal residual enclosures, and an exact solution (`[r] ∋ 0` of
/// width zero): the bound follows them down.
#[test]
fn subnormal_and_zero_residuals() {
    let a = small();
    let m = a.csr();
    let eta = f64::from_bits(1);
    for r in [
        [Interval::ZERO, Interval::ZERO],
        [Interval::point(eta), Interval::point(-3.0 * eta)],
        [
            Interval::mid_rad(0.0, 5.0 * eta),
            Interval::mid_rad(eta, eta),
        ],
    ] {
        for opts in [REFINE, PLAIN] {
            let bound = Closure::Lu.bound(&a, &r, opts);
            assert!(bound.is_certified(), "{bound:?}");
            assert!(bound.norm_inf <= 1e-320, "{bound:?}");
            // Every r ∈ [r] at the endpoints: e = A⁻¹ r.
            for r0 in [r[0].inf(), r[0].sup()] {
                for r1 in [r[1].inf(), r[1].sup()] {
                    let e = a.exact_solve(&[q(r0), q(r1)]).expect("nonsingular");
                    check(&bound, Some(&e), "subnormal r").expect("Theorem E");
                }
            }
        }
    }
    let exact = residual(&m, &[1.0, 2.0], &[4.0, 7.0]).expect("dimensions");
    let bound = Closure::Lu.bound(&a, &exact, REFINE);
    assert!(bound.is_certified() && bound.norm_inf < 1e-300, "{bound:?}");
}

/// The same bits in two calls, for every closure (invariant E4).
#[test]
fn deterministic() {
    let case = Case::new(dense_kappa(8, 1e8, 3), None);
    for (_, r, _) in case.solutions() {
        for c in Closure::ALL {
            for opts in [REFINE, PLAIN] {
                let first = c.bound(&case.a, &r, opts);
                let second = c.bound(&case.a, &r, opts);
                assert_eq!(first, second, "{c:?}");
                let bits = |b: &ForwardErrorBound| {
                    (
                        b.norm_inf.to_bits(),
                        b.contraction.to_bits(),
                        b.componentwise
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                    )
                };
                assert_eq!(bits(&first), bits(&second));
            }
        }
    }
}

/// The zero solve: `R = 0`, `C = I`, `α = 1` — and only the column `q = p`
/// of step E3 sees it.
#[test]
fn the_zero_solve_does_not_contract() {
    let a = small();
    let r = residual(&a.csr(), &[0.4, 0.2], &[1.0, 1.0]).expect("dimensions");
    for opts in [REFINE, PLAIN] {
        let bound = Closure::Zero.bound(&a, &r, opts);
        assert_eq!(
            (bound.outcome, bound.solves),
            (
                ForwardErrorOutcome::NotContracting,
                if opts.refine_rows { 2 } else { 1 }
            )
        );
        // `|c_pp| = 1` exactly; the enclosure V3 adds its rounding.
        assert!((1.0..1.0 + 1e-15).contains(&bound.contraction), "{bound:?}");
    }
}

#[test]
fn defaults() {
    assert_eq!(ForwardErrorOptions::default(), REFINE);
}

// ---------------------------------------------------------------------
// Random systems
// ---------------------------------------------------------------------

const FRACTION: u64 = (1 << 52) - 1;

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

/// A square matrix of order at most `max` with about `density` of its
/// entries stored (a strong diagonal with probability `dominant`), and a
/// right-hand side.
fn square(
    max: usize,
    density: f64,
    value: fn() -> BoxedStrategy<f64>,
) -> impl Strategy<Value = (Dense, Vec<f64>)> {
    (1..=max, any::<bool>()).prop_flat_map(move |(n, dominant)| {
        let entry = (value(), 0.0..1.0_f64).prop_map(move |(v, t)| (t < density).then_some(v));
        let entries = prop::collection::vec(entry, n * n);
        let b = prop::collection::vec(value(), n);
        (entries, b).prop_map(move |(entries, b)| {
            let mut a: Vec<f64> = entries.iter().map(|v| v.unwrap_or(0.0)).collect();
            if dominant {
                for i in 0..n {
                    let s: f64 = (0..n).map(|j| a[i * n + j].abs()).sum();
                    a[i * n + i] = (2.0 * s + 1.0).min(f64::MAX);
                }
            }
            (Dense { n, a }, b)
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

fn closure() -> impl Strategy<Value = Closure> {
    prop_oneof![
        4 => Just(Closure::Lu),
        1 => (0.1..1.9_f64).prop_map(Closure::Scaled),
        1 => prop_oneof![Just(1e-12), Just(1e-6), Just(1e-3), Just(0.3)].prop_map(Closure::Noisy),
        1 => Just(Closure::Identity),
        1 => Just(Closure::Zero),
        1 => Just(Closure::Garbage),
        1 => prop_oneof![Just(1e-10), Just(0.5)].prop_map(Closure::Perturbed),
    ]
}

proptest! {
    #![proptest_config(config(128))]

    /// Theorem E on small dense and sparse systems of ordinary data,
    /// `x̃` in `f64` and polished, a random closure.
    #[test]
    fn small_systems_never_get_a_false_bound(
        (a, b) in square(12, 0.6, moderate),
        c in closure(),
    ) {
        let case = Case::new(a, Some(b));
        case.check_all(&[c, Closure::Lu]).map_err(TestCaseError::fail)?;
    }

    #[test]
    fn small_dense_systems_never_get_a_false_bound(
        (a, b) in square(8, 1.0, moderate),
        c in closure(),
    ) {
        let case = Case::new(a, Some(b));
        case.check_all(&[c, Closure::Lu]).map_err(TestCaseError::fail)?;
    }
}

proptest! {
    #![proptest_config(config(64))]

    /// The same with data whose products overflow and underflow.
    #[test]
    fn small_wild_systems_never_get_a_false_bound(
        (a, b) in square(8, 0.6, wild),
        c in closure(),
    ) {
        let case = Case::new(a, Some(b));
        case.check_all(&[c, Closure::Lu]).map_err(TestCaseError::fail)?;
    }

    /// Diagonally dominant systems are certified by the LU closure, with
    /// and without refinement, for both solutions.
    #[test]
    fn dominant_systems_are_certified(
        (n, off, b) in (1_usize..=10).prop_flat_map(|n| (
            Just(n),
            prop::collection::vec(prop::option::weighted(0.4, -1.0..1.0_f64), n * n),
            prop::collection::vec(moderate(), n),
        )),
    ) {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = if i == j { 2.0 * n as f64 } else { off[i * n + j].unwrap_or(0.0) };
            }
        }
        let case = Case::new(Dense { n, a }, Some(b));
        for (label, r, x) in case.solutions() {
            let e = exact_errors(&case.a, &case.b, &x.expect("finite")).expect("nonsingular");
            for opts in [REFINE, PLAIN] {
                let bound = Closure::Lu.bound(&case.a, &r, opts);
                prop_assert!(bound.is_certified(), "{label}: {bound:?}");
                prop_assert!(bound.contraction < 1e-12, "{label}: {bound:?}");
                prop_assert_eq!(bound.solves, if opts.refine_rows { 2 * n } else { n });
                check(&bound, Some(&e), label).map_err(TestCaseError::fail)?;
            }
        }
    }
}
