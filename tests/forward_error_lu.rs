//! The falsification tests of Algorithm E′ (`verify::forward_error_bound_lu`,
//! Chapter 11.10, Theorem E′; task I06, issue xingo-extprec#4): the bound
//! **never** lies below the true error `x̃ − A⁻¹b`, computed in exact
//! rational arithmetic (`dashu`) for small systems and bracketed rigorously
//! (a polished solution and Algorithm E, compared in rationals) for medium
//! ones — with the factors of `xingo_sparse::Lu` under several options
//! (`drop_tol` active, tiny pivots, row singletons with tiny pivots and so
//! unbounded multipliers), hand-made factors with huge `|L|`, and perturbed
//! or adversarial factors (another matrix's, noisy, scaled, permutations
//! exchanged). Exactly singular matrices are never certified; invalid
//! factors are an error; the result is deterministic.
//!
//! `campaign_*` (ignored; `cargo test -p xingo-extprec --release --test
//! forward_error_lu -- --ignored --nocapture`) run the large campaigns and
//! print the measurements (certification rates, `β`, the ratio bound / true
//! error, the comparison with Algorithm E and the operation counts).

#![allow(clippy::disallowed_methods)] // L2.4: libm in tests, never as a reference (clippy.toml)
#![allow(clippy::float_cmp)] // exact comparisons are the subject
#![allow(clippy::cast_precision_loss)] // small indices as f64
#![allow(clippy::too_many_lines)]
#![allow(clippy::needless_range_loop)] // index notation of the dense references
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // quantile positions

use dashu_ratio::RBig;
use xingo_extprec::verify::{
    FactorBoundOptions, ForwardErrorBound, ForwardErrorOptions, ForwardErrorOutcome, PolishOptions,
    VerifyError, forward_error_bound, forward_error_bound_lu, polish, residual, residual_of_dd,
};
use xingo_extprec::{DoubleDouble, Interval};
use xingo_sparse::{
    Col, CscMatrix, CsrMatrix, Lu, LuFactor, LuOptions, Perm, SparseVec, TolScale,
    TriangularFactors,
};

type Dd = DoubleDouble;

// ---------------------------------------------------------------------
// Exact arithmetic and dense matrices
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

#[derive(Clone, Debug)]
struct Dense {
    n: usize,
    a: Vec<f64>,
}

impl Dense {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.a[i * self.n + j]
    }

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

    fn csc(&self) -> CscMatrix<f64> {
        self.csr().to_csc()
    }

    fn transpose(&self) -> Self {
        let n = self.n;
        Self {
            n,
            a: (0..n * n).map(|k| self.at(k % n, k / n)).collect(),
        }
    }

    /// `A⁻¹ b` in rational arithmetic; `None` if `A` is singular.
    fn exact_solve(&self, b: &[f64]) -> Option<Vec<RBig>> {
        let n = self.n;
        let mut m: Vec<Vec<RBig>> = (0..n)
            .map(|i| {
                let mut row: Vec<RBig> = (0..n).map(|j| q(self.at(i, j))).collect();
                row.push(q(b[i]));
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

/// A dense LU with partial pivoting in `f64`: the solves that produce `x̃`
/// and the closure of Algorithm E (independent of the factors under test).
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

/// A deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Uniform in `[−1, 1)`.
    fn sym(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(n).expect("small")).expect("small")
    }

    /// `±10^(spread · U(−1, 1))` times a uniform mantissa.
    fn spread(&mut self, spread: f64) -> f64 {
        let v = self.sym() * 10f64.powf(spread * self.sym());
        if v == 0.0 { 1.0 } else { v }
    }

    fn shuffle(&mut self, v: &mut [u32]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------
// Factors
// ---------------------------------------------------------------------

/// The exported factors of `xingo_sparse::Lu` with `opts`, if it factorizes.
fn lu_factors(a: &Dense, opts: LuOptions) -> Option<TriangularFactors<f64>> {
    let mut lu = Lu::<f64>::new(opts);
    lu.factorize(&a.csc()).ok()?;
    lu.triangular_factors()
}

/// A dense matrix (rows of vectors) as a CSC, nonzeros only.
fn csc_of(n: usize, d: &[Vec<f64>]) -> CscMatrix<f64> {
    let mut col_ptr = vec![0];
    let (mut row_idx, mut val) = (Vec::new(), Vec::new());
    for j in 0..n {
        for (i, row) in d.iter().enumerate() {
            if row[j] != 0.0 {
                row_idx.push(u32::try_from(i).expect("small"));
                val.push(row[j]);
            }
        }
        col_ptr.push(row_idx.len());
    }
    CscMatrix {
        n_rows: n,
        col_ptr,
        row_idx,
        val,
    }
}

fn dense_of(m: &CscMatrix<f64>) -> Vec<Vec<f64>> {
    let n = m.n_rows;
    let mut d = vec![vec![0.0; n]; n];
    for j in 0..m.n_cols() {
        let (rows, vals) = m.col(Col::from_usize(j));
        for (&i, &v) in rows.iter().zip(vals) {
            d[i as usize][j] = v;
        }
    }
    d
}

/// Factors with the given parts: `TriangularFactors` is `#[non_exhaustive]`,
/// so a value of order `n` is exported (the identity) and its public fields
/// are overwritten — what any caller can do.
fn make_factors(
    n: usize,
    l: CscMatrix<f64>,
    u: CscMatrix<f64>,
    p: Vec<u32>,
    q: Vec<u32>,
) -> TriangularFactors<f64> {
    let id = Dense {
        n,
        a: (0..n * n)
            .map(|k| if k % (n + 1) == 0 { 1.0 } else { 0.0 })
            .collect(),
    };
    let mut f = lu_factors(&id, LuOptions::default()).expect("the identity");
    f.l = l;
    f.u = u;
    f.row_perm = Perm::from_fwd(p).expect("a permutation");
    f.col_perm = Perm::from_fwd(q).expect("a permutation");
    f
}

/// Hand-made factors with `|l| ≤ 10^lmag` and `|u_kk|` down to
/// `10^-umag`, random permutations, and `A = fl(Pᵀ L U Qᵀ)` (each entry
/// rounded once), plus `extra` random perturbations of relative size `δ`
/// (entries "dropped" or changed): the defect is not only rounding.
fn hand_made(
    n: usize,
    lmag: f64,
    umag: f64,
    fill: f64,
    extra: usize,
    delta: f64,
    rng: &mut Rng,
) -> (Dense, TriangularFactors<f64>) {
    let mut l = vec![vec![0.0; n]; n];
    let mut u = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if j < i && rng.unit() < fill {
                l[i][j] = rng.sym() * 10f64.powf(lmag * rng.unit());
            }
            if j > i && rng.unit() < fill {
                u[i][j] = rng.spread(1.0);
            }
        }
        u[i][i] = rng.sym().signum() * 10f64.powf(-umag * rng.unit());
    }
    let mut p: Vec<u32> = (0..u32::try_from(n).expect("small")).collect();
    let mut qp = p.clone();
    rng.shuffle(&mut p);
    rng.shuffle(&mut qp);
    let mut a = vec![0.0; n * n];
    for s in 0..n {
        for t in 0..n {
            // fl of the exact sum: the entries are summed in double-double
            // and rounded once (a sum within 2⁻¹⁰⁶ relative, then RN).
            let mut acc = Dd::from_f64(u[s][t]);
            for k in 0..s.min(t + 1) {
                acc += Dd::from_f64(l[s][k]) * Dd::from_f64(u[k][t]);
            }
            a[p[s] as usize * n + qp[t] as usize] = acc.hi();
        }
    }
    for _ in 0..extra {
        let k = rng.below(n * n);
        a[k] += delta * (1.0 + a[k].abs()) * rng.sym();
    }
    let f = make_factors(n, csc_of(n, &l), csc_of(n, &u), p, qp);
    f.check_invariants().expect("(T1)");
    (Dense { n, a }, f)
}

/// The ways to spoil good factors (each keeps (T1)).
#[derive(Clone, Copy, Debug)]
enum Spoil {
    None,
    /// Every stored value times `1 + t·U(−1, 1)`.
    Noise(f64),
    /// `U` times `s`.
    ScaleU(f64),
    /// Two entries of the row permutation exchanged.
    SwapRows,
    /// Two entries of the column permutation exchanged.
    SwapCols,
    /// `L = I`, `U` its diagonal only.
    DiagonalOnly,
    /// The values replaced by random ones (same pattern).
    Garbage,
}

const SPOILS: [Spoil; 8] = [
    Spoil::None,
    Spoil::Noise(1e-10),
    Spoil::Noise(1e-3),
    Spoil::ScaleU(1.5),
    Spoil::SwapRows,
    Spoil::SwapCols,
    Spoil::DiagonalOnly,
    Spoil::Garbage,
];

fn spoil(f: &TriangularFactors<f64>, how: Spoil, rng: &mut Rng) -> TriangularFactors<f64> {
    let mut g = f.clone();
    let n = f.order();
    let fix = |v: f64| if v == 0.0 || !v.is_finite() { 1.0 } else { v };
    match how {
        Spoil::Noise(t) => {
            for v in g.l.val.iter_mut().chain(g.u.val.iter_mut()) {
                *v = fix(*v * (1.0 + t * rng.sym()));
            }
        }
        Spoil::ScaleU(s) => {
            for v in &mut g.u.val {
                *v = fix(*v * s);
            }
        }
        Spoil::SwapRows | Spoil::SwapCols if n >= 2 => {
            let i = rng.below(n);
            let j = (i + 1 + rng.below(n - 1)) % n;
            let perm = if matches!(how, Spoil::SwapRows) {
                &mut g.row_perm
            } else {
                &mut g.col_perm
            };
            let mut fwd = perm.fwd.clone();
            fwd.swap(i, j);
            *perm = Perm::from_fwd(fwd).expect("a permutation");
        }
        Spoil::None | Spoil::SwapRows | Spoil::SwapCols => {}
        Spoil::DiagonalOnly => {
            let du = dense_of(&f.u);
            let diag: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| if i == j { du[i][i] } else { 0.0 })
                        .collect()
                })
                .collect();
            g.l = csc_of(n, &vec![vec![0.0; n]; n]);
            g.u = csc_of(n, &diag);
        }
        Spoil::Garbage => {
            for v in g.l.val.iter_mut().chain(g.u.val.iter_mut()) {
                *v = fix(rng.sym());
            }
        }
    }
    g.check_invariants().expect("(T1) kept");
    g
}

// ---------------------------------------------------------------------
// The check, and the measurements
// ---------------------------------------------------------------------

/// The default options (first-order correction on).
fn on() -> FactorBoundOptions {
    FactorBoundOptions::default()
}

/// Theorem E′ against exact errors `e` (`None`: `A` singular or `x̃` not
/// real), and the shape of invariant E3.
fn check(bound: &ForwardErrorBound, e: Option<&[RBig]>, what: &str) -> Result<(), String> {
    if bound.is_certified() {
        let Some(e) = e else {
            return Err(format!("{what}: certified without a solution: {bound:?}"));
        };
        if !(bound.norm_inf.is_finite() && (0.0..1.0).contains(&bound.contraction)) {
            return Err(format!("{what}: E3: {bound:?}"));
        }
        if bound.solves != 0 {
            return Err(format!("{what}: solves = {}", bound.solves));
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
                    "{what}: FALSE BOUND |e_{p}| = {:e} > ε_{p} = {bp:e} ({bound:?})",
                    ep.to_f64().value()
                ));
            }
            largest = largest.max(ep);
        }
        if largest > q(bound.norm_inf) {
            return Err(format!(
                "{what}: FALSE BOUND ‖e‖∞ = {:e} > ε = {:e}",
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

fn norm_f64(e: &[RBig]) -> f64 {
    e.iter()
        .map(|v| abs(v.clone()))
        .max()
        .unwrap_or_else(zero)
        .to_f64()
        .value()
}

/// Counts and ratios over a campaign.
#[derive(Default)]
struct Stats {
    runs: usize,
    certified: usize,
    not_contracting: usize,
    not_finite: usize,
    /// `ε / ‖e‖∞` of the certified runs with `e ≠ 0`.
    ratios: Vec<f64>,
    betas: Vec<f64>,
    /// Algorithm E on the same data: certified, and `ε_{E′}/ε_E` when both.
    e_certified: usize,
    vs_e: Vec<f64>,
    /// Certified by E but not by E′.
    e_only: usize,
}

impl Stats {
    fn add(&mut self, b: &ForwardErrorBound, err: Option<f64>) {
        self.runs += 1;
        match b.outcome {
            ForwardErrorOutcome::Certified => {
                self.certified += 1;
                self.betas.push(b.contraction);
                if let Some(e) = err
                    && e > 0.0
                {
                    self.ratios.push(b.norm_inf / e);
                }
            }
            ForwardErrorOutcome::NotContracting => self.not_contracting += 1,
            _ => self.not_finite += 1,
        }
    }

    fn add_e(&mut self, lu: &ForwardErrorBound, e: &ForwardErrorBound) {
        if e.is_certified() {
            self.e_certified += 1;
            if lu.is_certified() {
                self.vs_e.push(lu.norm_inf / e.norm_inf);
            } else {
                self.e_only += 1;
            }
        }
    }

    fn quantiles(v: &mut [f64]) -> String {
        if v.is_empty() {
            return "—".into();
        }
        v.sort_by(f64::total_cmp);
        let at = |t: f64| v[((v.len() - 1) as f64 * t).round() as usize];
        format!(
            "{:.3e} / {:.3e} / {:.3e} / {:.3e}",
            at(0.0),
            at(0.5),
            at(0.99),
            at(1.0)
        )
    }

    fn print(&mut self, label: &str) {
        println!(
            "| {label} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            self.runs,
            self.certified,
            self.not_contracting,
            self.not_finite,
            Self::quantiles(&mut self.betas),
            Self::quantiles(&mut self.ratios),
            self.e_certified,
            self.e_only,
            Self::quantiles(&mut self.vs_e),
        );
    }
}

fn print_header() {
    println!(
        "| family | runs | E′ certified | not contracting | not finite | β min/med/p99/max | ε/‖e‖ min/med/p99/max | E certified | E only | ε_E′/ε_E min/med/p99/max |"
    );
    println!("|---|---:|---:|---:|---:|---|---|---:|---:|---|");
}

/// One small system: `x̃` in `f64` (dense partial pivoting) and polished,
/// the exact errors in rationals, Algorithm E′ with the given factors and
/// both options, checked; Algorithm E on the `f64` solution for comparison
/// when `with_e`.
fn run_small(
    a: &Dense,
    b: &[f64],
    factors: &TriangularFactors<f64>,
    stats: &mut Stats,
    with_e: bool,
    what: &str,
) -> Result<(), String> {
    let m = a.csr();
    let lu = DenseLu::new(a);
    let mut x = b.to_vec();
    lu.solve(&mut x);
    if !x.iter().all(|v| v.is_finite()) {
        x = vec![0.0; a.n];
    }
    let mut xd: Vec<Dd> = x.iter().map(|&v| Dd::from_f64(v)).collect();
    let _ = polish(
        &m,
        b,
        &mut xd,
        &mut |s| lu.solve(s),
        &PolishOptions::default(),
    )
    .expect("dimensions");
    if !xd.iter().all(|v| v.hi().is_finite() && v.lo().is_finite()) {
        xd = x.iter().map(|&v| Dd::from_f64(v)).collect();
    }
    let star = a.exact_solve(b);
    let errors = |xt: Vec<RBig>| {
        star.as_ref().map(|s| {
            xt.into_iter()
                .zip(s)
                .map(|(a, b)| a - b.clone())
                .collect::<Vec<_>>()
        })
    };
    let e64 = errors(x.iter().map(|&v| q(v)).collect());
    let edd = errors(xd.iter().map(|&v| qd(v)).collect());
    let r64 = residual(&m, &x, b).expect("dimensions");
    let rdd = residual_of_dd(&m, &xd, b).expect("dimensions");
    for (label, r, e) in [("f64", &r64, &e64), ("dd", &rdd, &edd)] {
        for correct in [true, false] {
            let mut opts = on();
            opts.first_order_correction = correct;
            let bound = forward_error_bound_lu(&m, r, factors, &opts).expect("checked data");
            check(
                &bound,
                e.as_deref(),
                &format!("{what}/{label}/correct={correct}"),
            )?;
            if correct {
                stats.add(&bound, e.as_deref().map(norm_f64));
                if with_e && label == "f64" {
                    let lut = DenseLu::new(&a.transpose());
                    let be = forward_error_bound(
                        &m,
                        r,
                        &mut |c| lut.solve(c),
                        &ForwardErrorOptions::default(),
                    )
                    .expect("dimensions");
                    stats.add_e(&bound, &be);
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------

/// Random sparse: about `density` of the entries, magnitudes `10^±spread`;
/// with `diag`, a nonzero diagonal.
fn sparse_random(n: usize, density: f64, spread: f64, diag: bool, rng: &mut Rng) -> Dense {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if (diag && i == j) || rng.unit() < density {
                a[i * n + j] = rng.spread(spread);
            }
        }
    }
    Dense { n, a }
}

/// Dense `U Σ Vᵀ` by random reflectors, singular values 1 to `1/κ`.
fn dense_kappa(n: usize, kappa: f64, rng: &mut Rng) -> Dense {
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

/// A row singleton with a tiny pivot `τ` in row 0, column 0, and large
/// entries below it in column 0: `xingo-sparse` pivots row singletons
/// without the threshold test, so the multipliers are `≈ 1/τ`.
fn row_singleton(n: usize, tau: f64, rng: &mut Rng) -> Dense {
    let mut a = sparse_random(n, 0.4, 1.0, true, rng).a;
    for j in 0..n {
        a[j] = 0.0;
    }
    a[0] = tau;
    for i in 1..n {
        a[i * n] = rng.spread(1.0);
    }
    Dense { n, a }
}

/// Integers in `−9..=9`, the last row the sum of the first two (exactly
/// singular; `n ≥ 3`, since for `n = 2` the "last row" is the second).
fn integers_singular(n: usize, rng: &mut Rng) -> Dense {
    assert!(n >= 3);
    let mut a: Vec<f64> = (0..n * n)
        .map(|_| f64::from(u8::try_from(rng.next() % 19).expect("small")) - 9.0)
        .collect();
    for j in 0..n {
        a[(n - 1) * n + j] = a[j] + a[n + j];
    }
    Dense { n, a }
}

fn rhs(n: usize, rng: &mut Rng) -> Vec<f64> {
    (0..n).map(|_| rng.spread(2.0)).collect()
}

fn lu_opts(kind: usize) -> LuOptions {
    match kind {
        // the default
        0 => LuOptions::default(),
        // drop_tol active (absolute and by columns)
        1 => LuOptions {
            drop_tol: 1e-4,
            ..LuOptions::default()
        },
        2 => LuOptions {
            drop_tol: 1e-6,
            tol_scale: TolScale::Column,
            ..LuOptions::default()
        },
        // tiny pivots accepted: large multipliers
        _ => LuOptions {
            rel_pivot_tol: 1e-12,
            abs_pivot_tol: 1e-300,
            drop_tol: 0.0,
            ..LuOptions::default()
        },
    }
}

/// The small families, `count` systems each, every one checked; returns
/// the stats per family (printed by the campaign).
fn small_families(count: usize, seed: u64, with_e: bool) -> Result<Vec<(String, Stats)>, String> {
    let mut out = Vec::new();
    let mut rng = Rng(seed);
    // (1) Lu factors of random sparse systems, four option sets, spoiled.
    for kind in 0..4 {
        for how in SPOILS {
            let mut st = Stats::default();
            for c in 0..count {
                let n = 1 + rng.below(12);
                let a = sparse_random(
                    n,
                    0.1 + 0.5 * rng.unit(),
                    3.0 * rng.unit(),
                    c % 3 != 0,
                    &mut rng,
                );
                let Some(f) = lu_factors(&a, lu_opts(kind)) else {
                    continue;
                };
                let f = spoil(&f, how, &mut rng);
                let b = rhs(n, &mut rng);
                run_small(
                    &a,
                    &b,
                    &f,
                    &mut st,
                    with_e,
                    &format!("lu{kind}/{how:?}/{c}"),
                )?;
            }
            out.push((format!("Lu opts {kind}, {how:?}"), st));
        }
    }
    // (2) Ill-conditioned dense, κ up to 1e16.
    let mut st = Stats::default();
    for c in 0..count {
        let n = 2 + rng.below(10);
        let kappa = 10f64.powf(2.0 + 14.0 * rng.unit());
        let a = dense_kappa(n, kappa, &mut rng);
        let b = rhs(n, &mut rng);
        if let Some(f) = lu_factors(&a, lu_opts(c % 4)) {
            run_small(&a, &b, &f, &mut st, with_e, &format!("kappa/{c}"))?;
        }
    }
    out.push(("dense κ ∈ [1e2, 1e16]".into(), st));
    // (3) Row singletons with tiny pivots (unbounded |L|).
    let mut st = Stats::default();
    for c in 0..count {
        let n = 2 + rng.below(10);
        let tau = 10f64.powf(-1.0 - 9.0 * rng.unit());
        let a = row_singleton(n, tau, &mut rng);
        let b = rhs(n, &mut rng);
        if let Some(f) = lu_factors(&a, lu_opts(c % 4)) {
            run_small(&a, &b, &f, &mut st, with_e, &format!("singleton/{c}"))?;
        }
    }
    out.push(("row singleton, tiny pivot".into(), st));
    // (4) Hand-made factors: huge |L|, tiny |u_kk|, extra defect.
    let mut st = Stats::default();
    for c in 0..count {
        let n = 1 + rng.below(12);
        let (a, f) = hand_made(
            n,
            8.0 * rng.unit(),
            8.0 * rng.unit(),
            0.5 * rng.unit(),
            rng.below(3),
            10f64.powf(-16.0 + 12.0 * rng.unit()),
            &mut rng,
        );
        let b = rhs(n, &mut rng);
        run_small(&a, &b, &f, &mut st, with_e, &format!("hand/{c}"))?;
    }
    out.push((
        "hand-made, |l| ≤ 1e8, |u_kk| ≥ 1e-8, extra defect".into(),
        st,
    ));
    // (5) Another matrix's factors.
    let mut st = Stats::default();
    for c in 0..count {
        let n = 1 + rng.below(12);
        let a = sparse_random(n, 0.4, 1.0, true, &mut rng);
        let mut other = a.clone();
        let k = rng.below(n * n);
        other.a[k] += 10f64.powf(-12.0 + 12.0 * rng.unit()) * rng.sym();
        if let Some(f) = lu_factors(&other, LuOptions::default()) {
            let b = rhs(n, &mut rng);
            run_small(&a, &b, &f, &mut st, with_e, &format!("other/{c}"))?;
        }
    }
    out.push(("factors of a perturbed matrix".into(), st));
    // (6) Exactly singular matrices with the factors of a nearby one.
    let mut st = Stats::default();
    for c in 0..count {
        let n = 3 + rng.below(7);
        let a = integers_singular(n, &mut rng);
        let mut near = a.clone();
        near.a[(n - 1) * n] += 10f64.powf(-14.0 + 13.0 * rng.unit());
        if let Some(f) = lu_factors(&near, lu_opts(3)) {
            let b = rhs(n, &mut rng);
            run_small(&a, &b, &f, &mut st, false, &format!("singular/{c}"))?;
            if st.certified > 0 {
                return Err(format!("singular/{c}: certified"));
            }
        }
    }
    out.push(("exactly singular, nearby factors".into(), st));
    Ok(out)
}

// ---------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------

#[test]
fn small_systems_never_get_a_false_bound() {
    let families =
        small_families(25, 0x9e37_79b9_7f4a_7c15, false).unwrap_or_else(|e| panic!("{e}"));
    // Something is certified in the ordinary families.
    assert!(families[0].1.certified > 0);
}

#[test]
fn a_well_conditioned_system_is_certified_tightly() {
    let mut rng = Rng(7);
    let mut st = Stats::default();
    for c in 0..50 {
        let n = 2 + rng.below(10);
        let mut a = sparse_random(n, 0.3, 0.5, true, &mut rng);
        for i in 0..n {
            a.a[i * n + i] = 4.0 * n as f64;
        }
        let f = lu_factors(&a, LuOptions::default()).expect("dominant");
        let b = rhs(n, &mut rng);
        run_small(&a, &b, &f, &mut st, true, &format!("dominant/{c}"))
            .unwrap_or_else(|e| panic!("{e}"));
    }
    assert_eq!(st.certified, st.runs);
    assert_eq!(st.e_only, 0);
    let mut r = st.ratios.clone();
    r.sort_by(f64::total_cmp);
    // f64 solutions with the correction: the bound follows the error.
    assert!(r[r.len() / 2] < 2.0, "median ratio {}", r[r.len() / 2]);
}

#[test]
fn exactly_singular_matrices_are_never_certified() {
    let mut rng = Rng(11);
    for c in 0..200 {
        let n = 3 + rng.below(7);
        let a = integers_singular(n, &mut rng);
        let mut near = a.clone();
        near.a[(n - 1) * n] += 10f64.powf(-15.0 + 14.0 * rng.unit());
        let Some(f) = lu_factors(&near, lu_opts(3)) else {
            continue;
        };
        let x: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
        let r = residual(&a.csr(), &x, &rhs(n, &mut rng)).expect("dimensions");
        for correct in [true, false] {
            let mut o = on();
            o.first_order_correction = correct;
            let b = forward_error_bound_lu(&a.csr(), &r, &f, &o).expect("checked");
            assert!(!b.is_certified(), "singular/{c}: {b:?}");
        }
    }
}

#[test]
fn the_empty_system_is_certified() {
    let a = CsrMatrix::from_raw_parts(0, vec![0], vec![], vec![]).expect("empty");
    let f = lu_factors(&Dense { n: 0, a: vec![] }, LuOptions::default());
    let Some(f) = f else {
        return; // the LU of order 0 exports nothing: nothing to test
    };
    let b = forward_error_bound_lu(&a, &[], &f, &on()).expect("dimensions");
    assert!(b.is_certified() && b.norm_inf == 0.0 && b.contraction == 0.0);
}

#[test]
fn dimensions_and_invalid_factors_are_errors() {
    let mut rng = Rng(3);
    let a = sparse_random(4, 0.5, 0.5, true, &mut rng);
    let f = lu_factors(&a, LuOptions::default()).expect("nonsingular");
    let r = residual(&a.csr(), &[0.0; 4], &[1.0; 4]).expect("dimensions");
    assert_eq!(
        forward_error_bound_lu(&a.csr(), &r[..3], &f, &on()),
        Err(VerifyError::Dimension {
            expected: 4,
            got: 3
        })
    );
    let small = sparse_random(3, 0.5, 0.5, true, &mut rng);
    let f3 = lu_factors(&small, LuOptions::default()).expect("nonsingular");
    assert_eq!(
        forward_error_bound_lu(&a.csr(), &r, &f3, &on()),
        Err(VerifyError::Dimension {
            expected: 4,
            got: 3
        })
    );
    // An entry of `l` above the diagonal, a missing diagonal of `u`.
    let mut bad = f.clone();
    bad.l = csc_of(
        4,
        &[
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0; 4],
        ],
    );
    assert!(matches!(
        forward_error_bound_lu(&a.csr(), &r, &bad, &on()),
        Err(VerifyError::InvalidFactors(_))
    ));
    let mut bad = f.clone();
    let mut du = dense_of(&f.u);
    du[2][2] = 0.0;
    bad.u = csc_of(4, &du);
    assert!(matches!(
        forward_error_bound_lu(&a.csr(), &r, &bad, &on()),
        Err(VerifyError::InvalidFactors(_))
    ));
    let mut bad = f;
    bad.u.val[0] = f64::NAN;
    assert!(matches!(
        forward_error_bound_lu(&a.csr(), &r, &bad, &on()),
        Err(VerifyError::InvalidFactors(_))
    ));
}

#[test]
fn unbounded_or_empty_residuals() {
    let mut rng = Rng(5);
    let a = sparse_random(3, 0.5, 0.5, true, &mut rng);
    let f = lu_factors(&a, LuOptions::default()).expect("nonsingular");
    let mut r = residual(&a.csr(), &[0.0; 3], &[1.0; 3]).expect("dimensions");
    r[1] = Interval::ENTIRE;
    for correct in [true, false] {
        let mut o = on();
        o.first_order_correction = correct;
        let b = forward_error_bound_lu(&a.csr(), &r, &f, &o).expect("checked");
        assert_eq!(b.outcome, ForwardErrorOutcome::NotFinite);
        assert!(b.norm_inf == f64::INFINITY && b.contraction < 1.0);
    }
    r[1] = Interval::EMPTY;
    let b = forward_error_bound_lu(&a.csr(), &r, &f, &on()).expect("checked");
    assert_eq!(b.outcome, ForwardErrorOutcome::NotFinite);
}

#[test]
fn deterministic() {
    let mut rng = Rng(13);
    let a = sparse_random(40, 0.1, 2.0, true, &mut rng);
    let f = lu_factors(&a, LuOptions::default()).expect("nonsingular");
    let b = rhs(40, &mut rng);
    let mut x = b.clone();
    DenseLu::new(&a).solve(&mut x);
    let r = residual(&a.csr(), &x, &b).expect("dimensions");
    let one = forward_error_bound_lu(&a.csr(), &r, &f, &on()).expect("checked");
    let two = forward_error_bound_lu(&a.csr(), &r, &f, &on()).expect("checked");
    assert_eq!(one.norm_inf.to_bits(), two.norm_inf.to_bits());
    assert!(
        one.componentwise
            .iter()
            .zip(&two.componentwise)
            .all(|(x, y)| x.to_bits() == y.to_bits())
    );
    assert_eq!(one.contraction.to_bits(), two.contraction.to_bits());
}

#[test]
fn the_defect_of_dropped_entries_is_enclosed() {
    // A cyclic bidiagonal with off-diagonal 3e-5: its fill in the last row,
    // 9e-10, is dropped by drop_tol = 1e-4, so the factors are those of
    // another matrix, and the bound must account for it.
    let n = 6;
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = 1.0;
        a[i * n + (i + 1) % n] = 3e-5;
    }
    let a = Dense { n, a };
    let f = lu_factors(&a, lu_opts(1)).expect("nonsingular");
    let b = vec![1.0; n];
    let mut st = Stats::default();
    run_small(&a, &b, &f, &mut st, false, "dropped").unwrap_or_else(|e| panic!("{e}"));
    assert!(st.certified > 0);
    // The contraction reflects the dropped fill (9e-10), not the rounding.
    assert!(st.betas.iter().all(|&b| b > 1e-10), "{:?}", st.betas);
}

/// The rounding of the comparison solves matters: factors of an M-matrix
/// `A = L U` held exactly (dyadic entries, so `P A Q − L U = 0` and `β` is
/// the underflow floor of Dot2Err only), `⟨L⟩ = L`, `⟨U⟩ = U`, so that
/// `⟨U⟩⁻¹⟨L⟩⁻¹ = F⁻¹ ≥ 0`; with `x̃ = 0` and `b < 0` the residual is the
/// point `−b > 0` and, without the correction, the bound is the exact error
/// `F⁻¹(−b)` plus rounding. Upward rounding keeps it above the error in
/// every component; rounding to nearest — a deliberate mutation of the
/// solves in task I06 — put a component below it (case 40 of the 60), and
/// none of the other tests caught that mutation.
#[test]
fn the_comparison_solves_must_round_upward() {
    let mut rng = Rng(23);
    let mut tight = 0;
    for c in 0..60 {
        let n = 8 + rng.below(16);
        let mut l = vec![vec![0.0; n]; n];
        let mut u = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if j < i && rng.unit() < 0.5 {
                    l[i][j] = -((1 + rng.below(7)) as f64) / 8.0;
                }
                if j > i && rng.unit() < 0.5 {
                    u[i][j] = -((1 + rng.below(5)) as f64) / 4.0;
                }
            }
            u[i][i] = (1 + rng.below(4)) as f64;
        }
        let mut a = vec![0.0; n * n];
        for s in 0..n {
            for t in 0..n {
                let mut v = u[s][t];
                for k in 0..s.min(t + 1) {
                    v += l[s][k] * u[k][t]; // exact: dyadic, small
                }
                a[s * n + t] = v;
            }
        }
        let id: Vec<u32> = (0..u32::try_from(n).expect("small")).collect();
        let f = make_factors(n, csc_of(n, &l), csc_of(n, &u), id.clone(), id);
        let a = Dense { n, a };
        let b: Vec<f64> = (0..n).map(|_| -(0.5 + 0.5 * rng.unit())).collect();
        let x = vec![0.0; n];
        let r = residual(&a.csr(), &x, &b).expect("dimensions");
        let star = a.exact_solve(&b).expect("nonsingular");
        let e: Vec<RBig> = star.into_iter().map(|s| zero() - s).collect();
        let mut o = on();
        o.first_order_correction = false;
        let bound = forward_error_bound_lu(&a.csr(), &r, &f, &o).expect("checked");
        assert!(
            bound.is_certified() && bound.contraction < 1e-200,
            "{c}: {bound:?}"
        );
        check(&bound, Some(&e), &format!("m-matrix/{c}")).unwrap_or_else(|m| panic!("{m}"));
        // Sharp: within a few ulps of the true error in every component.
        for (ep, &bp) in e.iter().zip(&bound.componentwise) {
            let ep = abs(ep.clone()).to_f64().value();
            assert!(bp <= ep * (1.0 + 1e-13), "{c}: not tight, {bp:e} vs {ep:e}");
            if bp > ep {
                tight += 1;
            }
        }
    }
    assert!(tight > 0);
}

// ---------------------------------------------------------------------
// Medium systems: a rigorous bracket of the true error
// ---------------------------------------------------------------------

/// For a medium system: `x̃` (f64), a polished `x̂` and Algorithm E's
/// certified `ε̂_p ≥ |x̂_p − x*_p|`; so `|x̃_p − x*_p| ∈ [|x̃_p − x̂_p| − ε̂_p,
/// |x̃_p − x̂_p| + ε̂_p]` (in rationals). A bound below the lower end is a
/// proven falsification; one inside the bracket is "undecided".
fn run_medium(
    a: &Dense,
    b: &[f64],
    f: &TriangularFactors<f64>,
    st: &mut Stats,
    undecided: &mut usize,
    what: &str,
) -> Result<(), String> {
    let m = a.csr();
    let lu = DenseLu::new(a);
    let mut x = b.to_vec();
    lu.solve(&mut x);
    if !x.iter().all(|v| v.is_finite()) {
        return Ok(());
    }
    let mut xd: Vec<Dd> = x.iter().map(|&v| Dd::from_f64(v)).collect();
    let _ = polish(
        &m,
        b,
        &mut xd,
        &mut |s| lu.solve(s),
        &PolishOptions::default(),
    )
    .expect("dimensions");
    let lut = DenseLu::new(&a.transpose());
    let rd = residual_of_dd(&m, &xd, b).expect("dimensions");
    let refb = forward_error_bound(
        &m,
        &rd,
        &mut |c| lut.solve(c),
        &ForwardErrorOptions::default(),
    )
    .expect("dimensions");
    let r = residual(&m, &x, b).expect("dimensions");
    let lub = forward_error_bound_lu(&m, &r, f, &on()).expect("checked");
    let eb = forward_error_bound(
        &m,
        &r,
        &mut |c| lut.solve(c),
        &ForwardErrorOptions::default(),
    )
    .expect("dimensions");
    if !refb.is_certified() {
        // No bracket: only the shape and the singularity can be checked.
        check(&lub, Some(&vec![zero(); a.n]), what)?;
        st.add(&lub, None);
        st.add_e(&lub, &eb);
        return Ok(());
    }
    let mut largest = 0.0_f64;
    if lub.is_certified() {
        for p in 0..a.n {
            let d = abs(q(x[p]) - qd(xd[p]));
            let lo = d.clone() - q(refb.componentwise[p]);
            let hi = d.clone() + q(refb.componentwise[p]);
            let bp = q(lub.componentwise[p]);
            if bp < lo {
                return Err(format!(
                    "{what}: FALSE BOUND at {p}: ε_p = {:e} < |e_p| ≥ {:e}",
                    lub.componentwise[p],
                    lo.to_f64().value()
                ));
            }
            if bp < hi {
                *undecided += 1;
            }
            largest = largest.max(d.to_f64().value());
        }
    }
    check(&lub, Some(&vec![zero(); a.n]), what)?;
    st.add(&lub, Some(largest));
    st.add_e(&lub, &eb);
    Ok(())
}

fn medium_families(count: usize, sizes: &[usize], seed: u64) -> Result<(), String> {
    let mut rng = Rng(seed);
    print_header();
    let mut undecided_all = 0;
    for &n in sizes {
        for kind in 0..4 {
            let mut st = Stats::default();
            let mut und = 0;
            for c in 0..count {
                let density = (3.0 + 5.0 * rng.unit()) / n as f64;
                let a = sparse_random(n, density, 3.0 * rng.unit(), true, &mut rng);
                let Some(f) = lu_factors(&a, lu_opts(kind)) else {
                    continue;
                };
                let b = rhs(n, &mut rng);
                run_medium(&a, &b, &f, &mut st, &mut und, &format!("n{n}/lu{kind}/{c}"))?;
            }
            st.print(&format!("sparse n = {n}, Lu opts {kind}"));
            undecided_all += und;
        }
        let mut st = Stats::default();
        let mut und = 0;
        for c in 0..count {
            let a = row_singleton(n, 10f64.powf(-1.0 - 9.0 * rng.unit()), &mut rng);
            let Some(f) = lu_factors(&a, lu_opts(c % 4)) else {
                continue;
            };
            let b = rhs(n, &mut rng);
            run_medium(
                &a,
                &b,
                &f,
                &mut st,
                &mut und,
                &format!("n{n}/singleton/{c}"),
            )?;
        }
        st.print(&format!("row singleton n = {n}"));
        undecided_all += und;
    }
    println!("\nundecided components (bound inside the bracket): {undecided_all}");
    Ok(())
}

#[test]
fn medium_systems_never_get_a_false_bound() {
    medium_families(3, &[30, 60], 0x1234_5678).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
#[ignore = "the falsification campaign; run in release with --ignored --nocapture"]
fn campaign_small() {
    let count: usize = std::env::var("I06_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let families =
        small_families(count, 0xdead_beef_cafe_f00d, true).unwrap_or_else(|e| panic!("{e}"));
    print_header();
    for (label, mut st) in families {
        st.print(&label);
    }
}

#[test]
#[ignore = "the medium campaign; run in release with --ignored --nocapture"]
fn campaign_medium() {
    let count: usize = std::env::var("I06_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    medium_families(count, &[50, 100, 200, 400], 0xfeed_face).unwrap_or_else(|e| panic!("{e}"));
}

/// The reference solve of these tests agrees with the FTRAN of `Lu` on a
/// well-conditioned system (a sanity check of the harness).
#[test]
fn ftran_agrees() {
    let mut rng = Rng(17);
    let a = sparse_random(10, 0.3, 0.3, true, &mut rng);
    let mut lu = Lu::<f64>::new(LuOptions::default());
    lu.factorize(&a.csc()).expect("nonsingular");
    let b = rhs(10, &mut rng);
    let mut v = SparseVec::from_dense(&b);
    lu.ftran(&mut v);
    let mut x = b.clone();
    DenseLu::new(&a).solve(&mut x);
    for (u, w) in v.values().iter().zip(&x) {
        assert!((u - w).abs() <= 1e-8 * (1.0 + w.abs()));
    }
}

/// Where `β` comes from, on the medium random family: the defect `ē` (in
/// f64, approximate), `‖⟨U⟩⁻¹⟨L⟩⁻¹ ē‖∞` (what Algorithm E′ bounds) against
/// `‖|F⁻¹| ē‖∞` (what an exact `|F⁻¹|` would give; `F⁻¹` formed by `n`
/// solves), with and without `drop_tol` and for three pivot thresholds. A measurement, not a check.
#[test]
#[ignore = "a diagnostic; run in release with --ignored --nocapture"]
fn diagnose_beta() {
    let mut rng = Rng(0xfeed_face);
    println!(
        "| n | drop_tol | rel_pivot_tol | β (E′) | ‖ē‖∞ | ‖Ĝ ē‖∞ | ‖|F⁻¹| ē‖∞ | ‖Ĝ 𝟙‖∞ | ‖|F⁻¹| 𝟙‖∞ | max |l| |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for &n in &[50_usize, 100, 200, 400] {
        for c in 0..6 {
            let density = (3.0 + 5.0 * rng.unit()) / n as f64;
            let a = sparse_random(n, density, 3.0 * rng.unit(), true, &mut rng);
            for (drop, rel) in [(1e-12, 1e-2), (0.0, 1e-2), (0.0, 0.1), (0.0, 1.0)] {
                let opts = LuOptions {
                    drop_tol: drop,
                    rel_pivot_tol: rel,
                    ..LuOptions::default()
                };
                let Some(f) = lu_factors(&a, opts) else {
                    continue;
                };
                let (l, u) = (dense_of(&f.l), dense_of(&f.u));
                let p = &f.row_perm.fwd;
                let qp = &f.col_perm.fwd;
                // ē in f64 from double-double entries.
                let mut ebar = vec![0.0; n];
                for s in 0..n {
                    for t in 0..n {
                        let mut acc = Dd::from_f64(u[s][t]);
                        for k in 0..s.min(t + 1) {
                            acc += Dd::from_f64(l[s][k]) * Dd::from_f64(u[k][t]);
                        }
                        acc -= Dd::from_f64(a.at(p[s] as usize, qp[t] as usize));
                        ebar[s] += acc.hi().abs();
                    }
                }
                let comp = |v: &[f64]| -> Vec<f64> {
                    let mut y = v.to_vec();
                    for k in 0..n {
                        for s in k + 1..n {
                            y[s] += l[s][k].abs() * y[k];
                        }
                    }
                    for k in (0..n).rev() {
                        y[k] /= u[k][k].abs();
                        for s in 0..k {
                            y[s] += u[s][k].abs() * y[k];
                        }
                    }
                    y
                };
                // |F⁻¹| in step numbering: |U⁻¹ L⁻¹| by n solves.
                let mut finv = vec![vec![0.0; n]; n];
                for j in 0..n {
                    let mut y = vec![0.0; n];
                    y[j] = 1.0;
                    for k in 0..n {
                        for s in k + 1..n {
                            y[s] -= l[s][k] * y[k];
                        }
                    }
                    for k in (0..n).rev() {
                        y[k] /= u[k][k];
                        for s in 0..k {
                            y[s] -= u[s][k] * y[k];
                        }
                    }
                    for i in 0..n {
                        finv[i][j] = y[i].abs();
                    }
                }
                let apply = |v: &[f64]| -> f64 {
                    (0..n)
                        .map(|i| (0..n).map(|j| finv[i][j] * v[j]).sum::<f64>())
                        .fold(0.0, f64::max)
                };
                let norm = |v: &[f64]| v.iter().fold(0.0_f64, |m, &x| m.max(x));
                let ones = vec![1.0; n];
                let r = residual(&a.csr(), &vec![0.0; n], &vec![1.0; n]).expect("dims");
                let bl = forward_error_bound_lu(&a.csr(), &r, &f, &on()).expect("checked");
                let maxl = f.l.val.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                println!(
                    "| {n} | {drop:e} | {rel} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {maxl:.1e} |",
                    bl.contraction,
                    norm(&ebar),
                    norm(&comp(&ebar)),
                    apply(&ebar),
                    norm(&comp(&ones)),
                    apply(&ones),
                );
                let _ = c;
            }
        }
    }
}
