//! Which thresholds of `xingo-sparse` are calibrated for `f64`, and what
//! they should be in double-double (task L2.2b, Chapter 11.8 of the
//! crate): the measurements behind Rule P of `xingo-sparse` (its Chapter
//! 5.3, `LuOptions::for_scalar`) and the scaled KKT defaults (Chapter 5.14).
//!
//! Each table factorizes the same matrices in `DoubleDouble` under several
//! *profiles* of options and prints what came out: whether the matrix was
//! accepted, the residual of the solves (computed in double-double), the
//! growth, the fill and the pivot counts. Nothing is timed (`dd_bench` is
//! the timing harness), so the output is deterministic.
//!
//! Run with `cargo run --release -p xingo-extprec --example dd_profile`.

use xingo_extprec::DoubleDouble;
use xingo_sparse::{
    Amd, Col, ColumnRef, CscMatrix, KktForm, KktSolver, Lu, LuError, LuFactor, LuOptions,
    RefactorReason, Refinement, Regularization, SparseVec, SupernodalLdl, SupernodalOptions,
    Triplets, UpdateOutcome,
};

type Dd = DoubleDouble;

/// `ε_dd / ε_f64 = 2⁻⁵²`: the factor by which a threshold proportional to
/// the machine epsilon shrinks from `f64` to double-double.
const EPS_RATIO: f64 = f64::EPSILON;

/// Knuth's MMIX generator; `unit` is uniform in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    #[allow(clippy::cast_precision_loss)] // 53 random bits
    fn unit(&mut self) -> f64 {
        (self.next() as f64) / (1_u64 << 52) as f64 - 1.0
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).expect("below n")
    }
}

fn idx(i: usize) -> u32 {
    u32::try_from(i).expect("fits u32")
}

type Entries = Vec<(usize, usize, f64)>;

fn to_csc(n_rows: usize, n_cols: usize, entries: &[(usize, usize, f64)]) -> CscMatrix<Dd> {
    let mut t = Triplets::new(n_rows, n_cols);
    for &(i, j, v) in entries {
        t.push(idx(i), idx(j), Dd::from_f64(v));
    }
    t.to_csc().expect("valid triplets")
}

/// The random sparse basis of `tests/sparse_dd.rs`.
fn random_entries(n: usize, per_col: usize, seed: u64) -> Entries {
    let mut rng = Lcg(seed);
    let mut e = Vec::new();
    for j in 0..n {
        let d = 3.0 + rng.unit();
        e.push((j, j, if rng.next().is_multiple_of(2) { d } else { -d }));
        for _ in 0..per_col.min(n - 1) {
            let i = rng.below(n);
            if i != j {
                e.push((i, j, rng.unit()));
            }
        }
    }
    e
}

/// `random_entries` with column `j` scaled by `10^(−decades · j / n)`: the
/// columns of tiny entries of an unscaled model (sherman3 in the
/// documentation of `LuOptions::abs_pivot_tol`).
fn graded_entries(n: usize, decades: f64, seed: u64) -> Entries {
    #[allow(clippy::cast_precision_loss)] // small
    random_entries(n, 3, seed)
        .into_iter()
        .map(|(i, j, v)| (i, j, v * 10f64.powf(-decades * j as f64 / n as f64)))
        .collect()
}

/// A basis of small integers, as in an LP: `±1`, `±2`, a dominant diagonal.
fn integer_entries(n: usize, seed: u64) -> Entries {
    let mut rng = Lcg(seed);
    let mut e = Vec::new();
    for j in 0..n {
        e.push((j, j, 5.0));
        for _ in 0..3 {
            let i = rng.below(n);
            if i != j {
                #[allow(clippy::cast_precision_loss)] // 1 or 2
                let v = (1 + rng.below(2)) as f64;
                e.push((i, j, if rng.next().is_multiple_of(2) { v } else { -v }));
            }
        }
    }
    e
}

/// `random_entries` with column 1 replaced by column 0 plus `delta` times
/// a unit vector: a basis that is singular to `f64` when `delta` is small
/// and regular in double-double.
fn nearly_dependent_entries(n: usize, delta: f64, seed: u64) -> Entries {
    let mut e: Entries = random_entries(n, 3, seed)
        .into_iter()
        .filter(|&(_, j, _)| j != 1)
        .collect();
    let col0: Entries = e.iter().filter(|&&(_, j, _)| j == 0).copied().collect();
    for (i, _, v) in col0 {
        e.push((i, 1, v));
    }
    e.push((n - 1, 1, delta));
    e
}

fn norm_inf(v: &[Dd]) -> Dd {
    v.iter().fold(Dd::ZERO, |acc, x| acc.max(x.abs()))
}

fn max_abs(a: &CscMatrix<Dd>) -> Dd {
    (0..a.n_cols()).fold(Dd::ZERO, |acc, j| {
        acc.max(norm_inf(a.col(Col::from_usize(j)).1))
    })
}

/// `B v − b` for the basis `heading` of `a`.
fn basis_residual(heading: &[ColumnRef], a: &CscMatrix<Dd>, v: &[Dd], b: &[Dd]) -> Vec<Dd> {
    let mut r: Vec<Dd> = b.iter().map(|&x| -x).collect();
    for (k, c) in heading.iter().enumerate() {
        c.for_each_entry(a, |i, b_ik| r[i as usize] += b_ik * v[k]);
    }
    r
}

fn structural(n: usize) -> Vec<ColumnRef> {
    (0..n)
        .map(|j| ColumnRef::Structural(Col::from_usize(j)))
        .collect()
}

/// The profiles of `LuOptions` under test.
fn lu_profiles() -> Vec<(&'static str, LuOptions)> {
    let d = LuOptions::default();
    vec![
        ("f64 (default)", d),
        ("drop_tol = 0", LuOptions { drop_tol: 0.0, ..d }),
        ("dd (Rule P)", LuOptions::for_scalar::<Dd>()),
        (
            "dd, tols = 0",
            LuOptions {
                drop_tol: 0.0,
                abs_pivot_tol: 0.0,
                ..LuOptions::for_scalar::<Dd>()
            },
        ),
    ]
}

/// One factorization and one FTRAN: a row of the LU table.
fn lu_row(name: &str, n: usize, entries: &[(usize, usize, f64)], exact_ones: bool) {
    let a = to_csc(n, n, entries);
    let heading = structural(n);
    // b = A·1 when the solution is known to be the vector of ones.
    let b: Vec<Dd> = if exact_ones {
        let ones = vec![Dd::ONE; n];
        let zero = vec![Dd::ZERO; n];
        basis_residual(&heading, &a, &ones, &zero)
    } else {
        let mut rng = Lcg(0xabcd ^ n as u64);
        (0..n).map(|_| Dd::from_f64(rng.unit())).collect()
    };
    for (label, opts) in lu_profiles() {
        let mut lu = Lu::<Dd>::new(opts);
        match lu.factorize(&a) {
            Ok(()) => {
                let mut x = SparseVec::from_dense(&b);
                lu.ftran(&mut x);
                let r = basis_residual(&heading, &a, x.values(), &b);
                let res = norm_inf(&r);
                let scale = max_abs(&a) * norm_inf(x.values()) + norm_inf(&b);
                #[allow(clippy::cast_precision_loss)] // n is small
                let scaled = (res / (Dd::EPSILON * scale)).to_f64() / n as f64;
                let err = if exact_ones {
                    let e: Vec<Dd> = x.values().iter().map(|&v| v - 1.0).collect();
                    format!("{:.1e}", norm_inf(&e).to_f64())
                } else {
                    "—".to_string()
                };
                let s = lu.stats();
                println!(
                    "| {name} | {label} | ok | {:.1e} | {scaled:.1e} | {err} | {:.1e} | {} | {} |",
                    res.to_f64(),
                    s.growth,
                    s.nnz_l + s.nnz_u,
                    s.markowitz_pivots
                );
            }
            Err(LuError::Singular {
                rank_deficiency, ..
            }) => println!(
                "| {name} | {label} | **singular, deficiency {rank_deficiency}** | — | — | — | — | — | — |"
            ),
            Err(e) => println!("| {name} | {label} | error {e:?} | — | — | — | — | — | — |"),
        }
    }
}

/// The drop tolerance alone, over several matrices of one family: the
/// median and the worst scaled residual of FTRAN and the mean fill.
fn drop_sweep(name: &str, n: usize, make: &dyn Fn(u64) -> Entries) {
    let seeds: Vec<u64> = (0..15).map(|s| 1000 + 17 * s).collect();
    for (label, drop_tol) in [
        ("0", 0.0),
        ("2⁻¹⁰⁰ = 7.9e-31 (16 ε_dd)", 2f64.powi(-100)),
        ("1e-29", 1e-29),
        ("2.2e-28 (default × 2⁻⁵²)", 1e-12 * EPS_RATIO),
        ("1e-24", 1e-24),
        ("1e-12 (f64 default)", 1e-12),
    ] {
        let opts = LuOptions {
            drop_tol,
            ..LuOptions::for_scalar::<Dd>()
        };
        let mut scaled: Vec<f64> = Vec::new();
        let (mut nnz, mut growth, mut failed) = (0_usize, 0.0_f64, 0);
        for &seed in &seeds {
            let a = to_csc(n, n, &make(seed));
            let heading = structural(n);
            let mut rng = Lcg(seed ^ 0xabcd);
            let b: Vec<Dd> = (0..n).map(|_| Dd::from_f64(rng.unit())).collect();
            let mut lu = Lu::<Dd>::new(opts);
            if lu.factorize(&a).is_err() {
                failed += 1;
                continue;
            }
            let mut x = SparseVec::from_dense(&b);
            lu.ftran(&mut x);
            let res = norm_inf(&basis_residual(&heading, &a, x.values(), &b));
            let scale = max_abs(&a) * norm_inf(x.values()) + norm_inf(&b);
            #[allow(clippy::cast_precision_loss)] // n is small
            scaled.push((res / (Dd::EPSILON * scale)).to_f64() / n as f64);
            nnz += lu.stats().nnz_l + lu.stats().nnz_u;
            growth = growth.max(lu.stats().growth);
        }
        scaled.sort_by(f64::total_cmp);
        let ok = scaled.len();
        if ok == 0 {
            println!("| {name} | {label} | {failed} | — | — | — | — |");
            continue;
        }
        println!(
            "| {name} | {label} | {failed} | {:.2e} | {:.2e} | {} | {growth:.1e} |",
            scaled[ok / 2],
            scaled[ok - 1],
            nnz / ok
        );
    }
}

#[allow(clippy::cast_precision_loss)] // small integers
fn hilbert(n: usize) -> CscMatrix<Dd> {
    let mut t = Triplets::new(n, n);
    for i in 0..n {
        for j in 0..n {
            t.push(idx(i), idx(j), Dd::ONE / Dd::from_f64((i + j + 1) as f64));
        }
    }
    t.to_csc().expect("valid triplets")
}

/// The Hilbert rows of the LU table (entries that are not `f64`: built in
/// double-double).
fn hilbert_row(n: usize) {
    let a = hilbert(n);
    let heading = structural(n);
    let ones = vec![Dd::ONE; n];
    let zero = vec![Dd::ZERO; n];
    let b = basis_residual(&heading, &a, &ones, &zero);
    for (label, opts) in lu_profiles() {
        let mut lu = Lu::<Dd>::new(opts);
        match lu.factorize(&a) {
            Ok(()) => {
                let mut x = SparseVec::from_dense(&b);
                lu.ftran(&mut x);
                let r = basis_residual(&heading, &a, x.values(), &b);
                let res = norm_inf(&r);
                let scale = max_abs(&a) * norm_inf(x.values()) + norm_inf(&b);
                #[allow(clippy::cast_precision_loss)] // n is small
                let scaled = (res / (Dd::EPSILON * scale)).to_f64() / n as f64;
                let e: Vec<Dd> = x.values().iter().map(|&v| v - 1.0).collect();
                let s = lu.stats();
                println!(
                    "| hilbert {n} | {label} | ok | {:.1e} | {scaled:.1e} | {:.1e} | {:.1e} | {} | {} |",
                    res.to_f64(),
                    norm_inf(&e).to_f64(),
                    s.growth,
                    s.nnz_l + s.nnz_u,
                    s.markowitz_pivots
                );
            }
            Err(LuError::Singular {
                rank_deficiency, ..
            }) => println!(
                "| hilbert {n} | {label} | **singular, deficiency {rank_deficiency}** | — | — | — | — | — | — |"
            ),
            Err(e) => println!("| hilbert {n} | {label} | error {e:?} | — | — | — | — | — | — |"),
        }
    }
}

/// The scaled residual `‖Bx − b‖∞ / (ε_dd (max|b_ij| ‖x‖∞ + ‖b‖∞))` of an
/// FTRAN with the factor `lu` of the basis `heading`.
fn ftran_quality(lu: &mut Lu<Dd>, heading: &[ColumnRef], a: &CscMatrix<Dd>) -> f64 {
    let n = heading.len();
    let b: Vec<Dd> = (0..n)
        .map(|i| Dd::from_f64(f64::from(u32::try_from((i * 37) % 11).expect("small")) - 5.0))
        .collect();
    let mut x = SparseVec::from_dense(&b);
    lu.ftran(&mut x);
    let res = norm_inf(&basis_residual(heading, a, x.values(), &b));
    let scale = max_abs(a) * norm_inf(x.values()) + norm_inf(&b);
    (res / (Dd::EPSILON * scale)).to_f64()
}

/// A sequence of Forrest–Tomlin exchanges under a given stability
/// tolerance: how many updates were applied or refused, and — for the
/// applied ones — how much worse the updated factor solves than a fresh
/// factorization of the same basis (the ratio of their scaled residuals:
/// what the stability test is there to bound).
fn update_row(n: usize, weak: f64, stability_tol: f64, label: &str) {
    let exchanges = 90;
    let mut entries = random_entries(n, 3, 97);
    let mut rng = Lcg(0x55);
    // Entering columns: sparse random ones, and — every third — the column
    // it replaces times `weak` plus a combination of two other basis
    // columns: the pivot `α_r` is `weak`, the case in which an update
    // loses accuracy.
    let base: Entries = entries.clone();
    for k in 0..exchanges {
        let j = n + k;
        let r = (7 * k + 3) % n;
        if k % 3 == 2 {
            let (o1, o2) = ((r + 1) % n, (r + 5) % n);
            for &(i, c, v) in &base {
                if c == o1 {
                    entries.push((i, j, v));
                }
                if c == o2 {
                    entries.push((i, j, -0.5 * v));
                }
                if c == r {
                    entries.push((i, j, v * weak));
                }
            }
        } else {
            for _ in 0..4 {
                entries.push((rng.below(n), j, rng.unit()));
            }
            entries.push((r, j, 2.0 + rng.unit()));
        }
    }
    let a = to_csc(n, n + exchanges, &entries);
    let opts = LuOptions {
        update_stability_tol: stability_tol,
        refactor_interval: 1000,
        fill_factor: 1e9,
        ..LuOptions::for_scalar::<Dd>()
    };
    let mut heading = structural(n);
    let mut lu = Lu::<Dd>::new(opts);
    lu.factorize(&(heading.as_slice(), &a))
        .expect("nonsingular");
    let (mut applied, mut unstable, mut singular, mut other) = (0, 0, 0, 0);
    let mut ratios: Vec<f64> = Vec::new();
    let mut worst = 0.0_f64;
    for k in 0..exchanges {
        let r = (7 * k + 3) % n;
        let entering = ColumnRef::Structural(Col::from_usize(n + k));
        let mut alpha = SparseVec::new(n);
        entering.for_each_entry(&a, |i, v| alpha.set(i, alpha.get(i) + v));
        lu.ftran_for_update(&mut alpha);
        let alpha_r = alpha.get(idx(r));
        let mut trial = heading.clone();
        trial[r] = entering;
        // The reference: a fresh factorization of the new basis.
        let mut fresh = Lu::<Dd>::new(opts);
        let regular = fresh.factorize(&(trial.as_slice(), &a)).is_ok();
        match lu.update(r, alpha_r) {
            Ok(UpdateOutcome::Updated) if regular => {
                applied += 1;
                heading = trial;
                let q = ftran_quality(&mut lu, &heading, &a);
                let q0 = ftran_quality(&mut fresh, &heading, &a);
                ratios.push(q / q0.max(1.0));
                worst = worst.max(q);
            }
            Ok(UpdateOutcome::RefactorNeeded(RefactorReason::Unstable)) => {
                unstable += 1;
                if regular {
                    heading = trial;
                }
                lu.factorize(&(heading.as_slice(), &a))
                    .expect("a regular basis");
            }
            Ok(_) => {
                other += 1;
                if regular {
                    heading = trial;
                }
                lu.factorize(&(heading.as_slice(), &a))
                    .expect("a regular basis");
            }
            Err(_) => {
                singular += 1;
                lu.factorize(&(heading.as_slice(), &a))
                    .expect("the old basis");
            }
        }
    }
    ratios.sort_by(f64::total_cmp);
    let (med, max) = if ratios.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        (ratios[ratios.len() / 2], ratios[ratios.len() - 1])
    };
    println!(
        "| {weak:.0e} | {label} | {applied} | {unstable} | {singular} | {other} | {med:.1e} | {max:.1e} | {worst:.1e} |"
    );
}

/// `[−D, Aᵀ; A, 0]` (lower triangle) of `tests/kkt_workspace.rs` in
/// `xingo-sparse`: a banded `A` (m × n) and `D` with a spread of
/// `10^{±spread}`.
fn augmented(m: usize, n: usize, spread: f64) -> Entries {
    let mut t = Vec::new();
    for j in 0..n {
        #[allow(clippy::cast_precision_loss)] // small
        let u = (j as f64 * 0.618_033_988_75).fract();
        t.push((j, j, -(10f64.powf(spread * (2.0 * u - 1.0)))));
        for i in [j % m, (j + 1) % m] {
            #[allow(clippy::cast_precision_loss)] // small
            let v = 1.0 + (i + j) as f64 / 7.0;
            t.push((n + i, j, v));
        }
    }
    t
}

/// `K x − b` for a symmetric `K` given by its lower triangle.
fn symmetric_residual(k_lower: &CscMatrix<Dd>, x: &[Dd], b: &[Dd]) -> Vec<Dd> {
    let mut r: Vec<Dd> = b.iter().map(|&v| -v).collect();
    for j in 0..k_lower.n_cols() {
        let (rows, vals) = k_lower.col(Col::from_usize(j));
        for (&i, &v) in rows.iter().zip(vals) {
            let i = i as usize;
            r[i] += v * x[j];
            if i != j {
                r[j] += v * x[i];
            }
        }
    }
    r
}

/// The profiles of the KKT solver: regularization, refinement target and
/// the two thresholds of the pivoted path.
struct KktProfile {
    label: &'static str,
    reg: Regularization<Dd>,
    target: f64,
    opts: SupernodalOptions,
}

fn kkt_profiles() -> Vec<KktProfile> {
    let d = SupernodalOptions::default();
    let scaled = SupernodalOptions {
        zero_pivot_tol: d.zero_pivot_tol * EPS_RATIO,
        pivot_threshold: d.pivot_threshold * EPS_RATIO.sqrt(),
        ..d
    };
    let reg = Regularization::<Dd>::default();
    // ρδ = 10⁻¹⁴ ≈ 45 ε_f64 becomes ≈ 45 ε_dd; each shift scaled by √(ε_dd/ε_f64).
    let s = EPS_RATIO.sqrt();
    let reg_dd = Regularization::<Dd> {
        primal: Dd::from_f64(1e-6 * s),
        dual: Dd::from_f64(1e-8 * s),
        pivot_tol: Some(Dd::from_f64(1e-13 * EPS_RATIO)),
        bump: Dd::from_f64(2e-7 * s),
        ..reg
    };
    vec![
        KktProfile {
            label: "f64 reg., target 1e-12",
            reg,
            target: 1e-12,
            opts: d,
        },
        KktProfile {
            label: "f64 reg., target 2.2e-28",
            reg,
            target: 1e-12 * EPS_RATIO,
            opts: d,
        },
        KktProfile {
            label: "dd reg., target 2.2e-28",
            reg: reg_dd,
            target: 1e-12 * EPS_RATIO,
            opts: scaled,
        },
        KktProfile {
            label: "no reg. (LBLᵀ), f64 thresholds",
            reg: Regularization::none(),
            target: 1e-12 * EPS_RATIO,
            opts: d,
        },
        KktProfile {
            label: "no reg. (LBLᵀ), dd thresholds",
            reg: Regularization::none(),
            target: 1e-12 * EPS_RATIO,
            opts: scaled,
        },
    ]
}

fn kkt_row(name: &str, n: usize, form: KktForm, entries: &[(usize, usize, f64)]) {
    let k = to_csc(n, n, entries);
    let b: Vec<Dd> = (0..n)
        .map(|i| Dd::from_f64(f64::from(u32::try_from((i * 37) % 11).expect("small")) - 5.0))
        .collect();
    for p in kkt_profiles() {
        let solver = SupernodalLdl::<Dd>::new(xingo_sparse::NaiveDense, p.opts);
        let sym = solver
            .analyze(&k.pattern(), form, &Amd::default())
            .expect("symmetric lower pattern");
        match solver.factorize(&sym, &k, p.reg) {
            Ok((num, inertia)) => {
                // The plain solve with the (regularized) factor …
                let mut x0 = b.clone();
                solver.solve(&num, &mut x0);
                let r0 = norm_inf(&symmetric_residual(&k, &x0, &b)).to_f64();
                // … and the refined one against the unregularized K.
                let mut x = vec![Dd::ZERO; n];
                let rep = solver.solve_refined(
                    &num,
                    &k,
                    &b,
                    &mut x,
                    Refinement {
                        max_iters: 30,
                        target_residual: Dd::from_f64(p.target),
                    },
                );
                let r = norm_inf(&symmetric_residual(&k, &x, &b)).to_f64();
                println!(
                    "| {name} | {} | {} | {} | {} | {r0:.1e} | {} | {} | {r:.1e} |",
                    p.label,
                    if inertia == form.expected_inertia(n) {
                        "ok"
                    } else {
                        "**wrong**"
                    },
                    num.bumped,
                    num.delayed(),
                    rep.iters,
                    if rep.converged { "yes" } else { "**no**" },
                );
            }
            Err(e) => println!(
                "| {name} | {} | error {e:?} | — | — | — | — | — | — |",
                p.label
            ),
        }
    }
}

fn main() {
    println!("## LU (`Lu<DoubleDouble>`): factorization and FTRAN under four profiles\n");
    println!(
        "Profiles: `f64 (default)` = `LuOptions::default()`; `drop_tol = 0` = the options of the \
         tests of L2.2a; `dd (Rule P)` = `LuOptions::for_scalar::<DoubleDouble>()` = {:?}; `dd, tols = 0` = \
         the same with `drop_tol = abs_pivot_tol = 0`.\n",
        LuOptions::for_scalar::<Dd>()
    );
    println!(
        "| matrix | profile | outcome | ‖Bx − b‖∞ | scaled (units of n·ε_dd) | ‖x − 1‖∞ | growth | nnz(L+U) | Markowitz pivots |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    lu_row("random 300", 300, &random_entries(300, 3, 7), false);
    lu_row("random 800", 800, &random_entries(800, 2, 8), false);
    lu_row("integers 500", 500, &integer_entries(500, 9), true);
    lu_row(
        "graded 200, 12 decades",
        200,
        &graded_entries(200, 12.0, 10),
        false,
    );
    lu_row(
        "graded 200, 30 decades",
        200,
        &graded_entries(200, 30.0, 10),
        false,
    );
    for delta in [1e-6, 1e-11, 1e-20, 1e-27] {
        lu_row(
            &format!("nearly dependent 100, δ = {delta:.0e}"),
            100,
            &nearly_dependent_entries(100, delta, 11),
            false,
        );
    }
    for n in [8, 10, 12, 16, 20, 24] {
        hilbert_row(n);
    }

    println!("\n## LU: the drop tolerance alone\n");
    println!(
        "15 matrices per family (seeds 1000, 1017, …); the other options are Rule P's dd \
         profile. `scaled` is `‖Bx − b‖∞ / (n ε_dd (max|b_ij| ‖x‖∞ + ‖b‖∞))`.\n"
    );
    println!(
        "| family | `drop_tol` | singular | median scaled | worst scaled | mean nnz(L+U) | worst growth |"
    );
    println!("|---|---|---|---|---|---|---|");
    drop_sweep("random 300", 300, &|s| random_entries(300, 3, s));
    drop_sweep("random 800", 800, &|s| random_entries(800, 2, s));
    drop_sweep("integers 500", 500, &|s| integer_entries(500, s));
    drop_sweep("graded 200, 12 decades", 200, &|s| {
        graded_entries(200, 12.0, s)
    });

    println!("\n## Forrest–Tomlin updates: the stability tolerance\n");
    println!(
        "90 exchanges on a random basis of order 150 (other options: Rule P's dd profile, \
         with the update and fill limits lifted); every third entering column has pivot `α_r = \
         weak`. `ratio` is the scaled residual of an FTRAN with the updated factor over that of \
         a fresh factorization of the same basis (floored at one unit of `ε_dd`), over the \
         applied updates; `worst` is the largest scaled residual (units of `ε_dd`) of an \
         updated factor.\n"
    );
    println!(
        "| weak | `update_stability_tol` | applied | refused: unstable | refused: singular | other | median ratio | max ratio | worst |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for weak in [1e-2, 1e-6, 1e-10, 1e-14, 1e-18, 1e-22] {
        for (tol, label) in [
            (1e-8, "1e-8 (f64 default)"),
            (1e-8 * EPS_RATIO.sqrt(), "1.5e-16 (× 2⁻²⁶)"),
            (1e-8 * EPS_RATIO, "2.2e-24 (× 2⁻⁵²)"),
            (1e-28, "1e-28"),
        ] {
            update_row(150, weak, tol, label);
        }
    }

    println!("\n## KKT (`SupernodalLdl<DoubleDouble>`): regularization and refinement\n");
    println!(
        "The augmented systems of `xingo-sparse/tests/kkt_workspace.rs` (`[−D Aᵀ; A 0]`, `D` \
         spread over `10^±spread`), refined against the unregularized `K` (at most 30 steps).\n"
    );
    println!(
        "| matrix | profile | inertia | bumped | delayed | plain-solve residual | refinement steps | converged | final ‖Kx − b‖∞ |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for (m, n, spread) in [(40, 60, 0.0), (40, 60, 4.0), (40, 60, 8.0), (200, 300, 4.0)] {
        kkt_row(
            &format!("augmented {m}×{n}, spread {spread}"),
            n + m,
            KktForm::Augmented {
                n_primal: n,
                n_dual: m,
            },
            &augmented(m, n, spread),
        );
    }
}
