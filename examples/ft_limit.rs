//! The experiment that chose the default of
//! `LuOptions::update_multiplier_limit` (task L1.8, section 5.4 of the
//! contract, D-0113): Forrest–Tomlin exchanges with tiny pivots, in `f64`
//! and in double-double, under the multiplier limit of step U4′ of Chapter
//! 5.5 of `xingo-sparse`.
//!
//! A basis of order 150 (three families: random, small integers, columns
//! graded over 8 decades — over 12 the basis is singular to the `f64`
//! absolute pivot tolerance; five seeds each) goes through 90 exchanges;
//! every third entering column is the column it replaces times `weak` plus
//! a combination of two other basis columns, so that its pivot `α_r` is
//! about `weak` (the sequence of `dd_profile`'s update table, from the
//! pilot of the contract). After each applied update an FTRAN with the
//! updated factor is compared with one with a fresh factorization of the
//! same basis: `ratio` is the quotient of their scaled residuals, both
//! computed **in double-double** (`‖Bx − b‖∞ / (ε_T (max|b_ij| ‖x‖∞ +
//! ‖b‖∞))`, the fresh one floored at 1). A refusal by U4′ is told apart
//! from one by the stability test of U4 by repeating the update on a copy
//! of the factor without the limit.
//!
//! Output: one tab-separated line per run,
//!
//! ```text
//! RUN <scalar> <family> <seed> <weak> <limit> <attempts> <applied> <unstable> <mult> <singular> <other> <eligible> <mult_eligible> <median> <max> <max_mu> <growth> <since>
//! ```
//!
//! where `eligible` counts the exchanges with `|α_r| / ‖α‖∞ ≥ 10⁻⁵` and
//! `mult_eligible` the U4′ refusals among them; `max_mu` is the largest
//! `UpdateStats::max_multiplier` seen; `growth` and `since` are
//! `UpdateStats::update_growth` and the number of updates since the last
//! factorization at the worst applied update. Nothing is timed.
//!
//! Run with `cargo run --release -p xingo-extprec --example ft_limit`
//! (optionally `-- f64` or `-- dd` for one scalar).

use xingo_extprec::DoubleDouble;
use xingo_sparse::{
    Col, ColumnRef, CscMatrix, Lu, LuFactor, LuOptions, Real, RefactorReason, SparseVec, Triplets,
    UpdateOutcome,
};

type Dd = DoubleDouble;
type Entries = Vec<(usize, usize, f64)>;

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

fn graded_entries(n: usize, decades: f64, seed: u64) -> Entries {
    #[allow(clippy::cast_precision_loss)] // small
    random_entries(n, 3, seed)
        .into_iter()
        .map(|(i, j, v)| (i, j, v * 10f64.powf(-decades * j as f64 / n as f64)))
        .collect()
}

fn to_csc<T: Real>(n_rows: usize, n_cols: usize, entries: &[(usize, usize, f64)]) -> CscMatrix<T> {
    let mut t = Triplets::new(n_rows, n_cols);
    for &(i, j, v) in entries {
        t.push(idx(i), idx(j), T::from_f64(v));
    }
    t.to_csc().expect("valid triplets")
}

fn norm_inf_dd(v: &[Dd]) -> Dd {
    v.iter().fold(Dd::ZERO, |m, x| m.max(x.abs()))
}

/// The scaled residual of an FTRAN with `lu`, computed in double-double
/// on the entries of the basis as stored in `T`, in units of `ε_T`.
fn quality<T: Real>(
    lu: &mut Lu<T>,
    heading: &[ColumnRef],
    entries_by_col: &[Vec<(usize, Dd)>],
    amax: Dd,
) -> f64 {
    let n = heading.len();
    let b: Vec<f64> = (0..n)
        .map(|i| f64::from(u32::try_from((i * 37) % 11).expect("small")) - 5.0)
        .collect();
    let mut x = SparseVec::from_dense(&b.iter().map(|&v| T::from_f64(v)).collect::<Vec<T>>());
    lu.ftran(&mut x);
    let xd: Vec<Dd> = x.values().iter().map(|v| to_dd(*v)).collect();
    let mut r: Vec<Dd> = b.iter().map(|&v| Dd::from_f64(-v)).collect();
    for (k, c) in heading.iter().enumerate() {
        let ColumnRef::Structural(col) = c else {
            unreachable!("structural headings only")
        };
        for &(i, v) in &entries_by_col[col.index()] {
            r[i] += xd[k] * v;
        }
    }
    let bn = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let scale = amax * norm_inf_dd(&xd) + bn;
    (norm_inf_dd(&r) / scale).to_f64() / T::EPSILON.to_f64()
}

/// A value of `T` as a double-double (exact for `f64` and `Dd`).
fn to_dd<T: Real>(v: T) -> Dd {
    // `T` is `f64` or `Dd`: its value is the sum of at most two `f64`.
    let hi = v.to_f64();
    let lo = (v - T::from_f64(hi)).to_f64();
    Dd::from_f64(hi) + lo
}

struct Outcome {
    attempts: usize,
    applied: usize,
    unstable: usize,
    mult: usize,
    singular: usize,
    other: usize,
    eligible: usize,
    mult_eligible: usize,
    ratios: Vec<f64>,
    max_mu: f64,
    /// At the worst applied update: the update growth and the number of
    /// updates since the last factorization.
    worst: (f64, f64, usize),
}

#[allow(clippy::too_many_lines)] // one exchange sequence, as in dd_profile
fn run<T: Real>(base: &Entries, n: usize, weak: f64, limit: f64, seed: u64) -> Outcome {
    let exchanges = 90;
    let mut entries = base.clone();
    let mut rng = Lcg(0x55 ^ seed);
    for k in 0..exchanges {
        let j = n + k;
        let r = (7 * k + 3) % n;
        if k % 3 == 2 {
            let (o1, o2) = ((r + 1) % n, (r + 5) % n);
            for &(i, c, v) in base {
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
    let a: CscMatrix<T> = to_csc(n, n + exchanges, &entries);
    // The columns as stored in `T` (duplicates summed in `T`), exactly.
    let by_col: Vec<Vec<(usize, Dd)>> = (0..n + exchanges)
        .map(|j| {
            let (rows, vals) = a.col(Col::from_usize(j));
            rows.iter()
                .map(|&i| i as usize)
                .zip(vals.iter().map(|&v| to_dd(v)))
                .collect()
        })
        .collect();
    let amax = by_col
        .iter()
        .flatten()
        .fold(Dd::ZERO, |m, &(_, v)| m.max(v.abs()));
    let opts = LuOptions {
        refactor_interval: 1000,
        fill_factor: 1e9,
        update_multiplier_limit: limit,
        ..LuOptions::for_scalar::<T>()
    };
    let unlimited = LuOptions {
        update_multiplier_limit: f64::INFINITY,
        ..opts
    };
    let mut heading: Vec<ColumnRef> = (0..n)
        .map(|j| ColumnRef::Structural(Col::from_usize(j)))
        .collect();
    let mut lu = Lu::<T>::new(opts);
    lu.factorize(&(heading.as_slice(), &a))
        .expect("nonsingular");
    let mut o = Outcome {
        attempts: 0,
        applied: 0,
        unstable: 0,
        mult: 0,
        singular: 0,
        other: 0,
        eligible: 0,
        mult_eligible: 0,
        ratios: Vec::new(),
        max_mu: 0.0,
        worst: (0.0, 0.0, 0),
    };
    for k in 0..exchanges {
        let r = (7 * k + 3) % n;
        let entering = ColumnRef::Structural(Col::from_usize(n + k));
        let mut alpha = SparseVec::new(n);
        entering.for_each_entry(&a, |i, v| alpha.set(i, alpha.get(i) + v));
        lu.ftran_for_update(&mut alpha);
        let alpha_r = alpha.get(idx(r));
        let alpha_inf = alpha
            .values()
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs().to_f64()));
        let eligible = alpha_r.abs().to_f64() >= 1e-5 * alpha_inf;
        o.attempts += 1;
        if eligible {
            o.eligible += 1;
        }
        let mut trial = heading.clone();
        trial[r] = entering;
        let mut fresh = Lu::<T>::new(opts);
        let regular = fresh.factorize(&(trial.as_slice(), &a)).is_ok();
        let before = lu.clone();
        match lu.update(r, alpha_r) {
            Ok(UpdateOutcome::Updated) if regular => {
                o.applied += 1;
                heading = trial;
                o.max_mu = o.max_mu.max(lu.update_stats().max_multiplier);
                let q = quality(&mut lu, &heading, &by_col, amax);
                let q0 = quality(&mut fresh, &heading, &by_col, amax);
                let ratio = q / q0.max(1.0);
                o.ratios.push(ratio);
                if ratio > o.worst.0 {
                    let u = lu.update_stats();
                    o.worst = (ratio, u.update_growth, u.updates);
                }
            }
            Ok(UpdateOutcome::RefactorNeeded(RefactorReason::Unstable)) => {
                // U4 or U4′? The same update without the limit.
                let mut probe = before;
                probe.set_options(unlimited);
                let by_mult = !matches!(
                    probe.update(r, alpha_r),
                    Ok(UpdateOutcome::RefactorNeeded(RefactorReason::Unstable)) | Err(_)
                );
                if by_mult {
                    o.mult += 1;
                    if eligible {
                        o.mult_eligible += 1;
                    }
                } else {
                    o.unstable += 1;
                }
                if regular {
                    heading = trial;
                }
                lu.factorize(&(heading.as_slice(), &a))
                    .expect("a regular basis");
            }
            Ok(_) => {
                o.other += 1;
                if regular {
                    heading = trial;
                }
                lu.factorize(&(heading.as_slice(), &a))
                    .expect("a regular basis");
            }
            Err(_) => {
                o.singular += 1;
                lu.factorize(&(heading.as_slice(), &a))
                    .expect("the old basis");
            }
        }
    }
    o
}

fn family(name: &str, n: usize, seed: u64) -> Entries {
    match name {
        "random" => random_entries(n, 3, seed),
        "integers" => integer_entries(n, seed),
        "graded" => graded_entries(n, 8.0, seed),
        _ => unreachable!("three families"),
    }
}

fn sweep<T: Real>(label: &str) {
    let n = 150;
    let weaks = [
        1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16, 1e-18, 1e-20, 1e-22,
    ];
    let limits = [f64::INFINITY, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8];
    for fam in ["random", "integers", "graded"] {
        for s in 0..5_u64 {
            let seed = 97 + 1000 * s;
            let base = family(fam, n, seed);
            for &weak in &weaks {
                for &limit in &limits {
                    let mut o = run::<T>(&base, n, weak, limit, s);
                    o.ratios.sort_by(f64::total_cmp);
                    let (med, max) = if o.ratios.is_empty() {
                        (f64::NAN, f64::NAN)
                    } else {
                        (o.ratios[o.ratios.len() / 2], o.ratios[o.ratios.len() - 1])
                    };
                    println!(
                        "RUN\t{label}\t{fam}\t{seed}\t{weak:.0e}\t{limit:.0e}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{med:.3e}\t{max:.3e}\t{:.3e}\t{:.3e}\t{}",
                        o.attempts,
                        o.applied,
                        o.unstable,
                        o.mult,
                        o.singular,
                        o.other,
                        o.eligible,
                        o.mult_eligible,
                        o.max_mu,
                        o.worst.1,
                        o.worst.2
                    );
                }
            }
        }
    }
}

fn main() {
    let which = std::env::args().nth(1).unwrap_or_default();
    if which.is_empty() || which == "f64" {
        sweep::<f64>("f64");
    }
    if which.is_empty() || which == "dd" {
        sweep::<Dd>("dd");
    }
}
