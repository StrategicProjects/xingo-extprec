//! Timing harness of the double-double kernels and factorizations (task
//! L2.2b): what the SIMD kernels of Chapter 11.6 and the dense kernel
//! `DdDense` of Chapter 11.7 buy, and what double-double costs over `f64`.
//!
//! One run measures one *group* and prints one line per measurement:
//!
//! ```text
//! RESULT <backend> <group> <name> <n> <median ns per unit> <min ns per unit> <samples>
//! ```
//!
//! (tab-separated). The scalar and the SIMD kernels are two builds of this
//! example — `--no-default-features --features sparse` and the default —
//! run one after the other; `backend` tells them apart. The unit is one
//! element for the vector kernels, one entry update `c_ij ← c_ij − a_il
//! b_lj` for `gemm_sub`, and one whole factorization or solve for the
//! `lu` and `ldl` groups.
//!
//! Each measurement is the median of [`SAMPLES`] samples of about
//! [`SAMPLE_NANOS`] each (the repetition count is calibrated first).
//! Nothing here asserts a speed: the numbers are for the tables of
//! `DECISIONS.md`.
//!
//! Run with `cargo run --release -p xingo-extprec --example dd_bench --
//! <group>`, `<group>` one of `k8`, `k64`, `k1024`, `k65536`, `lu`, `ldl`.

use std::hint::black_box;
use std::time::Instant;

use xingo_extprec::{DdDense, DoubleDouble, kernels};
use xingo_sparse::{
    Amd, CscMatrix, DenseKernel, KktForm, KktSolver, Lu, LuFactor, LuOptions, NaiveDense, Real,
    Regularization, SparseVec, SupernodalLdl, SupernodalOptions, Triplets,
};

type Dd = DoubleDouble;

/// Samples per measurement.
const SAMPLES: usize = 15;
/// Target duration of one sample.
const SAMPLE_NANOS: u128 = 20_000_000;

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

    fn f64s(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.unit()).collect()
    }

    /// Double-doubles with full significands.
    fn dds(&mut self, n: usize) -> Vec<Dd> {
        (0..n)
            .map(|_| Dd::from_f64(self.unit()) / (3.0 + self.unit()))
            .collect()
    }
}

/// Times `f` and prints the `RESULT` line: the median and the minimum over
/// the samples, per `units` units of work per call.
#[allow(clippy::cast_precision_loss)] // durations and counts far below 2^53
fn measure(group: &str, name: &str, n: usize, units: usize, mut f: impl FnMut()) {
    // Calibrate: double the repetitions until a sample is long enough.
    let mut reps: u128 = 1;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let dt = t.elapsed().as_nanos();
        if dt >= SAMPLE_NANOS / 2 || reps >= 1 << 30 {
            reps = (reps * SAMPLE_NANOS / dt.max(1)).max(1);
            break;
        }
        reps *= 2;
    }
    let mut per_unit: Vec<f64> = (0..SAMPLES)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed().as_nanos() as f64 / (reps as f64 * units as f64)
        })
        .collect();
    per_unit.sort_by(f64::total_cmp);
    println!(
        "RESULT\t{}\t{group}\t{name}\t{n}\t{:.4}\t{:.4}\t{SAMPLES}",
        kernels::backend(),
        per_unit[SAMPLES / 2],
        per_unit[0]
    );
}

/// The vector kernels on `n` elements.
fn kernel_group(n: usize) {
    let group = format!("k{n}");
    let mut rng = Lcg(0xbe7c ^ n as u64);
    let (x, y) = (rng.f64s(n), rng.f64s(n));
    let (p, q) = (rng.dds(n), rng.dds(n));
    let alpha = Dd::ONE / 3.0;

    // The f64 baselines: a plain sum and a plain dot product (one
    // accumulator, the loop the compiler does not reorder).
    measure(&group, "f64 sum", n, n, || {
        black_box(black_box(&x).iter().sum::<f64>());
    });
    measure(&group, "f64 dot", n, n, || {
        black_box(
            black_box(&x)
                .iter()
                .zip(black_box(&y))
                .map(|(a, b)| a * b)
                .sum::<f64>(),
        );
    });
    measure(&group, "sum2", n, n, || {
        black_box(kernels::sum2(black_box(&x)));
    });
    measure(&group, "dot2", n, n, || {
        black_box(kernels::dot2(black_box(&x), black_box(&y)));
    });
    measure(&group, "dot2_err", n, n, || {
        black_box(kernels::dot2_err(black_box(&x), black_box(&y)));
    });
    measure(&group, "dot", n, n, || {
        black_box(kernels::dot(black_box(&p), black_box(&q)));
    });
    measure(&group, "dot_mixed", n, n, || {
        black_box(kernels::dot_mixed(black_box(&p), black_box(&x)));
    });
    // axpy with α and −α in turn, so that y stays bounded.
    let mut z = q.clone();
    let mut sign = alpha;
    measure(&group, "axpy", n, n, || {
        kernels::axpy(black_box(sign), black_box(&p), black_box(&mut z));
        sign = -sign;
    });
    // gemm_sub: C (m × 4) ← C − A (m × 8) B (8 × 4), B negated in turn.
    let (m, k, cols) = (n, 8, 4);
    let a = rng.dds(m * k);
    let b = rng.dds(k * cols);
    let minus_b: Vec<Dd> = b.iter().map(|&v| -v).collect();
    let mut c = rng.dds(m * cols);
    let mut flip = false;
    measure(&group, "gemm_sub", n, m * k * cols, || {
        let bb = if flip { &minus_b } else { &b };
        kernels::gemm_sub(m, k, cols, black_box(&a), black_box(bb), black_box(&mut c));
        flip = !flip;
    });
}

fn idx(i: usize) -> u32 {
    u32::try_from(i).expect("fits u32")
}

fn to_csc<T: Real>(n: usize, entries: &[(usize, usize, f64)]) -> CscMatrix<T> {
    let mut t = Triplets::new(n, n);
    for &(i, j, v) in entries {
        t.push(idx(i), idx(j), T::from_f64(v));
    }
    t.to_csc().expect("valid triplets")
}

/// The random sparse basis of `tests/sparse_dd.rs`.
fn random_entries(n: usize, per_col: usize, seed: u64) -> Vec<(usize, usize, f64)> {
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

/// `Lu<T>`: one factorization, and one FTRAN with a dense right-hand side.
fn lu_in<T: Real>(label: &str, n: usize, per_col: usize, opts: LuOptions) {
    let a: CscMatrix<T> = to_csc(n, &random_entries(n, per_col, 7));
    let mut rng = Lcg(0xabcd);
    let b: Vec<T> = (0..n).map(|_| T::from_f64(rng.unit())).collect();
    let mut lu = Lu::<T>::new(opts);
    measure("lu", &format!("{label} factorize"), n, 1, || {
        lu.factorize(black_box(&a)).expect("nonsingular");
    });
    measure("lu", &format!("{label} ftran"), n, 1, || {
        let mut x = SparseVec::from_dense(black_box(&b));
        lu.ftran(&mut x);
        black_box(&x);
    });
}

/// The lower triangle of the five-point Laplacian of a `g × g` grid,
/// shifted to be safely positive definite.
fn grid_lower(g: usize) -> Vec<(usize, usize, f64)> {
    let mut e = Vec::new();
    for y in 0..g {
        for x in 0..g {
            let j = x + y * g;
            e.push((j, j, 4.25));
            if x + 1 < g {
                e.push((j + 1, j, -1.0));
            }
            if y + 1 < g {
                e.push((j + g, j, -1.0));
            }
        }
    }
    e
}

/// The lower triangle of a dense symmetric positive definite matrix: one
/// supernode, the dense kernels alone.
fn dense_lower(n: usize) -> Vec<(usize, usize, f64)> {
    let mut rng = Lcg(0xd15e);
    let mut e = Vec::new();
    for j in 0..n {
        #[allow(clippy::cast_precision_loss)] // small
        e.push((j, j, n as f64 + rng.unit()));
        for i in j + 1..n {
            e.push((i, j, rng.unit()));
        }
    }
    e
}

/// The lower triangle of the seven-point Laplacian of a `g × g × g` grid,
/// shifted: enough fill for the default options to choose dense blocks.
fn grid3_lower(g: usize) -> Vec<(usize, usize, f64)> {
    let mut e = Vec::new();
    for z in 0..g {
        for y in 0..g {
            for x in 0..g {
                let j = x + g * (y + g * z);
                e.push((j, j, 6.25));
                if x + 1 < g {
                    e.push((j + 1, j, -1.0));
                }
                if y + 1 < g {
                    e.push((j + g, j, -1.0));
                }
                if z + 1 < g {
                    e.push((j + g * g, j, -1.0));
                }
            }
        }
    }
    e
}

/// `SupernodalLdl<T, K>`: one numeric factorization, and one solve.
fn ldl_in<T: Real, K: DenseKernel<T>>(
    label: &str,
    name: &str,
    kernel: K,
    opts: SupernodalOptions,
    n: usize,
    entries: &[(usize, usize, f64)],
) {
    let k: CscMatrix<T> = to_csc(n, entries);
    let solver = SupernodalLdl::<T, K>::new(kernel, opts);
    let form = KktForm::NormalEquations;
    let sym = solver
        .analyze(&k.pattern(), form, &Amd::default())
        .expect("symmetric lower pattern");
    let (num, _) = solver
        .factorize(&sym, &k, Regularization::none())
        .expect("positive definite");
    let b: Vec<T> = (0..n)
        .map(|i| T::from_f64(f64::from(u32::try_from(i % 11).expect("small")) - 5.0))
        .collect();
    measure("ldl", &format!("{name}: {label} factorize"), n, 1, || {
        black_box(
            solver
                .factorize(&sym, black_box(&k), Regularization::none())
                .expect("positive definite"),
        );
    });
    measure("ldl", &format!("{name}: {label} solve"), n, 1, || {
        let mut x = b.clone();
        solver.solve(&num, black_box(&mut x));
        black_box(&x);
    });
    if label == "f64 naive" {
        eprintln!(
            "# {name}: n = {n}, {} supernodes, simplicial = {}",
            sym.n_supernodes(),
            sym.is_simplicial()
        );
    }
}

fn ldl_group() {
    let default = SupernodalOptions::default();
    // Dense blocks whatever the sparsity (the default switch keeps the 2-D
    // grids simplicial: no dense kernel is called there).
    let forced = SupernodalOptions {
        supernodal_switch: 0.0,
        ..default
    };
    let cases = [
        ("grid2d 80 (simplicial)", default, 6400, grid_lower(80)),
        (
            "grid2d 80 (forced supernodal)",
            forced,
            6400,
            grid_lower(80),
        ),
        ("grid3d 16", default, 4096, grid3_lower(16)),
        ("dense 200", default, 200, dense_lower(200)),
    ];
    for (name, opts, n, entries) in &cases {
        ldl_in::<f64, _>("f64 naive", name, NaiveDense, *opts, *n, entries);
        ldl_in::<Dd, _>("dd naive", name, NaiveDense, *opts, *n, entries);
        ldl_in::<Dd, _>("dd DdDense", name, DdDense, *opts, *n, entries);
    }
}

fn lu_group() {
    for (n, per_col) in [(800, 2), (2000, 3)] {
        lu_in::<f64>(&format!("f64 n={n}"), n, per_col, LuOptions::default());
        lu_in::<Dd>(&format!("dd n={n}"), n, per_col, LuOptions::for_scalar::<Dd>());
    }
}

fn main() {
    let group = std::env::args().nth(1).unwrap_or_default();
    match group.as_str() {
        "k8" => kernel_group(8),
        "k64" => kernel_group(64),
        "k1024" => kernel_group(1024),
        "k65536" => kernel_group(65_536),
        "lu" => lu_group(),
        "ldl" => ldl_group(),
        _ => {
            eprintln!("usage: dd_bench <k8|k64|k1024|k65536|lu|ldl>");
            std::process::exit(2);
        }
    }
}
