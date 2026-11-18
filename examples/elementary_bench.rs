//! Timing harness of the elementary functions of `DoubleDouble` (Chapter
//! 11.11, "Cost"; the table of L2.4 in the solver's benchmark log): the 23
//! functions, the four arithmetic operations for scale, and `f64::exp` as
//! the reference of the platform's libm.
//!
//! Each row is a throughput, not a latency: [`ARGS`] independent arguments
//! drawn once from the row's interval by a fixed generator, and one *round*
//! calls the operation on all of them (repeated, inside the round, as many
//! times as a calibration found necessary for a round of about
//! [`ROUND_NANOS`]). The table gives the median, the minimum and the
//! maximum over [`ROUNDS`] rounds, in nanoseconds per call. Results go
//! through [`std::hint::black_box`], so that no call is optimized away.
//!
//! The header records what the numbers depend on: the SIMD backend of the
//! build (`kernels::backend()`; the elementary functions themselves are
//! scalar), the target, and the load average of the machine at the start
//! (`/proc/loadavg`, else `sysctl -n vm.loadavg`, else "unknown"). Nothing
//! here asserts a speed.
//!
//! Run with `cargo run --release -p xingo-extprec --example
//! elementary_bench` (no feature is needed).

#![allow(clippy::cast_precision_loss)] // durations and counts far below 2^53

use std::hint::black_box;
use std::time::Instant;

use xingo_extprec::{DoubleDouble, kernels};

type Dd = DoubleDouble;

/// Independent arguments per row.
const ARGS: usize = 1024;
/// Rounds per row (the median, minimum and maximum are over these).
const ROUNDS: usize = 21;
/// Target duration of one round.
const ROUND_NANOS: u128 = 2_000_000;

/// Knuth's MMIX linear congruential generator: deterministic, no
/// dependency.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    /// Uniform in `[0, 1)`, 53 bits.
    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1_u64 << 53) as f64
    }

    /// A double-double uniform in `[a, b)`, with a full trailing component.
    fn uniform(&mut self, a: f64, b: f64) -> Dd {
        let h = a + (b - a) * self.unit();
        let ulp = f64::from_bits(h.abs().to_bits() + 1) - h.abs();
        Dd::from_sum(h, (self.unit() - 0.5) * ulp)
    }

    /// A double-double with magnitude log-uniform in `[2^e0, 2^e1)`: the
    /// leading component's exponent uniform, its significand uniform.
    fn log_uniform(&mut self, e0: i32, e1: i32) -> Dd {
        let span = u64::try_from(e1 - e0).expect("e1 > e0");
        let e = e0 + i32::try_from(self.next() % span).expect("small");
        self.uniform(1.0, 2.0).ldexp(e)
    }

    /// `ARGS` arguments from `sample`.
    fn args(&mut self, mut sample: impl FnMut(&mut Self) -> Dd) -> Vec<Dd> {
        (0..ARGS).map(|_| sample(self)).collect()
    }
}

/// The rows `[1e-3, 1e3]` draw from the binades `2^[-10, 10)`, that is
/// `[0.000 98, 1024)`, log-uniformly.
const BINADES_OF_1E3: i32 = 10;

/// One row: `name`, then median, min, max in ns per call.
struct Row {
    name: String,
    median: f64,
    min: f64,
    max: f64,
}

/// Times `pass` (one call per argument, `ARGS` calls) over `ROUNDS` rounds.
fn measure(name: &str, mut pass: impl FnMut()) -> Row {
    // Calibrate: double the repetitions until a round is long enough.
    let mut reps: u128 = 1;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            pass();
        }
        let dt = t.elapsed().as_nanos();
        if dt >= ROUND_NANOS / 2 || reps >= 1 << 24 {
            reps = (reps * ROUND_NANOS / dt.max(1)).max(1);
            break;
        }
        reps *= 2;
    }
    let mut per_call: Vec<f64> = (0..ROUNDS)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                pass();
            }
            t.elapsed().as_nanos() as f64 / (reps as f64 * ARGS as f64)
        })
        .collect();
    per_call.sort_by(f64::total_cmp);
    Row {
        name: name.to_string(),
        median: per_call[ROUNDS / 2],
        min: per_call[0],
        max: per_call[ROUNDS - 1],
    }
}

/// A function of one double-double over `xs`.
fn unary(name: &str, xs: &[Dd], f: impl Fn(Dd) -> Dd) -> Row {
    measure(name, || {
        for &x in xs {
            black_box(f(black_box(x)));
        }
    })
}

/// An operation on the pairs `(xs[i], ys[i])`.
fn binary(name: &str, xs: &[Dd], ys: &[Dd], f: impl Fn(Dd, Dd) -> Dd) -> Row {
    measure(name, || {
        for (&x, &y) in xs.iter().zip(ys) {
            black_box(f(black_box(x), black_box(y)));
        }
    })
}

/// The reference: the platform's `f64::exp` (libm), which the crate itself
/// never calls (Chapter 11.11, invariant F2).
#[allow(clippy::disallowed_methods)]
fn f64_exp(xs: &[f64]) -> Row {
    measure("`f64::exp` (libm, reference)", || {
        for &x in xs {
            black_box(black_box(x).exp());
        }
    })
}

/// The load average at the start, as the system reports it.
fn load_average() -> String {
    if let Ok(s) = std::fs::read_to_string("/proc/loadavg") {
        return s.split_whitespace().take(3).collect::<Vec<_>>().join(" ");
    }
    match std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
    {
        Ok(out) if out.status.success() => {
            let s = String::from_utf8_lossy(&out.stdout);
            let s = s
                .trim()
                .trim_start_matches('{')
                .trim_end_matches('}')
                .trim();
            if s.is_empty() {
                "unknown".to_string()
            } else {
                s.to_string()
            }
        }
        _ => "unknown".to_string(),
    }
}

fn main() {
    let load = load_average();
    let mut rng = Lcg(0x1124_2026);
    let mut rows = Vec::new();

    // The arithmetic, for scale: operands in [0.5, 2) (√ of the first).
    let a = rng.args(|r| r.uniform(0.5, 2.0));
    let b = rng.args(|r| r.uniform(0.5, 2.0));
    rows.push(binary("`dd · dd`", &a, &b, |x, y| x * y));
    rows.push(binary("`dd + dd`", &a, &b, |x, y| x + y));
    rows.push(binary("`dd / dd`", &a, &b, |x, y| x / y));
    rows.push(unary("`√dd`", &a, Dd::sqrt));

    // The intervals of the table of L2.4.
    let wide = rng.args(|r| r.uniform(-700.0, 700.0));
    let unit = rng.args(|r| r.uniform(-1.0, 1.0));
    let ten = rng.args(|r| r.uniform(-10.0, 10.0));
    let decades = rng.args(|r| r.log_uniform(-BINADES_OF_1E3, BINADES_OF_1E3));
    let near_one = rng.args(|r| r.uniform(0.5, 2.0));
    let huge = rng.args(|r| r.log_uniform(60, 960));
    let open_unit = rng.args(|r| r.uniform(-0.999_999, 0.999_999));
    let beyond_one = rng.args(|r| r.uniform(1.0, 1000.0));
    rows.push(unary("`exp [-700, 700]`", &wide, Dd::exp));
    rows.push(unary("`exp [-1, 1]`", &unit, Dd::exp));
    rows.push(unary("`exp2 [-700, 700]`", &wide, Dd::exp2));
    rows.push(unary("`exp_m1 [-1, 1]`", &unit, Dd::exp_m1));
    rows.push(unary("`ln [1e-3, 1e3]`", &decades, Dd::ln));
    rows.push(unary("`ln_1p [-1, 1]`", &open_unit, Dd::ln_1p));
    rows.push(unary("`log2 [1e-3, 1e3]`", &decades, Dd::log2));
    rows.push(unary("`log10 [1e-3, 1e3]`", &decades, Dd::log10));
    rows.push(unary("`powi(x, 7)`", &near_one, |x| x.powi(7)));
    rows.push(unary("`powi(x, -100)`", &near_one, |x| x.powi(-100)));
    let third = Dd::ONE / 3.0;
    rows.push(unary("`powf(x, 1/3)`", &decades, |x| x.powf(third)));
    rows.push(unary(
        "`sin [-1, 1] (no reduction below π/4)`",
        &unit,
        Dd::sin,
    ));
    rows.push(unary("`sin [-10, 10]`", &ten, Dd::sin));
    rows.push(unary("`cos [-10, 10]`", &ten, Dd::cos));
    rows.push(unary("`tan [-10, 10]`", &ten, Dd::tan));
    rows.push(unary("`sin_cos [-10, 10]`", &ten, |x| {
        let (s, c) = x.sin_cos();
        black_box(c);
        s
    }));
    rows.push(unary("`sin 2^[60, 960]`", &huge, Dd::sin));
    rows.push(unary("`atan [1e-3, 1e3]`", &decades, Dd::atan));
    rows.push(unary("`atan2(x, 1/3)`", &ten, |x| x.atan2(third)));
    rows.push(unary("`asin [-1, 1]`", &open_unit, Dd::asin));
    rows.push(unary("`acos [-1, 1]`", &open_unit, Dd::acos));
    rows.push(unary("`sinh [-1, 1]`", &unit, Dd::sinh));
    rows.push(unary("`cosh [-1, 1]`", &unit, Dd::cosh));
    rows.push(unary("`tanh [-1, 1]`", &unit, Dd::tanh));
    rows.push(unary("`asinh [-1, 1]`", &unit, Dd::asinh));
    rows.push(unary("`acosh [1, 1e3]`", &beyond_one, Dd::acosh));
    rows.push(unary("`atanh [-1, 1]`", &open_unit, Dd::atanh));
    let wide_f64: Vec<f64> = wide.iter().map(|x| x.hi()).collect();
    rows.push(f64_exp(&wide_f64));

    println!(
        "elementary_bench: backend {}, target {}-{}, load average at start: {load}",
        kernels::backend(),
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!(
        "ns per call, median / min / max of {ROUNDS} rounds over {ARGS} independent arguments (throughput)"
    );
    println!();
    println!("| operation | median ns | min | max |");
    println!("|---|---|---|---|");
    for r in rows {
        println!(
            "| {} | {:.1} | {:.1} | {:.1} |",
            r.name, r.median, r.min, r.max
        );
    }
}
