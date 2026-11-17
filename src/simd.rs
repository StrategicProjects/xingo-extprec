//! # Chapter 11.6 — SIMD kernels (`pulp`)
//!
//! ## Purpose
//! The kernels of Chapter 11.5 on vector registers, with **the same
//! results bit for bit** (invariant K1). Nothing here defines a kernel: the
//! definitions are Algorithms C and B of Chapter 11.5 and their scalar
//! reference; this chapter says how four of those scalar computations are
//! carried side by side, and why nothing changes when they are.
//!
//! ## Notation
//! As in Chapters 11.1, 11.2 and 11.5. `W = 4`. A *quad* is four `f64`
//! values `v = (v_0, v_1, v_2, v_3)`, its *lanes*; an operation on quads
//! (`+`, `−`, `·`, the fused `a b + c`, negation) is the `f64` operation of
//! the same name on each lane, correctly rounded — what NEON and AVX2 + FMA
//! provide. A *double-double quad* is a pair of quads `(h, ℓ)`. `2Sum`,
//! `Fast2Sum`, `2Prod` and the algorithms A2, M1, M2 of Chapter 11.2 **on
//! quads** are the same sequences of operations with every `f64` operation
//! replaced by the quad one, and **without step E**. A slice is cut into
//! *chunks* of four consecutive elements (element `4c + k` of the slice is
//! lane `k` of chunk `c`) and a *tail* of fewer than four.
//!
//! A quad is one 256-bit register on x86 (AVX2) and two 128-bit registers
//! on `aarch64` (NEON): `W = 4` on every target, the narrower one emulating
//! it (Chapter 11.5).
//!
//! ## Algorithm V (vectorized accumulation: `sum2`, `dot2`, `dot2_err`)
//! V1. \[Lanes.\] `p ← 0`, `s ← 0`, `e ← 0` (quads).
//! V2. \[Chunks.\] For each chunk `x` (and `y`): step C2 on quads — `sum2`:
//!     `(p, q) ← 2Sum(p, x)`, `t ← q + 0`; `dot2`: `(g, r) ← 2Prod(x, y)`,
//!     `(p, q) ← 2Sum(p, g)`, `t ← q + r` — then `s ← s + t` and, for
//!     `dot2_err` only, `e ← e + |t|`.
//! V3. \[Tail and result.\] Store the quads as the lanes `p_k`, `s_k`, `e_k`
//!     of Algorithm C, run step C2 on the tail (element `j` of the tail in
//!     lane `j`) and steps C3–C5, all in the scalar code of Chapter 11.5.
//!
//! Lane `k` of the quads goes through exactly the operations that the
//! reference applies to the elements `i ≡ k (mod 4)`, in the same order,
//! each correctly rounded: the lanes hold the same bits after V2, and V3
//! *is* the reference. (`sum2` and `dot2` do not compute `e`, which their
//! result does not use.)
//!
//! ## Algorithm W (vectorized double-double BLAS)
//! W1. \[`axpy`, the column update `c ← c ⊖ a ⊗ t`.\] For each chunk:
//!     compute `z ← y ⊕ (α ⊗ x)` (resp. `c ⊕ (−(a ⊗ t))`) on quads; if
//!     every lane of `z_h` and of `z_ℓ` is finite, store `z`; otherwise
//!     compute the four elements of the chunk with the scalar operations
//!     of Chapter 11.2 (from the operands, which were not yet
//!     overwritten). The tail is scalar.
//! W2. \[`dot`, `dot_mixed`, `dot4`.\] `a ← 0` (a double-double quad);
//!     for each chunk `a ← a ⊕ (x ⊗ y)` on quads. If some lane of `a_h` or
//!     `a_ℓ` is not finite, discard everything and run the reference kernel
//!     on the whole input. Otherwise store `a` as the four accumulators of
//!     step B1 (B4) and finish in scalar code: the tail, then `((a_0 ⊕ a_1)
//!     ⊕ a_2) ⊕ a_3` (B4: the fifth accumulator over the tail and `((a_0 ⊕
//!     a_1) ⊕ (a_2 ⊕ a_3)) ⊕ t`).
//!
//! **Why step E can be left out (the proof of K1 for Algorithm W).** The
//! scalar operations end in step E, which is the identity when the
//! leading and the trailing component of the result are both finite, and
//! which is the only place where the scalar code branches. Two facts about
//! A2, M1 and M2 *without* step E, for arbitrary `f64` operands:
//!
//! (N1) *Finite out, finite inside.* If `z_h` and `z_ℓ` are finite, every
//!      intermediate quantity was finite. (Every intermediate enters the
//!      result through additions, subtractions and products only; none of
//!      those maps a non-finite operand to a finite result.)
//! (N2) *Non-finite is absorbing.* If an operand has a non-finite
//!      component, or an intermediate quantity is not finite, then `z_h`
//!      is not finite. (In A2: a non-finite `s_h` makes `s_ℓ` a NaN, a
//!      non-finite `t_h` or `s_ℓ` makes `c`, then `v_h`, then `z_h`
//!      non-finite; an overflow of `v_h` gives `v_ℓ = −∞` or NaN and `z_h`
//!      a NaN. In M1 and M2: every term enters `c_ℓ3`, and `z_h = c_h +
//!      c_ℓ3`.)
//!
//! Now compare the quad computation of one chunk with the scalar one,
//! operation by operation. As long as the scalar code meets no non-finite
//! value, each of its step E is the identity and the two computations are
//! the same sequence of correctly rounded operations: the same bits. At
//! the first non-finite value of the scalar computation the quad
//! computation holds the same non-finite value, and by (N2), applied to
//! the operation where it occurs and to every later one that consumes its
//! result, the quad result of that lane is not finite: W1 then recomputes
//! the chunk, and W2 the whole kernel, *with the scalar code itself*.
//! Conversely, if the quad result is finite then by (N1) no non-finite
//! value occurred. In W2 "later operations" are the accumulations `a ← a ⊕
//! …` of the following chunks: a lane of `a` that became non-finite stays
//! so until the test. Hence the stored result is always the reference's —
//! NaN signs and payloads included in Algorithm W, since those results
//! come from the scalar code. In Algorithm V a NaN is produced by the
//! vector instruction, and only "NaN where the reference has a NaN" is
//! claimed (invariant K1 of Chapter 11.5).
//!
//! The argument does not use normalization (D1). The tests nevertheless
//! use valid double-doubles only: on other pairs the reference itself is
//! outside its contract (its `Fast2Sum` asserts its precondition in debug
//! builds).
//!
//! ## The vector arithmetic: [`Lanes`]
//! The algorithms are written once, over the trait [`Lanes`] — a quad type
//! and its operations — with two implementations on `pulp`'s safe
//! interface: [`X4`] (one four-lane register; `pulp::x86::V3`) and [`X2`]
//! (two two-lane registers; `pulp::aarch64::Neon`). The fused
//! multiply-add is `pulp`'s `mul_add_f64s`, which is `vfmaq_f64` on NEON
//! and `_mm256_fmadd_pd` on x86 — a true FMA on both, never a
//! multiplication followed by an addition (`mul_add_e_f64s` is the one
//! that may be either, and is not used). A double-double slice is read
//! and written through the accessors of [`DoubleDouble`], four elements at
//! a time: the crate has no `unsafe`, so there is no reinterpretation of
//! `&[DoubleDouble]` as `&[f64]`; the compiler turns the gathers into
//! shuffles.
//!
//! ## Invariants
//! (K1) of Chapter 11.5, tested there against the reference on random and
//!      exceptional inputs of every length `0..=67` and some large ones.
//! (L1) A [`Lanes`] operation is the correctly rounded `f64` operation on
//!      each lane; `mul_add` is fused. (The contract of the two
//!      implementations; checked lane by lane against `f64`, on the
//!      backend of the machine, by the test of this module.)
//! (L2) [`X4`] is built only on a four-lane backend and [`X2`] only on a
//!      two-lane one: checked at compile time by their constructors.
//!
//! ## What was run where
//! The NEON path is the one tested and measured (Apple M2). The AVX2 path
//! is the same generic code instantiated with [`X4`]; it has been
//! type-checked only as generic code and **has not been compiled for, or
//! run on, an x86 target** in this cycle — the tests of Chapter 11.5 run
//! it wherever `backend()` is `"avx2"`.
//!
//! ## References
//! Chapter 11.5 (the definitions) and Chapter 11.2 (A2, M1, M2, step E).
//! Ogita, Rump & Oishi (2005), §8 (they note that Sum2 and Dot2 use only
//! `+`, `−`, `·` and so vectorize). `pulp` 0.22.3: `Simd`, `WithSimd`,
//! `aarch64::Neon`, `x86::V3` (the API, read: `src/aarch64.rs`,
//! `src/x86/v3.rs`).

#![allow(clippy::float_cmp)]
// the finiteness test compares an exact zero
// `pulp` enables the target features (AVX2, FMA) for the code *inlined*
// into `vectorize`: a kernel that is not inlined there would be compiled
// without them. Every function of this module is on that path.
#![allow(clippy::inline_always)]

use pulp::Simd;

use crate::dd::DoubleDouble;
use crate::eft::two_prod;
use crate::kernels::{ColumnUpdate, Compensated, LANES, reference};

/// Four `f64` lanes and their arithmetic (invariant L1).
pub(crate) trait Lanes: Copy {
    /// A quad.
    type V: Copy;
    /// `(x, x, x, x)`.
    fn splat(self, x: f64) -> Self::V;
    /// The quad with lanes `a`.
    fn load(self, a: [f64; LANES]) -> Self::V;
    /// The lanes of `v`.
    fn store(self, v: Self::V) -> [f64; LANES];
    /// `RN(a + b)` on each lane.
    fn add(self, a: Self::V, b: Self::V) -> Self::V;
    /// `RN(a − b)` on each lane.
    fn sub(self, a: Self::V, b: Self::V) -> Self::V;
    /// `RN(a b)` on each lane.
    fn mul(self, a: Self::V, b: Self::V) -> Self::V;
    /// `RN(a b + c)` on each lane, fused.
    fn mul_add(self, a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    /// `−a` on each lane (exact).
    fn neg(self, a: Self::V) -> Self::V;
    /// `|a|` on each lane (exact).
    fn abs(self, a: Self::V) -> Self::V;
    /// Every lane of `a` and of `b` is finite.
    fn all_finite(self, a: Self::V, b: Self::V) -> bool;
}

/// A quad in one four-lane register of the backend `S` (AVX2).
#[derive(Clone, Copy, Debug)]
pub(crate) struct X4<S: Simd>(S);

impl<S: Simd> X4<S> {
    /// Invariant (L2): `S` has four `f64` lanes.
    #[inline(always)]
    #[cfg_attr(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        allow(dead_code)
    )]
    pub(crate) fn new(simd: S) -> Self {
        const { assert!(S::F64_LANES == LANES) };
        Self(simd)
    }
}

impl<S: Simd> Lanes for X4<S> {
    type V = S::f64s;

    #[inline(always)]
    fn splat(self, x: f64) -> Self::V {
        self.0.splat_f64s(x)
    }
    #[inline(always)]
    fn load(self, a: [f64; LANES]) -> Self::V {
        pulp::cast(a)
    }
    #[inline(always)]
    fn store(self, v: Self::V) -> [f64; LANES] {
        pulp::cast(v)
    }
    #[inline(always)]
    fn add(self, a: Self::V, b: Self::V) -> Self::V {
        self.0.add_f64s(a, b)
    }
    #[inline(always)]
    fn sub(self, a: Self::V, b: Self::V) -> Self::V {
        self.0.sub_f64s(a, b)
    }
    #[inline(always)]
    fn mul(self, a: Self::V, b: Self::V) -> Self::V {
        self.0.mul_f64s(a, b)
    }
    #[inline(always)]
    fn mul_add(self, a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        self.0.mul_add_f64s(a, b, c)
    }
    #[inline(always)]
    fn neg(self, a: Self::V) -> Self::V {
        self.0.neg_f64s(a)
    }
    #[inline(always)]
    fn abs(self, a: Self::V) -> Self::V {
        self.0.abs_f64s(a)
    }
    #[inline(always)]
    fn all_finite(self, a: Self::V, b: Self::V) -> bool {
        // x − x is +0 for a finite x and NaN otherwise.
        let s = self.0;
        let z = s.add_f64s(s.sub_f64s(a, a), s.sub_f64s(b, b));
        s.reduce_sum_f64s(z) == 0.0
    }
}

/// A quad in two two-lane registers of the backend `S` (NEON): lanes 0–1
/// and lanes 2–3.
#[derive(Clone, Copy, Debug)]
pub(crate) struct X2<S: Simd>(S);

impl<S: Simd> X2<S> {
    /// Invariant (L2): `S` has two `f64` lanes.
    #[inline(always)]
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn new(simd: S) -> Self {
        const { assert!(2 * S::F64_LANES == LANES) };
        Self(simd)
    }
}

impl<S: Simd> Lanes for X2<S> {
    type V = [S::f64s; 2];

    #[inline(always)]
    fn splat(self, x: f64) -> Self::V {
        [self.0.splat_f64s(x); 2]
    }
    #[inline(always)]
    fn load(self, a: [f64; LANES]) -> Self::V {
        [pulp::cast([a[0], a[1]]), pulp::cast([a[2], a[3]])]
    }
    #[inline(always)]
    fn store(self, v: Self::V) -> [f64; LANES] {
        let (lo, hi): ([f64; 2], [f64; 2]) = (pulp::cast(v[0]), pulp::cast(v[1]));
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[inline(always)]
    fn add(self, a: Self::V, b: Self::V) -> Self::V {
        [self.0.add_f64s(a[0], b[0]), self.0.add_f64s(a[1], b[1])]
    }
    #[inline(always)]
    fn sub(self, a: Self::V, b: Self::V) -> Self::V {
        [self.0.sub_f64s(a[0], b[0]), self.0.sub_f64s(a[1], b[1])]
    }
    #[inline(always)]
    fn mul(self, a: Self::V, b: Self::V) -> Self::V {
        [self.0.mul_f64s(a[0], b[0]), self.0.mul_f64s(a[1], b[1])]
    }
    #[inline(always)]
    fn mul_add(self, a: Self::V, b: Self::V, c: Self::V) -> Self::V {
        [
            self.0.mul_add_f64s(a[0], b[0], c[0]),
            self.0.mul_add_f64s(a[1], b[1], c[1]),
        ]
    }
    #[inline(always)]
    fn neg(self, a: Self::V) -> Self::V {
        [self.0.neg_f64s(a[0]), self.0.neg_f64s(a[1])]
    }
    #[inline(always)]
    fn abs(self, a: Self::V) -> Self::V {
        [self.0.abs_f64s(a[0]), self.0.abs_f64s(a[1])]
    }
    #[inline(always)]
    fn all_finite(self, a: Self::V, b: Self::V) -> bool {
        // x − x is +0 for a finite x and NaN otherwise.
        let s = self.0;
        let z0 = s.add_f64s(s.sub_f64s(a[0], a[0]), s.sub_f64s(b[0], b[0]));
        let z1 = s.add_f64s(s.sub_f64s(a[1], a[1]), s.sub_f64s(b[1], b[1]));
        s.reduce_sum_f64s(s.add_f64s(z0, z1)) == 0.0
    }
}

/// A double-double quad `(h, ℓ)`.
type Dq<L> = (<L as Lanes>::V, <L as Lanes>::V);

/// Algorithm S of Chapter 11.1 on quads.
#[inline(always)]
fn two_sum_q<L: Lanes>(l: L, a: L::V, b: L::V) -> (L::V, L::V) {
    let s = l.add(a, b);
    let a1 = l.sub(s, b);
    let b1 = l.sub(s, a1);
    let da = l.sub(a, a1);
    let db = l.sub(b, b1);
    (s, l.add(da, db))
}

/// Algorithm F of Chapter 11.1 on quads.
#[inline(always)]
fn fast_two_sum_q<L: Lanes>(l: L, a: L::V, b: L::V) -> (L::V, L::V) {
    let s = l.add(a, b);
    let z = l.sub(s, a);
    (s, l.sub(b, z))
}

/// Algorithm P of Chapter 11.1 on quads.
#[inline(always)]
fn two_prod_q<L: Lanes>(l: L, a: L::V, b: L::V) -> (L::V, L::V) {
    let p = l.mul(a, b);
    (p, l.mul_add(a, b, l.neg(p)))
}

/// Algorithm A2 of Chapter 11.2 on quads, without step E.
#[inline(always)]
fn add_q<L: Lanes>(l: L, x: Dq<L>, y: Dq<L>) -> Dq<L> {
    let (sh, sl) = two_sum_q(l, x.0, y.0);
    let (th, tl) = two_sum_q(l, x.1, y.1);
    let c = l.add(sl, th);
    let (vh, vl) = fast_two_sum_q(l, sh, c);
    let w = l.add(tl, vl);
    fast_two_sum_q(l, vh, w)
}

/// Algorithm M2 of Chapter 11.2 on quads, without step E.
#[inline(always)]
fn mul_q<L: Lanes>(l: L, x: Dq<L>, y: Dq<L>) -> Dq<L> {
    let (ch, cl1) = two_prod_q(l, x.0, y.0);
    let tl0 = l.mul(x.1, y.1);
    let tl1 = l.mul_add(x.0, y.1, tl0);
    let cl2 = l.mul_add(x.1, y.0, tl1);
    let cl3 = l.add(cl1, cl2);
    fast_two_sum_q(l, ch, cl3)
}

/// Algorithm M1 of Chapter 11.2 on quads, without step E.
#[inline(always)]
fn mul_f64_q<L: Lanes>(l: L, x: Dq<L>, y: L::V) -> Dq<L> {
    let (ch, cl1) = two_prod_q(l, x.0, y);
    let cl3 = l.mul_add(x.1, y, cl1);
    fast_two_sum_q(l, ch, cl3)
}

/// The double-double quad of a chunk.
#[inline(always)]
fn load_dd<L: Lanes>(l: L, c: &[DoubleDouble; LANES]) -> Dq<L> {
    (
        l.load([c[0].hi(), c[1].hi(), c[2].hi(), c[3].hi()]),
        l.load([c[0].lo(), c[1].lo(), c[2].lo(), c[3].lo()]),
    )
}

/// The four double-doubles of a quad (finite, hence the pairs the scalar
/// operations return: no invariant is re-checked).
#[inline(always)]
fn store_dd<L: Lanes>(l: L, z: Dq<L>) -> [DoubleDouble; LANES] {
    let (h, lo) = (l.store(z.0), l.store(z.1));
    [
        DoubleDouble::from_raw(h[0], lo[0]),
        DoubleDouble::from_raw(h[1], lo[1]),
        DoubleDouble::from_raw(h[2], lo[2]),
        DoubleDouble::from_raw(h[3], lo[3]),
    ]
}

/// The double-double quad `(x, x, x, x)`.
#[inline(always)]
fn splat_dd<L: Lanes>(l: L, x: DoubleDouble) -> Dq<L> {
    (l.splat(x.hi()), l.splat(x.lo()))
}

/// Algorithm V for `sum2`: the lanes after steps V1–V3 (C1–C2).
#[inline(always)]
pub(crate) fn sum2<L: Lanes>(l: L, x: &[f64]) -> Compensated {
    let (chunks, tail) = x.as_chunks::<LANES>();
    let zero = l.splat(0.0);
    let (mut p, mut s) = (zero, zero);
    for c in chunks {
        let (pn, q) = two_sum_q(l, p, l.load(*c));
        p = pn;
        s = l.add(s, l.add(q, zero));
    }
    let mut acc = Compensated {
        p: l.store(p),
        s: l.store(s),
        e: [0.0; LANES],
    };
    for (j, &xi) in tail.iter().enumerate() {
        acc.push(j, xi, 0.0);
    }
    acc
}

/// Algorithm V for `dot2` (`BOUND = false`: the lanes `e` are not
/// computed) and `dot2_err` (`BOUND = true`).
#[inline(always)]
pub(crate) fn dot2<L: Lanes, const BOUND: bool>(l: L, x: &[f64], y: &[f64]) -> Compensated {
    debug_assert_eq!(x.len(), y.len(), "dot2: the lengths differ");
    let (cx, tx) = x.as_chunks::<LANES>();
    let (cy, ty) = y.as_chunks::<LANES>();
    let zero = l.splat(0.0);
    let (mut p, mut s, mut e) = (zero, zero, zero);
    for (a, b) in cx.iter().zip(cy) {
        let (g, r) = two_prod_q(l, l.load(*a), l.load(*b));
        let (pn, q) = two_sum_q(l, p, g);
        let t = l.add(q, r);
        p = pn;
        s = l.add(s, t);
        if BOUND {
            e = l.add(e, l.abs(t));
        }
    }
    let mut acc = Compensated {
        p: l.store(p),
        s: l.store(s),
        e: l.store(e),
    };
    for (j, (&xi, &yi)) in tx.iter().zip(ty).enumerate() {
        let (h, r) = two_prod(xi, yi);
        acc.push(j, h, r);
    }
    acc
}

/// Step W2 for `dot`.
#[inline(always)]
pub(crate) fn dot<L: Lanes>(l: L, x: &[DoubleDouble], y: &[DoubleDouble]) -> DoubleDouble {
    debug_assert_eq!(x.len(), y.len(), "dot: the lengths differ");
    let (cx, tx) = x.as_chunks::<LANES>();
    let (cy, ty) = y.as_chunks::<LANES>();
    let mut a = splat_dd(l, DoubleDouble::ZERO);
    for (p, q) in cx.iter().zip(cy) {
        a = add_q(l, a, mul_q(l, load_dd(l, p), load_dd(l, q)));
    }
    if !l.all_finite(a.0, a.1) {
        return reference::dot(x, y);
    }
    let mut acc = store_dd(l, a);
    for (j, (&xi, &yi)) in tx.iter().zip(ty).enumerate() {
        acc[j] += xi * yi;
    }
    ((acc[0] + acc[1]) + acc[2]) + acc[3]
}

/// Step W2 for `dot4` (step B4 of Chapter 11.5): the chunks on quads, the
/// tail and the pairwise combination in scalar code, as the reference.
#[inline(always)]
#[cfg_attr(not(feature = "sparse"), allow(dead_code))]
pub(crate) fn dot4<L: Lanes>(l: L, x: &[DoubleDouble], y: &[DoubleDouble]) -> DoubleDouble {
    debug_assert_eq!(x.len(), y.len(), "dot4: the lengths differ");
    let (cx, tx) = x.as_chunks::<LANES>();
    let (cy, ty) = y.as_chunks::<LANES>();
    let mut a = splat_dd(l, DoubleDouble::ZERO);
    for (p, q) in cx.iter().zip(cy) {
        a = add_q(l, a, mul_q(l, load_dd(l, p), load_dd(l, q)));
    }
    if !l.all_finite(a.0, a.1) {
        return reference::dot4(x, y);
    }
    let acc = store_dd(l, a);
    let mut t = DoubleDouble::ZERO;
    for (&p, &q) in tx.iter().zip(ty) {
        t += p * q;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + t
}

/// Step W2 for `dot_mixed`.
#[inline(always)]
pub(crate) fn dot_mixed<L: Lanes>(l: L, x: &[DoubleDouble], y: &[f64]) -> DoubleDouble {
    debug_assert_eq!(x.len(), y.len(), "dot_mixed: the lengths differ");
    let (cx, tx) = x.as_chunks::<LANES>();
    let (cy, ty) = y.as_chunks::<LANES>();
    let mut a = splat_dd(l, DoubleDouble::ZERO);
    for (p, q) in cx.iter().zip(cy) {
        a = add_q(l, a, mul_f64_q(l, load_dd(l, p), l.load(*q)));
    }
    if !l.all_finite(a.0, a.1) {
        return reference::dot_mixed(x, y);
    }
    let mut acc = store_dd(l, a);
    for (j, (&xi, &yi)) in tx.iter().zip(ty).enumerate() {
        acc[j] += xi * yi;
    }
    ((acc[0] + acc[1]) + acc[2]) + acc[3]
}

/// Step W1 for `axpy`.
#[inline(always)]
pub(crate) fn axpy<L: Lanes>(
    l: L,
    alpha: DoubleDouble,
    x: &[DoubleDouble],
    y: &mut [DoubleDouble],
) {
    debug_assert_eq!(x.len(), y.len(), "axpy: the lengths differ");
    let (cx, tx) = x.as_chunks::<LANES>();
    let (cy, ty) = y.as_chunks_mut::<LANES>();
    let al = splat_dd(l, alpha);
    for (p, q) in cx.iter().zip(cy) {
        let z = add_q(l, load_dd(l, q), mul_q(l, al, load_dd(l, p)));
        if l.all_finite(z.0, z.1) {
            *q = store_dd(l, z);
        } else {
            reference::axpy(alpha, p, q);
        }
    }
    reference::axpy(alpha, tx, ty);
}

/// Step W1 for the column update of `gemm_sub` (step B3 of Chapter 11.5).
impl<L: Lanes> ColumnUpdate for L {
    #[inline(always)]
    fn column_sub(self, c: &mut [DoubleDouble], a: &[DoubleDouble], t: DoubleDouble) {
        debug_assert_eq!(c.len(), a.len(), "column_sub: the lengths differ");
        let l = self;
        let (cc, tc) = c.as_chunks_mut::<LANES>();
        let (ca, ta) = a.as_chunks::<LANES>();
        let tq = splat_dd(l, t);
        for (q, p) in cc.iter_mut().zip(ca) {
            let prod = mul_q(l, load_dd(l, p), tq);
            let z = add_q(l, load_dd(l, q), (l.neg(prod.0), l.neg(prod.1)));
            if l.all_finite(z.0, z.1) {
                *q = store_dd(l, z);
            } else {
                reference::Reference.column_sub(q, p, t);
            }
        }
        reference::Reference.column_sub(tc, ta, t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Invariant (L1) for a backend: every operation, lane by lane, against
    /// `f64`.
    fn lanes_follow_f64<L: Lanes>(l: L) {
        let a = [1.0 + f64::EPSILON, -3.5e200, 1e-310, f64::INFINITY];
        let b = [1.0 - f64::EPSILON / 2.0, 2.25e150, -7.0, 1.0];
        let c = [-1.0, f64::MAX, 3e-320, f64::NAN];
        let (va, vb, vc) = (l.load(a), l.load(b), l.load(c));
        let same = |v: L::V, f: &dyn Fn(usize) -> f64| {
            for (k, got) in l.store(v).into_iter().enumerate() {
                let want = f(k);
                assert!(
                    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                    "lane {k}: {got:e} != {want:e}"
                );
            }
        };
        same(va, &|k| a[k]);
        same(l.splat(0.1), &|_| 0.1);
        same(l.add(va, vb), &|k| a[k] + b[k]);
        same(l.sub(va, vb), &|k| a[k] - b[k]);
        same(l.mul(va, vb), &|k| a[k] * b[k]);
        same(l.mul_add(va, vb, vc), &|k| a[k].mul_add(b[k], c[k]));
        same(l.neg(vc), &|k| -c[k]);
        same(l.abs(vb), &|k| b[k].abs());
        // The FMA is fused: (1 + ε)(1 − ε/2) − 1 has a term ε²/2 that a
        // rounded product loses.
        let fused = l.store(l.mul_add(va, vb, l.splat(-1.0)))[0];
        assert_eq!(fused, a[0].mul_add(b[0], -1.0));
        assert_ne!(fused, a[0] * b[0] - 1.0);
        assert!(l.all_finite(l.splat(1.0), l.load([0.0, -0.0, f64::MAX, 5e-324])));
        for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            for k in 0..LANES {
                let mut v = [1.0; LANES];
                v[k] = bad;
                assert!(!l.all_finite(l.load(v), l.splat(0.0)));
                assert!(!l.all_finite(l.splat(0.0), l.load(v)));
            }
        }
    }

    #[test]
    fn the_native_backend_follows_f64() {
        #[cfg(target_arch = "aarch64")]
        if let Some(token) = pulp::aarch64::Neon::try_new() {
            token.vectorize(|| lanes_follow_f64(X2::new(token)));
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(token) = pulp::x86::V3::try_new() {
            token.vectorize(|| lanes_follow_f64(X4::new(token)));
        }
    }
}
