//! # Chapter 11.7 — `DdDense`: the dense kernel of `xingo-sparse` in double-double
//!
//! ## Purpose
//! `xingo_sparse::SupernodalLdl<T, K>` spends its time in the dense
//! operations of a `DenseKernel<T>` (Chapter 5.16 of `xingo-sparse`). For
//! `T = DoubleDouble` the only kernel so far was the generic
//! `NaiveDense`, one scalar double-double operation at a time. [`DdDense`]
//! is the same kernel with its inner loop — the column update of
//! Chapter 11.5, step B3 — on vector registers (Chapter 11.6). It returns
//! **the bits of `NaiveDense`**, so a factorization does not change when
//! the kernel is swapped; only its speed does.
//!
//! ## Notation
//! As in Chapter 5.16: a `DenseBlock` `A` is `rows × cols`, column-major;
//! square blocks are read and written in their lower triangle. `c ← c ⊖ a
//! ⊗ t` is the column update of Chapter 11.5 (`c`, `a` columns of equal
//! length, `t` a double-double), `⊘` the division D2 of Chapter 11.2.
//!
//! ## Algorithm K (the five operations)
//! K1. \[`gemm_sub`: `C ← C − A Bᵀ`.\] Step B3 of Chapter 11.5 with `t(j,
//!     l) = b_jl`, in its blocks: for each entry, `l` increasing.
//! K2. \[`syrk_sub`: `C ← C − A Aᵀ`, lower triangle.\] For `l` in blocks of
//!     `BLOCK_K`, for each column `j` of `C`: the column update of the
//!     rows `i ≥ j` with `a = A[j.., l]` and `t = a_jl`.
//! K3. \[`trsm_lower`, `B ← B L⁻ᵀ`.\] For `j = 0, …, n − 1`: for `l < j`,
//!     the column update of column `j` of `B` with `a` = column `l` of `B`
//!     (already solved) and `t = l_jl`; then `b_ij ← b_ij ⊘ l_jj`.
//! K4. \[`trsm_lower`, `B ← L⁻¹ B`.\] For each column `c` of `B`, for `l =
//!     0, …, n − 1`: `x ← b_lc ⊘ l_ll`, `b_lc ← x`, then the column update
//!     of the rows below `l` with `a = L[l + 1.., l]` and `t = x`.
//! K5. \[`potrf`, `sytrf`.\] Those of `NaiveDense`, unchanged: their inner
//!     loops are sequential sums along a *row* (`Σ_k l_ik l_jk`, summed
//!     left to right), which lanes would reorder.
//!
//! K1–K4 are the loops of `NaiveDense` (Chapter 5.16, "Loop order"), which
//! compute each entry as `x ← x − a · t` with the same operands in the
//! same order; the column update is that statement for a whole column, and
//! it is bit for bit the scalar one (invariant K1 of Chapter 11.5). The
//! product is `a ⊗ t` with the entry of the column first: the
//! double-double product M2 is not commutative in its last bits, so the
//! order of the operands is part of the definition.
//!
//! ## What the contract said
//! The approved contract describes `DdDense` as "`potrf`, `sytrf` and the
//! triangular solves are those of `NaiveDense`; `gemm_sub` and `syrk_sub`
//! go through `kernels::gemm_sub`". Two differences, neither visible in a
//! result: the triangular solves also use the vectorized column update
//! (their results are still `NaiveDense`'s, bit for bit), because the
//! supernodal factorization calls `trsm_lower` once per panel; and
//! `gemm_sub`/`syrk_sub` go through the column update of
//! `kernels::gemm_sub` rather than through that function, whose `B` is
//! `k × n` where the trait's is `n × k` (transposed).
//!
//! ## Invariants
//! (X1) *Same bits.* Every operation of [`DdDense`] leaves in its output
//!      block the bits `NaiveDense` would (NaN payloads aside, invariant K1
//!      of Chapter 11.5). Tested entry by entry on random blocks, and on
//!      whole factorizations in `tests/sparse_dd.rs`.
//! (X2) (D3) of Chapter 5.16: the strict upper triangle of a square block
//!      is untouched.
//!
//! ## References
//! Chapter 5.16 of `xingo-sparse` (`DenseKernel`, `NaiveDense`: the loop
//! order and the cache blocks); Chapters 11.5 and 11.6.

use xingo_sparse::{DenseBlock, DenseError, DenseKernel, Inertia, NaiveDense, Perm};

use crate::dd::DoubleDouble;
use crate::kernels::reference::Reference;
use crate::kernels::{BLOCK_K, ColumnUpdate, dispatch, gemm_blocked};

/// The dense kernel of `xingo-sparse` for double-double blocks, with the
/// results of `NaiveDense` bit for bit (Chapter 11.7): `gemm_sub`,
/// `syrk_sub` and the triangular solves run the column update of
/// [`crate::kernels::gemm_sub`] on SIMD registers (feature `simd`);
/// `potrf` and `sytrf` are those of `NaiveDense`. With it
/// `SupernodalLdl<DoubleDouble, DdDense>` spends its time in SIMD.
///
/// ```
/// use xingo_extprec::{DdDense, DoubleDouble};
/// use xingo_sparse::{DenseBlock, DenseKernel, NaiveDense};
/// let third = DoubleDouble::ONE / 3.0;
/// let a = DenseBlock { rows: 2, cols: 1, data: vec![third, DoubleDouble::ONE] };
/// let mut c = DenseBlock { rows: 2, cols: 2, data: vec![DoubleDouble::ONE; 4] };
/// let mut d = c.clone();
/// DdDense.syrk_sub(&a, &mut c);
/// NaiveDense.syrk_sub(&a, &mut d);
/// assert_eq!(c, d);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct DdDense;

/// `rows · cols` entries.
fn is_consistent(a: &DenseBlock<DoubleDouble>) -> bool {
    a.data.len() == a.rows * a.cols
}

/// The order of a consistent square block.
fn square(a: &DenseBlock<DoubleDouble>) -> Option<usize> {
    (is_consistent(a) && a.rows == a.cols).then_some(a.rows)
}

/// Step K2.
#[inline(always)] // into `pulp`'s `vectorize` (Chapter 11.6)
#[allow(clippy::inline_always)]
fn syrk<U: ColumnUpdate>(
    update: U,
    n: usize,
    a: &DenseBlock<DoubleDouble>,
    c: &mut DenseBlock<DoubleDouble>,
) {
    let kk = a.cols;
    let mut l0 = 0;
    while l0 < kk {
        let l1 = (l0 + BLOCK_K).min(kk);
        for j in 0..n {
            let cj = &mut c.data[j * n + j..(j + 1) * n];
            for l in l0..l1 {
                update.column_sub(cj, &a.data[l * n + j..(l + 1) * n], a.data[j + l * n]);
            }
        }
        l0 = l1;
    }
}

/// Step K3.
#[inline(always)] // into `pulp`'s `vectorize` (Chapter 11.6)
#[allow(clippy::inline_always)]
fn trsm_transposed<U: ColumnUpdate>(
    update: U,
    n: usize,
    l: &DenseBlock<DoubleDouble>,
    b: &mut DenseBlock<DoubleDouble>,
) {
    let rows = b.rows;
    for j in 0..n {
        let (done, rest) = b.data.split_at_mut(j * rows);
        let xj = &mut rest[..rows];
        for k in 0..j {
            update.column_sub(xj, &done[k * rows..(k + 1) * rows], l.get(j, k));
        }
        let d = l.get(j, j);
        for x in xj {
            *x /= d;
        }
    }
}

/// Step K4.
#[inline(always)] // into `pulp`'s `vectorize` (Chapter 11.6)
#[allow(clippy::inline_always)]
fn trsm_plain<U: ColumnUpdate>(
    update: U,
    n: usize,
    l: &DenseBlock<DoubleDouble>,
    b: &mut DenseBlock<DoubleDouble>,
) {
    for col in b.data.chunks_exact_mut(n.max(1)).take(b.cols) {
        for k in 0..n {
            let x = col[k] / l.get(k, k);
            col[k] = x;
            update.column_sub(&mut col[k + 1..], &l.data[k + 1 + k * n..(k + 1) * n], x);
        }
    }
}

impl DenseKernel<DoubleDouble> for DdDense {
    fn potrf(&self, a: &mut DenseBlock<DoubleDouble>) -> Result<(), DenseError> {
        NaiveDense.potrf(a)
    }

    fn sytrf(
        &self,
        a: &mut DenseBlock<DoubleDouble>,
        perm: &mut Perm,
        pivot_2x2: &mut Vec<bool>,
    ) -> Result<Inertia, DenseError> {
        NaiveDense.sytrf(a, perm, pivot_2x2)
    }

    fn trsm_lower(
        &self,
        l: &DenseBlock<DoubleDouble>,
        b: &mut DenseBlock<DoubleDouble>,
        transpose: bool,
    ) {
        let n = square(l).expect("trsm_lower: L must be square");
        assert!(is_consistent(b), "trsm_lower: B is inconsistent");
        if transpose {
            assert_eq!(b.cols, n, "trsm_lower: B must have n columns");
            dispatch!(|lanes| trsm_transposed(lanes, n, l, b));
            trsm_transposed(Reference, n, l, b);
        } else {
            assert_eq!(b.rows, n, "trsm_lower: B must have n rows");
            dispatch!(|lanes| trsm_plain(lanes, n, l, b));
            trsm_plain(Reference, n, l, b);
        }
    }

    fn gemm_sub(
        &self,
        a: &DenseBlock<DoubleDouble>,
        b: &DenseBlock<DoubleDouble>,
        c: &mut DenseBlock<DoubleDouble>,
    ) {
        assert!(
            is_consistent(a)
                && is_consistent(b)
                && is_consistent(c)
                && a.cols == b.cols
                && c.rows == a.rows
                && c.cols == b.rows,
            "gemm_sub: dimensions do not agree"
        );
        let (m, k, n) = (a.rows, a.cols, b.rows);
        dispatch!(
            |lanes| gemm_blocked(lanes, m, k, &a.data, &mut c.data, |j, l| b.data[j + l * n])
        );
        gemm_blocked(Reference, m, k, &a.data, &mut c.data, |j, l| {
            b.data[j + l * n]
        });
    }

    fn syrk_sub(&self, a: &DenseBlock<DoubleDouble>, c: &mut DenseBlock<DoubleDouble>) {
        let n = square(c).expect("syrk_sub: C must be square");
        assert!(
            is_consistent(a) && a.rows == n,
            "syrk_sub: dimensions do not agree"
        );
        dispatch!(|lanes| syrk(lanes, n, a, c));
        syrk(Reference, n, a, c);
    }

    fn name(&self) -> &'static str {
        "dd"
    }
}
