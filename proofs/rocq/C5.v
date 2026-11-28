(** * Step C5 of [dot2_err] (Chapter 11.5): the evaluation of the bound β

    Statement ([C5_binary64]). For an integer [n] with [1 <= n] and
    [n + 3 < 2^52] (the [MAX_TERMS] test of [result_with_bound]) and a real
    [e >= 0] (the accumulated magnitude of step C3), the value [β] computed
    in binary64 by [result_with_bound] (lines "C5." of [src/kernels.rs]):

      a     = RN(RN(n + 3) · u)          (terms as f64 * U)
      den   = RN(1 - RN(2 a))            (1.0 - 2.0 * a)
      δ     = RN(a / den)
      β     = RN(RN(RN(δ e) + 3η/u) · (1 + 8u))   ((delta * e + UNDERFLOW) * INFLATE)

    with [u = 2^-53], [η = 2^-1074], [3η/u = 6 · 2^-1022] and [1 + 8u = 1 +
    2^-50], satisfies

      (n + 3) u / (1 - (2n + 5) u) · e + n η / 2  <=  β,

    the bound that the proof of C5 in Chapter 11.5 requires. Moreover [a],
    [den] and [δ] are at most [2^52] (they never overflow).

    Not covered here: the overflow of [RN(δ e)], of the sum or of the
    product. In IEEE 754 an overflowing operation with positive operands
    returns [+∞], and [+∞ + 3η/u] and [+∞ · (1 + 8u)] are [+∞], so [β = +∞]
    and the inequality still holds; that argument is on paper. When no
    operation overflows the IEEE results are the [RN] values of this file
    ([Model.v]).

    Method: Flocq's relative error bounds for [RN] ([relative_error_N_FLT]
    for normal results, [error_N_FLT] for [RN(δ e)], which may be
    subnormal), exactness of [a], [2a] and [1 - 2a] by exhibiting them as
    integers times powers of two, then real arithmetic ([nra]). *)

From Stdlib Require Import Reals Psatz ZArith.
From Flocq Require Import Core Relative.
From XP Require Import Model.

Open Scope R_scope.

(** The constants of [result_with_bound]: [UNDERFLOW = 6.0 *
    f64::MIN_POSITIVE] and [INFLATE = 1.0 + 4.0 * f64::EPSILON], both exact
    in binary64. *)
Definition UNDERFLOW := 6 * pow2 (-1022).
Definition INFLATE := 1 + pow2 (-50).

(** The computation of [result_with_bound] after its tests. *)
Definition c5_steps (n : Z) (e : R) :=
  let terms := RN (IZR (n + 3)) in                   (* terms as f64 *)
  let a := RN (terms * pow2 (-53)) in                (* * U *)
  let den := RN (1 - RN (2 * a)) in                  (* 1.0 - 2.0 * a *)
  let delta := RN (a / den) in
  let beta := RN (RN (RN (delta * e) + UNDERFLOW) * INFLATE) in
  (a, den, delta, beta).

(** The bound of the proof of C5. *)
Definition c5_bound (n : Z) (e : R) :=
  IZR (n + 3) * pow2 (-53) / (1 - IZR (2 * n + 5) * pow2 (-53)) * e
  + IZR n * pow2 (-1074) / 2.

Lemma fmt_int : forall z f,
  (Z.abs z < 9007199254740992)%Z -> (-1074 <= f)%Z -> fmt (IZR z * pow2 f).
Proof.
intros z f Hz Hf. apply generic_format_FLT.
apply (FLT_spec _ _ _ _ (Float radix2 z f)); [reflexivity| |exact Hf].
simpl. exact Hz.
Qed.

Lemma RN_rel : forall y, pow2 (-1022) <= Rabs y ->
  Rabs (RN y - y) <= pow2 (-53) * Rabs y.
Proof.
intros y Hy.
pose proof (relative_error_N_FLT radix2 (-1074) 53 eq_refl (fun z => negb (Z.even z)) y Hy) as H.
replace (pow2 (-53)) with (/ 2 * pow2 (-53 + 1)).
- exact H.
- replace (-53 + 1)%Z with (1 + -53)%Z by ring. rewrite pow2_plus, pow2_1. field.
Qed.

(** The final inequality, as real arithmetic only. *)
Lemma c5_combine : forall u E K q d e p1 p2 beta Nr,
  0 < u -> 1024 * u <= 1 -> 0 < E -> 0 <= e -> 0 <= K -> K <= q ->
  (1 - u) * q <= d ->
  (1 - u) * (d * e) - u * E <= p1 ->
  (1 - u) * (p1 + 6 * E) <= p2 ->
  (1 - u) * (p2 * (1 + 8 * u)) <= beta ->
  Nr * u <= / 2 -> 0 <= Nr - 3 ->
  K * e + (Nr - 3) * (2 * u * E) / 2 <= beta.
Proof.
intros u E K q d e p1 p2 beta Nr Hu0 Hu10 HE0 He HK0 HKq Hd Hp1 Hp2 Hb HNu HN3.
set (M := (1 - u) * (1 + 8 * u)).
assert (HM : 0 <= M) by (unfold M; apply Rmult_le_pos; lra).
assert (S1 : M * p2 <= beta) by (unfold M; lra).
assert (S2 : M * ((1 - u) * (p1 + 6 * E)) <= beta).
{ apply Rle_trans with (M * p2); [apply Rmult_le_compat_l; lra|exact S1]. }
set (X := K * e).
assert (HX : 0 <= X) by (unfold X; apply Rmult_le_pos; lra).
assert (S3 : X <= q * e) by (unfold X; apply Rmult_le_compat_r; lra).
assert (S4 : (1 - u) * X <= d * e).
{ apply Rle_trans with ((1 - u) * (q * e)).
  - apply Rmult_le_compat_l; lra.
  - rewrite <- Rmult_assoc. apply Rmult_le_compat_r; lra. }
assert (S5 : (1 - u) * ((1 - u) * X) <= (1 - u) * (d * e)) by (apply Rmult_le_compat_l; lra).
set (Y := (1 - u) * ((1 - u) * X) + (6 - u) * E).
assert (S6 : Y <= p1 + 6 * E) by (unfold Y; lra).
assert (S7 : M * ((1 - u) * Y) <= beta).
{ apply Rle_trans with (M * ((1 - u) * (p1 + 6 * E))); [|exact S2].
  apply Rmult_le_compat_l; [exact HM|]. apply Rmult_le_compat_l; lra. }
assert (G1 : 1 <= M * ((1 - u) * ((1 - u) * (1 - u)))).
{ unfold M. clear - Hu0 Hu10. nra. }
assert (G2 : 5 <= M * ((1 - u) * (6 - u))).
{ unfold M. clear - Hu0 Hu10. nra. }
assert (G3 : X <= M * ((1 - u) * ((1 - u) * (1 - u))) * X).
{ rewrite <- (Rmult_1_l X) at 1. apply Rmult_le_compat_r; assumption. }
assert (G4 : 5 * E <= M * ((1 - u) * (6 - u)) * E) by (apply Rmult_le_compat_r; lra).
assert (G5 : (Nr - 3) * (2 * u * E) / 2 <= 5 * E).
{ assert (Nr * u * E <= / 2 * E) by (apply Rmult_le_compat_r; lra).
  assert (0 <= u * E) by (apply Rmult_le_pos; lra). nra. }
assert (EQ : M * ((1 - u) * Y)
             = M * ((1 - u) * ((1 - u) * (1 - u))) * X + M * ((1 - u) * (6 - u)) * E)
  by (unfold Y; ring).
lra.
Qed.

Theorem C5_binary64 : forall (n : Z) (e : R),
  (1 <= n)%Z -> (n + 3 < 4503599627370496)%Z -> 0 <= e ->
  let '(a, den, delta, beta) := c5_steps n e in
  c5_bound n e <= beta /\
  0 < a <= 1 /\ 0 < den <= 1 /\ 0 < delta <= pow2 52.
Proof.
intros n e Hn1 Hn2 He. unfold c5_steps.
set (u := pow2 (-53)).
assert (Hu0 : 0 < u) by apply pow2_pos.
assert (Hu10 : 1024 * u <= 1).
{ unfold u. replace 1024 with (pow2 10) by reflexivity. rewrite <- pow2_plus.
  rewrite <- pow2_0. apply pow2_le. lia. }
set (E := pow2 (-1022)).
assert (HE0 : 0 < E) by apply pow2_pos.
assert (HEu : E <= u) by (apply pow2_le; lia).
assert (H52 : pow2 (-52) = 2 * u).
{ unfold u. replace 2 with (pow2 1) at 1 by reflexivity. rewrite <- pow2_plus. reflexivity. }
assert (H50 : pow2 (-50) = 8 * u).
{ unfold u. replace 8 with (pow2 3) by reflexivity. rewrite <- pow2_plus. reflexivity. }
assert (Heta : pow2 (-1074) = 2 * u * E).
{ unfold u, E. replace 2 with (pow2 1) at 1 by reflexivity. rewrite <- !pow2_plus. reflexivity. }
(* The two constants enter only through these inequalities: the proof needs
   UNDERFLOW >= 3 eta / u and INFLATE >= 1 + 8u. *)
assert (HUF : 6 * E <= UNDERFLOW <= 8 * E) by (unfold UNDERFLOW; fold E; lra).
assert (HIN : 1 + 8 * u <= INFLATE <= 2) by (unfold INFLATE; rewrite ?H50; lra).
set (UF := UNDERFLOW) in *. set (I := INFLATE) in *.
assert (HP52 : pow2 52 * (2 * u) = 1).
{ rewrite <- H52, <- pow2_plus. reflexivity. }
(* the integer n + 3 *)
set (Nr := IZR (n + 3)).
assert (HN4 : 4 <= Nr) by (unfold Nr; apply IZR_le; lia).
assert (HNmax : Nr + 1 <= pow2 52).
{ replace (pow2 52) with (IZR 4503599627370496) by reflexivity.
  unfold Nr. rewrite <- plus_IZR. apply IZR_le. lia. }
assert (Hterms : RN Nr = Nr).
{ apply round_generic; [apply valid_rnd_N|].
  replace Nr with (IZR (n + 3) * pow2 0) by (rewrite pow2_0; unfold Nr; ring).
  apply fmt_int; lia. }
rewrite Hterms.
(* a = Nr u, 2a = Nr 2^-52 and 1 - 2a are exact *)
assert (Ha : RN (Nr * u) = Nr * u).
{ apply round_generic; [apply valid_rnd_N|]. apply fmt_int; lia. }
rewrite Ha.
assert (H2a : RN (2 * (Nr * u)) = Nr * (2 * u)).
{ replace (2 * (Nr * u)) with (IZR (n + 3) * pow2 (-52)) by (rewrite H52; unfold Nr; ring).
  rewrite <- H52. apply round_generic; [apply valid_rnd_N|]. apply fmt_int; lia. }
rewrite H2a.
assert (Hden : RN (1 - Nr * (2 * u)) = 1 - Nr * (2 * u)).
{ replace (1 - Nr * (2 * u)) with (IZR (4503599627370496 - (n + 3)) * pow2 (-52)).
  - apply round_generic; [apply valid_rnd_N|]. apply fmt_int; lia.
  - rewrite minus_IZR. fold Nr. rewrite H52.
    replace (IZR 4503599627370496) with (pow2 52) by reflexivity.
    transitivity (pow2 52 * (2 * u) - Nr * (2 * u)); [ring|rewrite HP52; ring]. }
rewrite Hden.
set (den := 1 - Nr * (2 * u)).
assert (Hden2 : 2 * u <= den) by (unfold den; nra).
assert (Hden1 : den <= 1) by (unfold den; nra).
(* delta: one rounding of a normal quotient *)
set (q := Nr * u / den).
assert (Hq0 : 0 < q) by (unfold q; apply Rdiv_lt_0_compat; nra).
assert (Hqden : q * den = Nr * u) by (unfold q; field; lra).
assert (Hq4 : 4 * u <= q) by nra.
assert (Hqmax : q <= pow2 51).
{ assert (Hp : pow2 52 = 2 * pow2 51) by (replace 2 with (pow2 1) by reflexivity; rewrite <- pow2_plus; reflexivity).
  assert (H1 : q * (2 * u) <= q * den) by (apply Rmult_le_compat_l; lra).
  assert (H2 : 2 * q * u <= Nr * u) by lra.
  assert (H3 : 2 * q <= Nr) by (apply Rmult_le_reg_r with u; lra).
  lra. }
set (d := RN q).
assert (Hd : Rabs (d - q) <= u * q).
{ rewrite <- (Rabs_pos_eq q) at 2 by lra. apply RN_rel. rewrite Rabs_pos_eq by lra. fold E. lra. }
apply Rabs_le_inv in Hd.
(* p1 = RN(d e): relative error u and absolute error eta/2 *)
destruct (error_N_FLT radix2 (-1074) 53 eq_refl (fun z => negb (Z.even z)) (d * e))
  as [eps [eta [Heps [Heta' [_ Hp1]]]]].
set (p1 := RN (d * e)).
assert (Heps' : Rabs eps <= u).
{ replace u with (/ 2 * pow2 (-53 + 1)); [exact Heps|].
  unfold u. replace (-53 + 1)%Z with (1 + -53)%Z by ring. rewrite pow2_plus, pow2_1. field. }
assert (Heta'' : Rabs eta <= u * E).
{ replace (u * E) with (/ 2 * pow2 (-1074)); [exact Heta'|]. rewrite Heta. field. }
apply Rabs_le_inv in Heps'. apply Rabs_le_inv in Heta''.
assert (Huq : u * q <= q / 2) by nra.
assert (Hd0 : 0 < d) by lra.
assert (Hde : 0 <= d * e) by (apply Rmult_le_pos; lra).
assert (Hp1lo : (1 - u) * (d * e) - u * E <= p1).
{ unfold p1. change (RN (d * e)) with (round radix2 fexp ZnearestE (d * e)).
  rewrite Hp1. nra. }
(* p2 = RN(p1 + 3eta/u): a normal number *)
set (p2 := RN (p1 + UF)).
assert (Hude : 0 <= (1 - u) * (d * e)) by (apply Rmult_le_pos; lra).
assert (HuE : u * E <= E / 1024).
{ pose proof (Rmult_le_compat_r E _ _ (Rlt_le _ _ HE0) Hu10). lra. }
assert (Hs : 5 * E <= p1 + UF) by lra.
assert (Hp2 : Rabs (p2 - (p1 + UF)) <= u * Rabs (p1 + UF)).
{ apply RN_rel. rewrite Rabs_pos_eq by lra. fold E. lra. }
rewrite (Rabs_pos_eq (p1 + UF)) in Hp2 by lra. apply Rabs_le_inv in Hp2.
(* beta = RN(p2 INFLATE): a normal number *)
assert (Hp2pos : 4 * E <= p2) by nra.
assert (Hp2I : p2 * (1 + 8 * u) <= p2 * I) by (apply Rmult_le_compat_l; lra).
assert (Hb : Rabs (RN (p2 * I) - p2 * I) <= u * Rabs (p2 * I)).
{ apply RN_rel. rewrite Rabs_pos_eq by nra. fold E. nra. }
rewrite (Rabs_pos_eq (p2 * I)) in Hb by nra. apply Rabs_le_inv in Hb.
set (beta := RN (p2 * I)) in *.
(* the inequality *)
split.
- unfold c5_bound.
  replace (IZR (2 * n + 5)) with (2 * Nr - 1)
    by (unfold Nr; rewrite <- (mult_IZR 2), <- minus_IZR; f_equal; ring).
  replace (IZR n) with (Nr - 3) by (unfold Nr; rewrite <- minus_IZR; f_equal; ring).
  fold u. rewrite Heta.
  set (K := Nr * u / (1 - (2 * Nr - 1) * u)).
  assert (HK : K * (den + u) = Nr * u) by (unfold K, den; field; nra).
  assert (HNu : Nr * u <= / 2) by (unfold den in Hden2; lra).
  assert (HK0 : 0 <= K).
  { unfold K. apply Rlt_le, Rdiv_lt_0_compat; [nra|]. unfold den in Hden2. lra. }
  assert (HKq : K <= q).
  { assert (Hqk : (q - K) * den = K * u) by (rewrite Rmult_minus_distr_r, Hqden; lra).
    assert (HKu : 0 <= K * u) by (apply Rmult_le_pos; lra).
    assert (Hdp : 0 < den) by lra.
    clear - Hqk HKu Hdp. nra. }
  assert (Hd' : (1 - u) * q <= d) by lra.
  assert (HN3 : 0 <= Nr - 3) by lra.
  assert (Hp2' : (1 - u) * (p1 + 6 * E) <= p2).
  { assert ((1 - u) * (6 * E) <= (1 - u) * UF) by (apply Rmult_le_compat_l; lra). lra. }
  assert (Hb' : (1 - u) * (p2 * (1 + 8 * u)) <= beta).
  { assert ((1 - u) * (p2 * (1 + 8 * u)) <= (1 - u) * (p2 * I)) by (apply Rmult_le_compat_l; lra).
    lra. }
  clear - Hu0 Hu10 HE0 He HK0 HKq Hd' Hp1lo Hp2' Hb' HNu HN3.
  apply (c5_combine u E K q d e p1 p2 beta Nr); assumption.
- split; [split|split; [split|split]].
  + unfold u. pose proof (pow2_pos (-53)). nra.
  + nra.
  + lra.
  + exact Hden1.
  + nra.
  + assert (pow2 51 * 2 = pow2 52) by (replace 2 with (pow2 1) at 1 by reflexivity; rewrite <- pow2_plus; reflexivity).
    nra.
Qed.

Print Assumptions C5_binary64.
