(** * Algorithm M1 ([dd · f64], DWTimesFP3, Chapter 11.2) in binary64

    Statement ([M1_binary64] at the end). Let [x_h], [x_l], [y] be binary64
    numbers with [x_h = RN(x_h + x_l)] and [x_h], [y] in the safe range
    ([2^-480 <= |x_h|, |y| <= 2^480]). Let the steps of M1 be, as in
    [impl Mul<f64> for DoubleDouble] ([src/dd.rs]):

    1. [c_h = RN(x_h y)]; [c_l1 = RN(x_h y - c_h)] (the FMA of 2Prod);
    2. [c_l3 = RN(x_l y + c_l1)] (one FMA);
    3. [z_h = RN(c_h + c_l3)]; [w = RN(z_h - c_h)]; [z_l = RN(c_l3 - w)].

    Then [|z_h + z_l - x y| <= 2u² (1 + 2^-37) |x y|] with [x = x_h + x_l],
    [u = 2^-53]; [c_l1] and the Fast2Sum are exact, [z_h = RN(z_h + z_l)],
    and every rounded intermediate result is at most [2^965] in magnitude.

    The constant is [2u² (1 + 2^-37)], not the [2u²] of [ERROR_BOUNDS]
    (Joldes, Muller & Popescu): Gappa 1.8.3 does not reach the exact [2u²]
    with the hints tried (see README.md and the report of task V03).

    Method: as for M3 ([M3.v]), with the two scales [2^(mag x_h - 1)] and
    [2^(mag y - 1)]. *)

From Stdlib Require Import Reals Psatz ZArith Classical_Prop.
From Flocq Require Import Core Mult_error Pff2Flocq.
From Gappa Require Import Gappa_definitions.
From XP Require Import Model M3.
Require XP.m1_normal XP.m1_underflow.
From XP Require Import m1_normal_unfold m1_underflow_unfold.

Open Scope R_scope.

Definition m1_steps (xh xl y : R) :=
  let ch := RN (xh * y) in
  let cl1 := RN (xh * y - ch) in
  let c := RN (xl * y + cl1) in
  let zh := RN (ch + c) in
  let w := RN (zh - ch) in
  let zl := RN (c - w) in
  (ch, cl1, c, zh, w, zl).

(** [2u² (1 + 2^-37)], as Gappa writes it. *)
Definition K1 := 137438953473 * pow2 (-142).

Definition m1_cert_goal (mx my c : R) :=
  let m := RNX mx in
  let ch := RNX (m * RNX my) in
  (exists e, -137438953473 * pow2 (-142) <= e <= 137438953473 * pow2 (-142) /\
             ch + c = mx * RNX my * (1 + e)) /\
  (0 <= Rabs c <= pow2 (-50)).

Lemma m1_cert_goal_use : forall mx my c, m1_cert_goal mx my c ->
  let m := RNX mx in
  let ch := RNX (m * RNX my) in
  Rabs (ch + c - mx * RNX my) <= K1 * Rabs (mx * RNX my) /\ Rabs c <= pow2 (-50).
Proof.
intros mx my c [[e [He Hr]] Hc] m ch.
split.
- fold m in Hr. fold ch in Hr. rewrite Hr.
  replace (mx * RNX my * (1 + e) - mx * RNX my) with (mx * RNX my * e) by ring.
  rewrite Rabs_mult, Rmult_comm. unfold K1.
  apply Rmult_le_compat_r; [apply Rabs_pos|]. apply Rabs_le. lra.
- exact (proj2 Hc).
Qed.

Lemma cert1_normal : forall mx my,
  1 <= Rabs (RNX mx) <= 2 -> 1 <= Rabs (RNX my) <= 2 ->
  let m := RNX mx in
  let ch := RNX (m * RNX my) in
  let c := RNX ((m * RNX my - ch) + (mx - m) * RNX my) in
  Rabs (ch + c - mx * RNX my) <= K1 * Rabs (mx * RNX my) /\ Rabs c <= pow2 (-50).
Proof.
intros mx my Hm Hy m ch c.
apply m1_cert_goal_use.
apply NNPP. intros H. unfold m1_cert_goal in H.
apply (XP.m1_normal.l1 mx my).
- unfold_m1_normal. ring.
- unfold_m1_normal. norm_pow2. split; [split; lra|]. tauto.
Qed.

Lemma cert1_underflow : forall mx my c,
  1 <= Rabs (RNX mx) <= 2 -> 1 <= Rabs (RNX my) <= 2 ->
  let m := RNX mx in
  let ch := RNX (m * RNX my) in
  let v := (m * RNX my - ch) + (mx - m) * RNX my in
  Rabs v <= pow2 (-62) -> Rabs (c - v) <= pow2 (-115) ->
  Rabs (ch + c - mx * RNX my) <= K1 * Rabs (mx * RNX my) /\ Rabs c <= pow2 (-50).
Proof.
intros mx my c Hm Hy m ch v Hv Hcv.
apply m1_cert_goal_use.
apply NNPP. intros H. unfold m1_cert_goal in H.
apply (XP.m1_underflow.l1 mx my c).
- unfold_m1_underflow. ring.
- unfold_m1_underflow. norm_pow2.
  assert (0 <= Rabs v) by apply Rabs_pos.
  assert (0 <= Rabs (c - v)) by apply Rabs_pos.
  unfold v, ch, m in *.
  split; [split; [split; [split|]|]|]; [lra|lra|lra|lra|tauto].
Qed.

(** A binary64 number in the safe range, written as [2^k] times a number of
    magnitude in [1, 2] that rounding to 53 bits leaves unchanged. *)
Lemma scale_operand : forall xh xl,
  xh = RN (xh + xl) -> pow2 (-480) <= Rabs xh <= pow2 480 ->
  let k := (mag radix2 xh - 1)%Z in
  let mx := (xh + xl) * pow2 (- k) in
  (-480 <= k <= 480)%Z /\ xh = pow2 k * RNX mx /\ xh + xl = pow2 k * mx /\
  1 <= Rabs (RNX mx) <= 2.
Proof.
intros xh xl Nrm Rng k mx.
assert (Hxh0 : xh <> 0).
{ intros H. rewrite H, Rabs_R0 in Rng. pose proof (pow2_pos (-480)). lra. }
assert (Hk1 : pow2 k <= Rabs xh) by (apply bpow_mag_le; exact Hxh0).
assert (Hk2 : Rabs xh < pow2 (k + 1)).
{ unfold k. replace (mag radix2 xh - 1 + 1)%Z with (mag radix2 xh : Z) by ring.
  apply bpow_mag_gt. }
assert (Hk : (-480 <= k <= 480)%Z).
{ split.
  - assert (-480 < k + 1)%Z; [|lia]. apply lt_bpow with radix2. lra.
  - apply le_bpow with radix2. lra. }
assert (Hkk : pow2 k * pow2 (- k) = 1).
{ rewrite <- pow2_plus. replace (k + - k)%Z with 0%Z by ring. apply pow2_0. }
assert (Hx : xh + xl = pow2 k * mx).
{ unfold mx. transitivity ((pow2 k * pow2 (- k)) * (xh + xl)); [rewrite Hkk; ring|ring]. }
assert (Hxbig : pow2 (-1022) <= Rabs (xh + xl)).
{ destruct (Rle_or_lt (pow2 (-1022)) (Rabs (xh + xl))) as [H|H]; [exact H|].
  assert (H' : Rabs (RN (xh + xl)) <= pow2 (-1022)) by (apply RN_abs_le; [lia|lra]).
  rewrite <- Nrm in H'.
  assert (pow2 (-1022) < pow2 (-480)) by (apply bpow_lt; lia). lra. }
assert (Hxhm : xh = pow2 k * RNX mx).
{ rewrite Nrm at 1. rewrite (RN_RNX _ Hxbig), Hx. exact (RNX_scale k mx). }
split; [exact Hk|split; [exact Hxhm|split; [exact Hx|]]].
rewrite Hxhm, Rabs_mult, (Rabs_pos_eq (pow2 k)) in Hk1, Hk2 by (apply Rlt_le, pow2_pos).
rewrite pow2_plus, pow2_1 in Hk2. pose proof (pow2_pos k). split; nra.
Qed.

Section Binary64.

Variables xh xl y : R.
Hypothesis Fxh : fmt xh.
Hypothesis Fy : fmt y.
Hypothesis Nrm : xh = RN (xh + xl).
Hypothesis Rngx : pow2 (-480) <= Rabs xh <= pow2 480.
Hypothesis Rngy : pow2 (-480) <= Rabs y <= pow2 480.

Lemma M1_main :
  let ch := RN (xh * y) in
  let cl1 := RN (xh * y - ch) in
  let c := RN (xl * y + cl1) in
  let zh := RN (ch + c) in
  let w := RN (zh - ch) in
  let zl := RN (c - w) in
  Rabs (zh + zl - (xh + xl) * y) <= K1 * Rabs ((xh + xl) * y) /\
  cl1 = xh * y - ch /\ zh + zl = ch + c /\ zh = RN (zh + zl) /\
  Rabs ch <= pow2 965 /\ Rabs cl1 <= pow2 965 /\ Rabs c <= pow2 965 /\
  Rabs zh <= pow2 965 /\ Rabs w <= pow2 965 /\ Rabs zl <= pow2 965.
Proof.
intros ch cl1 c zh w zl.
destruct (scale_operand xh xl Nrm Rngx) as [Hk1 [Hxhm [Hx Hm]]].
set (k1 := (mag radix2 xh - 1)%Z) in *.
set (mx := (xh + xl) * pow2 (- k1)) in *.
assert (Ny : y = RN (y + 0)) by (rewrite Rplus_0_r; symmetry; apply round_generic; [apply valid_rnd_N|exact Fy]).
destruct (scale_operand y 0 Ny Rngy) as [Hk2 [Hym [_ Hmy]]].
set (k2 := (mag radix2 y - 1)%Z) in *.
set (my := (y + 0) * pow2 (- k2)) in *.
set (m := RNX mx) in *. set (ym := RNX my) in *.
set (S := pow2 (k1 + k2)).
assert (HS : S = pow2 k1 * pow2 k2) by (unfold S; apply pow2_plus).
assert (HSpos : 0 < S) by apply pow2_pos.
assert (HSle : S <= pow2 960) by (apply pow2_le; lia).
set (iS := pow2 (- (k1 + k2))).
assert (HiS : S * iS = 1).
{ unfold S, iS. rewrite <- pow2_plus. replace (k1 + k2 + - (k1 + k2))%Z with 0%Z by ring.
  apply pow2_0. }
assert (HiSle : iS <= pow2 960) by (apply pow2_le; lia).
assert (Hxl : xl = pow2 k1 * (mx - m)).
{ replace xl with ((xh + xl) - xh) by ring. rewrite Hx, Hxhm. ring. }
(* step 1 *)
set (chs := RNX (m * ym)).
assert (Hpr : xh * y = S * (m * ym)) by (rewrite Hxhm, Hym, HS; ring).
assert (Hmym : 1 <= Rabs (m * ym)) by (rewrite Rabs_mult; nra).
assert (Hch : ch = S * chs).
{ unfold ch, chs. rewrite Hpr, RN_RNX; [exact (RNX_scale (k1 + k2) (m * ym))|].
  rewrite Rabs_mult, (Rabs_pos_eq S) by lra.
  assert (pow2 (-1022) <= pow2 (-960)) by (apply pow2_le; lia).
  assert (pow2 (-960) <= S) by (apply pow2_le; lia). nra. }
assert (Hcl1 : cl1 = xh * y - ch).
{ unfold cl1. apply round_generic; [apply valid_rnd_N|].
  replace (xh * y - ch) with (- (RN (xh * y) - xh * y)) by (unfold ch; ring).
  apply generic_format_opp.
  apply (mult_error_FLT radix2 (-1074) 53 ZnearestE xh y Fxh Fy).
  intros _. rewrite Hpr, Rabs_mult, (Rabs_pos_eq S) by lra.
  assert (pow2 (-1074 + 2 * 53 - 1) <= pow2 (-960)) by (apply pow2_le; lia).
  assert (pow2 (-960) <= S) by (apply pow2_le; lia). nra. }
(* step 2 *)
set (vs := (m * ym - chs) + (mx - m) * ym).
assert (Hv : xl * y + cl1 = S * vs).
{ rewrite Hcl1, Hch, Hpr, Hxl, Hym, HS. unfold vs. ring. }
set (cs := c * iS).
assert (Hc : c = S * cs) by (unfold cs; transitivity (S * iS * c); [rewrite HiS; ring|ring]).
assert (Hcert : Rabs (chs + cs - mx * ym) <= K1 * Rabs (mx * ym) /\ Rabs cs <= pow2 (-50)).
{ destruct (Rle_or_lt (pow2 (-1022)) (Rabs (S * vs))) as [Hn|Hu].
  - assert (Hcs : cs = RNX vs).
    { unfold cs, c. rewrite Hv, (RN_RNX _ Hn).
      change (RNX (pow2 (k1 + k2) * vs) * iS = RNX vs).
      rewrite (RNX_scale (k1 + k2) vs). fold S.
      transitivity (S * iS * RNX vs); [ring|rewrite HiS; ring]. }
    rewrite Hcs. exact (cert1_normal mx my Hm Hmy).
  - assert (He : Rabs (c - S * vs) <= pow2 (-1075)).
    { unfold c. rewrite Hv.
      apply Rle_trans with (/ 2 * ulp radix2 fexp (S * vs));
        [apply error_le_half_ulp; exact valid_exp_binary64|].
      rewrite ulp_FLT_small; [|reflexivity|].
      - replace (pow2 (-1075)) with (/ 2 * pow2 (-1074)); [lra|].
        replace (-1075)%Z with (-1 + -1074)%Z by reflexivity. rewrite pow2_plus. reflexivity.
      - apply Rlt_le_trans with (pow2 (-1022)); [exact Hu|]. apply pow2_le. lia. }
    apply (cert1_underflow mx my cs Hm Hmy).
    + change (Rabs vs <= pow2 (-62)).
      replace vs with ((S * vs) * iS) by (transitivity (S * iS * vs); [ring|rewrite HiS; ring]).
      rewrite Rabs_mult, (Rabs_pos_eq iS) by (apply Rlt_le, pow2_pos).
      replace (pow2 (-62)) with (pow2 (-1022) * pow2 960) by (rewrite <- pow2_plus; reflexivity).
      pose proof (pow2_pos (-1022)). assert (0 < iS) by apply pow2_pos. nra.
    + change (Rabs (cs - vs) <= pow2 (-115)).
      replace (cs - vs) with ((c - S * vs) * iS)
        by (unfold cs; transitivity (c * iS - S * iS * vs); [ring|rewrite HiS; ring]).
      rewrite Rabs_mult, (Rabs_pos_eq iS) by (apply Rlt_le, pow2_pos).
      replace (pow2 (-115)) with (pow2 (-1075) * pow2 960) by (rewrite <- pow2_plus; reflexivity).
      pose proof (pow2_pos (-1075)). pose proof (Rabs_pos (c - S * vs)).
      assert (0 < iS) by apply pow2_pos. nra. }
destruct Hcert as [Hrel Hcs].
(* step 3 *)
assert (Hchs1 : 1 <= Rabs chs).
{ unfold chs. apply abs_round_ge_generic; [apply FLX_exp_valid; reflexivity|apply valid_rnd_N| |exact Hmym].
  replace 1 with (pow2 0) by reflexivity. apply generic_format_bpow. unfold FLX_exp. lia. }
assert (Hchs4 : Rabs chs <= 4).
{ unfold chs. apply abs_round_le_generic; [apply FLX_exp_valid; reflexivity|apply valid_rnd_N| |].
  - replace 4 with (pow2 2) by reflexivity. apply generic_format_bpow. unfold FLX_exp. lia.
  - rewrite Rabs_mult. nra. }
assert (Hcch : Rabs c <= Rabs ch).
{ rewrite Hc, Hch, !Rabs_mult, (Rabs_pos_eq S) by lra.
  assert (pow2 (-50) <= 1) by (rewrite <- pow2_0; apply pow2_le; lia). nra. }
assert (Hfs : zh + zl = ch + c).
{ unfold zl, w, zh.
  replace (RN (ch + c) - ch) with (- (ch - RN (ch + c))) by ring.
  rewrite round_NE_opp.
  replace (c - - RN (ch - RN (ch + c))) with (c + RN (ch - RN (ch + c))) by ring.
  exact (Fast2Sum_correct (-1074) 53 (fun z => negb (Z.even z)) eq_refl
           ltac:(lia) choice_NE_sym ch c (fmt_RN _) (fmt_RN _) Hcch). }
(* magnitudes *)
assert (P3 : pow2 963 = 8 * pow2 960) by (replace 8 with (pow2 3) by reflexivity; rewrite <- pow2_plus; reflexivity).
assert (P4 : pow2 964 = 16 * pow2 960) by (replace 16 with (pow2 4) by reflexivity; rewrite <- pow2_plus; reflexivity).
assert (P5 : pow2 965 = 32 * pow2 960) by (replace 32 with (pow2 5) by reflexivity; rewrite <- pow2_plus; reflexivity).
assert (P50 : pow2 (-50) <= 1) by (rewrite <- pow2_0; apply pow2_le; lia).
pose proof (pow2_pos 960).
assert (Bch : Rabs ch <= 4 * pow2 960).
{ rewrite Hch, Rabs_mult, (Rabs_pos_eq S) by lra. nra. }
assert (Bc : Rabs c <= pow2 960).
{ rewrite Hc, Rabs_mult, (Rabs_pos_eq S) by lra. nra. }
assert (Bpr : Rabs (xh * y) <= 4 * pow2 960).
{ rewrite Hpr, Rabs_mult, (Rabs_pos_eq S) by lra.
  assert (Rabs (m * ym) <= 4) by (rewrite Rabs_mult; nra).
  apply Rle_trans with (pow2 960 * 4); [apply Rmult_le_compat; try lra; apply Rabs_pos|lra]. }
assert (Bzh : Rabs zh <= pow2 963).
{ apply RN_abs_le; [lia|]. rewrite P3.
  apply Rle_trans with (Rabs ch + Rabs c); [apply Rabs_triang|lra]. }
assert (Bw : Rabs w <= pow2 964).
{ apply RN_abs_le; [lia|]. rewrite P4, P3 in *.
  unfold Rminus. apply Rle_trans with (Rabs zh + Rabs (- ch)); [apply Rabs_triang|].
  rewrite Rabs_Ropp. lra. }
assert (Bzl : Rabs zl <= pow2 965).
{ apply RN_abs_le; [lia|]. rewrite P5, P4 in *.
  unfold Rminus. apply Rle_trans with (Rabs c + Rabs (- w)); [apply Rabs_triang|].
  rewrite Rabs_Ropp. lra. }
repeat split.
- rewrite Hfs, Hch, Hc, Hx, Hym.
  replace (S * chs + S * cs - pow2 k1 * mx * (pow2 k2 * ym)) with (S * (chs + cs - mx * ym))
    by (rewrite HS; ring).
  replace (pow2 k1 * mx * (pow2 k2 * ym)) with (S * (mx * ym)) by (rewrite HS; ring).
  rewrite !(Rabs_mult S), (Rabs_pos_eq S) by lra.
  assert (0 <= K1) by (unfold K1; pose proof (pow2_pos (-142)); lra). nra.
- exact Hcl1.
- exact Hfs.
- rewrite Hfs. reflexivity.
- rewrite P5. lra.
- rewrite Hcl1, P5. unfold Rminus.
  apply Rle_trans with (Rabs (xh * y) + Rabs (- ch)); [apply Rabs_triang|].
  rewrite Rabs_Ropp. lra.
- rewrite P5. lra.
- assert (pow2 963 <= pow2 965) by (apply pow2_le; lia). lra.
- assert (pow2 964 <= pow2 965) by (apply pow2_le; lia). lra.
- exact Bzl.
Qed.

End Binary64.

Theorem M1_binary64 : forall xh xl y,
  fmt xh -> fmt y -> xh = RN (xh + xl) ->
  pow2 (-480) <= Rabs xh <= pow2 480 -> pow2 (-480) <= Rabs y <= pow2 480 ->
  let '(ch, cl1, c, zh, w, zl) := m1_steps xh xl y in
  Rabs (zh + zl - (xh + xl) * y) <= K1 * Rabs ((xh + xl) * y) /\
  cl1 = xh * y - ch /\ zh + zl = ch + c /\ zh = RN (zh + zl) /\
  Rabs ch <= pow2 965 /\ Rabs cl1 <= pow2 965 /\ Rabs c <= pow2 965 /\
  Rabs zh <= pow2 965 /\ Rabs w <= pow2 965 /\ Rabs zl <= pow2 965.
Proof.
intros xh xl y Fxh Fy Nrm Rngx Rngy.
exact (M1_main xh xl y Fxh Fy Nrm Rngx Rngy).
Qed.

Print Assumptions M1_binary64.
