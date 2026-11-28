(** * Algorithm M3 ([dd²], [DoubleDouble::square], Chapter 11.2) in binary64

    Statement ([M3_binary64] at the end). Let [x_h], [x_l] be binary64
    numbers with [x_h = RN(x_h + x_l)] (a normalized double-double, invariant
    D1) and [2^-480 <= |x_h| <= 2^480] (the safe range of Chapter 11.2, zero
    excluded). Let the steps of M3 be, as in [src/dd.rs]:

    1. [c_h = RN(x_h x_h)]; [c_l1 = RN(x_h x_h - c_h)] (the FMA of 2Prod);
    2. [t = RN(2 x_h)]; [c = RN(t x_l + c_l1)] (one FMA);
    3. [z_h = RN(c_h + c)]; [w = RN(z_h - c_h)]; [z_l = RN(c - w)] (Fast2Sum).

    Then, with [x = x_h + x_l]:
    - [|z_h + z_l - x²| <= 3 · 2^-106 · x²] (the bound [3u²] of
      [ERROR_BOUNDS]);
    - [c_l1], [t] and the Fast2Sum are exact, and [z_h = RN(z_h + z_l)]
      (the result is normalized);
    - every rounded intermediate result has magnitude at most [2^1000], so
      no IEEE operation overflows and [RN] is the IEEE result (see
      [Model.v]); step E of Chapter 11.2 then returns [(z_h, z_l)].

    Method. With [s = 2^(mag(x_h) - 1)], [1 <= |x_h / s| < 2]. Steps 1 and 3
    round normal numbers, so they are [s²] times roundings to 53 bits of the
    scaled values ([RN_RNX], [RNX_scale]); step 2 either does the same or
    underflows, with an absolute error at most [2^-1075]. The two cases are
    the Gappa certificates [m3_normal] and [m3_underflow]; the exactness of
    [c_l1] is Flocq's [mult_error_FLT], that of the Fast2Sum Flocq's
    [Fast2Sum_correct]. *)

From Stdlib Require Import Reals Psatz ZArith Classical_Prop.
From Flocq Require Import Core Mult_error Pff2Flocq.
From Gappa Require Import Gappa_definitions.
From XP Require Import Model.
Require XP.m3_normal XP.m3_underflow.
From XP Require Import m3_normal_unfold m3_underflow_unfold.

Open Scope R_scope.

(** Algorithm M3 as written in [src/dd.rs]: the intermediate results, in
    program order. *)
Definition m3_steps (xh xl : R) :=
  let ch := RN (xh * xh) in
  let cl1 := RN (xh * xh - ch) in
  let t := RN (2 * xh) in
  let c := RN (t * xl + cl1) in
  let zh := RN (ch + c) in
  let w := RN (zh - ch) in
  let zl := RN (c - w) in
  (ch, cl1, t, c, zh, w, zl).

(** ** The Gappa certificates, restated in plain terms *)

(** The two certificates conclude the same three facts. *)
Definition m3_cert_goal (mx c : R) :=
  let m := RNX mx in
  let ch := RNX (m * m) in
  (exists e, -3 * pow2 (-106) <= e <= 3 * pow2 (-106) /\ ch + c = mx * mx * (1 + e)) /\
  (0 <= Rabs c <= pow2 (-50)) /\ (1 <= ch <= 4).

Lemma m3_cert_goal_use : forall mx c, m3_cert_goal mx c ->
  let m := RNX mx in
  let ch := RNX (m * m) in
  Rabs (ch + c - mx * mx) <= 3 * pow2 (-106) * (mx * mx) /\
  Rabs c <= pow2 (-50) /\ 1 <= ch <= 4.
Proof.
intros mx c [[e [He Hr]] [Hc Hch]] m ch.
split; [|split].
- fold m in Hr. fold ch in Hr. rewrite Hr.
  replace (mx * mx * (1 + e) - mx * mx) with (mx * mx * e) by ring.
  rewrite Rabs_mult, (Rabs_pos_eq (mx * mx)) by nra.
  rewrite Rmult_comm. apply Rmult_le_compat_r; [nra|].
  apply Rabs_le. lra.
- exact (proj2 Hc).
- exact Hch.
Qed.

Lemma cert_normal : forall mx,
  1 <= Rabs (RNX mx) <= 2 ->
  let m := RNX mx in
  let ch := RNX (m * m) in
  let c := RNX (2 * m * (mx - m) + (m * m - ch)) in
  Rabs (ch + c - mx * mx) <= 3 * pow2 (-106) * (mx * mx) /\
  Rabs c <= pow2 (-50) /\ 1 <= ch <= 4.
Proof.
intros mx Hm m ch c.
apply m3_cert_goal_use.
apply NNPP. intros H. unfold m3_cert_goal in H.
apply (XP.m3_normal.l1 mx).
- unfold_m3_normal. ring.
- unfold_m3_normal. norm_pow2. split; [lra|]. tauto.
Qed.

Lemma cert_underflow : forall mx c,
  1 <= Rabs (RNX mx) <= 2 ->
  let m := RNX mx in
  let ch := RNX (m * m) in
  let v := 2 * m * (mx - m) + (m * m - ch) in
  Rabs v <= pow2 (-62) -> Rabs (c - v) <= pow2 (-115) ->
  Rabs (ch + c - mx * mx) <= 3 * pow2 (-106) * (mx * mx) /\
  Rabs c <= pow2 (-50) /\ 1 <= ch <= 4.
Proof.
intros mx c Hm m ch v Hv Hcv.
apply m3_cert_goal_use.
apply NNPP. intros H. unfold m3_cert_goal in H.
apply (XP.m3_underflow.l1 mx c).
- unfold_m3_underflow. ring.
- unfold_m3_underflow. norm_pow2.
  assert (0 <= Rabs v) by apply Rabs_pos.
  assert (0 <= Rabs (c - v)) by apply Rabs_pos.
  unfold v, ch, m in *.
  split; [split; [split|]|]; [lra|lra|lra|tauto].
Qed.

(** ** The binary64 theorem *)

Lemma choice_NE_sym : forall z : Z,
  negb (Z.even z) = negb (negb (Z.even (- (z + 1)))).
Proof.
intros z. rewrite Z.even_opp, Z.even_add. simpl.
destruct (Z.even z); reflexivity.
Qed.

Section Binary64.

Variables xh xl : R.
Hypothesis Fxh : fmt xh.
Hypothesis Fxl : fmt xl.
Hypothesis Nrm : xh = RN (xh + xl).
Hypothesis Rng : pow2 (-480) <= Rabs xh <= pow2 480.

Lemma M3_main :
  let ch := RN (xh * xh) in
  let cl1 := RN (xh * xh - ch) in
  let t := RN (2 * xh) in
  let c := RN (t * xl + cl1) in
  let zh := RN (ch + c) in
  let w := RN (zh - ch) in
  let zl := RN (c - w) in
  Rabs (zh + zl - (xh + xl) * (xh + xl)) <= 3 * pow2 (-106) * ((xh + xl) * (xh + xl)) /\
  cl1 = xh * xh - ch /\ t = 2 * xh /\ zh + zl = ch + c /\ zh = RN (zh + zl) /\
  Rabs ch <= pow2 965 /\ Rabs cl1 <= pow2 965 /\ Rabs t <= pow2 965 /\
  Rabs c <= pow2 965 /\ Rabs zh <= pow2 965 /\ Rabs w <= pow2 965 /\ Rabs zl <= pow2 965.
Proof.
intros ch cl1 t c zh w zl.
(* The binade of x_h: s = 2^k <= |x_h| < 2^(k+1), -480 <= k <= 480. *)
assert (Hxh0 : xh <> 0).
{ intros H. rewrite H, Rabs_R0 in Rng. pose proof (pow2_pos (-480)). lra. }
set (k := (mag radix2 xh - 1)%Z).
assert (Hk1 : pow2 k <= Rabs xh) by (apply bpow_mag_le; exact Hxh0).
assert (Hk2 : Rabs xh < pow2 (k + 1)).
{ unfold k. replace (mag radix2 xh - 1 + 1)%Z with (mag radix2 xh : Z) by ring.
  apply bpow_mag_gt. }
assert (Hk : (-480 <= k <= 480)%Z).
{ split.
  - assert (-480 < k + 1)%Z; [|lia].
    apply lt_bpow with radix2. lra.
  - apply le_bpow with radix2. lra. }
set (S := pow2 (k + k)).
assert (HS : S = pow2 k * pow2 k) by (unfold S; apply pow2_plus).
assert (HSpos : 0 < S) by apply pow2_pos.
assert (HSle : S <= pow2 960) by (apply pow2_le; lia).
assert (HSge : pow2 (-960) <= S) by (apply pow2_le; lia).
set (iS := pow2 (- (k + k))).
assert (HiS : S * iS = 1).
{ unfold S, iS. rewrite <- pow2_plus. replace (k + k + - (k + k))%Z with 0%Z by ring.
  apply pow2_0. }
assert (HiSle : iS <= pow2 960) by (apply pow2_le; lia).
(* The scaled operand: x = 2^k mx, x_h = 2^k m, 1 <= |m| <= 2. *)
set (x := xh + xl).
set (mx := x * pow2 (- k)).
assert (Hkk : pow2 k * pow2 (- k) = 1).
{ rewrite <- pow2_plus. replace (k + - k)%Z with 0%Z by ring. apply pow2_0. }
assert (Hx : x = pow2 k * mx) by (unfold mx; rewrite (Rmult_comm x), <- Rmult_assoc, Hkk; ring).
assert (Hxbig : pow2 (-1022) <= Rabs x).
{ destruct (Rle_or_lt (pow2 (-1022)) (Rabs x)) as [H|H]; [exact H|].
  assert (H' : Rabs (RN x) <= pow2 (-1022)) by (apply RN_abs_le; [lia|lra]).
  assert (HN : RN x = xh) by (symmetry; exact Nrm). rewrite HN in H'.
  assert (pow2 (-1022) < pow2 (-480)) by (apply bpow_lt; lia). lra. }
set (m := RNX mx).
assert (Hxhm : xh = pow2 k * m).
{ assert (HN : xh = RN x) by exact Nrm.
  rewrite HN at 1. rewrite (RN_RNX x Hxbig), Hx. exact (RNX_scale k mx). }
assert (Hm : 1 <= Rabs m <= 2).
{ rewrite Hxhm, Rabs_mult, (Rabs_pos_eq (pow2 k)) in Hk1, Hk2 by (apply Rlt_le, pow2_pos).
  rewrite pow2_plus, pow2_1 in Hk2. pose proof (pow2_pos k). split; nra. }
assert (Hxl : xl = pow2 k * (mx - m)).
{ replace xl with (x - xh) by (unfold x; ring). rewrite Hx, Hxhm. ring. }
(* Step 1: c_h = S RNX(m^2); c_l1 is exact. *)
set (chs := RNX (m * m)).
assert (Hsq : xh * xh = S * (m * m)) by (rewrite Hxhm, HS; ring).
assert (Hch : ch = S * chs).
{ unfold ch, chs. rewrite Hsq, RN_RNX; [exact (RNX_scale (k + k) (m * m))|].
  rewrite Rabs_mult, (Rabs_pos_eq S) by lra.
  assert (pow2 (-1022) <= pow2 (-960)) by (apply pow2_le; lia).
  assert (1 <= Rabs (m * m)) by (rewrite Rabs_mult; nra). nra. }
assert (Hcl1 : cl1 = xh * xh - ch).
{ unfold cl1. apply round_generic; [apply valid_rnd_N|].
  replace (xh * xh - ch) with (- (RN (xh * xh) - xh * xh)) by (unfold ch; ring).
  apply generic_format_opp.
  apply (mult_error_FLT radix2 (-1074) 53 ZnearestE xh xh Fxh Fxh).
  intros _. rewrite Hsq, Rabs_mult, (Rabs_pos_eq S) by lra.
  assert (pow2 (-1074 + 2 * 53 - 1) <= pow2 (-960)) by (apply pow2_le; lia).
  assert (1 <= Rabs (m * m)) by (rewrite Rabs_mult; nra). nra. }
(* Step 2: t = 2 x_h is exact; c = S cs. *)
assert (Ht : t = 2 * xh).
{ unfold t. apply round_generic; [apply valid_rnd_N|].
  replace (2 * xh) with (xh * pow2 1) by (rewrite pow2_1; ring).
  apply mult_bpow_exact_FLT; [exact Fxh|]. unfold k in Hk. lia. }
set (vs := 2 * m * (mx - m) + (m * m - chs)).
assert (Hv : t * xl + cl1 = S * vs).
{ rewrite Ht, Hcl1, Hch, Hsq, Hxhm, Hxl, HS. unfold vs. ring. }
set (cs := c * iS).
assert (Hc : c = S * cs) by (unfold cs; transitivity (S * iS * c); [rewrite HiS; ring|ring]).
assert (Hcert : Rabs (chs + cs - mx * mx) <= 3 * pow2 (-106) * (mx * mx) /\
                Rabs cs <= pow2 (-50) /\ 1 <= chs <= 4).
{ destruct (Rle_or_lt (pow2 (-1022)) (Rabs (S * vs))) as [Hn|Hu].
  - (* no underflow *)
    assert (Hcs : cs = RNX vs).
    { unfold cs, c. rewrite Hv, (RN_RNX _ Hn).
      change (RNX (pow2 (k + k) * vs) * iS = RNX vs).
      rewrite (RNX_scale (k + k) vs). fold S.
      transitivity (S * iS * RNX vs); [ring|rewrite HiS; ring]. }
    rewrite Hcs. exact (cert_normal mx Hm).
  - (* underflow: |c - S vs| <= 2^-1075 *)
    assert (He : Rabs (c - S * vs) <= pow2 (-1075)).
    { unfold c. rewrite Hv.
      apply Rle_trans with (/ 2 * ulp radix2 fexp (S * vs)); [apply error_le_half_ulp; exact valid_exp_binary64|].
      rewrite ulp_FLT_small; [|reflexivity|].
      - replace (pow2 (-1075)) with (/ 2 * pow2 (-1074)); [lra|].
        replace (-1075)%Z with (-1 + -1074)%Z by reflexivity. rewrite pow2_plus. reflexivity.
      - apply Rlt_le_trans with (pow2 (-1022)); [exact Hu|]. apply pow2_le. lia. }
    apply (cert_underflow mx cs Hm).
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
destruct Hcert as [Hrel [Hcs Hchs]].
(* Step 3: the Fast2Sum is exact. *)
assert (Hcch : Rabs c <= Rabs ch).
{ rewrite Hc, Hch, !Rabs_mult, (Rabs_pos_eq S), (Rabs_pos_eq chs) by lra.
  assert (pow2 (-50) <= 1) by (rewrite <- pow2_0; apply pow2_le; lia). nra. }
assert (Hfs : zh + zl = ch + c).
{ unfold zl, w, zh.
  replace (RN (ch + c) - ch) with (- (ch - RN (ch + c))) by ring.
  rewrite round_NE_opp.
  replace (c - - RN (ch - RN (ch + c))) with (c + RN (ch - RN (ch + c))) by ring.
  exact (Fast2Sum_correct (-1074) 53 (fun z => negb (Z.even z)) eq_refl
           ltac:(lia) choice_NE_sym ch c (fmt_RN _) (fmt_RN _) Hcch). }
(* Magnitudes. *)
assert (P2 : pow2 962 = 4 * pow2 960) by (replace 4 with (pow2 2) by reflexivity; rewrite <- pow2_plus; reflexivity).
assert (P3 : pow2 963 = 8 * pow2 960) by (replace 8 with (pow2 3) by reflexivity; rewrite <- pow2_plus; reflexivity).
assert (P4 : pow2 964 = 16 * pow2 960) by (replace 16 with (pow2 4) by reflexivity; rewrite <- pow2_plus; reflexivity).
assert (P5 : pow2 965 = 32 * pow2 960) by (replace 32 with (pow2 5) by reflexivity; rewrite <- pow2_plus; reflexivity).
assert (P480 : pow2 481 <= pow2 965) by (apply pow2_le; lia).
assert (P50 : pow2 (-50) <= 1) by (rewrite <- pow2_0; apply pow2_le; lia).
pose proof (pow2_pos 960).
assert (Bch : Rabs ch <= 4 * pow2 960).
{ rewrite Hch, Rabs_mult, (Rabs_pos_eq S), (Rabs_pos_eq chs) by lra. nra. }
assert (Bc : Rabs c <= pow2 960).
{ rewrite Hc, Rabs_mult, (Rabs_pos_eq S) by lra. nra. }
assert (Bsq : Rabs (xh * xh) <= 4 * pow2 960).
{ rewrite Hsq, Rabs_mult, (Rabs_pos_eq S) by lra.
  rewrite Rabs_mult. pose proof (Rabs_pos m).
  assert (Rabs m * Rabs m <= 4) by nra.
  apply Rle_trans with (pow2 960 * 4); [apply Rmult_le_compat; nra|lra]. }
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
- (* the relative error *)
  rewrite Hfs, Hch, Hc. fold x. rewrite Hx.
  replace (S * chs + S * cs - pow2 k * mx * (pow2 k * mx)) with (S * (chs + cs - mx * mx))
    by (rewrite HS; ring).
  replace (pow2 k * mx * (pow2 k * mx)) with (S * (mx * mx)) by (rewrite HS; ring).
  rewrite Rabs_mult, (Rabs_pos_eq S) by lra. nra.
- exact Hcl1.
- exact Ht.
- exact Hfs.
- rewrite Hfs. reflexivity.
- rewrite P5. lra.
- rewrite Hcl1, P5. unfold Rminus.
  apply Rle_trans with (Rabs (xh * xh) + Rabs (- ch)); [apply Rabs_triang|].
  rewrite Rabs_Ropp. lra.
- rewrite Ht, Rabs_mult, Rabs_pos_eq by lra.
  assert (Rabs xh <= pow2 480) by lra.
  replace (pow2 481) with (2 * pow2 480) in P480 by (replace 2 with (pow2 1) by reflexivity; rewrite <- pow2_plus; reflexivity). lra.
- rewrite P5. lra.
- assert (pow2 963 <= pow2 965) by (apply pow2_le; lia). lra.
- assert (pow2 964 <= pow2 965) by (apply pow2_le; lia). lra.
- exact Bzl.
Qed.

End Binary64.

(** The theorem, for Algorithm M3 as defined by [m3_steps]. *)
Theorem M3_binary64 : forall xh xl,
  fmt xh -> fmt xl -> xh = RN (xh + xl) -> pow2 (-480) <= Rabs xh <= pow2 480 ->
  let '(ch, cl1, t, c, zh, w, zl) := m3_steps xh xl in
  Rabs (zh + zl - (xh + xl) * (xh + xl)) <= 3 * pow2 (-106) * ((xh + xl) * (xh + xl)) /\
  cl1 = xh * xh - ch /\ t = 2 * xh /\ zh + zl = ch + c /\ zh = RN (zh + zl) /\
  Rabs ch <= pow2 965 /\ Rabs cl1 <= pow2 965 /\ Rabs t <= pow2 965 /\
  Rabs c <= pow2 965 /\ Rabs zh <= pow2 965 /\ Rabs w <= pow2 965 /\ Rabs zl <= pow2 965.
Proof.
intros xh xl Fxh Fxl Nrm Rng.
exact (M3_main xh xl Fxh Nrm Rng).
Qed.

Print Assumptions M3_binary64.
