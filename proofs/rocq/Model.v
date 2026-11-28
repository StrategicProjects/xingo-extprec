(** * The model of binary64 used by the proofs of `xingo-extprec`

    Binary64 is Flocq's format [FLT_exp (-1074) 53] (53-bit significands,
    subnormals down to [2^-1074]) with rounding to nearest, ties to even
    ([ZnearestE]). Flocq's format has no largest exponent: an operation that
    overflows in IEEE 754 is *not* modelled, so every theorem that uses
    [RN] also bounds the magnitude of every rounded intermediate result,
    and that bound, below [2^1024], is what licenses the model (an IEEE
    operation whose rounded result is below [2^1024] in magnitude returns
    exactly [RN] of the exact result; Flocq's [Binary] module proves this
    for its IEEE model, theorems [Bplus_correct], [Bmult_correct],
    [Bfma_correct], [Bdiv_correct]; that step is not repeated here).

    [RNX] is rounding to 53 bits with an unbounded exponent (Flocq's
    [FLX_exp 53]), Gappa's [float<53,ne>]. *)

From Stdlib Require Import Reals Psatz ZArith.
From Flocq Require Import Core.

Open Scope R_scope.
(* Abbreviations kept as [Notation] so that the file also builds before Rocq 9.2. *)
Set Warnings "-notation-for-abbreviation".

Notation fexp := (FLT_exp (-1074) 53).
Notation fmt := (generic_format radix2 fexp).
Notation RN := (round radix2 fexp ZnearestE).
Notation RNX := (round radix2 (FLX_exp 53) ZnearestE).
Notation pow2 := (bpow radix2).

#[global] Instance prec_gt_0_53 : Prec_gt_0 53.
Proof. unfold Prec_gt_0. lia. Qed.

#[global] Instance valid_exp_binary64 : Valid_exp fexp := FLT_exp_valid (-1074) 53.

(** Rounding with an unbounded exponent commutes with scaling by a power
    of two. *)
Lemma RNX_scale : forall j y, RNX (pow2 j * y) = pow2 j * RNX y.
Proof.
intros j y.
destruct (Req_dec y 0) as [Hy|Hy].
{ rewrite Hy, Rmult_0_r, round_0 by apply valid_rnd_N. ring. }
unfold round, F2R, scaled_mantissa, cexp; simpl.
rewrite (Rmult_comm (pow2 j) y), (mag_mult_bpow radix2 y j Hy).
unfold FLX_exp.
replace (y * pow2 j * pow2 (- (mag radix2 y + j - 53)))
  with (y * pow2 (- (mag radix2 y - 53))).
2:{ rewrite Rmult_assoc, <- bpow_plus. f_equal. f_equal. ring. }
replace (mag radix2 y + j - 53)%Z with (j + (mag radix2 y - 53))%Z by ring.
rewrite bpow_plus. ring.
Qed.

(** Above the subnormal range binary64 rounding is rounding to 53 bits. *)
Lemma RN_RNX : forall y, pow2 (-1022) <= Rabs y -> RN y = RNX y.
Proof.
intros y Hy. apply round_FLT_FLX. exact Hy.
Qed.

(** A rounded value is in the format, so a second rounding is the identity. *)
Lemma fmt_RN : forall y, fmt (RN y).
Proof.
intros y. apply generic_format_round.
- apply FLT_exp_valid. reflexivity.
- apply valid_rnd_N.
Qed.

(** Powers of two in the binary64 range are in the format. *)
Lemma fmt_pow2 : forall e, (-1074 <= e)%Z -> fmt (pow2 e).
Proof.
intros e He. apply generic_format_bpow. unfold FLT_exp. lia.
Qed.

(** Monotonicity, in the form used for the no-overflow bounds. *)
Lemma RN_abs_le : forall y e, (-1074 <= e)%Z -> Rabs y <= pow2 e -> Rabs (RN y) <= pow2 e.
Proof.
intros y e He Hy. apply abs_round_le_generic.
- apply FLT_exp_valid. reflexivity.
- apply valid_rnd_N.
- apply fmt_pow2. exact He.
- exact Hy.
Qed.

Lemma pow2_pos : forall e, 0 < pow2 e.
Proof. intros e. apply bpow_gt_0. Qed.

Lemma pow2_le : forall e f, (e <= f)%Z -> pow2 e <= pow2 f.
Proof. intros e f H. apply bpow_le. exact H. Qed.

Lemma pow2_plus : forall e f, pow2 (e + f) = pow2 e * pow2 f.
Proof. intros e f. apply bpow_plus. Qed.

Lemma pow2_1 : pow2 1 = 2.
Proof. reflexivity. Qed.

Lemma pow2_0 : pow2 0 = 1.
Proof. reflexivity. Qed.

Lemma pow2_2 : pow2 2 = 4.
Proof. reflexivity. Qed.

(** Small powers of two as numerals, for [lra]. *)
Ltac norm_pow2 := rewrite ?pow2_0, ?pow2_1, ?pow2_2, ?Rmult_1_l, ?Rmult_0_l.
Ltac norm_pow2_in H := rewrite ?pow2_0, ?pow2_1, ?pow2_2, ?Rmult_1_l, ?Rmult_0_l in H.
