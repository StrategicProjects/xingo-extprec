# Algorithm M3 (`dd²`), scaled as in `m3_normal.g`, in the case where the
# rounding of step 2 underflows: the binary64 operand 2 x_h x_l + c_l1 is
# below 2^-1022 in magnitude, so c may differ from it by up to 2^-1075 (half
# the smallest subnormal) instead of a relative 2^-53. With |x_h| >= 2^-480,
# s^2 >= 2^-960, so after scaling by s^-2: |v| <= 2^-62 and |c - v| <= 2^-115.
# `rocq/M3.v` proves these two facts; here c is otherwise unconstrained.

@rnd = float<53,ne>;

m   = rnd(mx);
ml  = mx - m;
ch  = rnd(m * m);
cl1 = m * m - ch;
v   = 2 * m * ml + cl1;

{ |m| in [1,2] /\ |v| <= 1b-62 /\ |c - v| <= 1b-115
  -> ch + c -/ mx * mx in [-3b-106,3b-106]
  /\ |c| <= 1b-50
  /\ ch in [1,4] }

ch + c - mx * mx -> - (ml * ml) + (c - v);
