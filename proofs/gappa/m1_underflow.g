# Algorithm M1 (`dd · f64`), scaled as in `m1_normal.g`, in the case where
# the rounding of step 2 underflows (|c_l1 + x_l y| < 2^-1022 in binary64);
# after scaling by (s t)^-1 >= 2^-960: |v| <= 2^-62, |c - v| <= 2^-115
# (`rocq/M1.v`).

@rnd = float<53,ne>;

m   = rnd(mx);
ml  = mx - m;
my  = rnd(my_);
ch  = rnd(m * my);
cl1 = m * my - ch;
v   = cl1 + ml * my;

{ |m| in [1,2] /\ |my| in [1,2] /\ |v| <= 1b-62 /\ |c - v| <= 1b-115
  -> ch + c -/ mx * my in [-0x2.000000001p-106,0x2.000000001p-106]
  /\ |c| <= 1b-50 }

ch + c - mx * my -> c - v;

# Split the product of the scaled operands into 64 intervals; with Gappa
# 1.8.3 this reaches 2u^2 (1 + 2^-37), not the 2u^2 of the literature (see
# README.md).
$ m * my in 64;
