# Algorithm M1 (`dd · f64`, DWTimesFP3, `impl Mul<f64> for DoubleDouble`,
# Chapter 11.2), scaled to the binades of both operands, in the case where
# step 2 does not underflow. As for M3 (`m3_normal.g`), `rocq/M1.v` reduces
# the binary64 statement to this one: with s = 2^(mag(x_h) - 1) and
# t = 2^(mag(y) - 1), m = x_h / s and my = y / t lie in [1, 2] in magnitude.
#
# mx = x / s (the exact double-double value, scaled); ml = x_l / s.

@rnd = float<53,ne>;

m   = rnd(mx);                 # x_h = RN(x_h + x_l): the operand is normalized
ml  = mx - m;                  # x_l
my  = rnd(my_);                # y, a binary64 number
ch  = rnd(m * my);             # step 1, P1: c_h = RN(x_h y)
cl1 = m * my - ch;             # step 1, P2: exact (`rocq/M1.v`, Flocq mult_error_FLT)
v   = cl1 + ml * my;           # step 2: the exact operand of the FMA
c   = rnd(v);                  # step 2: c_l3 = RN(c_l1 + x_l y), no underflow

{ |m| in [1,2] /\ |my| in [1,2]
  -> ch + c -/ mx * my in [-0x2.000000001p-106,0x2.000000001p-106]
  /\ |c| <= 1b-50 }

# The error of c_h + c_l3 is the rounding error of step 2.
ch + c - mx * my -> c - v;

# Split the product of the scaled operands into 64 intervals; with Gappa
# 1.8.3 this reaches 2u^2 (1 + 2^-37), not the 2u^2 of the literature (see
# README.md).
$ m * my in 64;
