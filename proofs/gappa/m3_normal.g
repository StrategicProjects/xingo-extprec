# Algorithm M3 (`dd²`, `DoubleDouble::square`, Chapter 11.2), scaled to the
# binade of the operand and in the case where step 2 does not underflow.
#
# The Rocq file `rocq/M3.v` reduces the binary64 statement to this one: with
# s = 2^(mag(x_h) - 1), m = x_h / s satisfies 1 <= |m| <= 2, every rounding of
# steps 1 and 3 is a binary64 rounding of a normal number, hence s^2 times a
# rounding to 53 bits with an unbounded exponent of the scaled value, and the
# rounding of step 2 is either that (this file) or an underflow
# (`m3_underflow.g`). Variable names follow `src/dd.rs`.
#
# mx = x / s (the exact double-double value, scaled); m = x_h / s; ml = x_l / s.

@rnd = float<53,ne>;

m   = rnd(mx);                 # x_h = RN(x_h + x_l): the operand is normalized
ml  = mx - m;                  # x_l
ch  = rnd(m * m);              # step 1, P1: c_h = RN(x_h * x_h)
cl1 = m * m - ch;              # step 1, P2: exact (`rocq/M3.v`, Flocq mult_error_FLT)
v   = 2 * m * ml + cl1;        # step 2: the exact operand of the FMA
c   = rnd(v);                  # step 2: c = RN(2 x_h x_l + c_l1), no underflow

{ |m| in [1,2]
  -> ch + c -/ mx * mx in [-3b-106,3b-106]
  /\ |c| <= 1b-50
  /\ ch in [1,4] }

# The error of c_h + c is the rounding error of step 2 minus x_l^2.
ch + c - mx * mx -> - (ml * ml) + (c - v);
