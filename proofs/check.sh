#!/bin/sh
# Recheck the machine-checked proofs of xingo-extprec (see README.md).
#
#   1. Gappa proves the two scaled cases of Algorithms M3 and M1, writing Rocq
#      certificates; Rocq checks them.
#   2. Rocq checks Model.v, M3.v and M1.v (binary64 theorems for M3 and M1)
#      and C5.v (the bound of step C5 of dot2_err), and that the theorems depend on no
#      assumption besides the standard axioms of Rocq's real numbers.
#   3. Every mutant (a wrong constant or a wrong step) must be REJECTED.
#
# Exit status 0 only if every proof is accepted and every mutant rejected.
# Needs `gappa` (1.8.x) and `rocq` (9.x) with Flocq and the Gappa support
# library on PATH, or an opam switch named by XINGO_PROOFS_SWITCH (default
# `xingo-proofs`). Scratch files go to proofs/_build (git-ignored).
set -u

here=$(cd "$(dirname "$0")" && pwd)
build="$here/_build"

if command -v gappa >/dev/null 2>&1 && command -v rocq >/dev/null 2>&1; then
  RUN=""
else
  RUN="opam exec --switch=${XINGO_PROOFS_SWITCH:-xingo-proofs} --"
fi

start=$(date +%s)
failures=0
ok()   { echo "  ok    $1"; }
bad()  { echo "  FAIL  $1"; failures=$((failures + 1)); }

rm -rf "$build"
mkdir -p "$build"
echo "== tools"
$RUN gappa --version 2>&1 | head -n 1
$RUN rocq --version 2>&1 | head -n 1

compile() { # compile <dir> <file.v>: Rocq-check one file with logical root XP
  (cd "$1" && $RUN rocq compile -R . XP "$2")
}

gappa_cert() { # gappa_cert <dir> <name> <script.g>: certificate + check
  $RUN gappa -Bcoq "$3" > "$1/$2.v" 2> "$1/$2.gappa.log" &&
    compile "$1" "$2.v" > "$1/$2.rocq.log" 2>&1 &&
    sh "$here/tools/mkunfold.sh" "$1/$2.v" "$2" &&
    compile "$1" "$2_unfold.v" >> "$1/$2.rocq.log" 2>&1
}

echo "== proofs"
for g in m3_normal m3_underflow m1_normal m1_underflow; do
  if gappa_cert "$build" "$g" "$here/gappa/$g.g"; then
    ok "gappa/$g.g (Gappa proof, certificate checked by Rocq)"
  else
    bad "gappa/$g.g"; cat "$build/$g.gappa.log" "$build/$g.rocq.log" 2>/dev/null
  fi
done

cp "$here"/rocq/*.v "$build/"
allowed='ClassicalDedekindReals.sig_not_dec ClassicalDedekindReals.sig_forall_dec FunctionalExtensionality.functional_extensionality_dep Classical_Prop.classic classic'
for v in Model M3 M1 C5; do
  if compile "$build" "$v.v" > "$build/$v.log" 2>&1; then
    extra=""
    for a in $(sed -n 's/^\([A-Za-z_][A-Za-z0-9_.]*\) :.*/\1/p' "$build/$v.log"); do
      case " $allowed " in *" $a "*) ;; *) extra="$extra $a" ;; esac
    done
    if [ -z "$extra" ]; then ok "rocq/$v.v"; else bad "rocq/$v.v depends on:$extra"; fi
  else
    bad "rocq/$v.v"; cat "$build/$v.log"
  fi
done

# Mutants: <name>|<file>|<sed expression>|<what is wrong>
echo "== mutants (each must be rejected)"
mutants='m3-bound-2.5u2|gappa/m3_normal.g|s/\[-3b-106,3b-106\]/[-5b-107,5b-107]/|M3: claims 2.5u^2 instead of 3u^2
m3-no-factor-2|gappa/m3_normal.g|s/^v   = 2 \* m \* ml + cl1;/v   = m * ml + cl1;/;s/^ch + c - mx \* mx -> - (ml \* ml) + (c - v);/ch + c - mx * mx -> - (ml * ml) - m * ml + (c - v);/|M3 step 2: c = RN(x_h x_l + c_l1), the factor 2 dropped (hint adjusted to stay true)
m3-no-factor-2-stale-hint|gappa/m3_normal.g|s/^v   = 2 \* m \* ml + cl1;/v   = m * ml + cl1;/|M3 step 2 without the factor 2, hint left as is (now false): Gappa assumes hints, the Rocq side must refute
m1-bound-1.5u2|gappa/m1_normal.g|s/\[-0x2.000000001p-106,0x2.000000001p-106\]/[-0x1.8p-106,0x1.8p-106]/|M1: claims 1.5u^2 instead of 2u^2 (1 + 2^-37)
m1-no-cl1|gappa/m1_normal.g|s/^v   = cl1 + ml \* my;/v   = ml * my;/;s/^ch + c - mx \* my -> c - v;/ch + c - mx * my -> (c - v) - cl1;/|M1 step 2: c_l3 = RN(x_l y), the error term c_l1 of 2Prod dropped (hint adjusted)
c5-inflate-1|rocq/C5.v|s/^Definition INFLATE := 1 + pow2 (-50)\./Definition INFLATE := 1./|C5: INFLATE = 1 instead of 1 + 8u
c5-underflow-0|rocq/C5.v|s/^Definition UNDERFLOW := 6 \* pow2 (-1022)\./Definition UNDERFLOW := 0./|C5: UNDERFLOW = 0 instead of 3 eta / u
c5-terms-n+2|rocq/C5.v|s/let terms := RN (IZR (n + 3)) in/let terms := RN (IZR (n + 2)) in/|C5: a = (n + 2)u instead of (n + 3)u'
echo "$mutants" | while IFS='|' read -r name file expr what; do
  mdir="$build/mutant-$name"
  mkdir -p "$mdir"
  base=$(basename "$file")
  sed "$expr" "$here/$file" > "$mdir/$base"
  if cmp -s "$here/$file" "$mdir/$base"; then
    echo "  FAIL  $name: the mutation did not apply ($what)"; echo x >> "$build/mutant-failures"; continue
  fi
  case "$file" in
    gappa/*)
      # A Gappa rewriting hint is a hypothesis of the certificate (it is
      # discharged by `ring` in M3.v), so the mutant goes through the whole
      # chain: Gappa, the certificate, and the binary64 theorem.
      cert=$(basename "$base" .g)
      if ! $RUN gappa "$mdir/$base" > "$mdir/out.log" 2>&1; then
        echo "  ok    $name rejected by Gappa ($what)"
        sed 's/^/          gappa: /' "$mdir/out.log" | grep -v 'not trivially equal\|Difference:' | head -n 3
      else
        for f in "$build"/*.v; do
          case "$(basename "$f")" in "$cert.v"|"${cert}_unfold.v") ;; *) cp "$f" "$mdir/" ;; esac
        done
        for f in "$build"/*.vo; do
          case "$(basename "$f")" in "$cert.vo"|"${cert}_unfold.vo"|M3.vo|M1.vo) ;; *) cp "$f" "$mdir/" ;; esac
        done
        glue=M3.v; case "$cert" in m1_*) glue=M1.v ;; esac
        if gappa_cert "$mdir" "$cert" "$mdir/$base" && compile "$mdir" "$glue" > "$mdir/glue.log" 2>&1; then
          echo "  FAIL  $name ACCEPTED ($what)"; echo x >> "$build/mutant-failures"
        else
          echo "  ok    $name accepted by Gappa alone, rejected by Rocq ($what)"
          grep -h -m 2 -E '^(File|Error)' "$mdir/glue.log" "$mdir/$cert.rocq.log" 2>/dev/null | sed 's/^/          rocq: /'
        fi
      fi ;;
    rocq/*)
      for f in "$build"/*.vo "$build"/*.v; do
        case "$(basename "$f")" in "$base"|"${base}o") ;; *) cp "$f" "$mdir/" ;; esac
      done
      if compile "$mdir" "$base" > "$mdir/out.log" 2>&1; then
        echo "  FAIL  $name ACCEPTED ($what)"; echo x >> "$build/mutant-failures"
      else
        echo "  ok    $name rejected ($what)"
        grep -m 2 -E '^(File|Error)' "$mdir/out.log" | sed 's/^/          rocq: /'
      fi ;;
  esac
done
if [ -f "$build/mutant-failures" ]; then
  failures=$((failures + $(wc -l < "$build/mutant-failures")))
fi

echo "== counterexamples to the mutated statements (exact rationals)"
if command -v python3 >/dev/null 2>&1 && python3 -I -c 'import math; math.fma' 2>/dev/null; then
  python3 -I "$here/mutants/counterexamples.py" || failures=$((failures + 1))
else
  echo "  skipped: needs python3 >= 3.13 (math.fma)"
fi

echo "== $(( $(date +%s) - start )) s"
if [ "$failures" -ne 0 ]; then
  echo "FAILED: $failures check(s)"
  exit 1
fi
echo "all proofs accepted, all mutants rejected"
