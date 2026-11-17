#!/usr/bin/env bash
# Usage: [TLC_HEAP=3g] [TLC_WORKERS=6] [HEAVY=1] run.sh /path/to/tla2tools.jar [filter-substring]
# Memory: TLC's JVM heap is capped at TLC_HEAP (default 3g) and it runs TLC_WORKERS threads
# (default 6); the state queue spills to disk inside a temp dir that is always removed on exit.
# Configs marked "heavy" (millions of states, GBs of spill) are skipped unless HEAVY=1 -- run those
# on a big-memory node, not on a workstation.
# Generates one wrapper module + cfg per configuration in a temp dir and runs TLC.
# "expect" is PASS (all invariants hold, no deadlock) or FAIL (the check must fire).
set -u
JAR=${1:?path to tla2tools.jar}
FILTER=${2:-}
SPEC_DIR=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT INT TERM
HEAP=${TLC_HEAP:-3g}
WORKERS=${TLC_WORKERS:-6}
cp "$SPEC_DIR/multicast_barrier.tla" "$WORK/"
cd "$WORK"

INVS="SlotBound TotalBound StartSafety EndSafety WarSafe CtrClean Quiescent TypeOK"
pass=0; bad=0

# name  expect  R  B  Cap  InOrder  kernels  shared  delta  signrel  relrel  notake  nonfifo  piggy
run() {
  local name=$1 expect=$2 R=$3 B=$4 CAP=$5 INORD=$6 KERN=$7 SH=$8 DELTA=$9 SIGN=${10} REL=${11} NOTAKE=${12} NONFIFO=${13} PIGGY=${14:-FALSE} HEAVYCFG=${15:-}
  [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && return
  if [[ "$HEAVYCFG" == heavy && "${HEAVY:-0}" != 1 ]]; then
    printf "%-34s skipped (heavy; HEAVY=1 to run, needs a big-memory node)\n" "$name"; return
  fi
  cat > "MC_$name.tla" <<EOF
---- MODULE MC_$name ----
EXTENDS multicast_barrier
KernelsDef == <<$KERN>>
DeltaDef == $DELTA
====
EOF
  cat > "MC_$name.cfg" <<EOF
SPECIFICATION Spec
CONSTANTS
  R = $R
  B = $B
  Cap = $CAP
  InOrder = $INORD
  M_SharedSlots = $SH
  M_NumBlocksDelta <- DeltaDef
  M_SignInRelaxed = $SIGN
  M_ReleaseRelaxed = $REL
  M_NoTake = $NOTAKE
  M_NonFifo = $NONFIFO
  M_StartTakeAtEnd = $PIGGY
  Kernels <- KernelsDef
INVARIANTS $INVS
EOF
  local out
  # java.io.tmpdir = our own temp dir: TLC unpacks the standard modules (Integers.tla, ...) there
  # instead of into a shared /tmp, so concurrent runs and other users do not interfere.
  out=$(java -Xmx$HEAP -XX:MaxDirectMemorySize=512m -XX:+UseParallelGC -Djava.io.tmpdir="$WORK" -cp "$JAR" tlc2.TLC -workers $WORKERS -cleanup "MC_$name" 2>&1)
  local states verdict
  # the final summary line ("N states generated, M distinct states found, 0 states left on queue.")
  states=$(echo "$out" | grep -E "^[0-9,]+ states generated, [0-9,]+ distinct states found" | tail -1 | sed -E 's/^[0-9,]+ states generated, ([0-9,]+) distinct.*/\1/' | tr -d ',')
  if echo "$out" | grep -q "No error has been found"; then verdict=PASS
  elif echo "$out" | grep -q "Deadlock reached"; then verdict="FAIL(deadlock)"
  elif echo "$out" | grep -qE "Invariant [A-Za-z]+ is violated"; then
    verdict="FAIL($(echo "$out" | grep -oE "Invariant [A-Za-z]+ is violated" | head -1 | awk '{print $2}'))"
  else verdict="ERROR"; echo "$out" | tail -15; fi
  local ok=no
  [[ "$expect" == PASS && "$verdict" == PASS ]] && ok=yes
  [[ "$expect" == FAIL && "$verdict" == FAIL* ]] && ok=yes
  [[ $ok == yes ]] && pass=$((pass+1)) || bad=$((bad+1))
  printf "%-34s expect=%-4s got=%-24s states=%-9s %s\n" "$name" "$expect" "$verdict" "${states:--}" "$([[ $ok == yes ]] && echo ok || echo '<== UNEXPECTED')"
}

SE='"SE","SE","SE"'
# ---- the design as written -------------------------------------------------
run base_SE_r2b2        PASS 2 2 2 TRUE  "$SE"                       FALSE 0 FALSE FALSE FALSE FALSE
run base_SE_r3b2 PASS 3 2 2 TRUE '"SE","SE"' FALSE 0 FALSE FALSE FALSE FALSE FALSE heavy
run base_SE_r2b3 PASS 2 3 3 TRUE '"SE","SE"' FALSE 0 FALSE FALSE FALSE FALSE FALSE heavy
run endonly_EEE_r2b2    PASS 2 2 2 TRUE  '"E","E","E","E"'           FALSE 0 FALSE FALSE FALSE FALSE
run endonly_EEE_r3b2 PASS 3 2 2 TRUE '"E","E","E"' FALSE 0 FALSE FALSE FALSE FALSE FALSE heavy
run startonly_SSS_r2b2  PASS 2 2 2 TRUE  '"S","S","S","S"'           FALSE 0 FALSE FALSE FALSE FALSE
run mixed_r2b2          PASS 2 2 2 TRUE  '"SE","E","S","SE","E","S"' FALSE 0 FALSE FALSE FALSE FALSE
run mixed_r3b2 PASS 3 2 2 TRUE '"S","SE","E","S"' FALSE 0 FALSE FALSE FALSE FALSE FALSE heavy
run single_block_E      PASS 3 1 1 TRUE  '"E","E","E"'               FALSE 0 FALSE FALSE FALSE FALSE
# ---- residency: the grid needs no co-residency (in-order dispatch) ---------
run cap1_inorder_SE     PASS 2 3 1 TRUE  '"SE","SE","E"'             FALSE 0 FALSE FALSE FALSE FALSE
run cap1_inorder_r3     PASS 3 2 1 TRUE  '"SE","E","S"'              FALSE 0 FALSE FALSE FALSE FALSE
# ---- assumptions the design relies on (expected to FAIL when removed) ------
run needs_inorder_dispatch FAIL 2 2 1 FALSE '"SE","SE"'              FALSE 0 FALSE FALSE FALSE FALSE
# ---- mutants: each design choice removed must be caught ---------------------
run mut_shared_slots    FAIL 2 2 2 TRUE  "$SE"                       TRUE  0 FALSE FALSE FALSE FALSE
run mut_shared_slots_E  FAIL 2 2 2 TRUE  '"SE","E","SE"'             TRUE  0 FALSE FALSE FALSE FALSE
run mut_numblocks_low   FAIL 2 2 2 TRUE  "$SE"                       FALSE -1 FALSE FALSE FALSE FALSE
run mut_numblocks_high  FAIL 2 2 2 TRUE  "$SE"                       FALSE 1 FALSE FALSE FALSE FALSE
run mut_signin_relaxed  FAIL 2 2 2 TRUE  "$SE"                       FALSE 0 TRUE  FALSE FALSE FALSE
run mut_release_relaxed FAIL 2 2 2 TRUE  "$SE"                       FALSE 0 FALSE TRUE  FALSE FALSE
run mut_no_take         FAIL 2 2 2 TRUE  "$SE"                       FALSE 0 FALSE FALSE TRUE FALSE

# ---- landing order: the design does not need in-order landing ---------------
run nonfifo_EEE_r2b2    PASS 2 2 2 TRUE  '"E","E","E","E"'           FALSE 0 FALSE FALSE FALSE TRUE
run nonfifo_SE_r2b2     PASS 2 2 2 TRUE  "$SE"                       FALSE 0 FALSE FALSE FALSE TRUE
run nonfifo_mixed_r2b2  PASS 2 2 2 TRUE  '"SE","E","S","SE","E"'    FALSE 0 FALSE FALSE FALSE TRUE
# ---- call-site obligation: a start barrier before writes that follow a reader
run war_start_protects  PASS 2 2 2 TRUE  '"SE","L","SE","S","SE","E"' FALSE 0 FALSE FALSE FALSE FALSE
run war_E_after_L       FAIL 2 2 2 TRUE  '"E","L","E"'               FALSE 0 FALSE FALSE FALSE FALSE
run war_E_after_S       FAIL 2 2 2 TRUE  '"S","E"'                   FALSE 0 FALSE FALSE FALSE FALSE
run war_L_between_EE    FAIL 2 2 2 TRUE  '"SE","L","E"'              FALSE 0 FALSE FALSE FALSE FALSE

# ---- proposed optimization: take the start arrivals in the end barrier ------
run piggy_SE_r2b2       PASS 2 2 2 TRUE  "$SE"                       FALSE 0 FALSE FALSE FALSE FALSE TRUE
run piggy_SE_r3b1_k3    PASS 3 1 1 TRUE  '"SE","SE","SE"'            FALSE 0 FALSE FALSE FALSE FALSE TRUE
run piggy_SE_r3b2_k1    PASS 3 2 2 TRUE  '"SE"'                      FALSE 0 FALSE FALSE FALSE FALSE TRUE
run piggy_SE_r3b2 PASS 3 2 2 TRUE '"SE","SE"' FALSE 0 FALSE FALSE FALSE FALSE TRUE heavy
run piggy_mixed_r2b2    PASS 2 2 2 TRUE  '"SE","E","S","SE","L","SE"' FALSE 0 FALSE FALSE FALSE FALSE TRUE
run piggy_cap1          PASS 2 3 1 TRUE  '"SE","SE","S"'             FALSE 0 FALSE FALSE FALSE FALSE TRUE
run piggy_b1            PASS 3 1 1 TRUE  '"SE","SE","SE"'            FALSE 0 FALSE FALSE FALSE FALSE TRUE
run piggy_notake        FAIL 2 2 2 TRUE  "$SE"                       FALSE 0 FALSE FALSE TRUE  FALSE TRUE

echo "(configs above run with M_NonFifo=FALSE unless named nonfifo)"
echo "---- $pass as expected, $bad unexpected"
rm -rf "$WORK"
[[ $bad -eq 0 ]]
