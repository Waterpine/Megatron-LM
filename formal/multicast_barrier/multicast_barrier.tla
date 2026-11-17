------------------------ MODULE multicast_barrier ------------------------
(***************************************************************************)
(* Model of the once-per-rank symmetric-memory barriers of                  *)
(* sidsingh-nvidia/Megatron-LM PR #23 (barrier.py):                         *)
(*   symm_mem_sync_start  -- block 0 arrives, every block spins on peers,   *)
(*                           the last block to sign in takes the arrivals   *)
(*   symm_mem_sync_end    -- last block to sign in arrives (release) and    *)
(*                           takes one arrival per rank (acquire)           *)
(*                                                                          *)
(* One rank runs a stream of kernels (Kernels = "SE" | "E" | "S" each).    *)
(* Each kernel launches B blocks; kernel k+1 starts on a rank only after    *)
(* kernel k has completed on that rank (no PDL).  multimem.red arrivals are *)
(* asynchronous: they sit in flight and land on each rank's copy at         *)
(* independent, arbitrary times, even after the issuing kernel finished.    *)
(*                                                                          *)
(* Memory ordering is modelled operationally, with the weakest behaviour    *)
(* allowed (no spontaneous propagation): a block's data write has stage     *)
(*   1 = issued, 2 = device visible (acq_rel sign-in of its block),         *)
(*   3 = system visible (sys-scope release by the last block).              *)
(* The sign-in flush and the release are the load-bearing assumptions; the  *)
(* M_* flags weaken them (or other design choices) to show the checks bite. *)
(*                                                                          *)
(* Landing order: increments of one kind (start / end) from one rank to one *)
(* copy land in issue order (successive kernels; coherence + kernel          *)
(* boundary).  A start and an end increment of the same kernel come from    *)
(* different threads and may land in either order.  Data written before a   *)
(* release increment is visible on a rank only once that increment has      *)
(* LANDED there (lnd); a relaxed increment gives no such guarantee.         *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    R,                 \* ranks
    B,                 \* blocks per rank
    Kernels,           \* sequence of "SE" | "E" | "S" | "L"
    Cap,               \* resident-block capacity per rank (1..B)
    InOrder,           \* TRUE: blocks are dispatched in index order
    M_SharedSlots,     \* TRUE: start and end arrivals share one slot array
    M_NumBlocksDelta,  \* num_blocks passed to the barrier = B + delta
    M_SignInRelaxed,   \* TRUE: sign-in does not flush the block's writes
    M_ReleaseRelaxed,  \* TRUE: last block's arrival is relaxed, not release
    M_NoTake,          \* TRUE: arrivals are never consumed
    M_NonFifo,         \* TRUE: same-kind increments may land out of issue order
    M_StartTakeAtEnd   \* TRUE: "SE" kernels take their start arrivals in the end
                       \*       barrier's last block; the start barrier has no sign-in

Ranks == 0..(R-1)
Blks  == 0..(B-1)
N     == Len(Kernels)
NB    == B + M_NumBlocksDelta
Kinds == {"S", "E"}

HasStart(k) == Kernels[k] \in {"SE", "S"}
HasEnd(k)   == Kernels[k] \in {"SE", "E"}
\* "SE"/"E" kernels write data to peers; "S" (a pull, like the MoE combine) and
\* "L" (a local consumer, no barrier at all) only read the symmetric buffer.
Writes(k)   == Kernels[k] \in {"SE", "E"}
Reads(k)    == Kernels[k] \in {"S", "L"}

\* Physical slot array used by a logical barrier kind.
PK(x) == IF M_SharedSlots THEN "E" ELSE x

Phases == {"new", "spin", "signS", "resetS", "takeS", "body",
           "signE", "resetE", "relE", "takeE", "done"}
PostSpin == {"signS", "resetS", "takeS", "body",
             "signE", "resetE", "relE", "takeE", "done"}

VARIABLES
    kid,     \* kid[r]: index of the kernel rank r is running (N+1 = finished)
    ph,      \* ph[r][b]: phase of block b of rank r
    seen,    \* seen[r][b]: peers whose start slot this block saw nonzero
    tk,      \* tk[r][b]: <<kind, q>> slots the last block still has to take from
    ctr,     \* ctr[r][x]: local sign-in counters (x in Kinds)
    slot,    \* slot[r][x][q]: value of slot q in rank r's copy
    fl,      \* fl[s][x][d]: multicast increments from s in flight to d
    wv,      \* wv[k][r][b]: visibility stage of block b's writes, kernel k
    sa,      \* ghost: sa[q] = kernels whose start arrival q has issued
    cdone,   \* ghost: cdone[r] = kernels rank r has completed
    lnd      \* ghost: lnd[d][s] = <<k, kind>> increments from s landed on d

vars == <<kid, ph, seen, tk, ctr, slot, fl, wv, sa, cdone, lnd>>

Init ==
    /\ kid   = [r \in Ranks |-> 1]
    /\ ph    = [r \in Ranks |-> [b \in Blks |-> "new"]]
    /\ seen  = [r \in Ranks |-> [b \in Blks |-> {}]]
    /\ tk    = [r \in Ranks |-> [b \in Blks |-> {}]]
    /\ ctr   = [r \in Ranks |-> [x \in Kinds |-> 0]]
    /\ slot  = [r \in Ranks |-> [x \in Kinds |-> [q \in Ranks |-> 0]]]
    /\ fl    = [s \in Ranks |-> [x \in Kinds |-> [d \in Ranks |-> <<>>]]]
    /\ wv    = [k \in 1..N |-> [r \in Ranks |-> [b \in Blks |-> 0]]]
    /\ sa    = [r \in Ranks |-> {}]
    /\ cdone = [r \in Ranks |-> {}]
    /\ lnd   = [d \in Ranks |-> [q \in Ranks |-> {}]]

Resident(r) == Cardinality({b \in Blks : ph[r][b] \notin {"new", "done"}})

\* Multicast add of 1 to slot r on every rank (including r): in flight.
Arrive(r, x) ==
    [fl EXCEPT ![r][PK(x)] =
        [d \in Ranks |-> Append(fl[r][PK(x)][d], [k |-> kid[r], kind |-> x])]]

RemoveAt(sq, i) == SubSeq(sq, 1, i - 1) \o SubSeq(sq, i + 1, Len(sq))

(* A block is dispatched.  Block 0 of a kernel with a start barrier arrives. *)
Dispatch(r, b) ==
    /\ kid[r] <= N
    /\ ph[r][b] = "new"
    /\ Resident(r) < Cap
    /\ (InOrder => \A c \in Blks : c < b => ph[r][c] /= "new")
    /\ IF HasStart(kid[r])
         THEN /\ ph' = [ph EXCEPT ![r][b] = "spin"]
              /\ IF b = 0
                   THEN /\ fl' = Arrive(r, "S")
                        /\ sa' = [sa EXCEPT ![r] = @ \cup {kid[r]}]
                   ELSE /\ UNCHANGED <<fl, sa>>
         ELSE /\ ph' = [ph EXCEPT ![r][b] = "body"]
              /\ UNCHANGED <<fl, sa>>
    /\ UNCHANGED <<kid, seen, tk, ctr, slot, wv, cdone, lnd>>

(* Start barrier: spin (read only) until a peer's start slot is nonzero.    *)
Spin(r, b) ==
    /\ ph[r][b] = "spin"
    /\ \E p \in (Ranks \ {r}) \ seen[r][b] :
         /\ slot[r][PK("S")][p] > 0
         /\ seen' = [seen EXCEPT ![r][b] = @ \cup {p}]
    /\ UNCHANGED <<kid, ph, tk, ctr, slot, fl, wv, sa, cdone, lnd>>

SpinDone(r, b) ==
    /\ ph[r][b] = "spin"
    /\ seen[r][b] = Ranks \ {r}
    /\ ph' = [ph EXCEPT ![r][b] =
                 IF M_StartTakeAtEnd /\ HasEnd(kid[r]) THEN "body" ELSE "signS"]
    /\ UNCHANGED <<kid, seen, tk, ctr, slot, fl, wv, sa, cdone, lnd>>

SignS(r, b) ==
    /\ ph[r][b] = "signS"
    /\ ctr' = [ctr EXCEPT ![r]["S"] = @ + 1]
    /\ ph' = [ph EXCEPT ![r][b] = IF ctr[r]["S"] + 1 = NB THEN "resetS" ELSE "body"]
    /\ UNCHANGED <<kid, seen, tk, slot, fl, wv, sa, cdone, lnd>>

ResetS(r, b) ==
    /\ ph[r][b] = "resetS"
    /\ ctr' = [ctr EXCEPT ![r]["S"] = 0]
    /\ tk'  = [tk EXCEPT ![r][b] = {"S"} \X Ranks]
    /\ ph'  = [ph EXCEPT ![r][b] = "takeS"]
    /\ UNCHANGED <<kid, seen, slot, fl, wv, sa, cdone, lnd>>

TakeS(r, b) ==
    /\ ph[r][b] = "takeS"
    /\ \E t \in tk[r][b] :
         /\ slot[r][PK(t[1])][t[2]] > 0
         /\ slot' = [slot EXCEPT ![r][PK(t[1])][t[2]] = IF M_NoTake THEN @ ELSE @ - 1]
         /\ tk'   = [tk EXCEPT ![r][b] = @ \ {t}]
    /\ UNCHANGED <<kid, ph, seen, ctr, fl, wv, sa, cdone, lnd>>

TakeSDone(r, b) ==
    /\ ph[r][b] = "takeS"
    /\ tk[r][b] = {}
    /\ ph' = [ph EXCEPT ![r][b] = "body"]
    /\ UNCHANGED <<kid, seen, tk, ctr, slot, fl, wv, sa, cdone, lnd>>

(* Kernel body: this block writes data to its peers (stage 1 = issued).     *)
Body(r, b) ==
    /\ ph[r][b] = "body"
    /\ wv' = IF Writes(kid[r]) THEN [wv EXCEPT ![kid[r]][r][b] = 1] ELSE wv
    /\ ph' = [ph EXCEPT ![r][b] = IF HasEnd(kid[r]) THEN "signE" ELSE "done"]
    /\ UNCHANGED <<kid, seen, tk, ctr, slot, fl, sa, cdone, lnd>>

(* End barrier sign-in: acq_rel at gpu scope flushes the block's writes.    *)
SignE(r, b) ==
    /\ ph[r][b] = "signE"
    /\ ctr' = [ctr EXCEPT ![r]["E"] = @ + 1]
    /\ wv'  = IF M_SignInRelaxed THEN wv ELSE [wv EXCEPT ![kid[r]][r][b] = 2]
    /\ ph'  = [ph EXCEPT ![r][b] = IF ctr[r]["E"] + 1 = NB THEN "resetE" ELSE "done"]
    /\ UNCHANGED <<kid, seen, tk, slot, fl, sa, cdone, lnd>>

ResetE(r, b) ==
    /\ ph[r][b] = "resetE"
    /\ ctr' = [ctr EXCEPT ![r]["E"] = 0]
    /\ ph'  = [ph EXCEPT ![r][b] = "relE"]
    /\ UNCHANGED <<kid, seen, tk, slot, fl, wv, sa, cdone, lnd>>

(* Last block: multimem.red.release.sys -- every write of this rank's blocks *)
(* that is device visible (plus its own) becomes system visible before the  *)
(* arrival can land anywhere.                                               *)
RelE(r, b) ==
    /\ ph[r][b] = "relE"
    /\ fl' = Arrive(r, "E")
    /\ wv' = IF M_ReleaseRelaxed THEN wv
             ELSE [wv EXCEPT ![kid[r]][r] =
                     [c \in Blks |-> IF c = b \/ wv[kid[r]][r][c] >= 2 THEN 3
                                     ELSE wv[kid[r]][r][c]]]
    /\ tk' = [tk EXCEPT ![r][b] =
                 ({"E"} \X Ranks) \cup
                 (IF M_StartTakeAtEnd /\ HasStart(kid[r]) THEN {"S"} \X Ranks ELSE {})]
    /\ ph' = [ph EXCEPT ![r][b] = "takeE"]
    /\ UNCHANGED <<kid, seen, ctr, slot, sa, cdone, lnd>>

TakeE(r, b) ==
    /\ ph[r][b] = "takeE"
    /\ \E t \in tk[r][b] :
         /\ slot[r][PK(t[1])][t[2]] > 0
         /\ slot' = [slot EXCEPT ![r][PK(t[1])][t[2]] = IF M_NoTake THEN @ ELSE @ - 1]
         /\ tk'   = [tk EXCEPT ![r][b] = @ \ {t}]
    /\ UNCHANGED <<kid, ph, seen, ctr, fl, wv, sa, cdone, lnd>>

TakeEDone(r, b) ==
    /\ ph[r][b] = "takeE"
    /\ tk[r][b] = {}
    /\ ph' = [ph EXCEPT ![r][b] = "done"]
    /\ UNCHANGED <<kid, seen, tk, ctr, slot, fl, wv, sa, cdone, lnd>>

(* A multicast increment lands on one rank's copy.                          *)
Land(s, x, d) ==
    /\ \E i \in 1..Len(fl[s][x][d]) :
         /\ (M_NonFifo \/ \A j \in 1..(i-1) : fl[s][x][d][j].kind /= fl[s][x][d][i].kind)
         /\ fl'   = [fl EXCEPT ![s][x][d] = RemoveAt(@, i)]
         /\ slot' = [slot EXCEPT ![d][x][s] = @ + 1]
         /\ lnd'  = [lnd EXCEPT ![d][s] = @ \cup {<<fl[s][x][d][i].k, fl[s][x][d][i].kind>>}]
    /\ UNCHANGED <<kid, ph, seen, tk, ctr, wv, sa, cdone>>

(* The kernel completes when all its blocks have exited.                    *)
KernelDone(r) ==
    /\ kid[r] <= N
    /\ \A b \in Blks : ph[r][b] = "done"
    /\ kid'   = [kid EXCEPT ![r] = @ + 1]
    /\ ph'    = [ph EXCEPT ![r] = [b \in Blks |-> "new"]]
    /\ seen'  = [seen EXCEPT ![r] = [b \in Blks |-> {}]]
    /\ tk'    = [tk EXCEPT ![r] = [b \in Blks |-> {}]]
    /\ cdone' = [cdone EXCEPT ![r] = @ \cup {kid[r]}]
    /\ UNCHANGED <<ctr, slot, fl, wv, sa, lnd>>

AllDone ==
    /\ \A r \in Ranks : kid[r] = N + 1
    /\ \A s \in Ranks, x \in Kinds, d \in Ranks : fl[s][x][d] = <<>>

Next ==
    \/ \E r \in Ranks, b \in Blks :
         \/ Dispatch(r, b) \/ Spin(r, b) \/ SpinDone(r, b)
         \/ SignS(r, b) \/ ResetS(r, b) \/ TakeS(r, b) \/ TakeSDone(r, b)
         \/ Body(r, b)
         \/ SignE(r, b) \/ ResetE(r, b) \/ RelE(r, b)
         \/ TakeE(r, b) \/ TakeEDone(r, b)
    \/ \E s \in Ranks, x \in Kinds, d \in Ranks : Land(s, x, d)
    \/ \E r \in Ranks : KernelDone(r)
    \/ (AllDone /\ UNCHANGED vars)

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Properties                                                               *)
(***************************************************************************)

\* The design note: slots hold 0, 1 or 2 and are consumed back to zero.
SlotBound == \A r \in Ranks, x \in Kinds, q \in Ranks : slot[r][x][q] <= 2

\* Landed plus in-flight arrivals never exceed 2 either.
TotalBound ==
    \A r \in Ranks, x \in Kinds, q \in Ranks : slot[r][x][q] + Len(fl[q][x][r]) <= 2

\* A block that passed the start wait of kernel k: every peer had entered k
\* (hence completed k-1), so nothing of the previous kernel is still in use.
StartSafety ==
    \A r \in Ranks, b \in Blks :
        (kid[r] <= N /\ ph[r][b] \in PostSpin /\ HasStart(kid[r]))
            => \A q \in Ranks \ {r} : kid[r] \in sa[q]

\* When rank r completes a kernel with an end barrier, an END increment of that
\* kernel or a later one has landed on r from every rank (a release covers all
\* earlier writes, so a later kernel's release also suffices; a relaxed START
\* increment never does), and every block of every rank has finished writing,
\* with its writes visible system wide.
EndSafety ==
    \A r \in Ranks : \A k \in cdone[r] :
        HasEnd(k) =>
            \A q \in Ranks :
                /\ \E k2 \in k..N : <<k2, "E">> \in lnd[r][q]
                /\ \A b \in Blks : wv[k][q][b] = 3

\* Call-site obligation (one symmetric buffer per rank group): data is written
\* into peers' buffers only after every rank has finished its earlier kernels
\* that merely read it ("S" pulls, "L" local consumers).  Collective kernels
\* with an end barrier need no such condition: the end barrier already waited
\* for every rank's reads.  A start barrier before the write provides it.
WarSafe ==
    \A j \in 1..N : \A q \in Ranks, b \in Blks :
        (Writes(j) /\ wv[j][q][b] >= 1) =>
            \A x \in Ranks, i \in 1..(j-1) : Reads(i) => i \in cdone[x]

\* No state outlives a kernel: both sign-in counters are zero at kernel start.
CtrClean ==
    \A r \in Ranks :
        (kid[r] <= N /\ \A b \in Blks : ph[r][b] = "new")
            => (ctr[r]["S"] = 0 /\ ctr[r]["E"] = 0)

\* "Stateless": once everything finished and landed, all slots are zero.
Quiescent ==
    AllDone => (/\ \A r \in Ranks, x \in Kinds, q \in Ranks : slot[r][x][q] = 0
                /\ \A r \in Ranks, x \in Kinds : ctr[r][x] = 0)

TypeOK ==
    /\ kid \in [Ranks -> 1..(N+1)]
    /\ ph \in [Ranks -> [Blks -> Phases]]
    /\ \A r \in Ranks : \A x \in Kinds : ctr[r][x] \in 0..(B + 1)

=============================================================================
