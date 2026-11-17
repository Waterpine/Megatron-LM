# TLA+ model of the once-per-rank multicast barriers

A [TLA+](https://lamport.azurewebsites.net/tla/tla.html) model of `symm_mem_sync_start` and
`symm_mem_sync_end` in
`megatron/core/inference/communication/torch_symm_triton/barrier.py` (once-per-rank barriers built
on one `multimem.red` arrival per rank per kernel), checked exhaustively with the TLC model
checker on small configurations.

The model checks the **protocol** (counters, slots, who waits for whom, memory-ordering steps). It
does not check the Triton/PTX code itself; see [Assumptions and limits](#assumptions-and-limits).

## What is modeled

`multicast_barrier.tla` models one stream of kernels running on each of `R` ranks. Every kernel
launches `B` blocks per rank, and kernel `k+1` starts on a rank only after kernel `k` has finished
there (no PDL). A kernel is one of:

| Kind | Meaning | Example |
|------|---------|---------|
| `SE` | start barrier, writes to peers, end barrier | all-gather, reduce-scatter |
| `E`  | writes to peers, end barrier only | |
| `S`  | start barrier, only reads the symmetric buffer | MoE combine |
| `L`  | only reads the symmetric buffer, no barrier | local consumer |

Each step of the code is one action of the model:

| In `barrier.py` | In the model |
|-----------------|--------------|
| block 0 `multimem.red` add 1 to `start slot[rank]` | `Dispatch` (block 0), arrival goes in flight |
| `_wait_nonzero` on every peer's start slot | `Spin`, `SpinDone` |
| sign-in `atomic_add` on the start/end counter | `SignS`, `SignE` |
| last block resets the counter | `ResetS`, `ResetE` |
| `multimem.red.release` on the end slot | `RelE` |
| `_wait_and_take_one` (spin until nonzero, then -1) | `TakeS`, `TakeE` |
| the multicast add reaching a rank's copy | `Land` |
| kernel exit | `KernelDone` |

What the model lets happen (what TLC explores):

- **Asynchronous multicast arrivals.** An arrival is in flight until it lands on each rank's copy,
  at an arbitrary time, possibly after the kernel that issued it has finished. Arrivals of the
  same kind between one pair of ranks land in issue order (the design relies on this); a start and
  an end arrival may land in either order.
- **Weakest memory behavior.** A block's writes are only "issued"; the block's `acq_rel` sign-in
  makes them device visible; the last block's `release.sys` arrival makes them system visible
  before the arrival can land. Nothing becomes visible by itself.
- **Any interleaving** of blocks, ranks and landings, with `Cap` resident blocks per rank. With
  `InOrder` blocks are dispatched in index order, otherwise in any order.

## What is checked

| Property | Meaning |
|----------|---------|
| deadlock freedom | every block eventually exits (TLC deadlock check) |
| `SlotBound`, `TotalBound` | every slot holds 0, 1 or 2; landed plus in-flight arrivals never exceed 2 |
| `StartSafety` | a block that passed the start wait has seen every peer enter the kernel, so every peer has finished the previous one |
| `EndSafety` | when a kernel with an end barrier completes on a rank, an end arrival of that kernel (or a later one) from every rank has landed, and every block's writes are system visible |
| `WarSafe` | call-site obligation: data is written into peers' buffers only after every rank finished earlier kernels that only read it |
| `CtrClean`, `Quiescent` | no state outlives a kernel: both counters are 0 at kernel start; all slots and counters are 0 at the end |
| `TypeOK` | variable ranges |

## Results

`run.sh` runs the configurations below. Each has an expected verdict (`PASS`: all properties hold
and no deadlock; `FAIL`: the named check must fire). The output of a healthy run ends with
`N as expected, 0 unexpected`.

| Group (config name prefix) | What it shows | Expected |
|----------------------------|---------------|----------|
| `base_`, `endonly_`, `startonly_`, `mixed_`, `single_block_` | the design as written, up to 3 ranks x 3 blocks, up to 6 kernels | PASS |
| `cap1_inorder_` | grid needs no co-residency (1 resident block per rank) when blocks are dispatched in order | PASS |
| `needs_inorder_dispatch` | with 1 resident block per rank and out-of-order dispatch the design deadlocks | FAIL (deadlock) |
| `mut_shared_slots*` | start and end arrivals sharing one slot array | FAIL (`EndSafety`) |
| `mut_numblocks_low` / `_high` | `num_blocks` passed to the barrier is one too small / one too large | FAIL (deadlock / `EndSafety`) |
| `mut_signin_relaxed` | sign-in does not flush the block's writes | FAIL (`EndSafety`) |
| `mut_release_relaxed` | last block's arrival is relaxed instead of release | FAIL (`EndSafety`) |
| `mut_no_take` | arrivals are never consumed | FAIL (`StartSafety`) |
| `nonfifo_` | same-kind arrivals may land out of issue order | PASS |
| `war_start_protects` | a start barrier before the writes that follow a read-only kernel | PASS |
| `war_E_after_L`, `war_E_after_S`, `war_L_between_EE` | an end-only kernel after a read-only kernel, without a start barrier | FAIL (`WarSafe`) |
| `piggy_` | optional variant: the start arrivals are taken by the end barrier's last block instead of the start barrier's last block (no start sign-in) | PASS; `piggy_notake` FAIL (`StartSafety`) |

The mutants and the FAIL configurations are there to show that the checks are not vacuous: each
design choice, once removed, is caught.

Largest runs (distinct states): `base_SE_r3b2` 42.9M, `mixed_r3b2` 39.0M, `piggy_SE_r3b2_k1` 378M,
`piggy_SE_r3b2` 778M (about 1 h 48 min on a 28-worker CPU node with a 64 GB heap).

### What the model says about the design

- No counterexample to the design as written, within the bounds below.
- **Start barrier before writes is required.** An end-only kernel that follows a kernel that only
  reads the symmetric buffer (`S` or `L`) can overwrite data a slow rank still reads (`WarSafe`).
  Collective kernels that end with an end barrier need no such condition: that barrier already
  waited for every rank's reads.
- **Blocks must be dispatched in index order** (block 0 first) if the grid does not fit on the GPU
  at once. CUDA does not guarantee this.
- **Separate start and end slots are required**, and so is the exact `num_blocks`.
- **In-order landing of same-kind arrivals is not needed** for the properties above.

## Assumptions and limits

- The model is hand-written from `barrier.py`; nothing checks that it matches the code.
- The visibility steps assume that `multimem.red.release.sys` after the `acq_rel` sign-ins makes
  every block's writes visible before the arrival lands (cumulativity). This was **not** checked
  against the PTX memory model.
- Bounded: up to 3 ranks, 3 blocks per rank, 6 kernels in a sequence (see the table in `run.sh`).
- Threads inside a block are collapsed into one step sequence; data is represented only by its
  visibility stage, not by values.
- No PDL: kernel `k+1` starts on a rank after kernel `k` has finished on it.
- One symmetric buffer per rank group; collectives on the same buffer do not run concurrently.

## Running

Requirements: `bash` and a Java runtime (the jar targets Java 8; tested with OpenJDK 21). No
Python or GPU is needed.

```bash
# TLA+ tools 1.7.4 (TLC 2.19); do not commit the jar
gh release download v1.7.4 --repo tlaplus/tlaplus --pattern tla2tools.jar
# sha256 936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88

cd formal/multicast_barrier
./run.sh /path/to/tla2tools.jar base_SE_r2b2   # the design, a few seconds
./run.sh /path/to/tla2tools.jar mut_           # all mutants, each must FAIL
./run.sh /path/to/tla2tools.jar                # every configuration except the heavy ones
HEAVY=1 TLC_HEAP=64g TLC_WORKERS=28 ./run.sh /path/to/tla2tools.jar base_SE_r3b2
```

The second argument keeps only the configurations whose name contains it. Environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `TLC_HEAP` | `3g` | JVM heap of TLC |
| `TLC_WORKERS` | `6` | TLC threads |
| `HEAVY` | unset | `1` also runs the five heavy configurations (tens of millions of states, GBs of disk spill; use a big-memory machine) |

Each run uses its own temporary directory, which is removed on exit, and the state queue spills to
disk there. Timing on a 6-thread workstation with the defaults: the quick filters above take
seconds; the whole non-heavy suite about 2 hours, most of it in the three largest non-heavy
configurations (`piggy_SE_r3b2_k1`, `piggy_b1`, `piggy_SE_r3b1_k3`) with peak memory 3.4 GB.
For FAIL configurations TLC stops at the first violation, so the printed state count varies between
runs; for PASS configurations it is deterministic.

## Adding a configuration

Add a `run` line to `run.sh`. The columns are, in order:

```
name  expect(PASS|FAIL)  R  B  Cap  InOrder  kernels  shared_slots  num_blocks_delta
      sign_in_relaxed  release_relaxed  no_take  non_fifo  [start_take_at_end]  [heavy]
```

`kernels` is a comma separated list of `"SE"`, `"E"`, `"S"`, `"L"`. The `M_*` constants of the
spec map to the boolean columns; the defaults model the design as written.

## Files

| File | Content |
|------|---------|
| `multicast_barrier.tla` | the model and the properties |
| `run.sh` | generates one wrapper module and `.cfg` per configuration, runs TLC, compares with the expected verdict |
