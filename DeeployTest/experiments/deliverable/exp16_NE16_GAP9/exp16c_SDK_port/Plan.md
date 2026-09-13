# exp16c_SDK_port — port the GAP9 SDK's NE16 techniques into our flow

Date opened: **2026-09-14** · Branch `feat/GAP9_w_NE16`
Driver: `ETH/WorkLog/GAP9_SDK_NE16_Corner_Cases.md`
Siblings: `exp16a_PW_single_layer`, `exp16b_Dense_single_layer`, `exp16c_PW_single_layer`

---

## 1. Goal

**Run SpeechNet QZO end-to-end on GAP9 with NE16.**

Everything so far has been *per-layer* fixtures, hand-built by a script. Two things stand between
that and the goal:

1. **Performance.** NE16 is 4.3–7.5× slower than `pulp_nn_conv`. The SDK exploration found two
   concrete defects in our dispatch loop and one better algorithm.
2. **Automation.** The NE16 form of a `1×K` conv (pre-encoded weight, `ne16_taps`, the encode node,
   the signed-input correction) is currently *authored by the fixture builder*. The real QZO graph
   has none of that. Nothing rewrites a plain `Conv` into the NE16 form automatically, so the full
   network cannot be compiled.

Item 2 is what "port into our flow" means, and it is the blocker for the goal.

## 2. What the SDK gives us

From `CNN_BasicKernels_NE16.c` / `CNN_Generators_NE16.c` (file:line refs in the findings doc):

| # | SDK technique | our status | value |
|---|---|---|---|
| A | **Pipelined dispatch** — acquire/program/trigger, block only when the 2-deep job queue is full; wait for completion once, at the end | we `resolve_wait` after *every* dispatch (under GVSoC = full queue drain) → zero overlap | high, cheap |
| B | **Minimal register writes** — 10 words steady-state, invariants written once per job context (`if (SubTileCount < 2)`) | `ne16_nnx_dispatch` rewrites all **24** descriptor words every time | medium, invasive (bypasses pulp-nnx's task API) |
| C | **im2col channel folding** — cores build a `Cin·K` column buffer, NE16 runs 1×1, double-buffered | not implemented; was our STEP 4 proposal | high, large |
| D | 1-D W-axis remapped onto the 3×3 output grid | N/A — our maps are 2-D | none |

**Safety note, verified before relying on A:** GVSoC's `fsm_end_handler` (`gap/ne16/src/fsm.cpp:88`)
decrements `job_pending`, flips `cxt_use_ptr`, and only then starts the next job; `fsm_start_handler`
sets `job_running = 1`. So queued jobs execute **one at a time, in order** — the queue is a
program-ahead buffer, not concurrency. Removing the per-dispatch `resolve_wait` therefore preserves
the tap-to-tap RAW dependency that `streamin` relies on.

## 3. Phases

| phase | what | gate |
|---|---|---|
| **1** | **Technique A** — pipeline the dispatch loop in `Conv1xKTemplate.py` and `Conv3x3ChunkTemplate.py`. | all five blocks still bit-exact; cycles measured |
| **2** | **Automatic graph pass** — `NE161xKDecompositionPass`: match a `1×K`/`K×1` `Conv` coloured NE16, insert `NE16WeightEncode` on its weight, set `ne16_taps` / `weight_offset` / `ne16_weight_preencoded`, and for a signed input insert `NE16SignedInputBias` + the `+128` offset. Replaces the fixture builder's hand-authoring. | a fixture built from the *plain* graph compiles and is bit-exact |
| **3** | **Full SpeechNet QZO on GAP9 + NE16** via `deeployMezoRunner_tiled_GAP9_w_NE16.py`. | runs end-to-end; output matches the cluster-only QZO reference |
| **4** | *(stretch)* **Technique C** — im2col channel folding. | only if phases 1–3 land with time to spare |

Phase 3 is the goal; phases 1–2 are what make it reachable. Technique B is recorded but not
scheduled — it means abandoning pulp-nnx's task API, which is a large change for a second-order win.

## 4. Non-goals

* Multi-step training or an accuracy number — single-step correctness is the bar, as in every
  previous exp16 experiment.
* Beating `pulp_nn_conv`. Phase 1 and 4 narrow the gap; closing it is not promised here.
* Touching the NE16 ISA. Everything stays on fields `ne16_task_t` already has, so it runs on real
  GAP9 silicon.

## 5. Success criteria

1. All five convs still bit-exact after phase 1, with a measured cycle delta.
2. A plain `Conv` graph — no NE16 attributes authored by hand — compiles to NE16 and is bit-exact.
3. SpeechNet QZO runs end-to-end on GAP9 with NE16 engaged, verified against the cluster reference.
4. `Findings.md` records results, a reproduction section, and every changed file by path.
