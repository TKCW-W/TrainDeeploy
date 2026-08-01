# FINDING — exp3 MaxPool argmax-mask (Part 4): correct, drift-free, relaxes L2 by ~16% (after transpose-dedup)

## UPDATE 2026-08-02 (resolves the "does NOT relax L2" below) — IT DOES, once the redundant transpose is deduped
The initial correct build showed +1.8% (1,825,356 B) — NOT because the approach is bad, but because the
NCHW→NHWC layout passes give each op its own **input transpose**, so `MaxPool`, `MaxPoolArgmax`,
`MaxPoolGradMask` each transposed the same 314 KB block-0 activation → a redundant 314 KB buffer. Fix:
`MergeSiblingTransposesPass` (merge `Transpose` nodes with identical input+perm; placed at the END of the
PULP lowering pipeline, after the last `TransposeSplitPass`). Result (same `random-max` settings, bit-exact
0/16):
- **L2 peak: 1,511,308 B vs baseline 1,793,800 B → −282,492 B (−15.7%).** Drop = exactly 314,048 B (one
  block-0 buffer), confirming the redundant transpose was the sole cause. Determinism: all re-runs identical.
- So the argmax-mask **does relax L2 (~16%)**, is bit-correct, and removes the MaxPool argmax-drift.
  Change is layout-convention-preserving (no NCHW-native kernels needed).

---
### (original, now-superseded verdict)
# FINDING — exp3 MaxPool argmax-mask (Part 4): correct, drift-free, but does NOT relax L2

**Action date/time:** 2026-08-02
**Branch:** TrainDeeploy `feat/BNFRozen_OptionB` · Onnx4Deeploy `feat/BNFRozen_OptionB`
**Safe checkpoint (revert point):** `b129e42`. **Nothing reverted** (per user instruction).

## What was built
The "Part 4" separate-`MaxPoolArgmax`-op design (three single-output nodes, no tiler override):
- Kernels `PULP_MaxPoolArgmax2d` (forward → within-window offset mask) + `PULP_MaxPoolGradMask2d`
  (scatter via stored offset, tiling-invariant) — MaxPool.c/.h.
- Codegen templates, registration (checker/binding/tiler/platform), and an Onnx4Deeploy graph rewrite
  (`--maxpool-argmax-mask`) that inserts `MaxPoolArgmax` and wires `MaxPoolGradMask(dY, mask)`.
- **Layout fix (`22be691`)** — the real unblock: the new ops had no NCHW→NHWC lowering pass, so their
  tensors stayed NCHW and the HWC channel-tiler tiled the wrong axis (`minimizeRectangle` assertion).
  Added `NCHWtoNHWCMaxPoolArgmaxPass`/`NCHWtoNHWCMaxPoolGradMaskPass` + pipeline reg + op-list + 2nd-input
  transpose. (This was a layout omission in OUR code — NOT the fundamental B-decompose tiler limitation.)

## Test result (one FT round, `--plotMemAlloc`, speechnet_train_argmask_b1)
- ✅ **Compiles + runs on GVSoC. Loss BIT-CORRECT: Errors 0/16, diff ≤ 4e-6 vs the ORT reference.** The
  mask-based MaxPoolGrad math + within-window-offset design are validated.
- ❌ **L2 peak = 1,825,356 B vs baseline 1,793,800 B → +31,556 B (+1.8%). Memory did NOT drop; it rose.**

## Why the peak didn't drop (mechanistic, evidenced by the buffer dump)
- The mechanism **worked**: the baseline's long-lived MaxPoolGrad recompute-input
  (`node_4_...MaxPool_GradMaxPool`, 314,048 B, 88-step lifetime) is **eliminated** in the argmax version.
- But the **L2 peak is set by a stack of ~6 co-resident 314,048 B block-0 activations** (Conv/BN/ReLU
  outputs + their grads), all overlapping at the fwd/bwd boundary — not by the single MaxPoolGrad stash.
  Removing one buffer doesn't lower that ceiling, and the argmax path slightly **lengthened** the Conv
  (95→98) and BN (93→96) lifetimes and added the (fp32) mask + MaxPoolArgmax intermediates → net +31 KB.
- **exp2's memory attribution was incomplete**: the MaxPoolGrad recompute-input was *one of many* equal
  314 KB block-0 buffers, not the peak driver.

## Takeaways
1. **The argmax-mask is a correctness/robustness win, not a memory win here.** It removes the MaxPool
   argmax-drift (fwd & bwd share one stored index) and is bit-exact — but it does not move L2 peak for
   SpeechNet, because the peak is the block-0 activation stack.
2. **uint8 mask** (the impl fell back to fp32 for type-friction) would be 4× smaller, but the mask is
   already < 314 KB so it won't change the peak — marginal.
3. **The real L2 lever** is the co-resident block-0 314 KB activation stack: recompute or aggressively
   retile the Conv/BN/ReLU activations across the fwd/bwd boundary (the same large-spatial block-0 that
   dominates *latency*). Bigger, separate change.

## Commits (all on `feat/BNFRozen_OptionB`)
TD: kernels `24c4a82`, templates `ca1952d`, registration `e844ada`, fp32 `e322b08`, layout fix `22be691`.
O4D: rewrite `56b3709`, fp32 `e6af342`, ceil_mode `3f8ca50`.
Artifacts: `logs/memtest_afterfix.log`, `.../speechnet_train_argmask_b1/deeployStates/memory_alloc.html`.

## Open decision (awaiting user)
(a) try uint8 mask, (b) pivot to the block-0 activation-stack lever, (c) keep argmax-mask for the
correctness/drift benefit + shelve the memory angle, (d) other.
