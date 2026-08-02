# exp3 — MaxPool argmax-mask to relieve L2 activation stash (on-device FT memory)

**Started:** 2026-08-01
**Branch:** TrainDeeploy `feat/BNFRozen_OptionB`
**Safe checkpoint (revert point):** `b129e42` ("QW-mark our MaxPool changes; safe checkpoint before Part-4").
`git reset --hard b129e42` returns to the clean pre-experiment state.

## Problem (from exp2)
On-device FT is L1/L2-bound. The dominant long-lived L2 buffer is the **forward activation feeding
MaxPool (the ReLU/conv output, ~314 KB in block-0)**, kept alive from forward to backward **only** because
Deeploy's `PULP_MaxPoolGrad2d` **recomputes the argmax from that forward input X** at backward time. If
MaxPoolGrad instead consumed a small **argmax index** buffer, that big activation could be freed right after
the forward pass. Goal: measure whether this relaxes the **L2 peak** (baseline peak L2 = **1,793,800 B**,
89.7 %; L1 = 127,808 B, 99.9 %).

## Why the naive "2-output MaxPool" is blocked (root cause, verified)
`Deeploy/TilingExtension/TileConstraint.py:134` — `wrapTilingSolution` asserts
`len(outputTensorMemoryConstraints) == 1` ("Expected node to have only one output!"). The tiler is
**single-output-driven**: it tiles *the one* output from its full shape, propagates that tiling down
L3→L2→L1, and derives every input tile from it. Two outputs → no policy for which output anchors the
schedule. (Overridable per-constraint — `MSELossTileConstraint:97` does — but the base machinery assumes one
output. Supervisor confirmed: Deeploy currently handles single-output nodes only.)

## Two options considered
- **Option A — 2-output MaxPool + `wrapTilingSolution` override.** MaxPool emits `[pooled, mask]`; override
  co-tiles the mask (same shape as pooled → identical cubes) and reattaches it. One fewer op, **but touches
  the shared tiler** (fragile, hard to verify).
- **Option B (CHOSEN, "Part 4") — separate single-output `MaxPoolArgmax` op.** Three single-output nodes:
  | node | inputs | single output | lifetime |
  |---|---|---|---|
  | `MaxPool` (unchanged) | X (ReLU act) | pooled `[N,C,Ho,Wo]` | **short** (freed after next layer) |
  | **`MaxPoolArgmax`** (new) | X (ReLU act) | uint8 mask `[N,C,Ho,Wo]` | **long** (to backward) |
  | `MaxPoolGrad` (modified) | dY, mask | dX | — |
  Every node single-output → **no tiler change**, standard machinery. `MaxPoolArgmax` reads X in the
  **forward** pass (X still live), so after MaxPool+MaxPoolArgmax run, X is freed; only the small uint8 mask
  survives. Cost: +1 forward-scan op (~2–3 % of step) − backward recompute saved → ~compute-neutral; memory
  win. (Earlier note claimed it "eliminates the MaxPool argmax-drift" — CORRECTED 2026-08-02: it does NOT.
  The device-vs-ORT drift is from `argmax(X)` tie-flips on fp-differing X; both recompute-grad and
  MaxPoolArgmax compute argmax from the same X, so the susceptibility is unchanged. See FINDING.md.)

## Correctness invariant (the key design decision)
Store the **within-window offset** (0…k-1, uint8), NOT a global input coordinate. The offset is
**tile-position-independent**: in backward, for each output element (position known in the current tile) the
target input = `window_origin(from out pos) + offset`, reconstructed locally — robust to how fwd/bwd tile.
A global coordinate would be fragile under tiling. Tie-break must be identical across MaxPool / MaxPoolArgmax
(strict `>`, first-max) so the stored winner matches the pooled value.

## Implementation steps (Part 4)
1. **Kernel** (`TargetLibraries/PULPOpen/src/MaxPool.c` + `.h`):
   - New `PULP_MaxPoolArgmax2d_*_HWC(X, H,W,C, kH,kW, sH,sW, mask, pads…)` → writes uint8 within-window
     offset per output element (same window scan as forward, same tie-break).
   - Modify `PULP_MaxPoolGrad2d_*_HWC` to take `mask` instead of `X` and scatter via the stored offset (no
     recompute). Also fix its (x,y)→(y,x) arg order (QW-TODO in the template).
2. **Template** (`FloatMaxPoolTemplate.py`): add `MaxPoolArgmax` template; rewrite grad template to pass mask.
3. **Parser** (`Generic/Parsers.py`), **TypeChecker** (`PULPOpen/TypeCheckers.py`), **Binding**
   (`PULPOpen/Bindings.py`), **Platform mapping** (`PULPOpen/Platform.py`): register `MaxPoolArgmax`
   (uint8 output); update MaxPoolGrad for the mask input.
4. **TileConstraint**: `MaxPoolArgmax` (single output, like MaxPool); update `MaxPoolGradTileConstraint` for
   mask input (was X).
5. **Graph rewrite** (Onnx4Deeploy `base_exporter.py::_rewire_maxpoolgrad_recompute`): instead of rewiring
   MaxPoolGrad→X and dropping the mask, **insert a `MaxPoolArgmax` node** (input = MaxPool's input) and wire
   `MaxPoolGrad(dY, argmax_mask)`. MaxPool stays single-output (pooled only).

## Test / acceptance
- Rebuild + run **one** FT round on the tiled Siracusa trainer with **`--plotMemAlloc`**; compare the new
  **L2 peak** to the 1,793,800 B baseline (expect a drop ≈ the freed 314 KB-class activation). Also confirm
  the training loss still matches the ORT reference (correctness) up to the MaxPool-drift onset.
- Keep the baseline `memory_alloc_training.html` (exp2) for side-by-side.

## Status log
- 2026-08-01: checkpoint `b129e42`; QW-marking done; PLAN written.
- 2026-08-01: **step 1 (kernels) DONE** — commit `24c4a82` (`PULP_MaxPoolArgmax2d` +
  `PULP_MaxPoolGradMask2d` in MaxPool.c/.h, additive/no caller yet). Next: step 2 templates →
  step 3 parser/checker/binding/platform → step 4 tile-constraints → step 5 graph rewrite → build+sim.

## Build result (2026-08-02, fork)
Implementation COMPLETE + mapping-clean, but blocked in the tiler. Progress:
1. ✅ Registration (checker/binding/tiler/platform) + graph rewrite (`--maxpool-argmax-mask`).
   Export verified: 3 `MaxPoolArgmax` (fp32) + 3 `MaxPoolGradMask` nodes, MaxPool single-output.
2. Fix A: uint8 mask hit type-friction in the all-fp32 training pipeline → switched to an **fp32
   offset mask** + a **distinct `MaxPoolGradMask` op** (avoids dtype-dispatch ambiguity).
3. Fix B: `MaxPoolArgmax` failed to parse — `MaxPoolParser.parseNode` requires a `ceil_mode` attr
   the rewrite hadn't copied. Added it → **frontend binding now PASSES** (both new ops map).
4. ⛔ BLOCKED in tiling-loop codegen: `TilingCodegen.py:537 minimizeRectangle`
   `AssertionError: offset should be zero when dims == reference`, rect
   `offset=(0,0,0,5) dims=(1,16,14,5) ref=(1,16,14,5)` — a full-axis non-zero offset in the
   channel-tiling path for the argmax-mask tensors. **Same tiler-invariant class that blocked
   B-decompose** (exp1 optionB). The `memory_alloc.html` from this run is a partial (920 B)
   snapshot (codegen crashed mid-allocation) → **L2 peak NOT measured yet**.

## Root-cause analysis (2026-08-02, refined — supersedes the fork's tiler diagnosis)
The blocker is **LAYOUT, not the fundamental B-decompose tiler limitation.** The new ops
`MaxPoolArgmax`/`MaxPoolGradMask` are **not registered in the NCHW→NHWC lowering pass**:
`LoweringOptimizationPasses.py` has a per-op pass (`NCHWtoNHWCMaxPoolPass`,
`NCHWtoNHWCMaxPoolGradPass`, …Conv/AveragePool…) but **none for the new ops**, and
`_NCHWtoNHWC_fun` line 232's spatialDims op-list also excludes them. So their tensors stay **NCHW**.
The PULP MaxPool channel-tiling constraint assumes **HWC (C=last)**; on the NCHW `(1,16,14,5)` mask
tensor (C=16=block-1, W=5) it tiles the **W axis** and stamps a full-axis offset → `minimizeRectangle`
assertion at `TilingCodegen.py:537`. (Tell: `(1,16,14,5)` is NCHW; NHWC would be `(1,14,5,16)`.)
Unlike B-decompose (genuine `(1,C,1,1)` broadcast tensors → real core limitation), the mask has the
**same shape as the pooled output**, which tiles cleanly — it only broke because it was left in the
wrong layout. **Fixable in our code.**

## Proposed fix (NOT yet applied — awaiting go-ahead)
1. Add `NCHWtoNHWCMaxPoolArgmaxPass` + `NCHWtoNHWCMaxPoolGradMaskPass` (mirror existing passes) and
   register them in the platform's optimization-pass pipeline.
2. Add `"MaxPoolArgmax"`/`"MaxPoolGradMask"` to `_NCHWtoNHWC_fun`'s spatialDims op-list (line 232).
3. Extend our QW 2nd-input transpose (`if node.op == "MaxPoolGrad"`) to also handle `MaxPoolGradMask`
   (mask 2nd input, pooled shape). Then rebuild + re-run `--plotMemAlloc` for the L2 number.

Commits: TD kernels 24c4a82, templates ca1952d, registration e844ada, fp32 e322b08; O4D rewrite
56b3709, fp32 e6af342, ceil_mode 3f8ca50.

## Build result AFTER layout fix (2026-08-02) — WORKS, but memory goal NOT met
Layout fix (commit `22be691`) cleared the tiler. Full build+GVSoC run PASSED, **loss bit-correct
(Errors 0/16, diff ≤4e-6 vs ORT ref)** — the argmax-mask MaxPoolGrad math + tiling-invariant offset
are validated. BUT:
- **L2 peak = 1,825,356 B vs baseline 1,793,800 B → +31,556 B (+1.8%). Memory did NOT drop; it rose.**
- Mechanism DID work: the baseline's long-lived MaxPoolGrad recompute-input
  (`node_4_...MaxPool_GradMaxPool`, 314,048 B, 88-step lifetime) is **eliminated** in the argmax version.
- Why peak still rose: the L2 peak is set by a **stack of ~6 co-resident 314,048 B block-0 activations**
  (Conv/BN/ReLU outputs + their grads), not by the single MaxPoolGrad stash. Removing one doesn't lower
  that ceiling; and the argmax path slightly **lengthened** the Conv (95→98) and BN (93→96) lifetimes and
  added the (fp32) mask + MaxPoolArgmax intermediates → net +31 KB.
- **Implication:** exp2's attribution was incomplete — the MaxPoolGrad recompute-input was *one of many*
  equal 314 KB block-0 buffers, not the peak driver. The real L2 lever is the co-resident block-0
  activation stack (recompute/retile Conv/BN/ReLU activations), a different & bigger change. The
  argmax-mask is correct, but does NOT remove the device-vs-ORT drift (corrected 2026-08-02; see FINDING).
- Possible small improvements: uint8 mask (4× smaller than the current fp32 mask) — helps a little but
  won't change the peak (mask < 314 KB). Artifacts: logs/memtest_afterfix.log, deeployStates/memory_alloc.html.
