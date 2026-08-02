# FINDING — exp3: MaxPool argmax-mask (Part 4) relaxes on-device FT L2 peak by ~16%

**Date:** 2026-08-01 → 2026-08-02
**Branches:** TrainDeeploy `feat/BNFRozen_OptionB` · Onnx4Deeploy `feat/BNFRozen_OptionB`
**Safe checkpoint (revert point):** `b129e42` (`git reset --hard b129e42`). Nothing reverted.

## Result (headline)
Replacing the recompute-from-input MaxPool backward with a stored **argmax mask** — and deduping the
redundant input transpose it introduced — gives, on `speechnet_train_argmask_b1` (same tiled-Siracusa
settings as the exp2 baseline, `--searchStrategy random-max`):

| build | L2 peak | vs baseline | loss |
|---|--:|--:|---|
| baseline (recompute MaxPoolGrad) | 1,793,800 B | — | bit-exact |
| argmax-mask, no transpose-dedup | 1,825,356 B | +1.8% | bit-exact 0/16 |
| **argmax-mask + transpose-dedup** | **1,511,308 B** | **−282,492 B (−15.7%)** | **bit-exact 0/16** |

Deterministic (identical across re-runs).

**Correction (2026-08-02): the argmax-mask does NOT eliminate the device-vs-ORT MaxPool argmax-drift**
(an earlier draft claimed it did — wrong). The exp1 drift comes from fp reduction-order differences in **X**
(the conv/BN output, device-tiled vs ORT-untiled, ~1e-6) flipping a MaxPool tie. Both the recompute
`MaxPoolGrad` and the new `MaxPoolArgmax` compute `argmax(X)` from the *same* stashed X, so the tie-flip
susceptibility is **unchanged** — the mask only guarantees device fwd/bwd agree, which they already did in
the recompute path (both scanned the same X). The 0/16 memtest was too short to see it (`--n-steps 4` = 16
micro-batches; exp1 round-1 drift onset ≈ step 133). So: memory win + math-correctness are validated, but
**drift is inherent and expected to persist** at full round length.

## Motivation
On-device FT is L1/L2-bound (exp2). Deeploy's `PULP_MaxPoolGrad2d` **recomputes the argmax from the forward
input X** at backward time, which forces the large block-0 activation to be **stashed forward→backward**.
Idea: have the forward pass emit a small **within-window argmax offset** buffer; the backward reads it and
scatters — so the big activation can be freed after the forward pass. Chosen route ("Part 4"): a **separate
single-output `MaxPoolArgmax` op** (keeps every node single-output → no tiler-core changes; the naive
2-output MaxPool is blocked by `TileConstraint.py:134`'s "one output" assertion).

Correctness invariant: store the **within-window offset** (`p*Q+q`), which is **tile-position-independent**
(the input position is reconstructed locally in backward from the output position + offset), so it survives
tiling. Tie-break (strict `>`, first-max) is identical across MaxPool/MaxPoolArgmax so the stored winner
matches the pooled value.

## Implementation — where and how we modified

### 1. Device kernels — `TargetLibraries/PULPOpen/src/MaxPool.c` (+ `inc/kernel/MaxPool.h`)
- **`PULP_MaxPoolArgmax2d_fp32_fp32_HWC`** — forward argmax: same window scan + tie-break as
  `PULP_MaxPool2d`, writes the within-window offset `p*Q+q` per output element into the mask.
- **`PULP_MaxPoolGradMask2d_fp32_fp32_HWC`** — backward: reads the offset from the mask, reconstructs the
  input position locally (`p=off/Q, q=off%Q`, `in = window_origin + (p,q)`), scatters `dY` — no recompute,
  no forward-activation read.
  (Offset stored as **fp32**, not uint8 — see Issue A.)

### 2. Codegen templates — `Deeploy/Targets/PULPOpen/Templates/FloatMaxPoolTemplate.py`
- `argmaxTemplate` (calls `PULP_MaxPoolArgmax2d`, output = mask, same `(H,W)=(y,x)` dim mapping as the
  forward MaxPool template) and `referenceGradMaskTemplate` (calls `PULP_MaxPoolGradMask2d`; 2nd input is
  the mask, pooled-shape, advanced by the pooled size).
- Also carries the QW note that the forward MaxPool template passes `(H,W)=(y,x)` (our earlier fix).

### 3. Op registration (PULPOpen)
- **TypeCheckers** (`TypeCheckers.py`): `PULPMaxPoolArgmaxChecker` (fp32→uint8; defined but **unused** after
  the fp32 switch — the fp32 path reuses `PULPMaxPoolChecker`).
- **Bindings** (`Bindings.py`): `PULPMaxPoolArgmaxBindings` (→ `argmaxTemplate`) and
  `PULPMaxPoolGradMaskBindings` (→ `referenceGradMaskTemplate`); the existing recompute-grad binding is kept.
- **Tiler** (`Tiler.py`): `PULPMaxPoolArgmaxTilingReadyBindings` (reuses the single-output MaxPool
  channel-tiling constraint) and `PULPMaxPoolGradMaskTilingReadyBindings`.
- **Platform** (`Platform.py`): `MaxPoolArgmaxMapper = NodeMapper(MaxPool2DParser(), …)`;
  `'MaxPoolArgmax': MaxPoolLayer([MaxPoolArgmaxMapper])` and a `MaxPoolGradMask` mapping.
- **Parsers**: `MaxPoolArgmax` reuses `MaxPool2DParser`; `MaxPoolGradMask` reuses `MaxPoolGradParser`.

### 4. Graph rewrite — Onnx4Deeploy `onnx4deeploy/core/base_exporter.py`
- New CLI/flag **`--maxpool-argmax-mask`**. When set, instead of `_rewire_maxpoolgrad_recompute` (which
  rewires `MaxPoolGrad(dY, X)` and drops the MaxPool mask output), the exporter **inserts a `MaxPoolArgmax`
  node** (input = the MaxPool's forward input; output = the offset mask) and wires
  `MaxPoolGradMask(dY, mask)`. `MaxPool` stays single-output (pooled only). Default path unchanged.

### 5. Layout passes (the tiler unblock) — `Deeploy/CommonExtensions/OptimizationPasses/TopologyOptimizationPasses/LoweringOptimizationPasses.py`
- Added `"MaxPoolArgmax"`/`"MaxPoolGradMask"` to `_NCHWtoNHWC_fun`'s `spatialDims` op-list.
- Extended the QW MaxPoolGrad 2nd-input transpose special-case to also cover `MaxPoolGradMask` (its 2nd
  input = the mask).
- Added `NCHWtoNHWCMaxPoolArgmaxPass` + `NCHWtoNHWCMaxPoolGradMaskPass` and **registered them in the
  `PULPNCHWtoNHWCPass` composite**.

### 6. Transpose-dedup (the memory win) — `LoweringOptimizationPasses.py` + `Deeploy/Targets/PULPOpen/Deployer.py`
- Added **`MergeSiblingTransposesPass`** (`Pass` + `@contextagnostic`): merges `Transpose` nodes with
  identical (input tensor, perm) into one, rewiring all consumers.
- Wired it as the **last** pass in `PULPDeployer`'s lowering pipeline (after the final `TransposeSplitPass`),
  so the merge is not re-split.

## Issues encountered and how we solved them

### Issue A — uint8 mask "type friction" (worked around, not fully root-caused)
The first design used a **uint8** mask (`PULP_MaxPoolArgmax2d_fp32_u8_HWC`, `PULPMaxPoolArgmaxChecker` →
uint8, ONNX `TensorProto.UINT8`) and reused the `MaxPoolGrad` op with dtype-dispatch (`[float32,uint8]` vs
`[float32,float32]`). It hit "type friction" and was switched (fork commit `e322b08`+`e6af342`) to **fp32
offset mask + a distinct `MaxPoolGradMask` op**. Two changes were bundled, so the exact blocker isn't
isolated — most likely a uint8 tensor flowing as an *activation* through the all-float32 transpose/tiling
pipeline, and/or the dtype-dispatch binding ambiguity (removed by the distinct op). **Deferred:** retry
uint8 now (distinct op + layout fix + dedup already in place) to isolate it; would shrink the long-lived
mask ~4× (not peak-changing).

### Issue B — tiler `minimizeRectangle` assertion (SOLVED, commit `22be691`)
First build failed at `TilingCodegen.py:537` (`offset should be zero when dims == reference`, rect
`(1,16,14,5)`). **Root cause:** the new ops had **no NCHW→NHWC lowering pass**, so their tensors stayed
**NCHW** while the PULP channel-tiler assumes **HWC (C = last)** → it tiled the wrong (W) axis and stamped a
full-axis offset. Fixed by §5 (add the layout passes). *Not* the fundamental B-decompose tiler limitation —
the mask has the same shape as the pooled output and tiles cleanly once in NHWC.

### Issue C — the "correct build looked like a memory REGRESSION" (+1.8%), and the debugging path (SOLVED, commit `c6aa5bc`)
After Issue B the build was correct (0/16) but L2 peak = 1,825,356 (**+1.8%**, higher than baseline). Debug:
1. **First wrong hypothesis (mine): random search.** Wrong — 4 fresh re-runs gave *identical* 1,825,356, so
   the peak is **deterministic**, not `random-max` noise.
2. **Second confusion: a pre-fix intermediate html** (`exp3/results/memory_alloc_argmask.html`, 1.52M) that
   *looked* like the expected reduction. It was written by the fork **before** the layout fix existed
   (timestamp before `22be691`), i.e. an incorrect NCHW build — **not trustworthy**.
3. **Root cause (buffer diff correct-vs-intermediate):** the correct build had **2 extra 314 KB block-0
   buffers** = per-op **input transposes**. `MaxPool`, `MaxPoolArgmax`, `MaxPoolGradMask` each transpose the
   *same* 314 KB block-0 activation to NHWC. The PULP pipeline's **`TransposeSplitPass` (runs twice)
   deliberately gives each consumer its own input transpose**, so the transpose is duplicated per op. The
   +1.8% = exactly one 314 KB redundant transpose buffer (1,825,356 − 1,518,728 ≈ 314,048).
4. **Fix:** `MergeSiblingTransposesPass` at the end of the pipeline (§6). The tiler **accepts the shared
   (fanout) transpose**. Result: **1,511,308 B (−15.7%)**, bit-exact, deterministic across re-runs.

## Post-fix memory analysis (where the peak is now)
- The forward mask's own layout round-trip (`_pre_transposed` name) was already **cancelled** by
  `TransposeMergePass`/`TransposeNoPermOptPass` — verified in the lowered graph: the mask flows
  `MaxPoolArgmax → MaxPoolGradMask` **directly**, no transpose between them (the name is vestigial).
- The **new L2 peak (1,511,308) is set in the FORWARD pass (x≈2..22)** by the **`Conv_input_*_transposed`**
  buffers of blocks 2–3, **not** by any MaxPool/argmax buffer. The backward `MaxPoolGradMask` output (dX,
  314 KB) and its (un-cancelled) transpose pair sit at x≈94 and pack **below** the peak → not peak-driving.
- Conclusion: the argmax-mask + dedup captured the MaxPool-side win; the remaining ceiling is the forward
  conv-input transposes — a separate target.

## Deferred follow-ons (agreed to save for later)
1. **Real uint8 mask** — shrink the long-lived mask ~4× (not peak-changing). Retry to isolate Issue A.
2. **Backward dX transpose pair** — the `MaxPoolGradMask` output→ReLU-grad transpose didn't cancel (two
   314 KB copies at x≈94), but it's **below the peak**, so no L2 gain.
3. **Forward `Conv_input_*_transposed`** (blocks 2–3) — the *current* peak driver; reducing L2 below 1.51M
   means addressing these (dedup/fuse, or NCHW-native), not the MaxPool path.

## Commits
- TrainDeeploy: checkpoint `b129e42`; kernels `24c4a82`; templates `ca1952d`; registration `e844ada`;
  fp32 `e322b08`; layout fix `22be691`; **dedup `c6aa5bc`**.
- Onnx4Deeploy: rewrite `56b3709`; fp32 `e6af342`; ceil_mode `3f8ca50`.
- Artifacts: `logs/memtest_afterfix.log`, `logs/memtest_dedup.log`,
  `results/memory_alloc_dedup.html` (+ `dedup_confirm_c{1,2}.html`), and the fixture's
  `deeployStates/memory_alloc.html`.
