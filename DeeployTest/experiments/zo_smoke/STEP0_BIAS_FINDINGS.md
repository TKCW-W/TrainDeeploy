# ZO 2-step smoke — step-0 input-form ~0.002 bias: debug findings (-- QW)

## ✅ RESOLVED
Root cause: the ZO perturb kernel's L1 `data_in`/`data_out` buffers OVERLAP for the odd-length fc_bias (9
elems) — the tiler places `data_out` only +4 B into `data_in`, so the element-wise perturb writes
`data_out[i]` (= `data_in[i+1]`) before reading it → odd fc_bias elements corrupted → odd fc logits wrong →
the whole ~0.002 step-0 bias. Input-form-specific (constant fc_bias in init-form uses a non-tiled path).

Fix (shipped): `TargetLibraries/PULPOpen/src/ApplyRademacherPerturbation` is now overlap-safe — when
`data_out` sits just ahead of `data_in` (small forward overlap, decision identical on every core), each core
copies its chunk to a private temp, a `pi_cl_team_barrier` guarantees all reads finish before any write, then
perturbs from the temp. z-stream/order unchanged ⇒ bit-identical to the non-overlapping path. (A tiler-level
fix — forcing full-size perturb tiles — was infeasible: weights feed the tiled conv, and biases feed tiled
consumers, so the perturb tensors can't be forced full-size.)

Verified end-to-end (Onnx4Deeploy fixture gen → TrainDeeploy MeZO runner on device, weights as inputs):
- 2-step / n_accum=1: step-0 bit-exact (0.016488 / 0.069274), step-1 within 1e-6. Errors 0/4.
- 8-step / n_accum=2: Errors 0/32 (max diff 3e-6). Multi-step + gradient accumulation validated.

Below is the full investigation trail that led here.

---


## Status
- **Update propagation (step 1): FIXED.** Option-B deploy-prep (`_prep_zo_train_for_deploy` in
  `testUtils/trainingUtils.py`) promotes the 22 trainable initializers → graph inputs; `testMVPTraining.py`
  + `codeGenerateTraining.py` now emit `TRAINING_NUM_WEIGHT_INPUTS`/`testInitWeights` for the no-grad ZO
  graph; `build_shared_buffer_maps` then aliases `zo_update`'s outputs onto the weight inputs. Step-1 loss
  moved from the no-update `1.635` toward the reference `1.9078` — the update reaches step 1.
- **Remaining: step-0 ~0.002 bias.** init-form (weights as initializers, `ZO_NO_PREP=1`) is **bit-exact**
  (loss+₀=0.016488, loss-₀=0.069274). input-form (promoted to inputs) gives loss+₀=**0.014460**,
  loss-₀=0.062160. All losses biased LOW and consistent.

## What is PROVEN correct in input-form (ruled out)
1. All 22 weights load bit-exact on device (`-D ZO_DUMP_FCW=1` dumps every weight input → all match).
2. The **perturbed fc_weight** the Gemm consumes matches the reference `_perturb_rademacher` **exactly**
   (`-D ZO_DUMP_PWOFF=102952`, all 40 elements <1e-5). So weights + perturbation + fc-Gemm-weight path
   are fully correct.
3. All perturb codegen is byte-identical between forms (chunk 36/core, `tile_seed_offset=0`, node ids).
4. Zero Transpose nodes on any weight; conv reads perturbed weight directly (NCHW).
5. Only the ODD fc output logits differ; even logits exactly unchanged (this was a red herring — see below).

## Localization (the real cause)
Since the perturbed fc_weight is correct but the fc output (log_prob) is wrong, the divergence is in the
**features** — the `GlobalAveragePool` output (`node_0_Reshape__0`, 32-dim) feeding fc. Features come from
conv/BN/relu/maxpool/globalpool over CORRECT weights and the SAME input. The only thing that differs
between init-form and input-form is the **graph shape (24 inputs vs 2)** → different **arena layout /
activation tiling**. A tiled reduction (GlobalAveragePool) or tiled conv/maxpool producing a real ~0.002
error under a different tiling is the leading hypothesis (matches the supervisor's "tiling" concern and the
known device-vs-ORT MaxPool drift in this repo).

Could NOT dump the features post-forward: the L2 arena at the feature offset (102824) is **reused** and holds
log_prob/loss by dump time (only late-written tensors like fc_weight@102952 survive).

## Debug tooling added (guarded, off by default; all `-- QW`)
- `deeploymezotest.c`: `ZO_DUMP_FCW` (dump all weight inputs), `ZO_DUMP_PWOFF` (dump an L2 arena offset
  after +eps forward). CMake passthroughs in `Platforms/Siracusa/CMakeLists.txt`.
- `trainingUtils.py`: `ZO_NO_PREP` env var → deploy raw initializer-form (init-form) for comparison.
- Note: `--eps` does NOT change the perturbation magnitude (baked into the ONNX perturb node's epsilon at
  export; `--eps`/`ZO_EPS` only feeds the FC-side g_proj denominator). `--eps 0` breaks the build (`ZO_EPS 0f`).

## Update — tiling ruled out as the obvious cause
- GlobalAveragePool input is `[1,32,2,5]` (320 elems) → **single-tile in both forms** (not the culprit).
- The 67-vs-45 L1 transfer gap = exactly the 22 weight-input loads → **activation tiling is IDENTICAL**.
- So: weights correct, perturbs identical, activation tiling identical, pool single-tile — yet the forward
  output differs by ~0.002. The divergence is a subtle runtime/codegen effect of the 24-input graph, NOT a
  visible tiling or weight difference. Needs a **differential activation trace**: dump the output of each op
  (conv0 → BN0 → … ) for init-form vs input-form and find the FIRST op whose output diverges. Requires
  IN-FORWARD instrumentation (copy the op output to a persistent global from the kernel), because the L2
  arena is reused post-forward (only late tensors like fc_weight survive).

## ROOT CAUSE FOUND (2-step step-0 bias)
Instrumented the fc `PULP_Gemm_fp32` to persist its actual read inputs (M=1,N=32,O=9). Result:
- transB=1 ✓, features A ✓ (== ref), weight B ✓ (== ref perturbed fc_weight row-major).
- **bias C is CORRUPTED on ODD indices** {1,3,5,7} (even indices correct):
  ```
  device C : [0.1077, 0.1177, 0.171,  0.181,  -0.0115, -0.0015, 0.0069, -0.0031, -0.0998]
  ref (pb) : [0.1077, 0.1611, 0.171, -0.1573, -0.0115,  0.0832, 0.0069,  0.1327, -0.0998]
  ```
  Pattern: `C[2k] = perturbed_bias[2k]` (correct), `C[2k+1] = CLEAN_bias[2k]` (stale, stride-2). This EXACTLY
  matches the odd-logit divergence — the whole ~0.002 step-0 bias is the **fc_bias (Gemm C) read wrong on
  odd indices in input-form**. init-form (constant bias) is unaffected → input-form specific.
- Instrumentation: `TargetLibraries/PULPOpen/src/Gemm.c` persists `g_zo_gemmA/B/C`, `g_zo_gemm_transB`;
  dumped by the harness under `ZO_TRACE_POOLIN`. `GlobalAveragePool.c` persists `g_zo_poolin/g_zo_poolout`.
- Open: is the perturbed fc_bias L2 tensor (offset 104104) itself interleaved (perturb/layout bug) or does
  the Gemm-C L2→L1 DMA introduce the stride-2 corruption? (build `inputform_biasl2.log`, `ZO_DUMP_PWOFF=104104`).
  Likely a bias-tiling/transfer bug for the 9-elem fc_bias when it is a perturbed INPUT vs a constant.

## COMPLETE ROOT CAUSE (confirmed)
- The perturbed fc_bias **L2 tensor itself is corrupted** (dumped offset 104104): even indices correctly
  perturbed, ODD indices wrong (stride-2). So it's the perturb DATA MOVEMENT, not the Gemm-C DMA.
- The perturb kernel + template are UNIFORM for all params (`ApplyRademacherPerturbation`, local_size split
  8 cores). So the kernel isn't the differentiator.
- **fc_bias is the ONLY odd-length param (9 elements).** fc_weight=288, BN γ/β = 8/16/16/32/32, conv biases =
  8/16/16/32/32 — all EVEN, all correct. So this is an **odd-length 1D transfer geometry bug**: the L2↔L1
  DMA for the 9-element fc_bias INPUT reads/writes with a stride-2 pattern that mishandles the odd length.
- It is **input-form specific**: in init-form fc_bias is a baked constant (perturbed correctly). It only breaks
  when fc_bias is a promoted graph INPUT (the deploy-prep promotion) — which is exactly the "make sure the
  promotion is ok" concern.
- Effect chain: fc_bias wrong on odd indices → fc Gemm C wrong on odd → logits wrong on odd → the entire
  ~0.002 step-0 bias. (conv biases ≈ 0 and BN γ/β even-length, so features stayed correct — that's why the
  whole conv/BN chain + pool + fc_weight all verified correct.)

## EXACT MECHANISM — L1 buffer overlap (definitive)
Generated `TrainingNetwork.c` L1 tile offsets:
- fc_bias (node 31): `data_in = MEMORYARENA_L1 + 0`, `data_out = MEMORYARENA_L1 + 4`  ← only **4 bytes (1 float)** apart!
- fc_weight (node 30): data_in +0, data_out **+1152** (ok). blocks_4_1_bias (32): +0 / **+128** (ok).
The fc_bias in/out buffers are each 36 bytes (9 floats) but placed 4 bytes apart → they **overlap**. The perturb
writes `data_out[i]` which equals `data_in[i+1]`, so each odd position reads the already-perturbed previous even
output → `data_out[2k+1] = data_out[2k] + z·ε`. This is a tiler **under-allocation** of the perturb's 1-D output
(sized as 1 element / 4 bytes instead of 9).

Root: `UnaryTileConstraint.addGeometricalConstraint` (Deeploy/Targets/Generic/TileConstraints/UnaryTileConstraint.py)
iterates `range(len(input1Shape))` and ties input dim == output dim, but the perturb OUTPUT `31_fc_bias` stores its
1-D shape as a **bare int** (per PerturbTileConstraint.py line 37's own note) while the promoted fc_bias INPUT is a
proper `(9,)` — the dim mismatch lets the tiler pick a 1-element output tile, under-allocating the L1 buffer and
overlapping data_in. Even-length tensors happen to still allocate enough to not overlap; only fc_bias (9, odd,
smallest) overlaps visibly. init-form's constant fc_bias uses a different (non-tiled) path, so it's unaffected.

## FIX DIRECTION
Fix the L2↔L1 transfer geometry for 1D (esp. odd-length) perturbed INPUT tensors so the fc_bias DMA is
contiguous. Candidate sites: the tiling transfer codegen for a rank-1 input buffer, and/or how the deploy-prep
promotes a 1D initializer to an input (shape/stride metadata: a constant (9,) is stored as a bare int; a
promoted input may carry a shape/stride that triggers a 2D/strided transfer). Quick check next: dump the
fc_bias perturb DATA_IN (L1) to see if the READ (L2 input_23 -> L1) is already stride-2 duplicated, vs the
WRITE (L1 -> L2). Then correct that transfer descriptor (or normalize the promoted 1D input shape).

## Next directions (not yet tried)
1. **Confirm the activation-tiling mechanism**: dump the GlobalAveragePool INPUT (last conv block output)
   via an in-forward instrumentation (copy to a persistent global from the pool kernel), OR compare the
   exact L1 tile dims of the last conv/pool between forms.
2. **Native input emission (user's end-goal path)**: re-enable `_promote_initializers_to_inputs` in
   Onnx4Deeploy `zo_transform.py` so the EXPORT emits weights-as-inputs (like BP), regenerate the fixture,
   and run WITHOUT the TrainDeeploy promotion. If the native input-form behaves the same → confirms it's a
   Deeploy tiling issue, not the promotion. If different → the promotion graph is subtly off.
3. If tiling-induced: make input-form tile activations like init-form (constrain the tiler / memory), or
   fix the tiled-op numerics.
