# PLAN — Option B: ORT reference loss on the SAME graph as the device (frozen BN)

**Started:** 2026-07-29
**Branch:** TrainDeeploy `feat/BNFRozen_OptionB` · Onnx4Deeploy `feat/BNFRozen_OptionB`
(new branches so the current state stays untouched on the old branches)

> **FINAL STATUS (2026-07-29): CLOSED — reverted to Option A per user.** B-decompose was validated on the
> ORT side (frozen reference matches Option A to 8.3e-7) but cannot run on device (tiler collapse + no
> per-channel broadcast kernel); the true same-graph fix is **B-fused** (~2-3 wk, moderate risk). See
> `FINDING.md` for the full write-up + recommendation. Onnx4Deeploy reverted in `c9285df`.

## Motivation (why not Option A)
Option A computes the per-step reference loss in a **PyTorch** `eval()` model, not by ORT on the
training graph. It is legit as a cross-check, but it is not *clean*: ORT and the device no longer run
the **same graph**, so the loss comparison mixes "engine difference (PyTorch vs device)" with the real
target — "loss difference due to the on-device constraints (frozen BN, SUM accumulation, fp32 tiling,
MaxPool argmax drift)". We want **ORT-vs-device on one identical graph** so the residual is purely the
on-device numerical effects.

## Goal
Make the ORT-computed reference loss use **frozen BN stats on the same `network_train.onnx`** the device
runs, so the bit-exact training test compares like-for-like.

## Key facts established
- The training graph's BN is a **single fused `BatchNormInternal`** node (inputs
  `[X, scale, bias, running_mean, running_var]`, outputs `[Y, running_mean_def, running_var_def,
  saved_mean_def, saved_inv_std]`, attrs `epsilon`, `momentum=0.9`, `training_mode=1`). It is emitted by
  ORT's `artifacts.generate_artifacts`. It is **NOT** decoupled in the ONNX graph.
- **Device BN kernel (confirmed):** `TargetLibraries/PULPOpen/src/BatchNorm.c`.
  - The *mapped* forward `PULP_BatchNormInternal_fp32()` is a **single FUSED pass** (stats reduction +
    affine in one loop), NOT decoupled. BUT decoupled halves **exist unused**:
    `PULP_WelfordReduce_fp32()` (batch stats) + `PULP_ChannelNormalize_fp32()` (affine
    `y=gamma*(x-mean)*inv_std+beta`), and on the backward side `PULP_BNGradReduce`/`PULP_BNGradNormalize`.
    So the supervisor's "decoupled stats+affine" is a **latent capability in the C library**, not the
    active graph mapping.
  - `g_bn_frozen_stats` is a **runtime gate** (global in BatchNorm.c:19; set to 1 by `-D BN_FROZEN_STATS`
    via `deeploytraintest.c`). Forward: skips the reduction, uses `running_mean/var`. Backward: collapses
    to the pure frozen affine grad **`dX = gamma*inv_std*dY`**, `dgamma=Σ dY·xhat`, `dbeta=Σ dY` (drops
    the batch-stat Jacobian terms). → This is **exactly the autodiff of a frozen affine**
    `y=gamma*(x-mu_const)*inv_std_const+beta`.
- **No standalone affine/scale-shift ONNX binding exists.** Decomposing BN→`Sub/Mul/Mul/Add` in the graph
  would map on device to **generic elementwise kernels** (loses the fused BN kernel) UNLESS we add a
  Deeploy fusion/binding onto the existing (but unbound) `PULP_ChannelNormalize` + its grad.

## The core tension (now precise)
1. ORT **cannot** compute frozen BN on the fused `BatchNormInternal` node — training_mode=1 always uses
   batch stats, and its `BatchNormalizationGrad` always applies the batch-stat Jacobian. No ORT knob
   (momentum, saved-stat injection) changes this.
2. So "ORT reference = frozen" **requires the graph to express BN as differentiable frozen affine**
   (decomposed `Sub/Mul/Mul/Add`, running stats as constants, γ/β trainable) — ORT autodiffs that to the
   exact frozen grad the device computes.
3. But "device still uses the BatchNorm kernel" wants the **fused** node kept.
→ (1)+(2)+(3) can only ALL hold via **B-fused-device**: decompose in the graph (so ORT is frozen) **and**
  add a Deeploy fusion/binding so the decomposed frozen-affine maps onto the split BN kernel
  (`PULP_ChannelNormalize` fwd + a channel-normalize grad) — a "BN kernel", just the affine half.
  Without the device binding it's **A2** (ORT reference on the decomposed graph; device keeps fused
  `BatchNormInternal`+flag) — ORT-computed reference, same frozen math, but reference graph ≠ device graph.

## DECISION (2026-07-29, after fact-finding)
Prototype **B-decompose**: a post-`generate_artifacts` rewrite (mirroring `_rewire_maxpoolgrad_recompute`,
base_exporter.py:710-755) that replaces each fused `BatchNormInternal(training_mode=1)` + its
`BatchNormalizationGrad` with the **frozen-affine decomposition** using the frozen running stats as
constants and γ/β as trainable inputs:
`xhat = (x − running_mean_const) · inv_std_const ; y = γ·xhat + β` (as `Sub → Mul → Mul → Add`).
ORT then autodiffs this **natively** → the per-step reference loss is **ORT-computed, frozen, on the same
`network_train.onnx` the device runs** (goals i + iii). This lets us drop both the Option-A PyTorch
reference and the `g_bn_frozen_stats` C flag.

Two empirical tests gate the path:
- **Test 1 (exporter):** does ORT run the decomposed `network_train.onnx` and produce a frozen reference
  loss that matches the Option-A PyTorch frozen reference (bit-close)? Validates goals i+iii.
- **Test 2 (device):** can the Deeploy tiled Siracusa training runner compile+run the decomposed BN
  training graph — i.e. are `Sub/Mul/Add` **and their grads** bound? If yes, B-decompose is fully viable
  (device stops using the fused BN kernel → elementwise kernels; goal ii met only in spirit). If the
  grads are missing / too slow, escalate to **B-fused**: a Deeploy fusion pass + binding mapping the
  frozen-affine pattern onto the existing-but-unbound `PULP_ChannelNormalize` fwd +
  `PULP_BNGradReduce/Normalize` bwd (device keeps a real "BN kernel"; goal ii met fully).
Fallback if both are impractical: keep **Option A** (per user).

## Test 2 result (device feasibility, confirmed 2026-07-29)
- **`Sub` is UNBOUND in PULPOpen** → a decomposed forward using `Sub(x, μ)` would fail to compile. But
  **`Add`, `Mul`, `ReduceSum`, `Reshape` ARE bound.** → **Sidestep:** write the affine with a *negated
  mean* constant so it's all Add/Mul: `y = (x + (−μ)_const)·inv_std_const·γ + β`. The `FrozenAffineBN`
  module should store `neg_running_mean = −running_mean` and use `x + neg_running_mean` so torch exports
  `Add`, not `Sub`. (Told the Test-1 fork's follow-up; Sub-vs-Add is irrelevant to the ORT reference, only
  to device compilation, so it can be applied after Test 1 confirms the ORT side.)
- ORT autodiff of the Add/Mul chain emits `Mul`/`ReduceSum`/`Reshape` (all bound) and possibly **`Expand`**
  (broadcast, **unbound**). Residual risk = one small `Expand` (or Reshape-based broadcast) binding, TBD by
  actually compiling the graph.
- **B-fused fallback** (device keeps a real "BN kernel"): the split kernels `PULP_ChannelNormalize_fp32`
  (fwd) + `PULP_BNGradReduce`/`PULP_BNGradNormalize` (bwd) already have C code **and Deeploy templates**
  (`FloatBatchNormTemplate.py:53-119`) but **no TypeCheckers/Bindings** — needs ~4 checkers + 4 Bindings
  entries + Platform.py mapping. Bigger than the Add/Expand route but keeps a fused kernel.

**Revised device plan:** try B-decompose with the all-Add/Mul graph first (smallest device delta — at most
an `Expand` binding). Only if that proves impractical, do B-fused (ChannelNormalize bindings).

## ✅ Test 1 result (2026-07-29) — B-decompose VALIDATED (see RESULT.md)
Onnx4Deeploy commit `0858180`: `--bn-decompose-frozen` swaps `nn.BatchNorm*` → `FrozenAffineBN`
(`y=(x+neg_running_mean)·inv_running_std·γ+β`, Add/Mul only) before export → ORT autodiffs it natively.
- Decomposed `network_train.onnx`: **0 `BatchNormInternal`**; ops = `Add/Mul/ReduceSum/Reshape` (+`Expand:1`
  which is already in the Option-A graph). **No new unbound ops.**
- **ORT reference (this graph) vs Option-A PyTorch frozen ref: max |Δloss| = 8.3e-7** over 36 steps. γ/β
  train, stats frozen. Reference routing unchanged (distinct config key → existing ORT path).
- Frozen BN is now **baked into the graph** → the `g_bn_frozen_stats` C flag is **no longer needed** on
  this path; ORT and device run the **same** decomposed graph.

## Next steps (Goal 1)
1. **Test 2 — real device smoke:** compile+run the tiled Siracusa trainer on a small `--bn-decompose-frozen`
   fixture (NO `BN_FROZEN_STATS` needed) → confirm it compiles (no unbound op) and device per-step loss
   matches the ORT reference within tolerance up to the MaxPool-drift onset. [IN PROGRESS]
2. If device smoke passes → run the full **b1→b5** incremental FT chain (S01 voc fold3) with ORT-vs-device
   loss comparison; reuse `../ondevice_sim_S01_fold3` extract/compare scripts. Accuracy chain for parity.
3. Write `FINDING.md` (dated) — the clean ORT-vs-device loss delta = pure on-device numerical effects.

## ⚠️ Test 2 result (2026-07-29) — device COMPILE blocked by a Deeploy tiler bug (see TEST2_RESULT.md)
- Export PASS, graph device-op-clean (0 BatchNormInternal; Add/Mul/ReduceSum/Reshape/Expand — all bound;
  `Expand:1` same as Option A). **NOT an unbound-op gap.**
- Compile FAIL: `Deeploy/TilingExtension/TilingCodegen.py:537 minimizeRectangle` asserts
  `offset should be zero when dims match` on `HyperRectangle(offset=(0,0,1,0), dims=(1,8,1,1))` ref `(1,8,1,1)`.
  The assertion is a **valid invariant**; the bug is **upstream** — a `(1,C,1,1)` per-channel BN
  broadcast/reduction tensor got `offset=1` on a **size-1 (H) axis**. Almost certainly the tiler propagates
  a tiled activation's spatial offset onto a broadcast tensor's size-1 axis (broadcast along a tiled axis
  must stay offset 0). Deterministic across search strategies.
- Note: run device compiles in **`deeploy_arm_mounted`** (GVSoC+RISC-V); export can be in either container
  (both mount host `/Users/qiwenwu/ETH/TrainDeeploy` → `/app/TrainDeeploy`). Smoke fixture already exported:
  `Tests/Models/Training/SpeechNet/speechnet_train_decompose_b1_fold3` (9 steps). Non-fatal: a
  duplicate-initializer / dangling-Reshape "Invalid model" ORT warning on the shared decompose Constants —
  clean up later.

## Path forward (chosen): fix the Deeploy broadcast-tiling bug
Attempt a **minimal, regression-guarded** fix so `(1,C,1,1)` broadcast/reduction tensors don't get non-zero
offsets on size-1 axes (zero the tile offset on broadcast/size-1 axes at the point offsets are assigned).
This directly unblocks the clean B-decompose device path and helps any broadcast-tensor training graph.
- If the fix is localized+safe (existing training fixtures still compile) → proceed to the full b1→b5
  device chain with ORT-vs-device loss comparison.
- If it proves deep/risky → fall back options: **B-fused** (bind `PULP_ChannelNormalize`+`BNGrad*` and add a
  Deeploy fusion so the device uses a single BN-affine kernel — avoids the broadcast DMA entirely and
  satisfies "still use the BN kernel"), or keep **Option A** for the device chain while shipping
  B-decompose as the ORT-validated reference (per user's "keep Option A" fallback).

## B-fused scope (2026-07-29, read-only investigation)
Feasible, ~**400-450 LOC, ~2-3 weeks, moderate risk**. Already present: the C kernels
(`PULP_ChannelNormalize`/`BNGradReduce`/`BNGradNormalize`, BatchNorm.c:159-278), their templates
(FloatBatchNormTemplate.py:68-119), **and their tile constraints** (BatchNormTileConstraint.py:307-646,
C/H/W free → avoids the BOP broadcast collapse). Missing: (a) a **fusion pass** (~200 LOC,
`ReplaceSequentialPatternPass`+`NonBranchingMatcher` in PULPOpen Passes.py) to match the decomposed
fwd `Add→Mul→Mul→Add` **and** the ORT-autodiff bwd (Mul/ReduceSum/Reshape) → the 3 fused nodes, reshaping
the `[1,C,1,1]` consts to `[C]`; (b) 3 TypeCheckers + 3 Parsers + 3 Bindings + Platform.py mapping + Tiler
bindings (boilerplate). Hardest/riskiest = the **backward pattern-match** (ORT node ordering variability);
must ship fwd+bwd together (training needs both).

## Decision point (raised to user 2026-07-29)
- **Core win is DONE:** B-decompose gives an ORT-computed frozen reference (matches Option A to 8.3e-7).
- **True "same graph on device" needs B-fused** (~2-3 wk). Interim **A2** (ORT decomposed reference +
  working fused-BN device via `BN_FROZEN_STATS`) is available now but reference-graph ≠ device-graph.
→ Asked user: implement B-fused now / ship B-decompose+A2 interim / revert to Option A.

## Open question the investigations must settle
Is there a **clean** way to make ORT compute the frozen-BN forward+backward on a graph the device runs
with its BN kernel? Candidate approaches (to be scored once fact-finding returns):
- **B1 — post-generation graph rewrite:** after `generate_artifacts`, surgically replace the fused
  `BatchNormInternal` (training_mode=1) with a frozen-stats form ORT differentiates natively while the
  device still maps it to the (decoupled) BN kernel with frozen stats.
- **B2 — decompose to affine (Mul/Add) + freeze:** replace BN with `xhat=(x-mean_frozen)*inv_std_frozen`,
  `y=gamma*xhat+beta` using frozen running stats as constants; ORT autodiffs it; device runs the affine
  part (needs a device affine/BN-affine kernel — under investigation).
- **B0 — keep Option A** if neither B1 nor B2 is clean.

## Deliverables
- Chosen approach + rationale; implementation in Onnx4Deeploy; smoke test; full b1→b5 incremental FT
  (S01 vocalized fold 3) with ORT-vs-device loss comparison.
- `FINDING.md` — dated, with the loss-difference decomposition and the same accuracy chain as the
  Option-A run for comparison.

*(This PLAN will be updated with the concrete approach once the three fact-finding investigations
report: exporter internals, device BN-kernel decoupling, profiling tooling.)*
