# FINDING — Option B exploration: ORT-computed frozen-BN reference on the same graph

**Action date/time:** 2026-07-29 (full day; investigations + prototype + device smoke + tiler analysis + scope).
**Branches:** Onnx4Deeploy `feat/BNFRozen_OptionB` · TrainDeeploy `feat/BNFRozen_OptionB`
(created so the working state on the old branches — `feat/speechnet/inference/maxpool_ondevice` /
`feat/speechnet/on-device-FT` — stays untouched).
**Outcome / decision:** **Reverted to Option A** (per user). B-decompose is documented here as the explored
approach; **B-fused** is the recommended future path to a true same-graph-on-device solution.

---

## Goal
Replace the Option-A reference (per-step loss computed by a **PyTorch** `eval()` model) with an
**ORT-computed** frozen-BN reference **on the same graph the device runs**, so the ORT↔device loss
comparison is clean — the residual = pure on-device numerical effects (fp32 tiling, SUM accumulation,
MaxPool argmax drift), not an engine confound — while **still using the BatchNorm kernel** on device.

## What we confirmed about the graph and kernels
- The training graph's BN is a **single fused `BatchNormInternal(training_mode=1)`** node (inputs
  `[X, γ, β, running_mean, running_var]`, outputs `[Y, running_mean_def, running_var_def, saved_mean,
  saved_inv_std]`), emitted by ORT's `generate_artifacts`. It is **NOT decoupled in the ONNX graph.**
- The supervisor's "decoupled into (1) compute batch stats, (2) affine normalization" describes the
  **device C kernel** (`TargetLibraries/PULPOpen/src/BatchNorm.c`): the *mapped* `PULP_BatchNormInternal_fp32`
  is a single fused pass, but decoupled halves exist unused — `PULP_WelfordReduce` (stats) +
  `PULP_ChannelNormalize` (affine), and `PULP_BNGradReduce`/`PULP_BNGradNormalize` on the backward.
- ORT **cannot** be made to compute frozen BN on the fused node: `training_mode=1` always uses batch stats
  and `BatchNormalizationGrad` always applies the batch-stat Jacobian. No ORT knob changes this. (This is
  the root of the original 2160/2160 mismatch that Option A worked around with a PyTorch reference.)

## The approach explored — "B-decompose" (VALIDATED on ORT)
Make ORT compute the frozen reference by expressing BN as a **differentiable frozen affine** *before*
`generate_artifacts`, so ORT autodiffs it natively:
- New flag `--bn-decompose-frozen` swaps every `nn.BatchNorm*` → a `FrozenAffineBN` module whose forward is
  `y = (x + neg_running_mean)·inv_running_std·γ + β` in **plain Add/Mul** (running stats baked as constants;
  `neg_running_mean=−μ`, `inv_running_std=1/√(σ²+ε)`; γ/β trainable). Deliberately Add/Mul only — avoids the
  unbound `Sub`/`Sqrt`/`Reciprocal`. (Onnx4Deeploy commit `0858180`, now reverted in `c9285df`.)
- **Result (Test 1, `RESULT.md`):** decomposed `network_train.onnx` has **0 `BatchNormInternal`**; ORT
  autodiffs the frozen affine and the per-step reference loss matches the Option-A PyTorch frozen reference
  to **max |Δ| = 8.3e-7** over 36 steps, γ/β/conv/fc all train, stats stay frozen. Frozen BN is now **baked
  into the graph** → the `g_bn_frozen_stats` C flag would no longer be needed on this path, and ORT + device
  would run the **same** graph.

So on the **reference/ORT side the clean approach works** — an ORT-native frozen reference is achievable and
numerically equivalent to Option A.

## Why it does NOT run on device (Test 2 + tiler analysis)
The decomposed graph **fails to compile** for Siracusa (`TEST2_RESULT.md`, `TILER_FIX.md`):
1. **Tiler collapse:** `BOPTileConstraint` (the binary-op constraint for the decomposed Add/Mul) pins
   `in1==in2==out` on every axis. The per-channel `(1,C,1,1)` BN operand pins H,W to 1, force-collapsing the
   whole activation tile → a malformed DMA rectangle (`offset=1` on a size-1 axis) rejected by the *valid*
   invariant in `TilingCodegen.py:537 minimizeRectangle`.
2. **Deeper — no per-channel broadcast kernel:** even with a tiler fix the graph would be **numerically
   wrong** — the bound `FloatAdd`/`FloatMul` kernels have **no per-channel `(1,C,1,1)` broadcast**
   (`FloatMul` uses `B[0]` scalar; `FloatAdd` reads out of bounds). So "all ops bound" was misleading.
3. **Regression:** the tiler is healthy for normal graphs — `speechnet_train` compiles bit-exact (0/36),
   and the Option-A fused-BN `speechnet_train_fullfrozen` compiles+runs. The failure is specific to the
   decomposed `(1,C,1,1)` broadcast.

## Recommendation for a true same-graph-on-device solution — "B-fused" (future work)
Keep the decomposed graph for the ORT reference, and add a **Deeploy fusion** that maps the decomposed
frozen-affine (fwd `Add→Mul→Mul→Add`, and the ORT-autodiff bwd) onto the **existing per-channel kernels**
`PULP_ChannelNormalize` (fwd) + `PULP_BNGradReduce`/`PULP_BNGradNormalize` (bwd) — which read `param[c]`
natively (no broadcast DMA, no tiler collapse). This also satisfies "still use the BatchNorm kernel."
- **Already present:** the C kernels (`BatchNorm.c:159-278`), templates (`FloatBatchNormTemplate.py:68-119`),
  **and tile constraints** (`BatchNormTileConstraint.py:307-646`, C/H/W free → avoids the BOP collapse).
- **To add:** a fusion pass (`ReplaceSequentialPatternPass` + `NonBranchingMatcher`, ~200 LOC; hardest part =
  matching ORT's autodiff **backward** subgraph, which can vary in node ordering) + 3 TypeCheckers + 3
  Parsers + 3 Bindings + Platform.py mapping + Tiler bindings (boilerplate). **Reshape `[1,C,1,1]`→`[C]`
  constants inside the fusion.** Must ship fwd+bwd together.
- **Scope estimate:** ~400-450 LOC, ~2-3 weeks, **moderate risk** (backward pattern-match).

## Interim alternative that was available (A2, not pursued)
Use the B-decompose **ORT** graph as the frozen reference while running the device chain with the working
fused `BatchNormInternal` + `-D BN_FROZEN_STATS=ON`. Both compute the **same frozen math**, so the ORT↔device
residual is still ~pure on-device numerical effects — but the reference graph ≠ device graph, so it does not
meet the strict "same graph" bar. (Superseded by the decision to revert to Option A.)

## Decision (user, 2026-07-29)
**Revert to Option A** (PyTorch-computed frozen reference — the working, validated device path) and document
the B-decompose exploration + B-fused recommendation. Done:
- Onnx4Deeploy: `--bn-decompose-frozen`/`FrozenAffineBN` reverted (`c9285df`); Option A (`--bn-frozen-stats`)
  is the sole path. No Deeploy source was modified (the tiler "fix" was deliberately **not** committed — it
  would have compiled but produced numerically wrong results).
- The exploration lives in git history + these docs; **B-fused** is the recommended next step if/when a
  strict same-graph ORT↔device loss comparison is wanted.

## Files (this dir)
`PLAN.md` (full decision trail), `RESULT.md` (Test 1 — ORT validation), `TEST2_RESULT.md` (device compile
failure), `TILER_FIX.md` (tiler root cause + why not fixed), `validate_test1.py`, `logs/`.
Related: exp2 (`../../exp2/`) — on-device FT latency/memory tutorial + profiling (the fused-BN Option-A path).
