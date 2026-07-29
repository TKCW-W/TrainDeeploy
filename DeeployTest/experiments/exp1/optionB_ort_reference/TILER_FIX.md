# TILER_FIX — B-decompose frozen-BN tiler blocker: root cause + decision

**Date:** 2026-07-29 · **Branch:** TrainDeeploy `feat/BNFRozen_OptionB`
**Container (device):** `deeploy_arm_mounted` (GVSoC + RISC-V/LLVM + installed Deeploy)
**Fixture:** `Tests/Models/Training/SpeechNet/speechnet_train_decompose_b1_fold3` (9 steps)

## VERDICT: STOP — do NOT commit a tiler offset patch. The blocker is not a minimal offset bug.

The `minimizeRectangle` assertion is a **valid invariant** and the offending offset is indeed injected
upstream, but the true blocker runs deeper than "zero the size-1-axis offset": the decomposed per-channel
`(1,C,1,1)` `Add`/`Mul` broadcast **has no correct device kernel**. A tiler-only fix would make the graph
COMPILE while producing **numerically wrong results** (out-of-bounds / wrong-channel reads). This requires a
design decision (B-fused device binding, or keep Option A), not a small commit.

---

## 1. Reproduction (deterministic)

```
cd /app/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA && \
python deeployTrainingRunner_tiled_siracusa.py -t Tests/Models/Training/SpeechNet/speechnet_train_decompose_b1_fold3 \
  --n-steps 9 --n-accum 4 --cores 8 --l1 128000 --l2 2000000 --defaultMemLevel L2 \
  --memAllocStrategy MiniMalloc --searchStrategy max -D DUMP_WEIGHTS=ON
```

Aborts in:
```
Deeploy/TilingExtension/TilingCodegen.py:537, in minimizeRectangle
    assert rectOffset == 0, "Rectangle offset should be zero when the dimensions are the same. ..."
AssertionError: ... HyperRectangle(offset=(0, 0, 1, 0), dims=(1, 8, 1, 1)) and reference shape (1, 8, 1, 1)
```
Call path: `backEnd → codeTransform → PULPClusterTiling.apply → TilingCodeGeneration.generateTilingLoop →
SingleBufferingTilingCodeGeneration._tilingLoop → _generateTransferScheduleCalls → _legalizeTransfers →
minimizeRectangle`.

## 2. Root cause — traced end to end (file:line)

The failing transfer belongs to node **`node_14_blocks_0_blocks_0_2_Relu_Relu`**, tensor `data_in` =
`node_0_blocks_0_blocks_0_1_Add_1__0_tensor`. In the deployer context this activation buffer — whose true
shape is `(1,8,14,701)` — has been **collapsed to `(1,8,1,1)`**, and the transfer schedule advances its
offset along the H axis (`(0,0,0,0),(0,0,1,0),(0,0,2,0),…`) even though its H dim is 1. That is the illegal
rectangle.

Shape-collapse trace (deeployStates exports):
- `middleware_post_lowering.onnx`: `Conv__0 / Add__0 / Add_1__0` = **`(1,8,14,701)`** (correct)
- `backend_post_parsing.onnx`: same tensors = **`(1,8,1,1)`** (collapsed)

Where the collapse is decided — the tile-dimension solution:
- **`Deeploy/Targets/Generic/TileConstraints/BOPTileConstraint.py:38-44`** — `addGeometricalConstraint`
  adds, for **every** axis, `inputDim1Var == inputDim2Var` and `inputDim1Var == outputDimVar`.
  For the decomposed BN, `data_in_2` is the per-channel `(1,8,1,1)` broadcast constant
  (`blocks_0_1_neg_running_mean`, `inv_running_std`, γ, β). Its H and W dim vars are pinned to 1 (its own
  size). Through the `==` chain this force-collapses the **activation** output tile (and, transitively,
  `Conv__0` and the whole block) to H=1, W=1.
- **`Deeploy/Targets/Generic/TileConstraints/BOPTileConstraint.py:70-71`** — `serializeTilingSolution`
  assigns the **same output cube** (with its H/W offsets) to **both** inputs, so a `(1,8,1,1)` broadcast
  operand inherits a non-zero H offset it must never have.
- The solved memory-constraint shape is then written per-dim from these dim vars in
  **`Deeploy/TilingExtension/TilerExtension.py:1044-1052`**, yielding `(1,8,1,1)` for the activation, which
  is what `_legalizeTransfers → minimizeRectangle` finally rejects.

Confirmed empirically (temporary diagnostics, since reverted): at BOP serialize time all four decomposed
Add/Mul ops report **in1 == in2 == out == (1,8,1,1)** — i.e. the broadcast operand has already dragged the
activation to `(1,8,1,1)` before the ReLU transfer is legalized.

## 3. Why a tiler-only offset fix is NOT sufficient (the real blocker)

Even if `BOPTileConstraint` were fixed to (a) exclude size-1 broadcast axes from the `input==input`
equality and (b) zero the broadcast operand's offset/dims on those axes — so the graph tiles the activation
as `(1,C_tile,H,W)` and keeps the param `(1,C_tile,1,1)` — the **device kernels cannot compute the
per-channel broadcast**:

- `Deeploy/Targets/PULPOpen/Templates/FloatAddTemplate.py` (bound for float Add) computes
  `data_out[i] = data_in_1[i] + data_in_2[i]` for `i in [0, size)`, where `size = prod(data_in_1.shape)`
  (the full tile). A `(1,C,1,1)` param has only `C` elements → this reads **out of bounds** on `data_in_2`
  for any tile with H·W > 1. There is **no per-channel / scalar Add** template.
- `Deeploy/Targets/PULPOpen/Templates/FloatMulTemplate.py` (bound for float Mul) computes
  `A[i] * B[0]` — a **single scalar** broadcast. It is correct only when the tile is one channel
  (`B[0]` = that channel's value); it silently applies channel-0's value to all channels for any C_tile > 1.

So the decomposed per-channel `Add`/`Mul` fundamentally does not map onto the existing PULP element-wise
kernels. Making it work needs **new broadcast-aware kernels** (or DMA replication of the `(1,C,1,1)`
operand to `(1,C,H,W)` per tile) plus a matching tile constraint — a deep, invasive change, exactly the
kind the task says to STOP on. This matches the pre-existing `PLAN.md` note that decomposing BN "would map
on device to generic elementwise kernels (loses the fused BN kernel)."

## 4. Smoke result (decompose fixture)

- Compile: **FAIL** (baseline, unchanged) — the assertion above.
- Run (GVSoC): did not run (blocked at compile).
- Loss match: N/A.
- No tiler patch was applied (correct decision — see verdict).

## 5. Regression guard — the tiler is healthy for non-broadcast training graphs

`Tests/Models/Training/SpeechNet/speechnet_train` (head/no-BN training graph; ops: Conv/Relu/MaxPool/
GlobalAveragePool/Gemm/SoftmaxCrossEntropyLoss + grads/ReduceSum/InPlaceAccumulatorV2; **0
BatchNormInternal, 0 decompose broadcast**) — same compile command, `--n-steps 9`:

```
Errors: 0 out of 36
BENCH train_cycles=609939530 opt_cycles=34437 weight_sram=1188
✓ Test speechnet_train PASSED - No errors found
```
Device per-step loss matches ORT bit-exactly (all diffs 0.0, TOL 1e-3). This confirms the failure is
**specific to the decompose `(1,C,1,1)` broadcast**, not a general tiler regression.

Additional check — fused-BN frozen fixture `Tests/Models/Training/SpeechNet/speechnet_train_fullfrozen`
(5 fused `BatchNormInternal`, Option-A path), same command + `-D BN_FROZEN_STATS=ON`, `--n-steps 9`:
**COMPILES and RUNS on GVSoC** (binary built, `train_cycles=1263363084`, 36 device losses produced). The
per-step losses do **not** match this fixture's `outputs.npz` (35/36 over TOL) — but that is the
**pre-existing frozen-BN ORT-reference mismatch** (device = frozen stats vs the fixture reference =
live-batch BN; the known open TODO that motivates Option B), **not** a tiler/compile issue and **not**
caused by anything here. The tiler compiled the fused-BN broadcast graph without any `minimizeRectangle`
error — reinforcing that the blocker is unique to the **decomposed element-wise** `(1,C,1,1)` broadcast.

## 6. Recommended fallback (per the user's decision points)

- **B-fused device** (cleanest for "still use a BN kernel"): keep the graph decomposed **only for the ORT
  reference**, but bind the existing-but-unused `PULP_ChannelNormalize_fp32` (fwd) + a channel-normalize
  grad and add a Deeploy fusion so the device runs a single per-channel affine kernel — no `(1,C,1,1)`
  broadcast DMA, no new element-wise kernels. This is the fix that actually unblocks the device path.
- **Option A (B0)**: ship B-decompose as the ORT-validated reference only, and run the device chain with the
  fused `BatchNormInternal` + `-D BN_FROZEN_STATS=ON` (already validated). Reference graph ≠ device graph,
  but frozen math is identical.

No changes were committed for this blocker; the working tree under `Deeploy/` is clean.
