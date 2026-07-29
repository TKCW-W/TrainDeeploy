# Test 2 RESULT — B-decompose frozen-BN device smoke (tiled Siracusa GVSoC trainer)

**Date:** 2026-07-29 · **Branch:** Onnx4Deeploy `feat/BNFRozen_OptionB` @ `0858180` · TrainDeeploy `feat/BNFRozen_OptionB` @ `2dc59a9`
**Containers:** export in `agitated_hugle` (has Onnx4Deeploy + SilentWear + data); device run in `deeploy_arm_mounted` (has GVSoC + RISC-V/LLVM toolchain; `/app/TrainDeeploy` is the shared host mount).

## Verdict: ❌ COMPILE FAIL — the decomposed frozen-BN training graph does NOT compile on the tiled Siracusa trainer.

The graph exports cleanly and is device-op-clean (no unbound ops), but **tiled code generation aborts** in
Deeploy's DMA transfer legalization on the decomposed-BN `(1, 8, 1, 1)` broadcast tensors. Because
generation never completes, **no binary is built, GVSoC never runs, and there is no device per-step loss to
compare against the ORT reference.**

- **Compile:** ❌ FAIL (deterministic — see below)
- **Run (GVSoC):** ❌ did not run (blocked by compile)
- **Loss match:** N/A (no device loss produced)
- **Unbound-op gap:** NONE — this is NOT an op-binding gap. All ops (Add/Mul/ReduceSum/Reshape/Expand)
  are bound; the failure is in the tiler.

## Export — PASS (fast smoke fixture)
`network_train.onnx` op audit (decomposed): **0 `BatchNormInternal`, 0 `BatchNormalization`.** BN region is
Add/Mul/ReduceSum/Reshape only; exactly **1 `Expand`** (same as Option A — not introduced by decompose).
Full op counts: Add 10, Conv 5, ConvGrad 5, Expand 1, Gemm 3, GlobalAveragePool 1, Identity 21,
InPlaceAccumulatorV2 22, MaxPool 3, MaxPoolGrad 3, Mul 25, ReduceSum 11, Relu 5, ReluGrad 5, Reshape 32,
Scale 1, Shape 1, SoftmaxCrossEntropyLoss 1, SoftmaxCrossEntropyLossGrad 1.
ORT reference losses (9 update steps): `[0.52765, 0.63712, 0.01564, 1.20471, 0.02929, 0.33212, 0.01547, 0.31906, ...]`.

## Compile FAILURE — exact error (the critical finding)

Fatal exception during backend `codeTransform` → `PULPClusterTiling` → DMA transfer legalization:

```
File ".../Deeploy/TilingExtension/TilingCodegen.py", line 537, in minimizeRectangle
    assert rectOffset == 0, f"Rectangle offset should be zero when the dimensions are the same. ..."
AssertionError: Rectangle offset should be zero when the dimensions are the same.
Received rectangle HyperRectangle(offset=(0, 0, 1, 0), dims=(1, 8, 1, 1)) and reference shape (1, 8, 1, 1)
```
Call path: `DeeployTypes.backEnd → codeTransform → MemoryLevelDeployer.codeTransform → layer.codeTransform →
PULPClusterTiling.apply → TilingCodeGeneration.generateTilingLoop → SingleBufferingTilingCodeGeneration._tilingLoop →
_generateTransferScheduleCalls → _legalizeTransfers → minimizeRectangle`.

**Root cause / interpretation.** The decomposed BN emits, per block, four `Reshape` ops that lift the `[C]`
frozen/affine vectors (`neg_running_mean`, `inv_running_std`, `weight`/γ, `bias`/β) to `(1, C, 1, 1)` and
broadcasts them via `Add`/`Mul` against the `(1, C, H, W)` feature map; the backward pass adds
`ReduceSum`+`Reshape` producing `(1, C, 1, 1)` gradient tensors. For the first block (C=8), the tiler produced a
**degenerate transfer rectangle** `offset=(0, 0, 1, 0)` on a size-1 axis (dim=1, ref=1) — a non-zero offset on a
dimension that fully spans its reference — which `minimizeRectangle` rejects. This is a **Deeploy tiler / DMA
legalization bug provoked by the `(1, C, 1, 1)` broadcast+reduction tensors of the decomposed BN**, not an
unbound operator.

**Deterministic (not a search artifact):** reproduced identically with `--searchStrategy random-max` AND
`--searchStrategy max` — same rectangle `offset=(0,0,1,0), dims=(1,8,1,1)` both times.

**Note (non-fatal, separate):** ORT shape-inference during partitioning also logs
`INVALID_ARGUMENT : Invalid model. Node input '...Reshape_1_Reshape' is not a graph input, initializer, or
output of a previous node.` and duplicate-initializer warnings for the shared `Constant` reshape targets. The
deployer proceeds past these; they are not the cause of the abort but hint the shared decompose `Constant`
reshape-shape initializers are duplicated across blocks and may warrant cleanup.

## Exact commands used

Export (in `agitated_hugle`, cwd `/app/Onnx4Deeploy`):
```
python3 Onnx4Deeploy.py -model SpeechNet -mode train \
  -o /app/TrainDeeploy/DeeployTest/Tests/Models/Training/SpeechNet/speechnet_train_decompose_b1_fold3 \
  --dataset silentwear --data-path /app/SilentWear/SilentWear_data/data_raw_and_filt \
  --pretrained-weights /app/SilentWear/SilentWear/artifacts/models/inter_session_ft/S01/vocalized/speechnet/w1400ms/model_1/leave_one_session_out_fold_3.pt \
  --subject S01 --session 3 --batch 1 --condition vocalized \
  --data-size 18 --n-epochs 2 --n-accum 4 --lr 0.0003 \
  --training-strategy full --bn-decompose-frozen --stratified
```
Device run (in `deeploy_arm_mounted`, cwd `/app/TrainDeeploy/DeeployTest`):
```
rm -rf TEST_SIRACUSA && python3 deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_train_decompose_b1_fold3 \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_optimizer_decompose_b1_fold3 \
  --n-steps 9 --n-accum 4 --cores 8 \
  --l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc \
  --searchStrategy random-max -D DUMP_WEIGHTS=ON \
  2>&1 | tee experiments/exp1/optionB_ort_reference/logs/test2_device_smoke.log
```
(`--optimizer-dir` passed explicitly per the `_optimizer` fixture convention. `-D BN_FROZEN_STATS` correctly
OMITTED — frozen semantics are baked into the decomposed graph.)

Full log: `logs/test2_device_smoke.log`.

## Environment note (setup, not a result)
`agitated_hugle` (a plain `python:3.10` image) has Onnx4Deeploy + SilentWear + data but NO RISC-V/GVSoC
toolchain and no installed `Deeploy` package — it cannot compile/run for Siracusa. The device chain must run
in `deeploy_arm_mounted` (GVSoC + LLVM/RISC-V present, TrainDeeploy on `feat/BNFRozen_OptionB`). Both
containers share the host `/Users/qiwenwu/ETH/TrainDeeploy` at `/app/TrainDeeploy`, so the fixture exported in
`agitated_hugle` is directly consumed in `deeploy_arm_mounted`.

## Blocker / next steps
- **BLOCKER:** decomposed frozen-BN training graph does not tile-compile on Siracusa. The gate for the
  expensive full chain is **NOT** met with the current tiler.
- The fix is in **Deeploy tiling**, not the exporter: the `(1, C, 1, 1)` broadcast/reduction tensors of the
  decomposed BN must not produce non-zero offsets on size-1 axes during DMA transfer legalization
  (`_legalizeTransfers` / `minimizeRectangle`). Options to investigate: (a) fix the rectangle construction so
  size-1 axes carry zero offset; (b) mark these broadcast/reduction tensors as non-tiled (replicate whole to
  L1); (c) collapse the `(1,C,1,1)` reshape/broadcast so the tensor is presented as `[C]` to the tiler.
- Until the tiler is fixed, Option B cannot be validated end-to-end on device; the Option-A path (which the
  exp1 chain already runs on device) remains the working device route.
