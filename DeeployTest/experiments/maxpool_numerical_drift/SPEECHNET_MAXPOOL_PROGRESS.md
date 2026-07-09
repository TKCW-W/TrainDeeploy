# SpeechNet MaxPool On-Device Deployment (Inference + Fine-Tuning)

Branch: `feat/speechnet/inference/maxpool_ondevice`

**Goal:** Deploy SpeechNet on Siracusa (GVSoC) **100% loyal to the paper** — i.e. with
**MaxPool** (the paper architecture) rather than the AvgPool revision — for **both
inference and on-device fine-tuning**, and overcome MaxPool's argmax-index storage need
by **recomputing the argmax from the forward input** instead of storing indices.

## Result — achieved

| Capability | Status | Evidence |
|---|---|---|
| MaxPool **inference** on-device | ✅ bit-exact | 18/18 samples `sim_errors=0` vs ORT; balanced acc 70.6% (full set), ORT == on-device |
| MaxPool **fine-tuning** on-device (tiled) | ✅ **bit-accurate** | 4-step run `Errors: 0/4` @ **TOL=0.01** (tight); all loss diffs **1e-6**; `train_cycles=137M` |
| No argmax/index storage | ✅ | MaxPoolGrad **recomputes** argmax from the retained forward input X |

On-device 4-step training losses vs ORT reference (after the MaxPoolGrad layout fix, see bug #4):

```
[loss 0] computed=2.863738  ref=2.863737  diff=0.000001
[loss 1] computed=1.609689  ref=1.609690  diff=0.000001
[loss 2] computed=2.550769  ref=2.550768  diff=0.000001
[loss 3] computed=1.482667  ref=1.482667  diff=0.000001
```
All steps match the reference to **1e-6** — bit-accurate, no tolerance loosening. (Before the
layout fix, steps 1-3 drifted ≈0.02; that was a real correctness bug, not accumulation — see #4.)

## Bugs found & fixed

1. **MaxPool inference produced wrong logits on-device** (9/9 mismatch).
   Root cause: `FloatMaxPoolTemplate` (PULPOpen) passed spatial dims in `x,y` order to the
   `PULP_MaxPool2d_*_HWC` kernel, but that kernel expects `W,H` order (as the working
   `FloatAveragePoolTemplate` does). For SpeechNet's non-square 14×700 activations with
   asymmetric `(1,N)` kernels this transposed the memory indexing; symmetric test cases hid it.
   **Fix:** swap to `y,x` order. `Deeploy/Targets/PULPOpen/Templates/FloatMaxPoolTemplate.py`
   (TrainDeeploy commit `496a7f4`).

2. **MaxPool training graph failed to generate** (`'PULPVariableBuffer' has no attribute '_type'`).
   Root cause: ORT autodiff emits `MaxPool→[Y, Indices]` and `MaxPoolGrad(dY, Indices)`, but
   PULP's MaxPool binding types only 1 output, leaving the Indices/mask untyped. Deeploy's
   `MaxPoolGrad` kernel doesn't use indices anyway — it **recomputes argmax from the forward
   input X**.
   **Fix:** `BaseONNXExporter._rewire_maxpoolgrad_recompute()` rewires each `MaxPoolGrad`'s 2nd
   input from the mask to the MaxPool's forward input and drops the orphaned mask output, on the
   final `network.onnx` only (Onnx4Deeploy commit `e7af382`).

3. **Tiled training crashed in `minimizeRectangle`** (degenerate past-end tile).
   Root cause: `MaxPoolGradCTileConstraint` tiles a 3rd tensor (forward input X); for the
   degenerate `(1,1)` pools (blocks 3–4, where input shape == output shape) the tiler enumerates
   an invalid tile. AvgPoolGrad never hit this (1 input, reuses `AveragePoolCTileConstraint`).
   **Fix:** emit `nn.Identity()` for `(1,1)` pools in `SpeechNetDeploy` — the paper uses *no*
   pooling there, so this is a numerically-exact simplification that removes the degenerate
   MaxPool/MaxPoolGrad nodes (Onnx4Deeploy commit `da0ab76`). Inference re-verified 70.6%.

4. **MaxPool training gradient numerically wrong (~0.025 loss drift/step), failing tight tolerance.**
   This was NOT float-accumulation: AvgPool+raw training is exact (≤2e-6), and PyTorch (recompute)
   losses == ORT reference exactly, while on-device differed — a real on-device bug. Isolated with a
   standalone MaxPoolGrad op test on non-square shapes (14×40, k=(1,8)): 984/4480 dX wrong, errors
   concentrated at window-position 0 → kernel reading wrong window values.
   Root cause: Deeploy's `_NCHWtoNHWC_fun` (`LoweringOptimizationPasses.py`) transposes a node's
   `inputs[0]` and `outputs[0]` to NHWC (and `inputs[1:]` only for Conv). **MaxPoolGrad's `inputs[1]`
   (the forward input X used to recompute the argmax) was never transposed**, so the HWC kernel read
   X in NCHW → gradients routed to wrong positions. Silent for square pools (hidden by the existing
   8×8 k2×2 test), exposed by SpeechNet's non-square/asymmetric pools.
   **Fix:** add a `MaxPoolGrad` branch in `_NCHWtoNHWC_fun` that transposes `inputs[1]` (X) with the
   same permutation (TrainDeeploy commit `b2c3735`). Isolated test 984/4480 → 0/4480; full MaxPool
   training all loss diffs 0.025 → **1e-6**, passes at **TOL=0.01**.

## How to reproduce

Inference (untiled accuracy eval):
```
python speechnet_accuracy_eval_untiled.py --infer-dir Tests/Models/speechnet_infer_original --cores 8
```

Fine-tuning (tiled, 4 steps):
```
python deeployTrainingRunner_tiled_siracusa.py \
  -t Tests/Models/Training/SpeechNet/speechnet_train_maxpool \
  --optimizer-dir Tests/Models/Training/SpeechNet/speechnet_optimizer_maxpool \
  --l1 128000 --l2 2000000 --cores 8 --n-steps 4 --tolerance 0.05
```

## Notes / caveats
- Use the **tiled** training runner. The **untiled** runner crashes in `pulp_im2row_fp32`
  (ConvGrad im2row), a pre-existing untiled-path limitation unrelated to MaxPool.
- Artifacts generated by Onnx4Deeploy with `use_maxpool=True, normalize_input=False`
  (paper setup: MaxPool + raw EMG) and the SilentWear LOSO fold-3 checkpoint.
