# MaxPool-SpeechNet on-device deployment — annotated commit log

Branch: `feat/speechnet/inference/maxpool_ondevice`
Repos: **Onnx4Deeploy** (PyTorch→ONNX exporter) · **TrainDeeploy** (Deeploy compiler + GVSoC runner)

**Goal:** deploy SpeechNet 100% paper-loyal (MaxPool, not the AvgPool revision) for **both
inference and on-device fine-tuning**, overcoming MaxPool's argmax-index storage by
**recomputing the argmax from the forward input**.

Ordered by the logical flow of the work, not by repo.

---

### `b60c389` [Onnx4Deeploy] — SpeechNet exporter: add use_maxpool, normalize_input, bn_recalibrate options

Foundation for paper-loyal exports. Adds config knobs so the exporter can emit MaxPool (paper)
vs AvgPool (revision) graphs, and raw vs per-window normalised EMG, instead of being hard-wired
to AvgPool + normalised data.

- `use_maxpool` — MaxPool2d vs AvgPool2d per block
- `normalize_input` — raw EMG (paper) vs per-window standardisation
- `bn_recalibrate` — recompute BN running stats when porting MaxPool-trained weights onto AvgPool
- `SilentWearDataSource` gains a `normalize` flag (was unconditional)

With `(use_maxpool=True, normalize_input=False)` + the LOSO fold-3 checkpoint, the exported
inference ONNX reproduces the paper's 70.6% in ORT.

---

### `496a7f4` [TrainDeeploy] — Fix FloatMaxPool HWC template: swap x/y dims to match AvgPool convention

**Problem.** MaxPool inference ONNX was correct in ORT (70.6%) but produced completely wrong
logits on Siracusa (9/9 mismatched).

**Root cause.** The kernel `PULP_MaxPool2d_fp32_fp32_HWC(src, W, H, C, Q, P, SQ, SP, …)` indexes
memory as `(h*W + w)*C + c` — its first dim arg is the true width (row stride). The parser sets
`dim_im_in_x = shape[2] = H`, `dim_im_in_y = shape[3] = W` (NCHW). The MaxPool template passed
`dim_im_in_x` (=H) as the kernel's `W` argument, transposing the traversal. Fatal for SpeechNet's
non-square 14×700 tensors with asymmetric `(1,N)` kernels; the working AvgPool template passes
`(y,x)=(W,H)`. Square test fixtures are invariant to the swap, hiding it.

**Fix.** Swap the MaxPool template arg order to match AvgPool. **Result:** 9/9 → 0/9 errors,
bit-exact with ORT.

---

### `e7af382` [Onnx4Deeploy] — export_training: rewire MaxPoolGrad to recompute-from-input for Deeploy

**Problem.** MaxPool training-graph generation failed in type-checking:
`AttributeError: 'PULPVariableBuffer' object has no attribute '_type'` at the MaxPoolGrad node.

**Root cause.** ORT autodiff differentiates MaxPool the ONNX-standard way: forward MaxPool emits
two outputs — `Y` (values) and `Indices` (argmax) — and the gradient is `MaxPoolGrad(dY, Indices)`.
PULP's forward-MaxPool type-checker declares only one output, so `Indices` is never typed and the
downstream MaxPoolGrad crashes. This is MaxPool's "store the argmax" problem.

**Approach (recompute, not store).** Deeploy's PULP MaxPoolGrad kernel does not use indices — it
**recomputes the argmax from the forward input X**. Rewire the graph: for each MaxPoolGrad, replace
its 2nd input (Indices) with the MaxPool's forward input X, and drop the orphaned mask output
(forward MaxPool → single-output). Applied only to the final `network.onnx`; `network_train.onnx`
keeps ORT's convention so the ORT reference losses still run (mathematically identical).

**Result.** Training graph binds and compiles. No index buffers; argmax recomputed on the fly.

---

### `da0ab76` [Onnx4Deeploy] — SpeechNet: emit nn.Identity for (1,1) pools (paper no-op) instead of pool op

**Problem.** Tiled training codegen crashed:
`AssertionError: Rectangle offset should be zero when the dimensions are the same` in
`minimizeRectangle`, on a `(1,32,2,5)` tile (block-4 region).

**Root cause.** Blocks 3–4 use no pooling in the paper, expressed as `(1,1)` pools (input shape ==
output shape). MaxPoolGrad's dedicated tile constraint tiles three tensors (dY, dX, and the forward
input X for recompute); for the degenerate equal-shape case the tiler enumerated an invalid
past-end tile. AvgPoolGrad never hit this (one input, reuses the forward-pool tiler).

**Fix.** Emit `nn.Identity()` for `(1,1)` pools in `SpeechNetDeploy` — numerically exact (a `(1,1)`
stride-1 pool is the identity) and paper-faithful (no pooling there). Removes the degenerate
MaxPool/MaxPoolGrad nodes; no params, so `state_dict`/block indices unchanged. Inference re-verified
70.6%.

> Side finding: the **untiled** runner crashes in `pulp_im2row_fp32` — a pre-existing ConvGrad
> im2col limitation unrelated to MaxPool. Use the tiled runner.

---

### `b2c3735` [TrainDeeploy] — Fix NCHWtoNHWC pass: transpose MaxPoolGrad's forward-input X (inputs[1])

**Problem.** Tiled MaxPool training ran end-to-end on GVSoC but per-step losses drifted ~0.025/step
from the reference, failing the tight tolerance. Accumulation limit, or a bug?

**Investigation (ruled out a float limit).** AvgPool training on the identical pipeline is exact to
≤2e-6 → the pipeline *can* be bit-accurate. PyTorch (recompute) losses equal the ORT reference
**exactly**, and both differ from on-device → the reference is correct and the on-device MaxPoolGrad
is wrong. Ruled out: data scale, BN-variance precision, the recompute method (matches autograd),
argmax tie-breaking (ties are ReLU-killed), gradient magnitude. Isolated with a standalone MaxPoolGrad
op test at block-0 shapes: 17030/78512 of dX wrong; a single-tile 14×40 case also failed → not
tiling. Error pattern: ~88% of windows misrouted, concentrated at window-position 0.

**Root cause.** The layout pass `_NCHWtoNHWC_fun` transposes a node's `inputs[0]` (dY) and
`outputs[0]` (dX) to NHWC; the `inputs[1:]` branch existed only for Conv (weights). MaxPoolGrad's
`inputs[1]` — the forward input X used for the argmax recompute — was **never transposed**, so the
HWC kernel read X in NCHW with wrong strides and routed the gradient to the wrong element (defaulting
to position 0). Silent for square pools, hidden by the existing 8×8/k2×2 test.

**Fix.** Add a MaxPoolGrad branch in the layout pass that transposes `inputs[1]` (X) with the same
permutation. **Result:** isolated test 984/4480 → 0/4480; full MaxPool training loss diffs 0.025 →
1e-6, passes the tight TOL=0.01; 20-step run 0/20 (max diff 1e-6). The drift was a real Deeploy
layout bug, **not** an accumulation limit.

---

### `2690390` / `5fe8e2c` [TrainDeeploy] — Documentation

`SPEECHNET_MAXPOOL_PROGRESS.md`: goal, results table, the four root-cause bugs with fixes, and
reproduction commands; updated to reflect the layout fix (training bit-accurate, TOL=0.01).

---

### `ff8e8a6` [TrainDeeploy] — Add 90-step MaxPool fine-tuning evidence (8-core): log, loss curve, data

End-to-end fine-tuning demo: 18 stratified samples (2/class) × 5 epochs, lr=1e-3, MaxPool + raw
(paper-loyal), 8 cores. On-device tracks the reference bit-exact (1e-6) for ~36 steps, then bounded
fp32 accumulation (max 0.064); **52/90 over TOL=0.001 — the same profile as the AvgPool b1_ft
baseline (48/90)**. Epoch-avg loss 2.54→2.18 (the model learns). Adds run log, loss-curve plot, and
`.npz` data.

---

## Outcome

- **Inference:** bit-exact on-device, **70.56%** balanced (all 180 samples) = ORT = paper zero-shot.
- **Fine-tuning:** bit-accurate gradients (1e-6, passes TOL=0.01); over many steps, same bounded
  fp32 accumulation as the AvgPool baseline.
- **Mechanism:** recompute-from-input MaxPoolGrad — no argmax/index storage.
- **Theme:** every bug came from **non-square / asymmetric pooling**, which the codebase's square
  test fixtures never exercised — MaxPool only *looked* unsupported; the kernels were fine, the
  layout/typing plumbing around them was incomplete.
