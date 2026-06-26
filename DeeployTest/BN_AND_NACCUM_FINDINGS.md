# BatchNorm & Gradient Accumulation on-device — Findings

What governs the BatchNorm behaviour during on-device SpeechNet fine-tuning, why the
on-device kernel diverges from inference, and what gradient accumulation (`n_accum`) can and
cannot do about it. All claims below are verified against the actual codegen / kernels / CLI,
with file references.

---

## TL;DR

- Training and inference use **two different BN kernels**. Training normalizes with **batch
  statistics recomputed from the current window**; inference normalizes with the **frozen
  pretrained running stats**. They are different functions in different libraries.
- On-device the forward batch is **forced to 1 window**, so the training BN normalizes each
  window by its *own* spatial statistics — nothing like the population stats inference assumes.
- `n_accum` (gradient accumulation) enlarges the **optimizer's** effective batch only. It does
  **not** change BN: each forward still sees a single window. It cannot close the BN gap.
- The only flag that would change BN statistics is `--batch-size`, and that is pinned to 1 by the
  **L2 memory budget** (the full model already needs tiling to fit one window).
- **Resolution: fold / freeze BN** — bake the pretrained running stats into the preceding Conv and
  remove the BN op from the training graph, so the training forward uses the *same* stats as
  inference. This is what makes head-only on-device FT bit-exact and effective.

---

## 1. Two BatchNorm kernels — train vs inference

This is the standard train/eval duality of BatchNorm, realized on-device as **two literally
different functions**:

| | Training | Inference |
|---|---|---|
| ONNX op | `BatchNormInternal` (train-mode) | `BatchNormalization` (eval-mode) |
| Kernel function | `PULP_BatchNormInternal_fp32` | `BatchNorm_fp32` |
| Source file | `TargetLibraries/PULPOpen/src/BatchNorm.c` | `TargetLibraries/Generic/src/BatchNorm_fp32.c` |
| mean/var used | **recomputed from input X** (this window) | **frozen pretrained running_mean/var**, used directly |
| Cores | 8-core (PULP) | single-core (`BEGIN_SINGLE_CORE`) |

### Training kernel (`PULP_BatchNormInternal_fp32`)
- Computes batch `mean`/`var` **from X** (`mean += x_nc[hw]; var += diff*diff`), then normalizes
  `y = (x − mean) * inv_std * γ + β`.
- `running_mean` / `running_var` **are passed as arguments but never dereferenced** in the
  normalization, and are **not updated** (the `updated_running_mean/var` outputs have no
  consumers — template comment in `FloatBatchNormTemplate.py`). They are dead inputs.

### Inference kernel (`BatchNorm_fp32`)
- Reads `c_mean = mean[c]; c_var = var[c]` — the **frozen pretrained running stats** (graph
  initializers), then `norm = (x − c_mean)/sqrt(c_var + ε); out = γ·norm + β`.
- Computes **nothing** from the batch.

### Verified in the actually-compiled binary
The generated, compiled `TrainingNetwork.c` from the real full-model run
(`TEST_SIRACUSA/.../speechnet_train_maxpool_90/TrainingNetwork.c`, with a matching `.c.obj`)
contains **5 calls** to `PULP_BatchNormInternal_fp32` (one per block), each passing
`running_mean_ref`/`running_var_ref` and a dims tail `…, 1, 1, 14, 701` → **N = 1**.
Codegen path: `BatchNormInternalParser` → `PULPBatchNormInternalBindings`
(`Bindings.py:362`) → `batchNormInternalTemplate` → emits `PULP_BatchNormInternal_fp32(...)`.

---

## 2. Why this is a problem on-device

The asymmetry is normally harmless: training-mode BN uses batch stats (so gradients flow through
the normalization) *and* accumulates running stats for later inference use. Two requirements make
that correct:

1. a **real batch size** (batch stats ≈ population stats), and
2. an **active running-stat update**, so inference's stats match what was trained.

On-device **both fail**:
- **Batch is forced to 1.** Each forward normalizes a single window by its own spatial mean/var
  per channel — a degenerate estimate, completely different from the population stats inference
  uses.
- **Running-stat update is dead.** Even the side-channel reconciliation never happens.

Result: the training forward sees a feature distribution that does not match the inference graph's
(`BatchNorm_fp32` with frozen stats), so whatever the trainer learns does not transfer — the
unfolded full-model fine-tune does not reliably improve accuracy and exhibits drift.

---

## 3. What `n_accum` is — and the crucial limitation

There are **two independent axes** that get conflated as "batch size":

| CLI flag | Meaning | Effect on **BN statistics** |
|---|---|---|
| `--batch-size` | **True batch** = windows in one forward/backward; sets the input tensor N dim (`onnx_utils.py:83` `randn(batch_size,1,C,T)`). | **This is the BN-relevant one.** SilentWear = 32, ours = 1. |
| `--n-accum` | **Gradient accumulation** = forward/backward passes summed before one SGD update ("effective batch size" *for the optimizer*). | **None.** Each forward still runs BN on a single window. |
| `--n-batches` | Total forward count = training length (`= n_steps × n_accum`). | None — duration, not batch. |
| `--data-size` | Distinct-window pool size that is cycled (controls epochs). | None — *how much* data, not batch. |

### What `n_accum` CAN do
Reduce gradient noise and give the **optimizer** a larger effective batch. With `n_accum=4`, four
windows' gradients are summed, then one SGD step — optimization-wise comparable to a batch-4
update. Good for update stability. (On-device the accumulator **sums** the micro-step gradients;
LR is baked into the optimizer ONNX; SGD only, no momentum/Adam.)

### What `n_accum` CANNOT do
Fix BatchNorm. Each of the `n_accum` forwards runs BN **independently on its single window** —
accumulation happens *after* the backward, on the **gradients**, not on the activations. So:

```
SilentWear batch=32 :  BN sees 32 windows   +  gradient over 32
ours  n_accum=4     :  BN sees  1 window     +  gradient over  4
```

`n_accum` moves only the *gradient* axis toward SilentWear; the **BN axis stays at batch-1**.
The only flag that would change BN is `--batch-size > 1`.

---

## 4. Why we cannot just raise `--batch-size`

There is **no hard assert** forcing batch-size = 1 (checked). It is pinned to 1 by the **device
memory budget**:

- Siracusa: **L1 = 128 KB**, **L2 = 2 MB** (`--l1 128000 --l2 2000000`).
- The full model already needs **tiling** to fit **one** window's activations: one window resides
  in L2, the tiler streams tiles into L1 per layer for forward/backward, writes back to L2, then
  the next window is loaded.
- A batch of 32 windows would need ≈32× the activation memory — it does not fit in 2 MB L2.

The kernel and exporter *nominally* support N > 1, but the platform cannot hold it. So on-device,
batch-1 is effectively mandatory, and (per §3) `n_accum` cannot substitute for it.

---

## 5. The resolution: fold / freeze BN

For a frozen feature extractor (training only the head, `--training-strategy last_layer`):

- **Fold** each block's BatchNorm into the preceding Conv using the **pretrained running stats**:
  `std = sqrt(running_var + ε); scale = γ/std;`
  `W ← W·scale; b ← (b − running_mean)·scale + β;` then replace BN with Identity.
- This is **exact in eval mode** (zero-shot accuracy unchanged) and **removes `BatchNormInternal`
  from the training graph entirely**.

Consequence: the training forward no longer computes any batch statistics — it uses the **same
baked-in running stats as inference**. Train ≡ inference for the frozen extractor, the batch-1 BN
pathology disappears, and on-device head-only FT becomes **bit-exact** and **effective**
(+4.44 pp in our runs). This is numerically identical to PyTorch `FrozenBatchNorm` / eval-mode BN
used during training.

---

## 6. Aside — if "stale running stats" were the real concern

"Stale running stats" and "stale weights" are different problems with different fixes:

- If only the **running stats** are stale (a covariate shift), the correct tool is **BN
  recalibration / AdaBN**: forward-pass the new data in train mode to **re-estimate `running_mean/
  var`**, no loss, no backprop. Cheap, and device-friendly in principle (no backward, no large
  batch needed for a running estimate).
- Gradient fine-tuning is for the **learnable** params (γ/β, conv filters, classifier) — a
  different kind of shift.

Our `last_layer` choice bets the feature extractor + its running stats transfer and only the
classifier head needs to move (empirically validated). A clean follow-up to isolate the
"stale-stats" hypothesis would be: re-estimate running stats on batch-2 windows → fold the
recalibrated stats → re-evaluate zero-shot. (Not yet run.)

---

## 7. File reference index

| Item | Path |
|---|---|
| Training BN kernel | `TargetLibraries/PULPOpen/src/BatchNorm.c` (`PULP_BatchNormInternal_fp32`) |
| Inference BN kernel | `TargetLibraries/Generic/src/BatchNorm_fp32.c` (`BatchNorm_fp32`) |
| Train BN template/binding | `Deeploy/Targets/PULPOpen/Templates/FloatBatchNormTemplate.py`, `Deeploy/Targets/PULPOpen/Bindings.py:362` |
| Inference BN template/binding | `Deeploy/Targets/Generic/Templates/BatchNormalizationTemplate.py`, `Deeploy/Targets/Generic/Bindings.py:311` |
| CLI flags | `Onnx4Deeploy/Onnx4Deeploy.py` (`--batch-size`, `--n-accum`, `--n-batches`, `--data-size`, `--n-epochs`, `--n-steps`) |
| Input tensor shape | `Onnx4Deeploy/onnx4deeploy/core/onnx_utils.py:83` |
| BN fold | `Onnx4Deeploy/onnx4deeploy/models/speechnet_exporter.py` (`_fold_bn_into_conv`) |
| Generated training C (verified) | `DeeployTest/TEST_SIRACUSA/.../speechnet_train_maxpool_90/TrainingNetwork.c` |
