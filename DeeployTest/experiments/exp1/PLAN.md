# exp1 — Fix the ORT reference so on-device full-training-with-frozen-BN passes bit-exact

**Date:** 2026-07-27
**Branch:** `feat/speechnet/on-device-FT` (TrainDeeploy) · Onnx4Deeploy `feat/speechnet/inference/maxpool_ondevice`
**Goal:** make the on-device (Siracusa/GVSoC) training runner's bit-exact loss check *meaningful* for the
chosen deployment recipe — **full-model fine-tuning with BatchNorm frozen at pretrained running stats** —
by fixing the ORT reference to compute loss/grads with **frozen** BN (matching the device), instead of the
current **live-batch** BN.

---

## 1. The issue (why the runner reports `Errors: 2160/2160`)

The training-CI runner compares the device's per-step loss trajectory (`stored_losses[]`) against a
reference trajectory `testLossRef[]` baked into `testoutputs.h`. That reference comes from
`outputs.npz['loss']` — a full **2160-step** trajectory (54 windows × 40 epochs) produced by the SpeechNet
exporter's `create_training_test_data`, which runs the whole training loop **in ORT on `network_train.onnx`**.

For **full training** (and K=1), the BN affine (γ/β) is trainable, so ORT's training-artifact generator
emits `BatchNormInternal(training_mode=1)` — which normalizes with **per-batch statistics**. So:

- **ORT reference:** every one of the 2160 losses/grads uses **live-batch BN**.
- **Device (`-D BN_FROZEN_STATS=ON`, `g_bn_frozen_stats=1`):** trains with **frozen** pretrained stats.

They disagree at every step → the runner prints `Errors: 2160/2160` and exits 1. **This is a false
failure**: the device trains correctly (weights improve, accuracy rises — validated in the K=1 on-device
study, `SilentWear/.../ondevice_simulation_lastblock/RESULTS.md`). Only the *reference* is wrong.

(Head-only does **not** hit this: its feature extractor is frozen, so the exporter **folds** BN into Conv
→ a BN-free graph → the ORT reference and device agree → `Errors: 0/2160`.)

Naive attempt that does **not** work: flipping the graph's `BatchNormInternal.training_mode → 0` — ORT
rejects it (`"number of op outputs should be 1 when Training_mode=False"`; the training BN is a 5-output op
welded to `BatchNormalizationGrad`).

## 2. Two options

### Option A — compute the reference in PyTorch with frozen BN (chosen)
When `bn_frozen_stats` is set, replace the per-step **ORT** forward/backward in
`speechnet_exporter.create_training_test_data` with **PyTorch** (`model.eval()` → frozen BN), keeping the
loop, window order, sum-accumulation, and SGD **identical**. The torch model *is* the exported model, so
the forward matches by construction; `model.eval()` freezes BN exactly as the device does.

| pros | cons |
|---|---|
| **Zero device change** — keeps the fast fused `BatchNormInternal(frozen)` kernel + `g_bn_frozen_stats` | reference engine becomes PyTorch, so non-BN ops are PyTorch-vs-device (within the 1e-3 tolerance) |
| Small, localized change (one method, gated on `bn_frozen_stats`) | reference and generated graph now differ in *how BN is computed* (graph still has `BatchNormInternal`; device overrides via C-flag) |
| Reference matches the device's frozen-BN math → check becomes meaningful | — |

### Option B — export frozen BN as explicit ops (`Sub/Mul-const → Mul γ → Add β`)
Rewrite frozen BN into primitive ops before grad generation, so ORT differentiates it natively (no
`BatchNormInternal`); graph = reference = kernels = device, and the `BN_FROZEN_STATS` C-flag retires.

| pros | cons |
|---|---|
| Whole pipeline consistent; C-flag retired; genuinely bit-exact | **un-fused** (no fp32 fusion pass / affine kernel exists): ~4× BN-layer activation DMA → real latency cost on early large-activation blocks |
| No BatchNorm op on device | recovering the speed needs a **new fused fp32 affine fwd+grad kernel** — the kernel B was meant to avoid |
| — | bigger change; re-validate the whole training suite |

**Decision: Option A.** It fixes the *reference* (the actual bug) with no device-side latency/memory cost;
Option B's un-fused chain adds ~4× DMA on the BN layers and would need a new kernel to claw back. Option B
stays a future "make frozen-BN a first-class graph construct" cleanup.

## 3. Our fine-tuning setting (deployment recipe)

Full-model FT with BatchNorm frozen at the pretrained running stats for **both** FT and inference
(train ≡ inference normalization). From the PyTorch study
`SilentWear/.../frozen_stat_full_training` (setting **S2**, best config) and the 4-subject reproduction:

| knob | value |
|---|---|
| trainable scope | **full model** (conv + BN γ/β + fc), no BN folding |
| BN stats | **frozen pretrained** (sessions 1+2), train == inference; device `BN_FROZEN_STATS=ON` |
| optimizer | **SGD**, no momentum, no weight decay |
| learning rate | **3e-4**, static |
| gradient accumulation | **n_accum = 4, SUM** (`w ← w − lr·Σ gradᵢ`, no ÷n_accum) |
| effective batch | 1 (one window per fwd/bwd) |
| epochs | 40 |
| FT data | 30% of each batch = **54 windows** (6/class × 9, seed 42) |
| accuracy (ref) | S2 4-subj b2–5 = **77.05 ± 9.40** (≈ head-only 76.83; GN Edge-FT 78.97; paper 80.02) |

## 4. Plan of work
1. **Implement Option A** in `Onnx4Deeploy/onnx4deeploy/models/speechnet_exporter.py`: a
   `bn_frozen_stats`-gated PyTorch frozen-BN reference loop replacing the ORT loop — identical window order,
   SUM accumulation, plain-SGD update; emit `outputs.npz` (per-step losses + updated weights).
2. **Smoke test (NOT the full 40-epoch run):** export a *small* full-training-frozen-BN fixture (few
   windows, few epochs), build + GVSoC, and check the per-step `[loss k] computed vs ref`:
   - **Before** (live-BN reference): expect large mismatches / `Errors: N/N`.
   - **After** (Option A frozen reference): expect `computed ≈ ref` and `Errors: 0/N`.
3. **FINDING.md:** report whether device loss matches the frozen ORT reference, the smoke-test numbers, and
   next steps (full 40-epoch run, then wire into the incremental on-device simulation for full training).
