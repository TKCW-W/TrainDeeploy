# On-Device Fine-Tuning — POSITIVE RESULT + Root-Cause Fix

**Goal (achieved):** find a fine-tuning setup that, run **on-device** (tiled GVSoC,
SGD, eff-batch-1), produces updated weights whose accuracy on **whole batch-2** beats
batch-2's own zero-shot (78.33%) by a meaningful margin.

## Headline result

| config (on-device, GVSoC) | batch-2 acc | Δ vs zero-shot | device-vs-ORT loss errors |
|---|---|---|---|
| zero-shot (pretrained) | 78.33% | — | — |
| **head-only + BN-fold, ep10** (135 steps) | **81.67%** | **+3.33 pp** | **0 / 540 (bit-exact)** |
| head-only + BN-fold, ep40 (540 steps) | 82.78% (ORT; on-device run in progress) | +4.44 pp | (expected bit-exact) |

The **actual on-device fine-tuned weights** (extracted from GVSoC, not the ORT
reference) evaluated on whole batch-2 give **+3.33 pp**. Device weights are bit-exact
to the ORT reference (max|Δ| = 2.4e-7).

## The key discovery: why naive on-device FT fails

Straightforward fine-tuning (full-model **or** head-only) does **not** improve batch-2
on-device — best was −0.56 to −1.1 pp. Root cause, proven in code:

- The ORT training graph uses **`BatchNormInternal`** (training-mode BN), and Deeploy's
  `TargetLibraries/PULPOpen/src/BatchNorm.c` kernel **recomputes batch mean/variance
  from the input** (lines 45–57) and does **not** use running stats (line 18).
- With on-device **batch-size-1**, every window is normalised by its *own* spatial
  statistics → the features during training differ completely from inference (which uses
  the frozen running stats). The classifier trains on corrupted features and cannot
  generalise.
- This is why **PyTorch sims were over-optimistic**: they used eval-mode BN (running
  stats). The ORT reference (= on-device, both batch-stat BN) is the only faithful
  predictor — and it correctly predicted the on-device failure, saving GVSoC hours.

## The fix

For a **frozen feature extractor** (`--training-strategy last_layer`), **fold BatchNorm
into the preceding Conv** and drop the BN op (`speechnet_exporter.py:create_model` →
`_fold_bn_into_conv`, auto-enabled for `last_layer`). This is exact in eval mode
(zero-shot unchanged) and makes the frozen training features match inference. Result in
the on-device-predictive ORT space: **+3.33 pp (ep10) … +4.44 pp (ep40)**, with the
training loss now actually converging (0.77 → 0.16).

Bonus: head-only freezes all MaxPool/Conv/BN, so the argmax-discreteness precision drift
becomes a *fixed, non-compounding* per-window offset and the only evolving tensor is the
linear `fc`. On-device this yields **0/540 loss errors — fully bit-exact to ORT.** The
precision problem that motivated the whole investigation is *eliminated* by this approach.

## Winning configuration

```
head-only (train fc only) + BN folded into Conv
data: 54 stratified batch-1 windows (6/class, = 30% of the rest-balanced batch)
n-accum 4, lr 0.01, n-epochs 10 (135 optimizer steps)   # ep40 for +4.44 pp
```
Generate:
```
Onnx4Deeploy.py -model SpeechNet -mode train -o <train_dir> \
  --dataset silentwear --data-path .../data_raw_and_filt --pretrained-weights .../fold_3.pt \
  --subject S01 --session 3 --batch 1 --condition vocalized \
  --stratified --data-size 54 --n-epochs 10 --n-accum 4 --lr 0.01 \
  --training-strategy last_layer
# optimizer dir auto-created (per-param SGD). DO NOT overwrite it with optimizer_model.onnx.
```
Run + extract weights:
```
rm -rf TEST_SIRACUSA
python deeployTrainingRunner_tiled_siracusa.py -t <train_dir> \
  --n-steps 135 --n-accum 4 --cores 8 -D DUMP_WEIGHTS=ON > run.log
# parse [WDUMP wi=0 n=288] fc_weight(9,32) + [WDUMP wi=1 n=9] fc_bias from run.log
```

## On-device weight extraction (new infra)

`Platforms/Siracusa/src/deeploytraintest.c`: after `run_optimizer_step()` (the spot the
supervisor pointed to), `dump_weights()` reads the persistent training-weight buffers and
prints each as raw 32-bit hex words (FPU-free, bit-exact) under `[WDUMP s=<step> wi=<i>
n=<#floats>]`. Gated by `-D DUMP_WEIGHTS=ON` (CMake option). For head-only this is just
`fc_weight` (288) + `fc_bias` (9). Parse with `struct.unpack('<f', struct.pack('<I', word))`.

## Files
- `speechnet_exporter.py` — BN-fold fix (`_fold_bn_into_conv`, auto for `last_layer`).
- `deeploytraintest.c` + `Platforms/Siracusa/CMakeLists.txt` — `DUMP_WEIGHTS` weight dump.
- Fixtures: `Tests/Models/Training/SpeechNet/speechnet_train_head_ep10` (+ `_ep40`, + optimizer dirs).
- Logs: `speechnet_head_ep10_ondevice.log` (0/540 errors, PASS), `speechnet_head_ep40_ondevice.log`.
- Search artifacts: `speechnet_ft_sim_search*.py`, `speechnet_ft_folded.py`, `speechnet_ft_ortsweep.py`.

## Caveats / honest notes
- The PyTorch sim search (`v2/v3/v4`) was **not predictive** (eval-mode BN); only the
  ORT/generation-space sweep (`speechnet_ft_folded.py`) reflects on-device. Trust the latter.
- ep10 = +3.33 pp is the *measured on-device* number; ep40 = +4.44 pp is the ORT
  prediction (on-device run finishing; will match given bit-exactness).
- This validates on subject S01 / session 3, batch-1→batch-2. Generalising to other
  subjects/sessions is future work.
