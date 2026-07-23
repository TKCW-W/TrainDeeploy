# K=1 (last-block + fc) on-device incremental FT — results

**Date:** 2026-07-23
**What:** Full on-device (Siracusa GVSoC, tiled, bit-exact inference) incremental fine-tuning of the K=1
scope — last conv block (`blocks.4`) + BN affine + `fc` (6 trainable tensors, ~7.5K params, frozen
pretrained BN stats) — S01 vocalized, fold 3, batches b1→b5, carried forward. Compared to the PyTorch
K=1 host reference on the *same* Onnx4Deeploy 54-window draws, plus head-only and the paper (fold 3).

## Chain (balanced accuracy, %)

| batch | note | head-only (fold3) | K=1 PyTorch (host) | **K=1 on-device (GVSoC)** | paper (fold3) |
|---|---|---|---|---|---|
| 1 | zero-shot | 80.56 | 80.56 | 80.56 | 80.56 |
| 2 | FT on b1 | 89.44 | 87.78 | **87.78** | 87.78 |
| 3 | FT on b1–2 | 80.00 | 78.89 | **78.33** | 86.67 |
| 4 | FT on b1–3 | 83.89 | 83.89 | **84.44** | 88.33 |
| 5 | FT on b1–4 | 82.22 | 85.56 | **86.11** | 87.78 |
| **mean b2–5** | | 83.89 | 84.03 | **84.17** | 87.64 |

## Verdict
**On-device K=1 training works and matches PyTorch.** The GVSoC chain (mean b2–5 = 84.17) tracks the
PyTorch K=1 host reference (84.03) to within ~0.14 pp overall and ≤0.55 pp (≈1 window) per batch — the
residual is float32 / tiling round-off accumulated over the incremental chain, not a training error. Each
round runs the true tiled backward through the last block on the accelerator; weights are extracted from
the `[WDUMP]` dump and validated by downstream accuracy.

## Reference-loss caveat (why the runner "fails" with exit 1)
`create_training_test_data` computes the reference LOSS/grads by running ORT on `network_train.onnx`, whose
BN is **live-batch** (`BatchNormInternal`, `training_mode=1`), whereas the device trains with **frozen** BN
stats (`BN_FROZEN_STATS=ON`, runtime `g_bn_frozen_stats=1`). So the runner's bit-exactness check reports
"Errors 2160/2160" and exits 1 — a **false failure**. The device is correct: `process_round_k1.py` extracts
the 6 trainable tensors from `[WDUMP]` (produced regardless of the exit code), and accuracy validates them
(b2 on-device 87.78 = PyTorch K=1 87.78). The per-round "DIVERGES" lines vs `outputs.npz` are the same
artifact (live-BN ORT reference), not a device error.

### Root cause (verified 2026-07-24)
The `--bn-frozen-stats` export flag *does* work — but only when BN is **not** trainable. Then eval-mode BN
is a pure affine and the exporter **folds it into the preceding conv** (0 BN nodes → frozen reference =
device; head-only fixtures pass cleanly). For K=1 (and full training) we make the **BN affine
`blocks_4_1_weight/bias` trainable**, so ORT's training-artifact generator **cannot fold BN** — it must use
training-mode `BatchNormInternal` (it feeds `saved_mean/var` to `BatchNormalizationGrad`). The eval export
is therefore overridden and the reference graph comes back live-batch. The device re-freezes at runtime via
the separate C flag, so device (frozen) ≠ ORT reference (live) ⇒ the false failure. Confirmed: head-only
fixtures have **0 BN nodes**; K=1/full fixtures have `BatchNormInternal(training_mode=1)`.

### TODO — fix the ORT reference so the bit-exact test passes (circle back later)
Goal: make `create_training_test_data` emit a **frozen-BN** reference for `bn_frozen_stats` fixtures.
- ❌ **Naive attribute patch does NOT work.** Flipping `BatchNormInternal.training_mode→0` on the reference
  graph makes ORT reject it: *"number of op outputs should be 1 when Training_mode = False"* — the training
  BN is a 5-output op welded to `BatchNormalizationGrad`; you can't freeze it in place.
- **Option A (targeted):** when `bn_frozen_stats`, compute the single-step reference in **PyTorch
  `model.eval()`** (frozen BN, trainable params get grads, one SGD step) instead of running the live graph —
  matches `train_lastk`/the device by construction. Small change in `create_training_test_data`; only caveat
  is non-BN ops become PyTorch-vs-device (within the 1e-3 tol).
- **Option B (principled):** export frozen BN as explicit ops — `xn=(x−running_mean)·(1/√(running_var+ε))`
  (constants) then `y=γ·xn+β` (trainable Mul/Add). ORT differentiates natively; Deeploy emits Mul/Add
  kernels; graph = reference = device, and the runtime `BN_FROZEN_STATS` C override becomes unnecessary.
  Bigger exporter change + full training-suite re-validation.
- Recommendation: Option A to pass the test; Option B for the proper long-term cleanup. **Does not affect the
  K=1 results above** — those are validated by accuracy, not this reference.

## Tiling
K=1's heavier backward needs explicit runner flags (head-only tiled with defaults):
`--l1 128000 --l2 2000000 --defaultMemLevel L2 --memAllocStrategy MiniMalloc --searchStrategy random-max`.

## Files
`export_k1_round.py`, `process_round_k1.py` (WDUMP → 6-tensor extract → carry ckpt → eval next batch),
`run_pytorch_k1_ondevicedata_chain.py` (host reference on the same draws), `k1_vs_headonly_vs_paper.csv`,
`logs/round{1..4}_gvsoc_train.log`, `logs/device_r*_*.npy`. See `ONDEVICE_SIMULATION_PLAN.md` for the
per-round export/run/extract procedure.
