# FINDING — on-device full-training + frozen-BN, S01 vocalized fold 3

**Action date/time:** 2026-07-28 ~18:00 local (chain ran 2026-07-27→28; each of the 4 training rounds was a
~4-5 h GVSoC full-model sim).
**Branch:** TrainDeeploy `feat/speechnet/on-device-FT` · Onnx4Deeploy `feat/speechnet/inference/maxpool_ondevice`

## What ran
The complete on-device (Siracusa/GVSoC, tiled) incremental fine-tuning chain for the deployment recipe
**full-model FT + BatchNorm frozen at pretrained running stats** (S2), S01 vocalized fold 3, b1→b5, carried
forward. Each round r: export train fixture from the previous carry (Onnx4Deeploy, `--bn-frozen-stats`, full
model, lr 3e-4, n_accum 4, 40 ep, 54 windows) → tiled GVSoC train with weight dump → extract the 22 device
tensors by weight-buffer order → carry forward → evaluate batch r+1. This is the first full-model on-device
FT chain; it relies on the Option A frozen-BN reference fix (`../FINDING.md`).

## Per-batch accuracy — on-device vs PyTorch (matched data)

| batch | on-device | PyTorch (matched) | Δ |
|---|---|---|---|
| 1 (zero-shot) | 80.56 | 80.56 | 0.00 |
| 2 (FT b1) | 87.22 | 87.22 | 0.00 |
| 3 (FT b1–2) | 81.11 | 81.11 | 0.00 |
| 4 (FT b1–3) | 82.78 | 83.33 | −0.55 |
| 5 (FT b1–4) | 81.67 | 83.33 | −1.66 |
| **mean b2–5** | **83.20** | **83.75** | **−0.55** |

On-device balanced accuracy is computed from the device-trained weights (WDUMP-extracted) via host inference,
which is **bit-exact to on-device inference** (established in the head-only/K=1 studies; the b2 on-device
inference-runner cross-check below re-confirms it for the full model). The PyTorch column is an independent
"ideal" chain on the *same* 54-window Onnx4Deeploy draws (`run_pytorch_fullfrozen_ondevicedata_chain.py`).

**On-device inference-runner confirmation (b2):** `speechnet_accuracy_eval_untiled.py` on the b2 inference
fixture (carry_b1 weights) → balanced accuracy = **87.22 %**, with per-window `Errors: 0/9` (bit-exact to
ORT). This **exactly matches** the host eval (87.22) and the PyTorch chain (87.22), confirming on-device
inference == host == PyTorch for the full model — so the accuracy chain above (computed via host inference on
the device-trained weights) is the true on-device accuracy.

## MaxPool argmax drift — per round (from the saved train logs)

| round | Errors / 2160 | onset step | max \|diff\| |
|---|---|---|---|
| 1 (b1) | **416** | 133 | 0.035 |
| 2 (b2) | **0** (bit-exact) | — | 0.0003 |
| 3 (b3) | **111** | 594 | 0.0086 |
| 4 (b4) | **0** (bit-exact) | — | 0.0003 |

Logs kept in `logs/round{1..4}_gvsoc_train.log` for later analysis.

## Conclusions
1. **On-device full-training-with-frozen-BN works and reproduces the PyTorch recipe closely.** The on-device
   chain **matches the ideal PyTorch chain exactly at b1–b3** and stays within **≤1.66 pp (≤3 windows/180)**
   at b4–b5; mean b2–5 differs by only **−0.55 pp**.
2. **The MaxPool argmax drift is data/trajectory-dependent** — 2 of the 4 rounds were **bit-exact** (0/2160),
   the other two had **111 and 416** breaches with small magnitudes (≤0.035 loss). It is NOT guaranteed every
   round; it depends on whether the specific batch/weight trajectory flips a pooled-value tie.
3. **The drift is loss-only per round; its accuracy effect is tiny and accumulates slowly.** Because it (plus
   fp32 tiling round-off) perturbs weights by only ~1e-2, individual-round accuracy is unaffected (b2/b3
   exact); the ≤1.66 pp late-chain gap is the *accumulated* effect over 4 incremental rounds, not a
   per-round failure.
4. **Validation is by accuracy, not strict per-step loss** — the correct acceptance criterion for this path,
   since the argmax drift makes long-run bit-exact loss impossible (as flagged in `../FINDING.md`).

## Files
`PLAN.md`, `FINDING.md`, `run_pytorch_fullfrozen_ondevicedata_chain.py`, `extract_device_weights.py`,
`results/ondevice_vs_pytorch_S01_fold3.csv`, `results/pytorch_fullfrozen_ondevicedata.csv`,
`logs/round{1..4}_gvsoc_train.log`.
