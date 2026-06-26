# Full-training (n_accum=8) on-device constrained experiment — progress

Goal (user): under on-device constraints, SGD-no-momentum, **full** training (like the paper),
`n_accum=8`, `30% data` (data_size 54). (1) Does the on-device-constrained setup improve accuracy
effectively? (2) With larger train size / step counts, does the precision drift affect the result,
or is it acceptable (tie-flip diverges from ORT but doesn't necessarily pick a wrong gradient)?

## Status

- **P0 ✓** Committed + pushed the two reference docs (`be1d14e`).
- **P1 ✓** Host/ORT predictive sweep — full training, SGD, n_accum 8, data 54, ep×lr grid,
  eval batch-2 (`speechnet_ft_full_naccum8_sweep.py` → `full8_sweep.log`, `..._sweep.json`):
  - zero-shot batch-2 = **78.33%**.
  - **Best = +2.78pp (81.11%)** at **lr 1e-3** (stable across ep10/20/40; +2.22pp at ep50).
  - lr ≥ 5e-3 → **collapse** (−21 to −35pp); lower training loss ⇒ worse batch-2 acc
    (train/inference BN-mismatch overfitting). 4/12 configs beat zero-shot.
  - Ceiling far below head-only folded (**+4.44pp**) and SilentWear (**+8.33pp**); fragile.
- **P2 ✓** Device config chosen: full training, data 54, n_accum 8, **lr 1e-3, n-epochs 20**
  → **1080 forwards / 135 steps** (12× the original 90-forward run = strong drift stress test).
  Fixture `speechnet_train_full8_e20` (+ `speechnet_optimizer_full8_e20`), verified
  5 BatchNormInternal + 22 InPlaceAccumulatorV2 (22 trainable params).
- **P3 ✓** Device GVSoC run done (135/135 updates, 1080 forwards), 22 weights dumped at final step
  (`full8_e20_device.log`). Per-weight drift vs ORT: worst 5.75% (one conv), most <1%.
- **P4 ✓** Reconstructed device weights (deterministic wi-order, size-verified) → eval batch-2.
- **P5 ✓ DONE** — full decomposition in `FULL_TRAINING_ONDEVICE_RESULTS.md`.

## RESULTS (complete)

| | weights | running-stats | batch-2 | Δ |
|---|---|---|---|---|
| A | pretrained | pretrained | 78.33% | zero-shot |
| B | ORT | ORT-updated | 81.11% | +2.78 (host sweep — OPTIMISTIC) |
| C | ORT | frozen | 63.33% | −15.00 |
| E | **device** | **frozen** | **60.56%** | **−17.78 (ACTUAL)** |

- **Goal 1 — NO:** on-device full-training **regresses −17.78pp**. Killer = BN **running-stats
  never updated** on device (−17.78pp, B→C), not the drift. Host sweep's +2.78pp was optimistic
  (it used ORT-updated running stats the device doesn't produce). ⇒ earlier full-model ORT sweep
  was also optimistic; real device full-model FT is negative.
- **Goal 2 — drift is acceptable:** over 1080 forwards (12×), drift cost only **−2.78pp** (C→E),
  weights <1% off ORT (worst 5.75%), bounded, no explosion. Tie-flips ≠ wrong gradients — confirmed.
- **Reframe:** drift is a red herring for full-model; the BN running-stat non-update is the real
  killer ⇒ head-only + BN-fold (drift-free *and* running-stat-safe, +4.44pp) is the right design.
