# SpeechNet on-device incremental fine-tuning — S01 vocalized fold 3 (b1→b5)

**Date:** 2026-07-23
**What:** End-to-end on-device (Siracusa/GVSoC) simulation of the deployed head-only + BN-fold recipe,
fine-tuning incrementally across session-3 batches b1→b5 for S01 vocalized fold 3, with the accuracy
compared to the host PyTorch simulation. This is the canonical home for the on-device results.

## Result (all four rounds trained on real GVSoC, bit-exact to ORT)

| batch | on-device (GVSoC) | PyTorch fold-3 (seed-42) | PyTorch on on-device data |
|---|---|---|---|
| b1 zero-shot | 80.56 | 80.56 | 80.56 |
| b2 | 89.44 | 90.56 | 89.44 |
| b3 | 80.00 | 80.56 | 80.56 |
| b4 | 83.89 | 88.89 | 83.89 |
| b5 | 82.22 | 81.67 | 82.22 |
| mean b2–5 | 83.89 | 85.42 | — |

- Every round passed `Errors: 0 / 2160`, loss bit-exact to ORT (max|Δ| 1.2–2.9e-6), device fc == ORT < 1e-6.
- Feeding the PyTorch sim the *same* Onnx4Deeploy 54-window draws reproduces on-device on **4/5 batches
  exactly**; the seed-42 gap was purely the different stratified draw. The lone b3 difference (1 window /
  180) is `finetune_head` vs ORT boundary numerics, not a GVSoC error.

## Files here
- `ONDEVICE_SIMULATION_PLAN.md` — reproducible cross-repo flow (file-by-file, command-by-command).
- `RESULTS.md` — full results + fidelity analysis + draw-alignment proof.
- `logs/round{1..4}_losses.csv` (+ `_epoch_mean_loss.csv`, `_accuracy.txt`) — per-round GVSoC vs ORT
  loss traces; `device_fc_r{2..4}_*.npy` — extracted on-device classifier weights.
- `*.csv` — `ondevice_predicted_fold3` (matched-draw host chain), `pytorch_fold3_headonly` (seed-42
  baseline), `pytorch_ondevicedata_vs_gvsoc` (the 4/5-exact cross-check).
- `*.py` — the driver scripts (reference copies; see note below).

## How to reproduce
See `ONDEVICE_SIMULATION_PLAN.md`. Two containers: `agitated_hugle` (Onnx4Deeploy export + PyTorch eval)
and `traindeeploy` (Deeploy codegen/compile + GVSoC). Per round: export the training fixture
(`Onnx4Deeploy.py -mode train ... --training-strategy last_layer`), GVSoC-train
(`deeployTrainingRunner_tiled_siracusa.py -t <fixture> --n-steps 540 --n-accum 4 -D DUMP_WEIGHTS=ON`),
extract the device fc (`process_round.py`), carry it into the next round's export.

## Note on the PyTorch driver scripts
The `.py` files import SpeechNet/windowing/FT helpers (`adabn_full_training`, `windowing`, `ondevice_ft`)
that live in **`SilentWear/SilentWear/PyTorch_for_On_Device/`** and run inside the `agitated_hugle`
container against the SilentWear dataset. The copies here are a reproducibility snapshot; the live,
runnable versions are in that SilentWear directory (`ondevice_simulation/`).
