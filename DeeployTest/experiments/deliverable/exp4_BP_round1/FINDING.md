# exp4 — BP (first-order) round-1 on-device fine-tuning — RESULT

Complete first-round incremental fine-tuning of SpeechNet on Siracusa/GVSoC: train on batch 1 (S01,
session 3, vocalized, fold 3), dump device weights → carry checkpoint, evaluate on the **whole batch 2**.

## Result — PASS
- **Batch-2 balanced accuracy: 0.8722 (87.22%)** — 157/180, all 9 classes (20/class).
- **All 180 eval windows bit-exact** to the ORT reference (`Errors: 0/9` each) → the device-trained model's
  on-device inference == its reference inference.
- Matches the documented reference exactly (`experiments/exp1/ondevice_sim_S01_fold3/FINDING.md`: b2 = 87.22%).

## Recipe (S2, full-model frozen-BN)
Full model (conv + BN γ/β + fc), frozen-BN, SGD lr 3e-4, n_accum 4 (SUM), **40 epochs**, 54 stratified
windows (6/class, seed 42) → **540 device update steps**, MaxPool **argmax-mask**, L1 128000 / L2 1500000 (GAP9).

## Artifacts
- `logs/round1_gvsoc_train.log` — 540-step device train (with `[WDUMP s=539 …]` weight dump, `DUMP_WEIGHTS=ON`).
- `results/carry_fullfrozen_b1_fold3.pt` — the 22 device-trained tensors over the official ckpt (BN stats frozen).
- `results/eval_b2.log`, `results/eval_b2_results.json` — per-window bit-exactness + balanced accuracy.

## Notes
- **`416 / 2160` per-step "errors" during training are expected, not a failure.** They are the inherent
  device-vs-ORT MaxPool argmax tie-flips (fp reduction-order differences flip a pooling-window max), onset
  ≈ step 133. The dumped weights remain valid and, as shown, yield the reference accuracy. Accuracy (not
  per-step training loss) is the acceptance criterion (`BP_FLOW.md §B.5`).
- **Latency:** `BENCH train_cycles` in the log is a `uint32` that **overflowed** (round ≈ 75.8 G cycles, ~17×
  past 2^32; printed value is the wrapped remainder). Faithful per-step from the single-step deliverable:
  ~130–140 M cyc/step × 540 ≈ **75.8 G cycles ≈ 3.4 min on GAP9 @ 370 MHz**. (The ZO harness now prints a
  `uint64` hi/lo split to avoid this; the BP harness could take the same 2-line fix if re-run.)
