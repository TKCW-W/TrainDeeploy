# exp5 — ZO (MeZO / zeroth-order) round-1 on-device fine-tuning — RESULT

Complete first-round incremental fine-tuning of SpeechNet on Siracusa/GVSoC using **forward-only zeroth-order
(MeZO)** optimization: train on batch 1 (S01, session 3, vocalized, fold 3), dump device weights → carry
checkpoint, evaluate on the **whole batch 2**. Trains on the **exact same 54 windows as BP exp4** (verified
byte-identical) — the two experiments differ only in the optimization method / graph.

## Result — PASS
- **Batch-2 balanced accuracy: 0.8833 (88.33%)** — 159/180, all 9 classes (20/class).
- **All 180 eval windows bit-exact** to the ORT reference (`Errors: 0/9` each).
- **Training fully bit-exact: `0 / 21,600` forward-pass errors over the entire 200-epoch round.** No drift at
  all (ZO is forward-only — no MaxPool backward, so none of the device-vs-ORT argmax tie-flips that BP shows).
- Exceeds the BP device round (87.22%), the exp18 PyTorch ZO sim (87.36%), and approaches/edges the paper
  target (87.64%).

## Recipe (exp18 PyTorch recipe, validated)
All 22 trainable params (10 conv + 10 BN γ/β + 2 fc), frozen-BN, **200 epochs**, 54 stratified windows
(6/class, seed 42) → **2700 device update steps**, n_accum 4, **lr 3e-6, ε 0.01, q 1, seed 42**.

## Latency (faithful — uint64 counter)
The ZO harness now prints a `uint64` hi/lo split (a `uint32` counter wraps ~90× on this round). From the log:
`train_cycles = 89·2³² + 2,670,025,895 = 384.92 G`; `opt = 388.4 M` → **≈ 385.3 G cycles ≈ 17.35 min @ 370 MHz**
(≈ 26.8 min @ 240 MHz). Matches the per-forward extrapolation (21,600 × 17.8 M) exactly.

## Artifacts
- `logs/round1_gvsoc_zo.log` — 2700-step device train, `[WDUMP s=2699 …]` (22 tensors), faithful `BENCH`.
- `results/carry_zo_b1_fold3.pt` — 22 device-trained tensors over the official ckpt (BN stats frozen; max move 1.15e-2).
- `results/eval_b2.log`, `results/eval_b2_results.json` — per-window bit-exactness + balanced accuracy.
- `extract_zo_weights.py` — the ZO WDUMP→carry extractor (mirrors the BP script).

## New infrastructure added for this run (all `-- QW`, on `feat/BP+ZO`)
- `deeploymezotest.c`: `dump_zo_weights()` under `-D DUMP_WEIGHTS=ON` (FPU-free hex; the CMake MEZO passthrough
  already existed) + `uint64` cycle counters printed as hi/lo halves. The MeZo runner needed no change
  (`-D DUMP_WEIGHTS=ON` flows through the existing `-D` passthrough). Dump validated single-step against the
  reference updated weights to 5.96e-08.

## BP (exp4) vs ZO (exp5) — round 1, same 54 windows, same batch-2 eval
| | BP (first-order) | ZO (MeZO) |
|---|--:|--:|
| epochs / device steps | 40 / 540 | 200 / 2700 |
| forward/backward | fwd+bwd | forward-only (2 fwd/sample) |
| training bit-exactness | 416 / 2160 (MaxPool argmax drift, expected) | **0 / 21600 (perfect)** |
| round latency @370 MHz | ≈ 3.4 min | ≈ 17.35 min (~5×, = 5× epochs) |
| per-step @ n_accum 4 | 130.8 M cyc | 142.6 M cyc |
| **batch-2 balanced accuracy** | **87.22%** | **88.33%** |
| L2 peak (single-step) | 1.511 MB (argmax) | 0.790 MB (forward-only) |

**Takeaways:** ZO matches/slightly beats BP accuracy on this round (88.33 vs 87.22) while being **fully
bit-exact on device** (no MaxPool-drift) and using **~half the L2** (no gradient/activation stash) — at the
cost of ~5× wall-clock (driven by 5× epochs; per-step cost is comparable since ZO trades the backward for a
second forward). The forward-only nature is exactly why ZO avoids the MaxPool argmax tie-flips that give BP its
416/2160 training drift.
