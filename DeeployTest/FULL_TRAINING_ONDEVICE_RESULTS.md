# Full-training under on-device constraints — results & conclusions

**Experiment.** Under the on-device constraints (SGD, no momentum, batch-1 forward), run **full
training** (all 22 params, like the paper) with `n_accum=8`, `data_size=54` (30%), `lr=1e-3`,
`n_epochs=20` → **1080 forwards / 135 SGD steps** (12× the original 90-forward run). Generate the
real Deeploy training graph, run it **on-device (GVSoC, tiled, 8 cores)** with weight dumping,
extract the actual device final weights, rebuild the infer graph, evaluate balanced accuracy on the
180 batch-2 windows. Goals: (1) does the on-device-constrained setup improve accuracy effectively?
(2) does the precision drift wreck the result at larger step counts, or is it acceptable?

## Headline result — full-training **regresses** on-device

Evaluated via the validated PyTorch path (zero-shot reproduces the established 78.33%). All four
weight/running-stat combinations on the **same** batch-2 windows:

| # | trained weights | running-stats (BN) | batch-2 bal-acc | Δ vs zero-shot |
|---|---|---|---|---|
| A | pretrained | pretrained | 78.33% | — (zero-shot) |
| B | ORT-sim | **ORT-updated** | 81.11% | **+2.78pp** ← host-sweep number (optimistic) |
| C | ORT-sim | frozen (pretrained) | 63.33% | −15.00pp |
| D | device | ORT-updated | 78.33% | +0.00pp |
| **E** | **device** | **frozen (pretrained)** | **60.56%** | **−17.78pp** ← ACTUAL on-device |

**Decomposition of the gap (B → E):**

| effect | rows | cost | nature |
|---|---|---|---|
| **BN running-stat non-update** | B → C | **−17.78pp** | architectural/kernel limitation (dominant) |
| **precision drift** (argmax tie-flip / fp) | C → E | **−2.78pp** | numerical, small & bounded |
| **net on-device full-training** | A → E | **−17.78pp** | **regression** |

## Goal 1 — Does the on-device-constrained full-training improve accuracy? **No — it regresses (−17.78pp).**

The actual on-device result is **60.56%**, i.e. **−17.78pp below zero-shot**. Full-model FT is
fundamentally broken on this device for two stacked reasons, both BatchNorm:

1. **Batch-1 forward** → the training-mode BN (`BatchNormInternal`) normalizes each window by its
   own spatial statistics (degenerate).
2. **Running stats are never updated on-device** (the kernel's EMA path is dead). So the BN affine
   params γ/β and the conv filters get trained against batch-1 normalization, but at inference the
   graph normalizes with the **frozen pretrained running stats** → a large train/inference
   mismatch that costs **−17.78pp** all by itself (row B→C).

Row **D is the tell**: device weights paired with *correct* (updated) running stats give exactly
**78.33% = zero-shot** — the trained weights themselves are essentially harmless; **100% of the
damage is the running-stat mismatch**, which the device cannot avoid.

### The host sweep was optimistic — a methodological correction
The host/ORT sweep reported **+2.78pp** (row B) because its evaluator overrode the running stats
with the **ORT-sim-updated** values (`outputs.npz` contains them — ORT's BatchNormInternal *does*
update running stats, with `running_var` drifting by up to 4×10⁴). The **device kernel does not**
update running stats, so the device-realistic evaluation is row C/E. **This means the earlier
full-model ORT sweep (best "+1.11pp") was also optimistic; the real device full-model FT is
negative.** (Head-only/folded results are unaffected — folding removes BN entirely, so there are
no running stats to diverge; the +4.44 pp there is device-exact and stands.)

### vs SilentWear (+8.33pp)
SilentWear reaches +8.33pp because host training gives it **both** things the device lacks:
batch-32 (proper BN statistics) **and** running-stat EMA updates (proper BN train/infer
consistency), plus Adam. The device can do **neither** (batch pinned to 1 by L2; RS update path
dead), so full-model FT regresses. This is the on-device-vs-paper gap, made concrete.

## Goal 2 — Does the precision drift wreck the result at larger step counts? **No — it is small and bounded.**

Over **1080 forwards** (12× the original 90-forward run), the device final weights stayed very
close to the drift-free ORT reference:

- **Per-weight drift:** worst **5.75%** relative (one conv layer), most layers **<1%**, biases
  ~1e-9. No divergence/explosion with more steps.
- **Accuracy cost of drift:** only **−2.78pp** (row C→E, and identically B→D). The argmax tie-flips
  diverge from ORT but do **not** systematically pick wrong gradients — the device trajectory
  tracks ORT closely. **This directly confirms the user's hypothesis:** a tie-flip is just a
  divergence from ORT's arbitrary tie-break, not a wrong gradient, and it stays bounded.

## The key reframe

For **full-model** on-device FT, the precision drift is a **red herring** (−2.78pp, bounded). The
real killer is the **BN running-stat non-update** (−17.78pp) — an architectural limitation, not a
numerical one. This is exactly why **head-only + BN-fold** is the correct design: folding bakes the
pretrained running stats into Conv and deletes BN from the training graph, so there is **nothing to
update and no batch-1 mismatch** — making head-only simultaneously drift-free *and*
running-stat-safe, hence the **+4.44pp that actually holds on-device**.

## Reproduction

```bash
# (1) generate full-training fixture (no --training-strategy => full; BN stays as BatchNormInternal)
docker exec agitated_hugle bash -lc "cd /app/Onnx4Deeploy && python3 Onnx4Deeploy.py \
  -model SpeechNet -mode train -o <Tests>/speechnet_train_full8_e20 \
  --dataset silentwear --data-path <DATA> --pretrained-weights <CKPT> \
  --subject S01 --session 3 --batch 1 --condition vocalized \
  --stratified --data-size 54 --n-epochs 20 --n-accum 8 --lr 0.001"
# (2) on-device run + weight dump (1080 forwards / 135 steps)
docker exec traindeeploy bash -lc "cd /app/ETH/TrainDeeploy/DeeployTest && rm -rf TEST_SIRACUSA && \
  python deeployTrainingRunner_tiled_siracusa.py -t Tests/.../speechnet_train_full8_e20 \
  --n-steps 135 --n-accum 8 --cores 8 --l1 128000 --l2 2000000 \
  --memAllocStrategy MiniMalloc --searchStrategy random-max -D DUMP_WEIGHTS=ON > full8_e20_device.log"
# (3) decomposition (A/B/C/D/E) — extract 22 device weights from [WDUMP], eval 4 weight×RS combos
#     speechnet_full_device_verify.py  (drift table + ONNX-path cross-check)
#     + the inline A/B/C/D/E decomposition (PyTorch path; zero-shot must reproduce 78.33%)
```

Artifacts: `full8_e20_device.log` (device WDUMP), `speechnet_ft_full_naccum8_sweep.{py,json,log}`
(host sweep), `speechnet_full_device_verify.py` (drift + reconstruction),
`Tests/Models/Training/SpeechNet/speechnet_train_full8_e20/` (fixture).
