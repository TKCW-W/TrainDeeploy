# Frozen-stat BN fine-tuning — a deployable full-FT path (saved for later)

## The idea (user's insight, confirmed)
Instead of BatchNorm computing **per-window (batch-1) statistics** during training — which caused
the −17.78 pp on-device regression via a train/inference mismatch — **normalize by the frozen
pretrained running statistics during training too** (BN in eval mode). Then train == inference,
batch-1 is no longer a problem (no per-window stats at all), and the whole thing is batch-independent.

This is **FrozenBatchNorm / eval-mode-BN fine-tuning** (a known small-batch transfer-learning trick).

## First result (host PyTorch, CPU; ep40, lr1e-3, 54 windows/30%, n_accum 8, summing conv)
| BN mode (batch-1, full training) | batch-2 | Δ |
|---|---|---|
| batch-stat BN (per-window) — broken | 49–55% | −17.78 |
| **frozen-stat BN, train all** | **81.67%** | **+3.33** |
| head-only + BN-fold (shipped) | 82.78% | +4.44 |
| true-batch ceiling | 85.56% | +7.22 |

Flipping the normalization source alone swings −17.78 → +3.33 pp (~21 pp) — confirms the diagnosis.

## Why it is deployable at batch-1
Folding the frozen-stat BN into Conv (frozen μ,σ,γ,β) yields a **BN-free** Conv-ReLU-Pool graph.
Training *all* the conv weights + fc on that graph is batch-independent, needs no running-stat
update, no two-pass BN, no batch>1. Forward = the inference kernel; backward has no cross-batch
coupling. **This is "full training on the BN-folded graph"** — the untested middle option between
our shipped head-only-on-folded-graph and full-training-on-unfolded-graph. Only remaining on-device
concern: MaxPool argmax drift from training conv (shown bounded & non-directional).

## Caveats seen so far
- ~= or slightly below head-only+fold (+3.33 vs +4.44) on only 54 windows; conv adaptation may
  overfit and re-adds drift. Promise is with **more data** (this search).
- lr-fragile: lr 5e-3 (summing, n_accum 8 → eff 4e-2) collapsed to 11%. Needs careful lr.

## Grid-search results (host CPU, `speechnet_frozenbn_gridsearch.py`, 60 configs, ep40)
lr × n_accum × data-size {30/40/50/70%}, averaging accumulation (lr = true step; device summing
lr = lr/n_accum). zero-shot = 78.33%.

**Best = +5.56 pp (83.89%)** — reached at BOTH `30% / lr 2e-3 / n_accum 1` and `70% / lr 5e-3 /
n_accum 1` (two independent points → a real ~+5.5 pp ceiling). **Beats head-only+fold (+4.44 pp)**,
and is the best *deployable* (batch-1) result so far.

Per-axis benefit:
- **lr — dominant.** Sweet spot **1e-3–5e-3**; ≥1e-2 collapses at low data, ≥2e-2 collapses
  everywhere. More data widens the safe-lr range (70% tolerates 5e-3–1e-2; 30% only ≤2e-3).
- **data-size — nearly flat.** best per size 30→+5.56, 40→+4.44, 50→+5.00, 70→+5.56. **More data
  does NOT raise the ceiling** — 30% already optimal; it only buys lr headroom.
- **n_accum — minor** (±1–2 pp, within ~±1.7 pp noise; n_accum=1 fine).

**Recommended config: 30% data, lr 2e-3, n_accum 1, frozen-stat BN, ep40 → +5.56 pp** (device lr
2e-3) — least data/compute, ties the best. vs head-only+fold +4.44, true-batch ceiling +7.22,
paper +8.33. Caveat: +5.56 vs +4.44 is ~1.1 pp (~2 windows) — within noise, so "beats head-only"
is suggestive; it is at least as good AND adapts conv, at the cost of lr-fragility.

Artifacts: `frozenbn_grid.log`, `speechnet_frozenbn_gridsearch.json`.

## Variance study — the grid's "+5.56 beats head-only" was a LUCKY DRAW (corrected)
`speechnet_frozenbn_variance.py`: 8 independent stratified 30% draws (6/class from the full 180-pool)
× 2 seeds, training BOTH recipes per draw (frozen-BN, batch-1, lr 2e-3, n_accum 1, ep40). Eval batch-2.

| recipe | mean Δ | std | range | never neg? |
|---|---|---|---|---|
| frozen-BN **full-FT** | **+2.60** | **3.04** | −2.78 … +7.78 | no (3 negative draws) |
| **head-only** | **+2.01** | **1.06** | +0.56 … +4.44 | **yes** |
| paired (full − head) | **+0.59** | 3.73 | — | wins **10/16** only |

**Verdict: full-FT does NOT robustly beat head-only.** Mean edge +0.59 pp (within noise), wins 62% of
draws, and **3× the variance** (unstable — result depends on which windows are sampled). The single-draw
+5.56 (grid) and +8.33 (fine lr sweep, lr 2.5e-3) are favorable draws, not reproducible. Finer lr sweep
on one draw is non-monotonic/jumpy (1e-3→+7.22, 2e-3→+4.44, 2.5e-3→+8.33, 4e-3→−2.78 collapse) — a
symptom of the high-variance regime.

**Conclusion:** frozen-stat BN full-FT is a valid BN fix (turns −17.78pp regression into +2.60 mean,
deployable at batch-1), but on this tiny FT set adapting conv is a high-variance gamble with **no
reliable gain** over head-only and it re-adds MaxPool drift. **Head-only + BN-fold stays the better
deployable choice** — reliable, low-variance, properly-tuned +4.44 pp exceeds full-FT's mean.
Artifacts: `frozenbn_variance.log`, `speechnet_frozenbn_variance.py`.

## Fair head-to-head — each recipe at its OWN best lr (`speechnet_frozenbn_fair.py`)
Same 8 draws × 2 seeds; each recipe swept over a small lr set, compared at its best-mean lr.

| recipe (best lr) | mean Δ | std | range |
|---|---|---|---|
| head-only @ 5e-3 | +3.40 | **0.52** | +2.22 … +3.89 |
| full-FT @ 1e-3 | **+3.99** | 2.95 | −1.67 … +8.89 |
| paired (full − head) | **+0.59** | 2.99 | full wins **9/16** |

**Refined verdict (risk/reward, not a clear winner):** full-FT has a marginally higher *mean*
(+3.99 vs +3.40, edge +0.59 pp — within noise, wins 9/16) but **~6× the variance** (std 2.95 vs 0.52,
can go negative). head-only is **rock-solid** (±0.52, never negative). So: full-FT = slightly higher
expected value / high variance; head-only = slightly lower / very reliable → **head-only is the safe
deployable default; full-FT only if you can validate per-deployment and want the higher ceiling.**
Both need lr tuning. Realistic expected gain for this task ≈ **+3.4–4.0 pp** (the single-draw
+4.44/+5.56/+8.33 peaks were all optimistic draws).
Artifacts: `frozenbn_fair.log`, `speechnet_frozenbn_fair.py`.

## Gradient-accumulation convention in TrainDeeploy = SUMMING (code-verified)
- `FloatInPlaceAccumulatorV2Template.py`: `accum_buffer += gradient` (first micro-batch resets, rest
  add) → `accum = Σ gradient_i`. `SGDTemplate.py`: `weight -= lr * grad_acc` — **no /n_accum**.
  Loss grad = 1/batch_size = 1 at batch-1. ⇒ **effective LR = lr × n_accum.**
- Consequence (device-faithful summing sweep, frozen-BN full-FT, 30% draw seed1000, ep40):
  - **[A] fix lr=1e-3, raise n_accum** → eff_lr = 1e-3/4e-3/8e-3 → 85.56 / 86.11 / **43.89% (collapse)**.
    n_accum is NOT a free knob: raising it at fixed lr blows up eff_lr and collapses.
  - **[B] matched eff_lr≈1e-3** (lr=1e-3/n_accum) → n1/n4/n8 = 85.56 / 87.22 / 86.67% — n_accum>1 is
    fine and marginally better (smoothing) when lr is compensated. (The averaging grid = view [B].)
- **Deployment rule:** keep eff_lr = lr × n_accum in ~1e-3…4e-3 (collapse ≳8e-3). For n_accum>1,
  divide the baked lr by n_accum. n_accum=1 needs no compensation. (Draw seed1000 is favorable →
  read the pattern, not the absolute numbers.)

## FINAL full-training deployment config = KERNEL MODIFICATION (frozen-stat BN)
Chosen path: modify `BatchNorm.c` so the training BN normalizes with the FROZEN pretrained running
stats, gated by `-D BN_FROZEN_STATS`. (The fold-into-Conv alternative was DROPPED — folding rescales
conv weights by γ/σ and, with block-0's huge running_var, collapses at the normal lr. The kernel mod
keeps γ/β at natural scale, so the validated lr transfers.) Commit `e87291b`.

- `PULP_BatchNormInternal_fp32` (fwd): frozen branch uses `running_mean/var`, saves them.
- `PULP_BatchNormGrad_fp32` (bwd): affine gradient `dX = γ·inv_std·dY` (no batch-stat Jacobian terms).
- Runtime `g_bn_frozen_stats` (default 0 = unchanged); harness sets it under `#ifdef BN_FROZEN_STATS`.
- **Validated bit-exact:** 20-step device run vs host frozen-BN reference → worst weight drift 0.000%.

```
# unfolded full-training fixture (keeps BatchNormInternal — the modified kernel handles it)
Onnx4Deeploy.py -model SpeechNet -mode train -o <Tests>/speechnet_train_fullfrozen \
  --dataset silentwear --data-path <DATA> --pretrained-weights <CKPT> \
  --subject S01 --session 3 --batch 1 --condition vocalized \
  --stratified --data-size 54 --n-epochs 40 --n-accum 1 --lr 0.001 --training-strategy full
# on-device train with frozen-stat BN + weight dump  (2160 forwards)
deeployTrainingRunner_tiled_siracusa.py -t <Tests>/speechnet_train_fullfrozen \
  --n-steps 2160 --n-accum 1 --cores 8 --l1 128000 --l2 2000000 \
  --memAllocStrategy MiniMalloc --searchStrategy random-max -D BN_FROZEN_STATS=ON -D DUMP_WEIGHTS=ON
```
Config: full training, batch 1, **n_accum 1, lr 0.001** (eff_lr 1e-3), data 54 (30%), ep40 → 2160 forwards.
Rules: eff_lr = lr×n_accum ∈ ~1e-3..4e-3; for n_accum>1 divide lr. Expected ~+4pp (sim +3.99±2.95, high
variance — validate the run). Validation harness: `speechnet_fullfrozen_validate.py`.

## TODO (deferred)
- Rename the "GPU" ablation script/wording → "host PyTorch (CPU)" (no CUDA in this env; runs were
  full-precision CPU, numerically GPU-equivalent). File: `speechnet_full_gpu_ablation.py`.
- ~~Onnx4Deeploy `--fold-bn` flag~~ DROPPED (superseded by the kernel modification).
