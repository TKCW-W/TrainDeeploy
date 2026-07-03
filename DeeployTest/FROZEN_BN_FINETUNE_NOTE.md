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

## TODO (deferred)
- Rename the "GPU" ablation script/wording → "host PyTorch (CPU)" (no CUDA in this env; runs were
  full-precision CPU, numerically GPU-equivalent). File: `speechnet_full_gpu_ablation.py`.
- Optionally confirm the best config **on-device** (full-train the BN-folded graph, extract weights,
  eval batch-2) and check drift.
